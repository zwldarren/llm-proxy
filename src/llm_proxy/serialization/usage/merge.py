"""How two observations of usage combine.

Two rules, deliberately separate because they answer different questions:

- :func:`merge_latest` — *one* provider response reported more than once. A
  streaming turn splits its accounting across events (Anthropic sends prompt
  tokens in ``message_start`` and completion tokens in ``message_delta``), and
  each event carries a cumulative snapshot. A field the later observation
  carries wins; a field it omits keeps its earlier value. Not a sum: summing
  cumulative snapshots double-counts.
- :func:`add_usage` — *two* provider responses are being billed together (the
  web-search continuation's second call). The turns are independent billed
  calls, so the facts add.

Both live here because the merge rule is part of the canonical record's
contract, not of whichever transformer happens to hold the accumulator.
"""

from __future__ import annotations

from dataclasses import fields, replace

from llm_proxy.models.types import CompletionTokensDetails, PromptTokensDetails, Usage

# Scalar facts a later observation may restate. ``total_tokens`` is included:
# a provider that reports it verbatim keeps ownership of it.
_SCALAR_FIELDS: tuple[str, ...] = tuple(
    field.name
    for field in fields(Usage)
    if field.name not in {"prompt_tokens_details", "completion_tokens_details"}
)

# Facts that add when two independent provider calls are billed together
# (the rule ``sum_usage`` implemented before this module owned it).
_SUMMED_FIELDS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
    "web_search_requests",
    # An audio endpoint billed per second: two independent calls transcribed two
    # durations, and their sum is what the provider charged.
    "audio_duration_seconds",
    # Two billed calls each reported their own cost: their sum is what the two
    # calls cost. Dropping it would leave a record whose tokens cover both
    # calls while its cost covers one, and cost.py treats a reported cost as
    # authoritative over the computed one.
    "provider_reported_cost",
)


# Accounting counters accumulate across a turn's events, so an observation
# that restates one can only raise it. Anthropic's ``message_delta`` repeats
# the prompt accounting without the cache counters reported beside it in
# ``message_start``; letting that restatement win would drop the cached prompt
# out of an inclusive ``input_tokens``.
_CUMULATIVE_FIELDS: frozenset[str] = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens",
        "web_search_requests",
    }
)


def merge_latest(target: Usage, source: Usage) -> None:
    """Fold a later observation of one response into ``target`` in place.

    Accounting counters take the larger of the two values (they are cumulative
    within a turn); every other field takes the later non-``None`` value.
    ``total_tokens`` is taken from the merged counters, because a dialect with no
    wire total (Anthropic) reports a per-event total that is only that event's
    partial sum — max-merging those leaves a total the merged counters contradict.
    A wire total larger than those counters keeps ownership of the field: a
    provider's own total may count dimensions the canonical counters do not
    (Gemini's ``totalTokenCount`` includes the tool-use prompt tokens a
    search-grounded turn bills through the per-request search fee instead), and
    recomputing it as ``input + output`` would undercount it.
    """
    for name in _SCALAR_FIELDS:
        if name == "total_tokens":
            # Handled below: taken with the others, a total synthesized from
            # one event's partial counters would win the max and then be
            # contradicted by the counters it was derived from.
            continue
        value = getattr(source, name)
        if value is None:
            continue
        if name in _CUMULATIVE_FIELDS:
            current = getattr(target, name)
            value = value if current is None else max(current, value)
        setattr(target, name, value)

    reported_total = getattr(source, "total_tokens", None)
    if target.input_tokens or target.output_tokens or reported_total is not None:
        # Cumulative counters never decrease, so the turn's largest total so far is
        # the turn's total: an event that omits the total does not withdraw it.
        target.total_tokens = max(
            target.total_tokens or 0,
            target.input_tokens + target.output_tokens,
            reported_total or 0,
        )

    target.prompt_tokens_details = _merge_details(
        target.prompt_tokens_details, source.prompt_tokens_details
    )
    target.completion_tokens_details = _merge_details(
        target.completion_tokens_details, source.completion_tokens_details
    )


def _merge_details[DetailsT: PromptTokensDetails | CompletionTokensDetails](
    target: DetailsT | None, source: DetailsT | None
) -> DetailsT | None:
    """Latest-wins merge for one nested details block, field by field."""
    if source is None:
        return target
    if target is None:
        return source
    for name, value in vars(source).items():
        if value is not None:
            setattr(target, name, value)
    return target


def _sum_details[DetailsT: PromptTokensDetails | CompletionTokensDetails](
    target: DetailsT | None, source: DetailsT | None
) -> DetailsT | None:
    """Field-by-field sum of one nested details block (two independent calls).

    The details blocks are not mirror expressions of the flat fields: they carry
    billable facts the record has nowhere else (``audio_tokens``,
    ``image_tokens``), and billing reads them, so a hop's details must add rather
    than be dropped. ``None`` on either side leaves the field as it is — unknown
    on one side of the sum is not zero — and two ``None``s stay ``None`` rather
    than collapsing into an empty block.
    """
    if source is None:
        return target
    if target is None:
        return source
    for name, value in vars(source).items():
        if value is None:
            continue
        current = getattr(target, name)
        setattr(target, name, value if current is None else current + value)
    return target


def copy_usage(usage: Usage) -> Usage:
    """A copy safe to keep: the record *and* its nested details blocks.

    ``dataclasses.replace`` alone would share the details objects, and both
    merge rules write into them, so a kept record would move under whichever
    caller still held it.
    """
    return replace(
        usage,
        prompt_tokens_details=(
            replace(usage.prompt_tokens_details) if usage.prompt_tokens_details else None
        ),
        completion_tokens_details=(
            replace(usage.completion_tokens_details) if usage.completion_tokens_details else None
        ),
    )


def add_usage(target: Usage, source: Usage) -> None:
    """Sum ``source`` into ``target`` in place (independent billed calls).

    A side that reported nothing contributes nothing: its ``None`` is "no known
    contribution", not an unknown that invalidates the other side's fact, so a
    value only one call reports is adopted rather than dropped (both ``None``
    stays ``None``). Reading it the other way made the sum order-dependent —
    ``add_usage(a, b)`` kept a fact that ``add_usage(b, a)`` lost. The nested
    details blocks are summed too, field by field: they carry billable facts of
    their own (``audio_tokens``, ``image_tokens``), not a mirror of the flat
    fields.
    """
    for name in _SUMMED_FIELDS:
        source_value = getattr(source, name)
        if source_value is None:
            continue
        target_value = getattr(target, name)
        setattr(
            target,
            name,
            source_value if target_value is None else target_value + source_value,
        )

    target.prompt_tokens_details = _sum_details(
        target.prompt_tokens_details, source.prompt_tokens_details
    )
    target.completion_tokens_details = _sum_details(
        target.completion_tokens_details, source.completion_tokens_details
    )
