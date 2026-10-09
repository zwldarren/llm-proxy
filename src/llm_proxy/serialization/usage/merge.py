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

from dataclasses import fields

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
    ``total_tokens`` is recomputed from the merged counters, because a dialect
    with no wire total (Anthropic) reports a per-event total that is only that
    event's partial sum — max-merging those leaves a total the merged counters
    contradict.
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
    if reported_total is not None and not target.input_tokens and not target.output_tokens:
        # A record that carries only a total keeps it (nothing contradicted it).
        target.total_tokens = reported_total
    elif target.input_tokens or target.output_tokens:
        target.total_tokens = target.input_tokens + target.output_tokens

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


def add_usage(target: Usage, source: Usage) -> None:
    """Sum ``source`` into ``target`` in place (independent billed calls).

    ``None`` on either side leaves the field untouched: unknown on one side of
    the sum is not zero.
    """
    for name in _SUMMED_FIELDS:
        target_value = getattr(target, name)
        source_value = getattr(source, name)
        if target_value is not None and source_value is not None:
            setattr(target, name, target_value + source_value)
