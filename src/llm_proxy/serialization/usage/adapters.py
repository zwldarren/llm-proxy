"""One adapter per provider wire shape.

The dialect is a wire *shape*, not a provider: one dialect serves many
providers (chat-completions covers OpenAI, OpenRouter, Ollama, vLLM, SGLang,
…), and the native-passthrough tier has no provider serializer in play at all.
An adapter owns two things:

- **its envelope** — where the usage block sits in the payload. Anthropic
  nests it under ``message`` on ``message_start`` and at the top level on
  ``message_delta``; Responses stream events nest it under ``response``;
  Realtime nests it two deep in a ``response.done`` event;
- **its aliases** — which dialect key expresses a canonical fact, folded at
  parse time so the record holds one field per fact (ADR-0006).

Extraction only: nothing here emits a client-facing shape, and nothing here
estimates. See ADR-0025.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from llm_proxy.models.types import CompletionTokensDetails, PromptTokensDetails, Usage
from llm_proxy.serialization.usage.folds import (
    fold_deepseek_cache_hits,
)

type UsageDialect = Literal["chat", "responses", "anthropic", "realtime"]

__all__ = ["UsageDialect", "parse_usage"]


# ---------------------------------------------------------------------------
# Tolerant readers — wire payloads carry whatever the provider felt like
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _int(value: Any) -> int | None:
    """An int the wire actually reported; booleans and floats are not counts."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _positive_int(value: Any) -> int | None:
    """A count the wire actually reported, and one that is a count (> 0)."""
    count = _int(value)
    return count if count is not None and count > 0 else None


def _web_search_requests(server_tool_use: Any) -> int | None:
    """``server_tool_use.web_search_requests`` — provider-reported search count."""
    block = _as_dict(server_tool_use)
    if block is None:
        return None
    return _positive_int(block.get("web_search_requests"))


def _prompt_details(raw: Any) -> PromptTokensDetails | None:
    details = _as_dict(raw)
    if details is None:
        return None
    return PromptTokensDetails(
        audio_tokens=_int(details.get("audio_tokens")),
        cached_tokens=_int(details.get("cached_tokens")),
        image_tokens=_int(details.get("image_tokens")),
        text_tokens=_int(details.get("text_tokens")),
        cache_write_tokens=_int(details.get("cache_write_tokens")),
        video_tokens=_int(details.get("video_tokens")),
    )


def _completion_details(raw: Any) -> CompletionTokensDetails | None:
    details = _as_dict(raw)
    if details is None:
        return None
    return CompletionTokensDetails(
        accepted_prediction_tokens=_int(details.get("accepted_prediction_tokens")),
        audio_tokens=_int(details.get("audio_tokens")),
        reasoning_tokens=_int(details.get("reasoning_tokens")),
        rejected_prediction_tokens=_int(details.get("rejected_prediction_tokens")),
        image_tokens=_int(details.get("image_tokens")),
    )


def _lift_cache_facts(
    usage: dict[str, Any],
    prompt_details: PromptTokensDetails | None,
) -> tuple[int | None, int | None, PromptTokensDetails | None]:
    """Fold the OpenAI-dialect cache aliases into the canonical flat fields.

    Returns ``(cache_read, cache_write, prompt_details)``. The flat field wins
    and the nested copy is cleared, because the same tokens in both fields are
    priced twice downstream. Client emission already tolerates the flat-only
    form: the OpenAI protocol formatter falls back to
    ``usage.cache_read_input_tokens`` and OpenResponses takes the max of the
    two (``protocols/openai/formatting.py``,
    ``protocols/openresponses/serializer.py``).
    """
    cache_read = _int(usage.get("cache_read_input_tokens"))
    if cache_read is None:
        nested_cached = prompt_details.cached_tokens if prompt_details is not None else None
        cache_read = fold_deepseek_cache_hits(nested_cached, usage.get("prompt_cache_hit_tokens"))
    if cache_read is not None and prompt_details is not None:
        prompt_details.cached_tokens = None

    cache_write = _int(usage.get("cache_creation_input_tokens"))
    if cache_write is None and prompt_details is not None:
        cache_write = prompt_details.cache_write_tokens
    if cache_write is not None and prompt_details is not None:
        prompt_details.cache_write_tokens = None

    # A block whose only facts were the lifted ones says nothing: an all-``None``
    # details object is not ``None`` to a caller testing truthiness, so leaving
    # it behind would hand downstream code an empty nested block to emit.
    if prompt_details is not None and all(value is None for value in vars(prompt_details).values()):
        prompt_details = None

    return cache_read, cache_write, prompt_details


def _lift_reasoning(
    usage: dict[str, Any],
    completion_details: CompletionTokensDetails | None,
) -> tuple[int | None, CompletionTokensDetails | None]:
    """Fold the thinking aliases into the canonical flat ``reasoning_tokens``.

    Thinking tokens are one fact, and the canonical record keeps it flat
    (CONTEXT.md -> Canonical usage record). Dialects express it nested (OpenAI's
    and the Responses API's ``completion_tokens_details.reasoning_tokens``) or
    top-level (SGLang), so both are read and the alias is cleared — the same
    treatment the cache facts get, for the same reason: an alias left in place is
    a second expression of one fact that a reader may pick up twice.

    Returns ``(reasoning, completion_details)``: the details block keeps the
    details that have no canonical field of their own, and is dropped when nothing
    is left.
    """
    nested = completion_details.reasoning_tokens if completion_details is not None else None
    reasoning = _int(nested)
    if reasoning is None:
        # SGLang reports the count top-level, and reports it as zero by default, so
        # only a positive one is a report; an explicit nested value wins.
        reasoning = _positive_int(usage.get("reasoning_tokens"))
    if reasoning is not None and completion_details is not None:
        completion_details.reasoning_tokens = None
        if all(value is None for value in vars(completion_details).values()):
            completion_details = None
    return reasoning, completion_details


def _provider_reported_cost(usage: dict[str, Any]) -> float | None:
    """Cost the provider itself reported (NanoGPT's ``nanogpt_cost``, OpenRouter's ``cost``)."""
    for key in ("nanogpt_cost", "cost"):
        value = usage.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _count_items(payload: dict[str, Any], item_type: str) -> int:
    """Count response items of one type in a non-streaming response body."""
    count = 0
    for container in (payload, _as_dict(payload.get("response")) or {}):
        items = container.get("output")
        if not isinstance(items, list):
            continue
        count += sum(
            1 for item in items if _as_dict(item) is not None and item.get("type") == item_type
        )
    return count


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


def _parse_chat(payload: Mapping[str, Any]) -> Usage | None:
    block = _as_dict(payload.get("usage"))
    if block is None:
        return None
    usage = dict(block)
    prompt_details = _prompt_details(usage.get("prompt_tokens_details"))
    cache_read, cache_write, prompt_details = _lift_cache_facts(usage, prompt_details)
    reasoning, completion_details = _lift_reasoning(
        usage, _completion_details(usage.get("completion_tokens_details"))
    )

    input_tokens = _int(usage.get("prompt_tokens")) or 0
    output_tokens = _int(usage.get("completion_tokens")) or 0
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_int(usage.get("total_tokens")) or (input_tokens + output_tokens),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        reasoning_tokens=reasoning,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
        provider_reported_cost=_provider_reported_cost(usage),
        web_search_requests=_web_search_requests(usage.get("server_tool_use")),
    )


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------


def _anthropic_usage_block(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """``message_start`` nests usage under ``message``; everything else is flat."""
    message = _as_dict(payload.get("message"))
    if message is not None:
        block = _as_dict(message.get("usage"))
        if block is not None:
            return block
    return _as_dict(payload.get("usage"))


def _parse_anthropic(payload: Mapping[str, Any]) -> Usage | None:
    usage = _anthropic_usage_block(payload)
    if usage is None:
        return None
    # Absent means "not reported", so both counters stay ``None``; the inclusive
    # input count needs a number, hence the local zero-default.
    cache_read = _int(usage.get("cache_read_input_tokens"))
    cache_write = _int(usage.get("cache_creation_input_tokens"))
    # The canonical invariant makes input_tokens INCLUDE the cache counters:
    # Anthropic reports them beside it, and emission subtracts them again.
    input_tokens = (_int(usage.get("input_tokens")) or 0) + (cache_read or 0) + (cache_write or 0)
    output_tokens = _int(usage.get("output_tokens")) or 0
    output_details = _as_dict(usage.get("output_tokens_details"))
    reasoning_tokens = _int(output_details.get("thinking_tokens")) if output_details else None

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        reasoning_tokens=reasoning_tokens,
        web_search_requests=_web_search_requests(usage.get("server_tool_use")),
    )


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------


def _responses_usage_block(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """A response body is flat; a stream event nests usage under ``response``."""
    response = _as_dict(payload.get("response"))
    if response is not None:
        block = _as_dict(response.get("usage"))
        if block is not None:
            return block
    return _as_dict(payload.get("usage"))


def _parse_responses(payload: Mapping[str, Any]) -> Usage | None:
    # The Responses wire has no usage-level search count: a response body
    # reports its calls as output items, so the items are the count whenever the
    # provider did not report one itself. A stream event announces its calls the
    # same way (:func:`_count_responses_events`).
    counted = _count_items(dict(payload), "web_search_call")
    usage = _responses_usage_block(payload)
    if usage is None:
        # Native web search calls bill per request even with no usage object.
        return Usage(web_search_requests=counted) if counted else None

    input_details = _as_dict(usage.get("input_tokens_details")) or _as_dict(
        usage.get("prompt_tokens_details")
    )
    output_details = _as_dict(usage.get("output_tokens_details")) or _as_dict(
        usage.get("completion_tokens_details")
    )
    prompt_details = _prompt_details(input_details)
    cache_read, cache_write, prompt_details = _lift_cache_facts(usage, prompt_details)
    reasoning, completion_details = _lift_reasoning(usage, _completion_details(output_details))
    input_tokens = _int(usage.get("input_tokens")) or 0
    output_tokens = _int(usage.get("output_tokens")) or 0

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_int(usage.get("total_tokens")) or (input_tokens + output_tokens),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        reasoning_tokens=reasoning,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
        provider_reported_cost=_provider_reported_cost(usage),
        web_search_requests=_positive_int(usage.get("web_search_requests")) or counted or None,
    )


def _count_responses_events(payload: Mapping[str, Any]) -> int:
    """Billable events a Responses stream announces as they happen.

    Only ``response.output_item.added`` announces a call — the same rule the
    converted tier counts by. The terminal ``response.completed`` event
    restates every item in ``response.output``, so counting that array here
    would double every search; a non-streaming body is counted by
    :func:`_parse_responses` from its own array instead.
    """
    if payload.get("type") != "response.output_item.added":
        return 0
    item = _as_dict(payload.get("item"))
    return 1 if item is not None and item.get("type") == "web_search_call" else 0


# ---------------------------------------------------------------------------
# Realtime
# ---------------------------------------------------------------------------


def _realtime_usage_block(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    response = _as_dict(payload.get("response"))
    if response is not None:
        block = _as_dict(response.get("usage"))
        if block is not None:
            return block
    return _as_dict(payload.get("usage"))


def _parse_realtime(payload: Mapping[str, Any]) -> Usage | None:
    usage = _realtime_usage_block(payload)
    if usage is None:
        return None
    # The Realtime dialect nests under singular "token" keys.
    input_details = _as_dict(usage.get("input_token_details")) or _as_dict(
        usage.get("input_tokens_details")
    )
    output_details = _as_dict(usage.get("output_token_details")) or _as_dict(
        usage.get("output_tokens_details")
    )
    prompt_details = _prompt_details(input_details)
    cache_read, cache_write, prompt_details = _lift_cache_facts(usage, prompt_details)
    reasoning, completion_details = _lift_reasoning(usage, _completion_details(output_details))
    input_tokens = _int(usage.get("input_tokens")) or 0
    output_tokens = _int(usage.get("output_tokens")) or 0

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_int(usage.get("total_tokens")) or (input_tokens + output_tokens),
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=_completion_details(output_details),
        audio_duration_seconds=usage.get("audio_duration_seconds"),
        web_search_requests=_positive_int(usage.get("web_search_requests")),
    )


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DialectAdapter:
    """What a dialect adapter must answer."""

    parse: Callable[[Mapping[str, Any]], Usage | None]
    #: Billable events the payload announces outside its usage block (a
    #: streaming terminal event may never restate them). Summed per turn.
    count_events: Callable[[Mapping[str, Any]], int] = lambda _payload: 0


_ADAPTERS: dict[str, DialectAdapter] = {
    "chat": DialectAdapter(_parse_chat),
    "responses": DialectAdapter(_parse_responses, _count_responses_events),
    "anthropic": DialectAdapter(_parse_anthropic),
    "realtime": DialectAdapter(_parse_realtime),
}


def parse_usage(dialect: UsageDialect, payload: Mapping[str, Any] | None) -> Usage | None:
    """Extract the canonical usage record from one provider wire payload.

    Args:
        dialect: the wire shape — declared once per conversion path, never
            re-decided at the call site (a provider serializer knows its own;
            a native-tier protocol declares its own).
        payload: the protocol envelope — a response body, a stream chunk, or a
            stream event. The adapter knows where its dialect hides the usage
            block inside it.

    Returns:
        The canonical record, or ``None`` when the payload carries no usage
        (a non-terminal chunk, or a response whose provider reported nothing).
    """
    if payload is None:
        return None
    adapter = _ADAPTERS.get(dialect)
    if adapter is None:
        raise ValueError(f"unknown usage dialect: {dialect!r}")
    return adapter.parse(payload)


def is_known_dialect(dialect: str) -> bool:
    """Whether a wire shape has an adapter (used by callers that declare one)."""
    return dialect in _ADAPTERS


def count_billable_events(dialect: UsageDialect, payload: Mapping[str, Any] | None) -> int:
    """Count billable events this payload announces outside its usage block."""
    if payload is None:
        return 0
    adapter = _ADAPTERS.get(dialect)
    if adapter is None:
        raise ValueError(f"unknown usage dialect: {dialect!r}")
    return adapter.count_events(payload)
