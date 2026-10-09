"""Parity ratchet: the usage intake seam must agree with the call sites it replaces.

ADR-0025 moves usage extraction behind one seam (``serialization/usage``) and
deletes the per-tier copies one dialect at a time. Deleting a copy is only safe
if the seam and the copy have been compared over the same wire frames first —
that comparison is this module's whole job.

A row pairs a wire dialect with a call site that still parses it locally:

* ``legacy`` drives the *shipped* code path and returns whatever that path
  produces — a ``StreamingUsage``, an ``EventContext`` or billing's
  ``TokenUsage``;
* the canonical side runs ``UsageAccumulator(dialect)`` over the same frames.

Both sides are then projected onto the billable **amounts**, not the field
placement: amounts are what billing multiplies. Field placement is pinned
separately by the module's own tests (``tests/test_serialization/test_usage_intake.py``),
which is why a row can pass while the canonical record expresses a fact in a
flatter field than the legacy one did.

When a migration slice lands, its driver stops being an independent parser and
becomes a projection onto the record the seam already fills, so comparing the two
only proves the projection is faithful. Such a row therefore keeps its driver *and*
gains ``recorded``: the amounts the shipped parser produced, measured from it
before the swap, which still fail if the seam or the projection moves an amount. A
row whose call site is deleted outright drops the driver and keeps ``recorded``
alone.

``declared`` names the facts a row is allowed to disagree on, with the reason.
The set is checked for exactness in both directions — a declaration without a
real difference fails, and a new difference fails — so a divergence cannot
quietly hide behind a stale declaration.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import orjson
import pytest

from llm_proxy.billing.tokens import TokenUsage, extract_tokens_from_usage
from llm_proxy.core.conversion import NativePassthroughHandler
from llm_proxy.models.types import Usage
from llm_proxy.observability.event_context import EventContext
from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer
from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer
from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer
from llm_proxy.serialization.anthropic.serializer import parse_usage_and_provider_extras
from llm_proxy.serialization.anthropic.streaming_converter import AnthropicChunkConverter
from llm_proxy.serialization.openai.serializer import parse_usage_from_response
from llm_proxy.serialization.openai.streaming_converter import OpenAIResponsesChunkConverter
from llm_proxy.serialization.usage import UsageAccumulator
from llm_proxy.streaming.transformer import StreamingUsage

#: The billable facts both sides are projected onto. ``cost`` is included
#: because a provider-reported cost outranks the local price table
#: (``observability/cost.py``), so dropping it changes a bill.
FACTS: tuple[str, ...] = (
    "input",
    "output",
    "total",
    "cache_read",
    "cache_write",
    "reasoning",
    "web_search",
    "cost",
)

Facts = dict[str, int | float]


def _amount(flat: Any, nested: Any) -> int:
    """One cache fact's billable amount, from whichever dialect field carries it.

    A cache fact has two expressions (flat Anthropic field vs nested OpenAI
    detail) and the same amount in both; billing multiplies the amount, so the
    ratchet compares there. Which field the canonical record *keeps* it in is
    the module's own contract, pinned by its own tests.
    """
    return int(flat or nested or 0)


def _facts_of_usage(usage: Usage | None) -> Facts:
    """Canonical record → billable facts."""
    if usage is None:
        return dict.fromkeys(FACTS, 0)
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "total": usage.total_tokens or usage.input_tokens + usage.output_tokens,
        "cache_read": _amount(
            usage.cache_read_input_tokens, getattr(prompt_details, "cached_tokens", None)
        ),
        "cache_write": _amount(
            usage.cache_creation_input_tokens, getattr(prompt_details, "cache_write_tokens", None)
        ),
        "reasoning": int(
            usage.reasoning_tokens or getattr(completion_details, "reasoning_tokens", None) or 0
        ),
        "web_search": int(usage.web_search_requests or 0),
        "cost": float(usage.provider_reported_cost or 0),
    }


def _facts_of_streaming_usage(usage: StreamingUsage | None) -> Facts:
    """Legacy ``StreamingUsage`` → billable facts."""
    if usage is None:
        return dict.fromkeys(FACTS, 0)
    prompt_details = usage.prompt_tokens_details or {}
    completion_details = usage.completion_tokens_details or {}
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "total": usage.total_tokens or usage.input_tokens + usage.output_tokens,
        "cache_read": _amount(usage.cache_read_input_tokens, prompt_details.get("cached_tokens")),
        "cache_write": _amount(
            usage.cache_creation_input_tokens, prompt_details.get("cache_write_tokens")
        ),
        "reasoning": int(completion_details.get("reasoning_tokens") or 0),
        "web_search": int(usage.web_search_requests or 0),
        "cost": float(usage.provider_reported_cost or 0),
    }


def _facts_of_event_context(context: EventContext) -> Facts:
    """Observability capture object → billable facts (the passthrough tier's view)."""
    return {
        "input": int(context.prompt_tokens or 0),
        "output": int(context.completion_tokens or 0),
        "total": int(
            context.total_tokens or (context.prompt_tokens or 0) + (context.completion_tokens or 0)
        ),
        "cache_read": _amount(context.cache_read_input_tokens, context.cached_prompt_tokens),
        "cache_write": _amount(context.cache_creation_input_tokens, context.cache_write_tokens),
        "reasoning": int(context.reasoning_tokens or 0),
        "web_search": int(context.web_search_requests or 0),
        "cost": float(context.provider_reported_cost or 0),
    }


def _facts_of_token_usage(usage: TokenUsage) -> Facts:
    """Billing's own read model → billable facts.

    The read model has no reasoning dimension: billing prices reasoning inside
    the output count, so a row comparing against it restricts ``facts``.
    """
    return {
        "input": usage.prompt_tokens,
        "output": usage.completion_tokens,
        "total": usage.total_tokens or usage.prompt_tokens + usage.completion_tokens,
        "cache_read": _amount(usage.cache_read_input_tokens, usage.cached_prompt_tokens),
        "cache_write": usage.cache_creation_input_tokens,
        "reasoning": 0,
        "web_search": usage.web_search_requests,
        "cost": 0.0,
    }


# ---------------------------------------------------------------------------
# The wire frames and the legacy drivers
# ---------------------------------------------------------------------------

_TERMINAL_USAGE: dict[str, Any] = {
    "input_tokens": 1000,
    "output_tokens": 50,
    "total_tokens": 1050,
    "input_tokens_details": {"cached_tokens": 400},
    "output_tokens_details": {"reasoning_tokens": 20},
}

_ANTHROPIC_FRAMES: tuple[dict[str, Any], ...] = (
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "usage": {"input_tokens": 100, "output_tokens": 1, "cache_read_input_tokens": 50},
        },
    },
    {
        "type": "message_delta",
        "usage": {
            "output_tokens": 40,
            "output_tokens_details": {"thinking_tokens": 7},
            "server_tool_use": {"web_search_requests": 2},
        },
    },
)


def _new_context() -> EventContext:
    return EventContext(request_id="req_parity", trace_id="trace_parity", model="test-model")


def _chat_transformer(frames: Sequence[dict[str, Any]]) -> StreamingUsage | None:
    """``protocols/openai/streaming.py`` — the client-facing chat tier's own parser."""
    transformer = OpenAIStreamingTransformer(model="gpt-4o", request_id="req_parity")
    for frame in frames:
        transformer.transform(frame)
    return transformer.get_usage()


def _anthropic_converter(frames: Sequence[dict[str, Any]]) -> StreamingUsage | None:
    """``serialization/anthropic/streaming_converter.py`` — provider frames → canonical usage."""
    converter = AnthropicChunkConverter(model="claude-3", request_id="msg_1")
    for frame in frames:
        converter.convert_chunk(frame)
    return converter.get_usage()


def _anthropic_protocol_transformer(frames: Sequence[dict[str, Any]]) -> StreamingUsage | None:
    """``protocols/anthropic/streaming.py`` — canonical chunks → the client's wire."""
    transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg_1")
    for frame in frames:
        transformer.transform(frame)
    return transformer.get_usage()


def _anthropic_passthrough(frames: Sequence[dict[str, Any]]) -> EventContext:
    """``core/conversion.py::process_anthropic_usage_event`` — the native tier's handler."""
    context = _new_context()
    for frame in frames:
        NativePassthroughHandler.process_anthropic_usage_event(frame["type"], frame, context)
    return context


def _anthropic_body_parser(frames: Sequence[dict[str, Any]]) -> Usage | None:
    """``serialization/anthropic/serializer.py`` — a non-streaming body → usage."""
    usage, _extras = parse_usage_and_provider_extras(frames[0])
    return usage


def _openresponses_protocol_transformer(
    frames: Sequence[dict[str, Any]],
) -> StreamingUsage | None:
    """``protocols/openresponses/streaming.py`` — canonical or native events → usage."""
    transformer = OpenResponsesStreamingTransformer(model="gpt-4", request_id="resp_1")
    for frame in frames:
        transformer.transform(frame)
    return transformer.get_usage()


def _responses_terminal_passthrough(frames: Sequence[dict[str, Any]]) -> EventContext:
    """``core/conversion.py::_apply_openresponses_usage`` — the native tier's handler."""
    context = _new_context()
    for frame in frames:
        NativePassthroughHandler._apply_openresponses_usage(frame, context)
    return context


def _responses_provider_converter(frames: Sequence[dict[str, Any]]) -> StreamingUsage | None:
    """``serialization/openai/streaming_converter.py`` — Responses wire events → usage."""
    converter = OpenAIResponsesChunkConverter(model="gpt-4o", request_id="resp_1")
    for frame in frames:
        converter.convert_chunk(frame)
    return converter.get_usage()


def _native_chat_frame(frames: Sequence[dict[str, Any]]) -> EventContext:
    """``core/conversion.py::_capture_openai_stream_usage`` — the native tier's handler."""
    context = _new_context()
    for frame in frames:
        sse = "data: " + orjson.dumps(frame).decode() + "\n\n"
        NativePassthroughHandler._capture_openai_stream_usage(sse, context)
    return context


def _realtime_billing_read_model(frames: Sequence[dict[str, Any]]) -> TokenUsage:
    """``realtime/usage.py`` hands the raw usage dict straight to cost calculation."""
    return extract_tokens_from_usage(frames[0]["response"]["usage"])


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One (dialect, call site) pair the ratchet holds in agreement."""

    id: str
    dialect: str
    frames: tuple[dict[str, Any], ...]
    legacy: Callable[[Sequence[dict[str, Any]]], Any] | None = None
    facts_of: Callable[[Any], Facts] = _facts_of_usage
    #: Facts this row compares. A legacy view that cannot express a fact (the
    #: billing read model has no reasoning dimension) restricts the set here
    #: rather than declaring a permanent divergence.
    facts: frozenset[str] = frozenset(FACTS)
    #: fact → why the two sides may legitimately differ. Usually this is the
    #: legacy side losing a fact the seam now keeps; the message names which.
    declared: Mapping[str, str] = field(default_factory=dict)
    #: Fact → why the migration deliberately moved it. The pre-swap call site did
    #: not report the fact (the reason a ``declared`` entry once carried) and the
    #: slice closed the gap; ``recorded`` then carries the amount the repair
    #: produces. Naming it keeps the change visible in the diff instead of leaving
    #: an amount that silently moved.
    closed: Mapping[str, str] = field(default_factory=dict)
    #: The row's contract for every fact it compares: the amount the shipped parser
    #: produced before the swap, or — for a fact in ``closed`` — the amount the
    #: repair produces. Frozen amounts, not a projection of the seam, because the
    #: shipped driver and the accumulator share the seam once a slice lands.
    recorded: Facts | None = None


def _thinking_tokens_are_unread(fact: str) -> str:
    return (
        f"the shipped site reads only the input/output/cache counters, so "
        f"``{fact}`` (Anthropic's ``output_tokens_details.thinking_tokens``) never reaches the "
        "recorded usage; the seam keeps it. Observability only — reasoning tokens are priced "
        "inside the output count."
    )


_COST_WAS_UNREAD = (
    "the shipped transformer reported no provider cost, so a streamed turn was billed from "
    "the local price table even when the provider sent its own (``usage.cost`` / "
    "``nanogpt_cost``)."
)

_SEARCH_COUNT_WAS_UNREAD = (
    "the shipped transformer read ``server_tool_use`` for client emission only "
    "(``_message_delta_usage``), so the record the proxy bills from lost the search count."
)

_FLAT_CACHE_COUNTERS_WERE_UNREAD = (
    "the shipped transformer read the usage block's nested details only, so a canonical "
    "chunk's flat cache counters never reached the record."
)

_NATIVE_SEARCHES_WERE_UNREAD = (
    "the shipped transformer's Responses branch read the token counters only, so a "
    "provider-executed search in the terminal snapshot was never counted — while the "
    "non-streaming body parser counted the same items."
)

_CHAT_SEARCHES_WERE_UNREAD = (
    "the shipped transformer never read the usage block's own search count; its "
    "emitter-side counter covers provider tool calls, not a count the provider "
    "reported in ``server_tool_use``."
)


CHAT_FLAT_AND_NESTED_COPY: dict[str, Any] = {
    "choices": [],
    "usage": {
        "prompt_tokens": 1500,
        "completion_tokens": 500,
        "total_tokens": 2000,
        "cache_read_input_tokens": 300,
        "cache_creation_input_tokens": 200,
        "prompt_tokens_details": {"cached_tokens": 300, "audio_tokens": 25},
    },
}

CHAT_NESTED_ONLY: dict[str, Any] = {
    "choices": [],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "total_tokens": 1050,
        "prompt_tokens_details": {"cached_tokens": 400},
    },
}

CHAT_DEEPSEEK_CACHE_HITS: dict[str, Any] = {
    "choices": [],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_cache_hit_tokens": 64,
        "prompt_cache_miss_tokens": 36,
    },
}

CHAT_COST_AND_SEARCH: dict[str, Any] = {
    "choices": [],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "total_tokens": 1050,
        "completion_tokens_details": {"reasoning_tokens": 20},
        "cost": 0.25,
        "server_tool_use": {"web_search_requests": 3},
    },
}

CHAT_OPENROUTER_FRAME: dict[str, Any] = {
    "choices": [],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "total_tokens": 1050,
        "prompt_tokens_details": {"cached_tokens": 300},
        "cost": 0.5,
        "server_tool_use": {"web_search_requests": 3},
    },
}

ANTHROPIC_BODY: dict[str, Any] = {
    "id": "msg_1",
    "usage": {
        "input_tokens": 100,
        "output_tokens": 40,
        "cache_read_input_tokens": 50,
        "output_tokens_details": {"thinking_tokens": 7},
        "server_tool_use": {"web_search_requests": 2},
    },
}

RESPONSES_BODY: dict[str, Any] = {
    "id": "resp_1",
    "usage": _TERMINAL_USAGE,
    "output": [{"type": "web_search_call"}, {"type": "message"}],
}

RESPONSES_TERMINAL_EVENT: dict[str, Any] = {
    "type": "response.completed",
    "response": {
        "id": "resp_1",
        "usage": _TERMINAL_USAGE,
        "output": [{"type": "web_search_call", "id": "ws_1"}, {"type": "message"}],
    },
}

# The same turn as it arrives at the provider-side converter, which is handed the
# SSE event type separately from its data (see ``convert_chunk``); the payload's own
# ``type`` field rides along, as it does on the wire.
RESPONSES_PROVIDER_EVENTS: tuple[dict[str, Any], ...] = (
    {
        "event_type": "response.output_item.added",
        "type": "response.output_item.added",
        "item": {"id": "ws_1", "type": "web_search_call"},
    },
    {
        "event_type": "response.completed",
        "type": "response.completed",
        "response": {"id": "resp_1", "status": "completed", "usage": _TERMINAL_USAGE},
    },
)

REALTIME_DONE: dict[str, Any] = {
    "type": "response.done",
    "response": {
        "id": "resp_1",
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
            "input_token_details": {"cached_tokens": 20, "audio_tokens": 10},
            "output_token_details": {"audio_tokens": 25},
        },
    },
}

_ROWS: tuple[Row, ...] = (
    Row(
        id="chat-protocol-transformer-flat-and-nested-copy",
        dialect="chat",
        # Slice 2 landed here (see the ``chat-passthrough-usage-frame`` row), so the
        # shipped driver now runs the migrated transformer: this row checks its
        # projection of the canonical record, and ``recorded`` anchors it to the
        # amounts the transformer's own parser produced before the swap.
        frames=(CHAT_FLAT_AND_NESTED_COPY,),
        legacy=_chat_transformer,
        facts_of=_facts_of_streaming_usage,
        recorded={
            "input": 1500,
            "output": 500,
            "total": 2000,
            "cache_read": 300,
            "cache_write": 200,
            "reasoning": 0,
            "web_search": 0,
            "cost": 0.0,
        },
    ),
    Row(
        id="chat-protocol-transformer-nested-only",
        dialect="chat",
        frames=(CHAT_NESTED_ONLY,),
        legacy=_chat_transformer,
        facts_of=_facts_of_streaming_usage,
        recorded={
            "input": 1000,
            "output": 50,
            "total": 1050,
            "cache_read": 400,
            "cache_write": 0,
            "reasoning": 0,
            "web_search": 0,
            "cost": 0.0,
        },
    ),
    Row(
        id="chat-protocol-transformer-deepseek-cache-hits",
        dialect="chat",
        frames=(CHAT_DEEPSEEK_CACHE_HITS,),
        legacy=_chat_transformer,
        facts_of=_facts_of_streaming_usage,
        recorded={
            "input": 100,
            "output": 10,
            "total": 110,
            "cache_read": 64,
            "cache_write": 0,
            "reasoning": 0,
            "web_search": 0,
            "cost": 0.0,
        },
    ),
    Row(
        id="chat-protocol-transformer-cost-and-search",
        dialect="chat",
        frames=(CHAT_COST_AND_SEARCH,),
        legacy=_chat_transformer,
        facts_of=_facts_of_streaming_usage,
        recorded={
            "input": 1000,
            "output": 50,
            "total": 1050,
            "cache_read": 0,
            "cache_write": 0,
            "reasoning": 20,
            "web_search": 3,
            "cost": 0.25,
        },
    ),
    Row(
        id="anthropic-provider-converter",
        dialect="anthropic",
        frames=_ANTHROPIC_FRAMES,
        legacy=_anthropic_converter,
        facts_of=_facts_of_streaming_usage,
        closed={"reasoning": _thinking_tokens_are_unread("reasoning")},
        recorded={
            "input": 150,
            "output": 40,
            "total": 190,
            "cache_read": 50,
            "cache_write": 0,
            "reasoning": 7,
            "web_search": 2,
            "cost": 0.0,
        },
    ),
    Row(
        id="anthropic-passthrough-event-context",
        dialect="anthropic",
        frames=_ANTHROPIC_FRAMES,
        legacy=_anthropic_passthrough,
        facts_of=_facts_of_event_context,
        closed={"reasoning": _thinking_tokens_are_unread("reasoning")},
        recorded={
            "input": 150,
            "output": 40,
            "total": 190,
            "cache_read": 50,
            "cache_write": 0,
            "reasoning": 7,
            "web_search": 2,
            "cost": 0.0,
        },
    ),
    Row(
        id="anthropic-protocol-transformer-cost-and-search",
        dialect="chat",
        # The canonical chunk shape an OpenAI-family provider converter hands the
        # Anthropic-dialect client's transformer. Its old parser dropped the two
        # facts the provider reported beside the tokens, so both are ``closed``.
        frames=(CHAT_COST_AND_SEARCH,),
        legacy=_anthropic_protocol_transformer,
        facts_of=_facts_of_streaming_usage,
        closed={"cost": _COST_WAS_UNREAD, "web_search": _SEARCH_COUNT_WAS_UNREAD},
        recorded={
            "input": 1000,
            "output": 50,
            "total": 1050,
            "cache_read": 0,
            "cache_write": 0,
            "reasoning": 20,
            "web_search": 3,
            "cost": 0.25,
        },
    ),
    Row(
        id="anthropic-protocol-transformer-flat-and-nested-copy",
        dialect="chat",
        frames=(CHAT_FLAT_AND_NESTED_COPY,),
        legacy=_anthropic_protocol_transformer,
        facts_of=_facts_of_streaming_usage,
        recorded={
            "input": 1500,
            "output": 500,
            "total": 2000,
            "cache_read": 300,
            "cache_write": 200,
            "reasoning": 0,
            "web_search": 0,
            "cost": 0.0,
        },
    ),
    Row(
        id="anthropic-body-parser",
        dialect="anthropic",
        frames=(ANTHROPIC_BODY,),
        legacy=_anthropic_body_parser,
        # The shipped parser put the search count in the ``provider_info`` extras
        # rather than the record; the fact was not lost (``update_provider_info``
        # reads the same block), but the record now carries it too.
        closed={
            "web_search": "the shipped parser returned the count through provider_info extras."
        },
        recorded={
            "input": 150,
            "output": 40,
            "total": 190,
            "cache_read": 50,
            "cache_write": 0,
            "reasoning": 7,
            "web_search": 2,
            "cost": 0.0,
        },
    ),
    Row(
        id="responses-body-parser",
        dialect="responses",
        frames=(RESPONSES_BODY,),
        legacy=lambda frames: parse_usage_from_response(frames[0]),
        # The shipped parser read the search count from the caller
        # (``web_search_call_count``); the adapter counts the body's own output
        # items, which is the same number for the same body.
        recorded={
            "input": 1000,
            "output": 50,
            "total": 1050,
            "cache_read": 400,
            "cache_write": 0,
            "reasoning": 20,
            "web_search": 1,
            "cost": 0.0,
        },
    ),
    Row(
        id="responses-passthrough-terminal-event",
        dialect="responses",
        frames=(RESPONSES_TERMINAL_EVENT,),
        legacy=_responses_terminal_passthrough,
        facts_of=_facts_of_event_context,
        closed={
            "web_search": (
                "the shipped handler read only the usage block, so a "
                "provider-executed search in the terminal snapshot was never "
                "counted on this tier; the non-streaming body parser already "
                "counted the same items."
            )
        },
        recorded={
            "input": 1000,
            "output": 50,
            "total": 1050,
            "cache_read": 400,
            "cache_write": 0,
            "reasoning": 20,
            "web_search": 1,
            "cost": 0.0,
        },
    ),
    Row(
        id="responses-provider-streaming-converter",
        dialect="responses",
        frames=RESPONSES_PROVIDER_EVENTS,
        legacy=_responses_provider_converter,
        facts_of=_facts_of_streaming_usage,
    ),
    Row(
        id="openresponses-protocol-transformer-flat-and-nested-copy",
        dialect="chat",
        frames=(CHAT_FLAT_AND_NESTED_COPY,),
        legacy=_openresponses_protocol_transformer,
        facts_of=_facts_of_streaming_usage,
        closed={"cache_write": _FLAT_CACHE_COUNTERS_WERE_UNREAD},
        recorded={
            "input": 1500,
            "output": 500,
            "total": 2000,
            "cache_read": 300,
            "cache_write": 200,
            "reasoning": 0,
            "web_search": 0,
            "cost": 0.0,
        },
    ),
    Row(
        id="openresponses-protocol-transformer-cost-and-search",
        dialect="chat",
        frames=(CHAT_COST_AND_SEARCH,),
        legacy=_openresponses_protocol_transformer,
        facts_of=_facts_of_streaming_usage,
        closed={"cost": _COST_WAS_UNREAD, "web_search": _CHAT_SEARCHES_WERE_UNREAD},
        recorded={
            "input": 1000,
            "output": 50,
            "total": 1050,
            "cache_read": 0,
            "cache_write": 0,
            "reasoning": 20,
            "web_search": 3,
            "cost": 0.25,
        },
    ),
    Row(
        id="openresponses-protocol-transformer-native-terminal-event",
        dialect="responses",
        frames=(RESPONSES_TERMINAL_EVENT,),
        legacy=_openresponses_protocol_transformer,
        facts_of=_facts_of_streaming_usage,
        closed={"web_search": _NATIVE_SEARCHES_WERE_UNREAD},
        recorded={
            "input": 1000,
            "output": 50,
            "total": 1050,
            "cache_read": 400,
            "cache_write": 0,
            "reasoning": 20,
            "web_search": 1,
            "cost": 0.0,
        },
    ),
    Row(
        id="chat-passthrough-usage-frame",
        dialect="chat",
        # The shape a native OpenAI-dialect tier actually forwards: an aggregator
        # that reports its own cost and its cache read through the nested detail.
        # Slice 1 of the migration (ADR-0025) landed here, so the shipped driver
        # is now the seam-backed handler: this row checks the projection into the
        # EventContext, and ``recorded`` anchors it to the pre-swap amounts.
        frames=(CHAT_OPENROUTER_FRAME,),
        legacy=_native_chat_frame,
        facts_of=_facts_of_event_context,
        recorded={
            "input": 1000,
            "output": 50,
            "total": 1050,
            "cache_read": 300,
            "cache_write": 0,
            "reasoning": 0,
            "web_search": 3,
            "cost": 0.5,
        },
    ),
    Row(
        id="realtime-usage-dict-to-billing",
        dialect="realtime",
        frames=(REALTIME_DONE,),
        legacy=_realtime_billing_read_model,
        facts_of=_facts_of_token_usage,
        # Billing's read model has no reasoning dimension, and the Realtime
        # dialect's cache read reaches it as ``cached_prompt_tokens``, which its
        # input-total heuristic does not add in — so this row compares every
        # fact the two can express, and agrees on all of them.
        facts=frozenset(FACTS) - {"reasoning", "cost"},
    ),
)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _seam_facts(row: Row) -> Facts:
    accumulator = UsageAccumulator(row.dialect)
    for frame in row.frames:
        accumulator.observe(frame)
    return {k: v for k, v in _facts_of_usage(accumulator.result()).items() if k in row.facts}


def _shipped_facts(row: Row) -> Facts:
    if row.legacy is None:
        assert row.recorded is not None, f"{row.id}: no legacy driver and no recorded facts"
        return {k: v for k, v in row.recorded.items() if k in row.facts}
    return {k: v for k, v in row.facts_of(row.legacy(row.frames)).items() if k in row.facts}


def _disagreements(seam: Facts, shipped: Facts) -> set[str]:
    return {
        fact
        for fact in seam
        if seam[fact] != shipped[fact]
        and not (
            isinstance(seam[fact], float)
            and isinstance(shipped[fact], float)
            and abs(seam[fact] - shipped[fact]) < 1e-9
        )
    }


@pytest.mark.parametrize("row", _ROWS, ids=lambda row: row.id)
def test_seam_agrees_with_the_call_site_it_replaces(row: Row) -> None:
    seam = _seam_facts(row)
    shipped = _shipped_facts(row)
    differing = _disagreements(seam, shipped)

    assert differing == set(row.declared), (
        f"{row.id}: the usage intake seam and its shipped call site disagree on "
        f"{sorted(differing)} (declared: {sorted(row.declared)}).\n"
        f"  seam:   {seam}\n"
        f"  shipped:{shipped}\n"
        + (
            "  declared reasons:\n"
            + "".join(f"    {fact}: {row.declared[fact]}\n" for fact in sorted(row.declared))
            if row.declared
            else ""
        )
        + "An undeclared fact is a divergence to fix on one side (or, when the "
        "shipped side is the one losing it, to record in ADR-0025), not to declare."
    )


def test_every_declared_divergence_carries_a_reason() -> None:
    for row in _ROWS:
        for fact, reason in row.declared.items():
            assert fact in FACTS, f"{row.id}: declared fact {fact!r} is not a billable fact"
            assert reason.strip(), f"{row.id}: declared {fact} without a reason"


def test_every_closed_gap_carries_a_reason() -> None:
    """A repaired fact states why it moved, and how it moved."""
    for row in _ROWS:
        for fact, reason in row.closed.items():
            assert fact in FACTS, f"{row.id}: closed fact {fact!r} is not a billable fact"
            assert reason.strip(), f"{row.id}: closed {fact} without a reason"
            assert fact in row.recorded, f"{row.id}: closed {fact} with no recorded amount"


def test_recorded_expectations_cover_the_facts_the_row_compares() -> None:
    """A partial ``recorded`` dict would anchor only the facts it happens to name."""
    for row in _ROWS:
        if row.recorded is None:
            continue
        missing = set(row.facts) - set(row.recorded)
        assert not missing, f"{row.id}: recorded facts do not cover {sorted(missing)}"


def test_row_ids_are_unique() -> None:
    ids = [row.id for row in _ROWS]
    assert len(ids) == len(set(ids))


def test_a_recorded_expectation_anchors_the_migrated_call_site() -> None:
    """A migrated row keeps comparing live code — against a frozen expectation.

    Once a slice lands, the shipped driver and the accumulator share the seam, so
    comparing them only proves the projection is self-consistent: a change to the
    seam moves both sides together. The recorded amounts are the contract as it
    stood *before* the swap, so they still fail when an amount moves.
    """
    anchored = [row for row in _ROWS if row.recorded is not None and row.legacy is not None]
    assert anchored, "no row anchors a recorded expectation: every migrated row is free to drift"
    for row in anchored:
        actual = {k: v for k, v in row.facts_of(row.legacy(row.frames)).items() if k in row.facts}
        expected = {k: v for k, v in row.recorded.items() if k in row.facts}
        assert actual == expected, (
            f"{row.id}: the migrated call site no longer produces the amounts this row "
            f"pins.\n  now:      {actual}\n  recorded: {expected}"
        )


def test_every_still_unmigrated_dialect_has_a_row() -> None:
    """The ratchet is only worth its weight while it covers the remaining slices.

    ADR-0025's landing order migrates chat, anthropic, responses and realtime
    call sites; a dialect with no row would be deleted without ever having been
    compared.
    """
    assert {row.dialect for row in _ROWS} == {"chat", "anthropic", "responses", "realtime"}
