# src/llm_proxy/models/finish_reasons.py
"""Unified finish reason mappings across providers."""

from enum import StrEnum
from typing import Literal


class FinishReason(StrEnum):
    """Standard OpenAI finish reason values."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    CONTEXT_LENGTH = "context_length"  # Anthropic: model_context_window_exceeded


# Gemini → OpenAI mapping. Enum values from the API discovery document
# (FinishReason): FINISH_REASON_UNSPECIFIED, STOP, MAX_TOKENS, SAFETY,
# RECITATION, LANGUAGE, OTHER, BLOCKLIST, PROHIBITED_CONTENT, SPII,
# MALFORMED_FUNCTION_CALL, IMAGE_SAFETY, IMAGE_PROHIBITED_CONTENT, IMAGE_OTHER,
# NO_IMAGE, IMAGE_RECITATION, UNEXPECTED_TOOL_CALL, TOO_MANY_TOOL_CALLS,
# MISSING_THOUGHT_SIGNATURE, MALFORMED_RESPONSE, ESCALATION,
# PUP_LIMITED_DISABLED, CONTINUATION.
GEMINI_TO_OPENAI: frozendict[str, str] = frozendict(
    {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "LANGUAGE": "content_filter",
        "MALFORMED_FUNCTION_CALL": "tool_calls",
        "IMAGE_SAFETY": "content_filter",
        "IMAGE_PROHIBITED_CONTENT": "content_filter",
        "IMAGE_OTHER": "stop",
        "IMAGE_RECITATION": "content_filter",
        "NO_IMAGE": "content_filter",
        "UNEXPECTED_TOOL_CALL": "tool_calls",
        "TOO_MANY_TOOL_CALLS": "tool_calls",
        "MISSING_THOUGHT_SIGNATURE": "stop",
        "MALFORMED_RESPONSE": "stop",
        "OTHER": "stop",
        "BLOCKLIST": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "SPII": "content_filter",
        "ESCALATION": "content_filter",
        # Prohibited Use Policy limit on the account: the candidate was filtered.
        "PUP_LIMITED_DISABLED": "content_filter",
        # The response hit the per-request cap but is not complete (long decode);
        # the candidate's ``continuationToken`` resumes it. OpenAI's nearest
        # client-visible meaning of "stopped at the limit" is ``length``. Both
        # paths surface the token itself — the non-streaming parser as
        # ``provider_info["continuationToken"]``, the streaming converter as a
        # ``continuation_token`` chunk field — so a truncated turn stays
        # resumable however it was delivered.
        "CONTINUATION": "length",
        "FINISH_REASON_UNSPECIFIED": "stop",
    }
)

# Anthropic → OpenAI mapping
ANTHROPIC_TO_OPENAI: frozendict[str, str] = frozendict(
    {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop_sequence": "stop_sequence",
        "pause_turn": "stop",
        "refusal": "content_filter",
        "model_context_window_exceeded": "context_length",
        # Beta compaction turns carry no OpenAI equivalent; preserve the value so
        # the Anthropic client still sees ``stop_reason: "compaction"``.
        "compaction": "compaction",
    }
)

# OpenAI → Anthropic mapping
OPENAI_TO_ANTHROPIC: frozendict[str, str] = frozendict(
    {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
        "pause_turn": "pause_turn",
        "refusal": "refusal",
        "context_length": "model_context_window_exceeded",
        "stop_sequence": "stop_sequence",
        "compaction": "compaction",
    }
)

#: The official Anthropic ``StopReason`` enum (docs/en/api — StopReason object)
#: plus the beta ``compaction`` turn the proxy preserves end-to-end. Any other
#: value reaching an Anthropic-protocol client is an illegal enum member.
ANTHROPIC_STOP_REASONS: frozenset[str] = frozenset(
    {
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "pause_turn",
        "refusal",
        "model_context_window_exceeded",
        # Beta compaction turns carry no OpenAI equivalent; the raw value is
        # preserved through the canonical channel (see ANTHROPIC_TO_OPENAI).
        "compaction",
    }
)


def to_anthropic_stop_reason(reason: str | None) -> str:
    """Coerce any upstream finish reason onto the official StopReason enum.

    Canonical-channel finish reasons are the OpenAI dialect, but upstreams
    (OpenRouter, older OpenAI models, internal converters) emit values outside
    it — ``function_call``, ``error``, ``eos``. Unknown values must not leak
    onto the Anthropic wire as an illegal ``stop_reason``; the ones that are
    already official Anthropic values (native values preserved through the
    canonical channel) pass through, everything else collapses to
    ``end_turn``.
    """
    if not reason:
        return "end_turn"
    if reason in ANTHROPIC_STOP_REASONS:
        return reason
    mapped = OPENAI_TO_ANTHROPIC.get(reason)
    return mapped if mapped is not None else "end_turn"


#: The official OpenAI Chat Completions ``finish_reason`` enum. Canonical-channel
#: values outside it must not reach an OpenAI-protocol client (see
#: :func:`to_openai_finish_reason`).
OPENAI_FINISH_REASONS: frozenset[str] = frozenset(
    {"stop", "length", "tool_calls", "content_filter", "function_call"}
)


def to_openai_finish_reason(reason: str | None) -> str | None:
    """Clamp any canonical finish reason onto the official OpenAI enum.

    The canonical channel preserves Anthropic-native values end-to-end
    (``compaction``, ``stop_sequence``) and carries the proxy-internal
    ``context_length`` sentinel; none of those exist on the OpenAI wire. The
    official Anthropic direction clamps unknowns onto ``end_turn`` — this is
    the mirror: legal values pass, canonical extras fold onto their closest
    official member, anything else collapses to ``stop``.
    """
    if not reason:
        return reason
    if reason in OPENAI_FINISH_REASONS:
        return reason
    if reason == "context_length":
        return "length"
    return "stop"


# Gemini → Anthropic mapping
GEMINI_TO_ANTHROPIC: frozendict[str, str] = frozendict(
    {
        "STOP": "end_turn",
        "MAX_TOKENS": "max_tokens",
        "SAFETY": "refusal",
        "RECITATION": "refusal",
        "LANGUAGE": "refusal",
        "MALFORMED_FUNCTION_CALL": "tool_use",
        "IMAGE_SAFETY": "refusal",
        "IMAGE_PROHIBITED_CONTENT": "refusal",
        "IMAGE_OTHER": "end_turn",
        "IMAGE_RECITATION": "refusal",
        "NO_IMAGE": "refusal",
        "UNEXPECTED_TOOL_CALL": "tool_use",
        "TOO_MANY_TOOL_CALLS": "tool_use",
        "MISSING_THOUGHT_SIGNATURE": "end_turn",
        "MALFORMED_RESPONSE": "end_turn",
        "OTHER": "end_turn",
        "BLOCKLIST": "refusal",
        "PROHIBITED_CONTENT": "refusal",
        "SPII": "refusal",
        "ESCALATION": "refusal",
        "PUP_LIMITED_DISABLED": "refusal",
        "CONTINUATION": "max_tokens",
        "FINISH_REASON_UNSPECIFIED": "end_turn",
    }
)


def map_finish_reason(
    reason: str | None,
    source: Literal["openai", "anthropic", "gemini"],
    target: Literal["openai", "anthropic"],
) -> str | None:
    """Map finish reason between provider formats.

    Args:
        reason: The source finish reason value
        source: Source provider format ("openai", "anthropic", or "gemini")
        target: Target provider format ("openai" or "anthropic")

    Returns:
        Mapped finish reason, or None if not found or input is None
    """
    if reason is None:
        return None

    if source == "gemini" and target == "openai":
        return GEMINI_TO_OPENAI.get(reason)
    elif source == "gemini" and target == "anthropic":
        return GEMINI_TO_ANTHROPIC.get(reason)
    elif source == "anthropic" and target == "openai":
        return ANTHROPIC_TO_OPENAI.get(reason)
    elif source == "openai" and target == "anthropic":
        return OPENAI_TO_ANTHROPIC.get(reason)
    elif source == target:
        return reason

    return None
