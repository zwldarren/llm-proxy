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
# MISSING_THOUGHT_SIGNATURE, MALFORMED_RESPONSE, ESCALATION.
GEMINI_TO_OPENAI: dict[str, str] = {
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
    "FINISH_REASON_UNSPECIFIED": "stop",
}

# Anthropic → OpenAI mapping
ANTHROPIC_TO_OPENAI: dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop_sequence",
    "pause_turn": "stop",
    "refusal": "content_filter",
    "model_context_window_exceeded": "context_length",
}

# OpenAI → Anthropic mapping
OPENAI_TO_ANTHROPIC: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
    "pause_turn": "pause_turn",
    "refusal": "refusal",
    "context_length": "model_context_window_exceeded",
    "stop_sequence": "stop_sequence",
}

# Gemini → Anthropic mapping
GEMINI_TO_ANTHROPIC: dict[str, str] = {
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
    "FINISH_REASON_UNSPECIFIED": "end_turn",
}


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
