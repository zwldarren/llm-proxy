"""Reasoning field normalization utilities for providers.

This module provides shared utilities for normalizing reasoning fields
across different provider adapters. Some providers use 'reasoning' while
others use 'reasoning_content' - these utilities handle the conversion.

IMPORTANT: All functions in this module mutate their input dictionaries
in-place AND return the mutated dict. This design enables method chaining
while avoiding unnecessary deep copies. The return value is the same
object that was passed in.
"""

from typing import Any

from llm_proxy.core.thinking import resolve_thinking

# Model name markers whose tool-call turns require DeepSeek-style reasoning echo.
# DeepSeek's thinking-mode API returns HTTP 400 when a request that carries a
# tool-call assistant message without ``reasoning_content`` is sent (the field
# must be echoed back in all subsequent turns). Kimi K2.5 enforces the same
# rule. Markers are matched case-insensitively against the model name.
REASONING_ECHO_MODEL_MARKERS: tuple[str, ...] = ("deepseek", "kimi")


# Fallback placeholder when a tool-call turn genuinely produced no reasoning.
def _placeholder_for_tool_calls(tool_calls: list[dict[str, Any]] | None) -> str:
    """Build a minimal, factual placeholder reasoning string.

    DeepSeek requires a non-empty ``reasoning_content`` on tool-call assistant
    messages, but a fabricated chain-of-thought degrades the model. The chosen
    text only states the fact that a tool call happened (mirroring the
    conversation transcript), so the model's subsequent reasoning is not
    derailed by invented content.
    """
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            return f"Calling {name}."
    return "Tool call."


def is_thinking_explicitly_disabled(request: Any) -> bool:
    """Return True when the request explicitly disables thinking mode.

    Thinking mode defaults to enabled for thinking-capable models, so only an
    explicit ``thinking.type == "disabled"`` or ``reasoning_effort: none``
    (which normalizes to ``ThinkingConfig(type="disabled")``) counts as
    disabled. When disabled, DeepSeek performs no reasoning-content validation,
    so no echo placeholder is needed.
    """
    config = resolve_thinking(request)
    if config is None:
        return False
    if config.type == "disabled":
        return True
    return (config.effort or "").lower() == "none"


def ensure_reasoning_echo(body: dict[str, Any], field: str, request: Any) -> dict[str, Any]:
    """Guarantee every tool-call assistant message carries ``field``.

    The conversation serializer already restores real reasoning from the
    call_id reasoning cache (see ``_restore_reasoning_from_cache``) and from
    ``ThinkingBlock`` history, so this only fills the gap for turns that
    genuinely produced no reasoning (e.g. served by another provider, or a
    cache miss after restart). It injects a minimal factual placeholder so
    DeepSeek-style thinking-mode validation never sees a tool-call assistant
    message without the reasoning field.

    Skipped entirely when thinking mode is explicitly disabled: no validation
    runs in that case, and an injected placeholder would only pollute context.

    Args:
        body: The fully built OpenAI-format request body (mutated in place).
        field: The provider's reasoning field name (``reasoning_content`` or
            ``reasoning``) after normalization.
        request: The internal request, used to resolve thinking mode.

    Returns:
        The same body with the reasoning field guaranteed on tool-call
        assistant messages.
    """
    if is_thinking_explicitly_disabled(request):
        return body
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        if msg.get(field):
            continue
        msg[field] = _placeholder_for_tool_calls(msg["tool_calls"])
    return body


def detect_reasoning_field_in_message(message: dict[str, Any]) -> str | None:
    """Detect the provider's reasoning field name from a response message.

    Returns ``"reasoning_content"``, ``"reasoning"``, or None when the
    message carries no reasoning field. ``reasoning_content`` wins when
    both fields are present.
    """
    has_reasoning_content = (
        "reasoning_content" in message and message["reasoning_content"] is not None
    )
    has_reasoning = "reasoning" in message and message["reasoning"] is not None
    if has_reasoning_content:
        return "reasoning_content"
    if has_reasoning:
        return "reasoning"
    return None


def _detect_reasoning_field_in_first_choice(
    payload: dict[str, Any], container_key: str
) -> str | None:
    """Detect the reasoning field name from the first choice's container.

    ``container_key`` selects ``message`` (non-streaming bodies) or ``delta``
    (streaming chunks); the field name is read from whichever container the
    provider populated.
    """
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    container = choice.get(container_key)
    if not isinstance(container, dict):
        return None
    return detect_reasoning_field_in_message(container)


def detect_reasoning_field_in_response_body(body: dict[str, Any]) -> str | None:
    """Detect the provider's reasoning field name from a non-streaming response body.

    Reads the first choice's message — the same message the
    ``OpenAIResponseParser`` inspects to seed the preference cache on the
    fully parsed tier.
    """
    return _detect_reasoning_field_in_first_choice(body, "message")


def detect_reasoning_field_in_stream_chunk(chunk: dict[str, Any]) -> str | None:
    """Detect the provider's reasoning field name from a streaming chunk.

    Inspects the first choice's delta while the provider's original field
    naming is still visible — i.e. before
    :func:`normalize_reasoning_in_stream_chunk` renames the field for the
    client. Mirrors :func:`detect_reasoning_field_in_response_body` for
    the non-streaming tier.
    """
    return _detect_reasoning_field_in_first_choice(chunk, "delta")


def normalize_reasoning_in_response_body(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize reasoning fields in a non-streaming response body.

    Converts 'reasoning' to 'reasoning_content' in every choice's message,
    mirroring :func:`normalize_reasoning_in_stream_chunk` so the wire-reuse
    (verbatim) response tier emits the same field names as both the streaming
    and the fully parsed paths. Preserves existing 'reasoning_content' if
    present.

    Args:
        body: Response body from provider API. Modified in-place.

    Returns:
        The same body object with normalized reasoning fields (mutated in-place).
    """
    choices = body.get("choices", [])
    if not isinstance(choices, list):
        return body
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if (
            isinstance(message, dict)
            and "reasoning" in message
            and "reasoning_content" not in message
        ):
            message["reasoning_content"] = message.pop("reasoning")
    return body


def normalize_reasoning_in_stream_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Normalize reasoning fields in streaming chunks.

    Converts 'reasoning' to 'reasoning_content' in streaming delta chunks.
    Preserves existing 'reasoning_content' if present.

    Note:
        This function processes only the first choice in the chunk, as streaming
        responses typically deliver one choice at a time.

    Args:
        chunk: Streaming chunk from provider API. Modified in-place.

    Returns:
        The same chunk object with normalized reasoning field (mutated in-place).
    """
    choices = chunk.get("choices", [])
    if not choices:
        return chunk
    delta = choices[0].get("delta", {})
    if "reasoning" in delta and "reasoning_content" not in delta:
        delta["reasoning_content"] = delta.pop("reasoning")
    return chunk
