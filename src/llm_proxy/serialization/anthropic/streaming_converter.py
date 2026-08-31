"""Anthropic provider-side streaming chunk converter.

Converts Anthropic SSE streaming events to canonical OpenAI
``chat.completion.chunk`` dict format.

This converter is the provider-side counterpart of the protocol-side
``AnthropicStreamingTransformer`` (which converts canonical chunks to
Anthropic SSE wire format).

Migrated from ``providers/anthropic/adapter.py:stream_chat_completion()``
to the serialization layer so that all provider-side chunk conversion
logic lives alongside the provider serializer.
"""

import time
from typing import Any

from llm_proxy.models.finish_reasons import ANTHROPIC_TO_OPENAI
from llm_proxy.streaming.transformer import StreamingTransformer, StreamingUsage


def _make_openai_chunk(
    response_id: str,
    model: str,
    created_at: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a canonical OpenAI ``chat.completion.chunk`` dict.

    ``created_at`` must be a stable per-stream timestamp (captured once in
    ``AnthropicChunkConverter.__init__``) so every chunk in a single stream
    shares the same ``created`` value, matching OpenAI's convention.
    """
    chunk: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": model,
    }
    chunk.update(kwargs)
    return chunk


def _map_stop_reason(anthropic_reason: str) -> str:
    """Map an Anthropic ``stop_reason`` to an OpenAI ``finish_reason``."""
    return ANTHROPIC_TO_OPENAI.get(anthropic_reason, anthropic_reason)


class AnthropicChunkConverter(StreamingTransformer):
    """Convert Anthropic SSE streaming events to canonical OpenAI chunk dicts.

    Replaces the ~200 lines of manual SSE parsing that previously lived
    inside ``AnthropicAdapter.stream_chat_completion()``.  The adapter
    is still responsible for reading the HTTP stream and framing SSE
    events; this converter only handles the *event → chunk* mapping.

    Usage::

        converter = AnthropicChunkConverter(model="claude-3", request_id="msg_1")
        async for frame in adapter._stream_raw_sse(request):
            data_str = _extract_data_from_sse_frame(frame)
            if data_str is None:
                continue
            event = orjson.loads(data_str)
            chunk = converter.convert_chunk(event)
            if chunk is not None:
                yield chunk
        yield "[DONE]"
    """

    def __init__(
        self,
        model: str = "",
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            request_id=request_id,
        )
        # Per-converter mutable state (was nonlocal variables in the adapter).
        self._response_id: str = request_id or ""
        self._model: str = model
        # Fixed per-stream timestamp so all chunks share the same ``created``.
        self._created_at: int = int(time.time())
        self._tool_call_index: int = 0
        self._current_tool_index: int = 0
        self._pending_stop_reason: str | None = None
        self._pending_stop_sequence: str | None = None
        self._pending_stop_details: dict[str, Any] | None = None
        self._pending_container: dict[str, Any] | None = None
        self._pending_usage: dict[str, Any] | None = None
        # Set once the final chunk (stop_reason + usage) has been emitted;
        # ``_pending_usage`` is intentionally kept afterwards so that
        # ``get_usage()`` still works when called after the stream ends.
        self._final_chunk_emitted: bool = False
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cache_read_input_tokens: int = 0
        self._cache_creation_input_tokens: int = 0
        # Buffer for thinking signature across content_block_start → content_block_stop.
        self._thinking_signature_buffer: str = ""

    # ------------------------------------------------------------------
    # Public API — called by the adapter
    # ------------------------------------------------------------------

    def convert_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a single parsed Anthropic SSE event to an OpenAI chunk dict.

        Args:
            chunk: A parsed JSON dict from an Anthropic SSE ``data:`` line.

        Returns:
            An OpenAI ``chat.completion.chunk`` dict suitable for yielding
            from ``stream_chat_completion()``, or ``None`` if the event
            should not produce a client-visible chunk (e.g. ``ping``,
            accumulation-only events).
        """
        event_type = chunk.get("type", "")
        handler = _EVENT_HANDLERS.get(event_type)
        if handler is not None:
            return handler(self, chunk)
        return None  # unknown event → skip

    def get_usage(self) -> StreamingUsage | None:
        """Return accumulated usage information from the streaming response."""
        if self._pending_usage is not None:
            # Web search request count (server_tool_use.web_search_requests)
            web_search_requests: int | None = None
            server_tool_use = self._pending_usage.get("server_tool_use")
            if isinstance(server_tool_use, dict):
                ws_count = server_tool_use.get("web_search_requests")
                if isinstance(ws_count, int) and ws_count > 0:
                    web_search_requests = ws_count

            return StreamingUsage(
                input_tokens=self._pending_usage.get("prompt_tokens", 0),
                output_tokens=self._pending_usage.get("completion_tokens", 0),
                total_tokens=self._pending_usage.get("total_tokens", 0),
                cache_read_input_tokens=self._pending_usage.get("cache_read_input_tokens"),
                cache_creation_input_tokens=self._pending_usage.get("cache_creation_input_tokens"),
                web_search_requests=web_search_requests,
            )
        return None

    # ------------------------------------------------------------------
    # Event handlers  (one method per Anthropic event type)
    # ------------------------------------------------------------------

    def _handle_message_start(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """message_start: extract model, usage; emit initial role chunk."""
        msg = event.get("message", {})
        self._response_id = msg.get("id", self._response_id)
        usage = msg.get("usage", {})
        self._input_tokens = usage.get("input_tokens", 0) or 0
        self._output_tokens = usage.get("output_tokens", 0) or 0
        self._cache_read_input_tokens = usage.get("cache_read_input_tokens") or 0
        self._cache_creation_input_tokens = usage.get("cache_creation_input_tokens") or 0

        chunk = _make_openai_chunk(
            self._response_id,
            self._model,
            self._created_at,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        )
        # Cache-diagnostics beta: the message-level ``diagnostics`` travels
        # with ``message_start``; forward it on the canonical channel so the
        # protocol transformer can replay it inside its message_start.
        diagnostics = msg.get("diagnostics")
        if diagnostics is not None:
            chunk["diagnostics"] = diagnostics
        return chunk

    def _handle_content_block_start(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """content_block_start: emit placeholder chunk for new content block."""
        block = event.get("content_block", {})
        block_type = block.get("type", "")

        if block_type == "text":
            return _make_openai_chunk(
                self._response_id,
                self._model,
                self._created_at,
                choices=[
                    {
                        "index": 0,
                        "delta": {"content": ""},
                        "finish_reason": None,
                    }
                ],
            )

        if block_type == "thinking":
            self._thinking_signature_buffer = ""
            return None  # No client-visible chunk; we accumulate deltas.

        if block_type == "redacted_thinking":
            self._thinking_signature_buffer = ""
            return _make_openai_chunk(
                self._response_id,
                self._model,
                self._created_at,
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "[redacted]",
                            "reasoning_is_redacted": True,
                        },
                        "finish_reason": None,
                    }
                ],
            )

        if block_type in ("tool_use", "server_tool_use"):
            tool_id = block.get("id", "")
            tool_name = block.get("name", "")
            self._current_tool_index = self._tool_call_index
            self._tool_call_index += 1
            return _make_openai_chunk(
                self._response_id,
                self._model,
                self._created_at,
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": self._current_tool_index,
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            )

        # Unknown block type (web_search_tool_result, web_fetch_tool_result,
        # container_upload, ...): forward the complete block losslessly in a
        # ``raw_content_block`` delta slot so native-shaped providers can
        # replay it on their SSE stream; other protocol transformers ignore
        # the key (same as before, but no silent block loss).
        return _make_openai_chunk(
            self._response_id,
            self._model,
            self._created_at,
            choices=[
                {
                    "index": 0,
                    "delta": {"raw_content_block": block},
                    "finish_reason": None,
                }
            ],
        )

    def _handle_content_block_delta(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """content_block_delta: emit delta chunk for an in-progress block."""
        delta = event.get("delta", {})
        delta_type = delta.get("type", "")

        if delta_type == "text_delta":
            text = delta.get("text", "")
            return _make_openai_chunk(
                self._response_id,
                self._model,
                self._created_at,
                choices=[
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            )

        if delta_type == "thinking_delta":
            text = delta.get("thinking", "")
            return _make_openai_chunk(
                self._response_id,
                self._model,
                self._created_at,
                choices=[
                    {
                        "index": 0,
                        "delta": {"reasoning_content": text},
                        "finish_reason": None,
                    }
                ],
            )

        if delta_type == "signature_delta":
            sig = delta.get("signature", "")
            self._thinking_signature_buffer += sig
            return None  # Accumulate; emitted on content_block_stop.

        if delta_type == "input_json_delta":
            partial = delta.get("partial_json", "")
            return _make_openai_chunk(
                self._response_id,
                self._model,
                self._created_at,
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": self._current_tool_index,
                                    "id": None,
                                    "function": {"arguments": partial},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            )

        if delta_type == "citations_delta":
            # Lossless citation passthrough (attached to the current text block).
            citation = delta.get("citation")
            return _make_openai_chunk(
                self._response_id,
                self._model,
                self._created_at,
                choices=[
                    {
                        "index": 0,
                        "delta": {"citations": citation},
                        "finish_reason": None,
                    }
                ],
            )

        return None  # Unknown delta type → skip

    def _handle_content_block_stop(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """content_block_stop: emit buffered signature if any."""
        sig = self._thinking_signature_buffer
        self._thinking_signature_buffer = ""
        if sig:
            return _make_openai_chunk(
                self._response_id,
                self._model,
                self._created_at,
                choices=[
                    {
                        "index": 0,
                        "delta": {"reasoning_signature": sig},
                        "finish_reason": None,
                    }
                ],
            )
        return None

    def _handle_message_delta(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """message_delta: capture stop_reason/stop_sequence/usage; no chunk emitted."""
        delta = event.get("delta", {})
        stop_reason = delta.get("stop_reason")
        if stop_reason:
            self._pending_stop_reason = _map_stop_reason(stop_reason)
        stop_sequence = delta.get("stop_sequence")
        if stop_sequence is not None:
            self._pending_stop_sequence = stop_sequence
        stop_details = delta.get("stop_details")
        if stop_details:
            self._pending_stop_details = stop_details
        # Container info (code execution) rides the canonical channel the
        # same way as stop_details, for the protocol transformer to replay.
        container = delta.get("container")
        if container is not None:
            self._pending_container = container

        usage = event.get("usage", {})
        if usage:
            self._output_tokens = usage.get("output_tokens", 0) or 0
            total_input = (
                self._input_tokens
                + self._cache_read_input_tokens
                + self._cache_creation_input_tokens
            )
            self._pending_usage = {
                "prompt_tokens": total_input,
                "completion_tokens": self._output_tokens,
                "total_tokens": total_input + self._output_tokens,
                "cache_read_input_tokens": self._cache_read_input_tokens,
                "cache_creation_input_tokens": self._cache_creation_input_tokens,
            }
            # Preserve server_tool_use for web search billing (Anthropic)
            server_tool_use = usage.get("server_tool_use")
            if server_tool_use is not None:
                self._pending_usage["server_tool_use"] = server_tool_use
            # Anthropic-native usage extensions (output_tokens_details.
            # thinking_tokens, service_tier, fast-mode "speed", compaction/
            # fallback "iterations") travel losslessly to the protocol
            # transformer's passthrough channel.
            for key in ("output_tokens_details", "service_tier", "speed", "iterations"):
                if usage.get(key) is not None:
                    self._pending_usage[key] = usage[key]
            # Also normalize provider-native counters into the OpenAI-dialect
            # details objects (cached_tokens / reasoning_tokens), so canonical-
            # channel consumers such as the OpenResponses usage folding can
            # read a single dialect instead of special-casing Anthropic keys.
            if self._cache_read_input_tokens:
                prompt_details = self._pending_usage.setdefault("prompt_tokens_details", {})
                prompt_details["cached_tokens"] = self._cache_read_input_tokens
            output_details = usage.get("output_tokens_details")
            thinking_tokens = (
                output_details.get("thinking_tokens") if isinstance(output_details, dict) else None
            )
            if thinking_tokens is not None:
                completion_details = self._pending_usage.setdefault("completion_tokens_details", {})
                completion_details["reasoning_tokens"] = thinking_tokens
        return None  # State-only event; chunk emitted on message_stop.

    def _handle_error(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """error: propagate mid-stream upstream errors instead of swallowing them.

        Raising a ProviderError lets the streaming pipeline surface the real
        failure (and its Anthropic error type) through the protocol
        transformer's ``error_frames`` instead of truncating the stream
        silently at this point.
        """
        from llm_proxy.core.exceptions import ProviderError

        error = event.get("error", {})
        error_type = error.get("type") or "api_error"
        message = error.get("message") or "Upstream error event in stream"
        raise ProviderError(message=message, error_type=error_type, provider_name="anthropic")

    def _build_final_chunk(self) -> dict[str, Any] | None:
        """Build a final chunk with pending stop fields and usage, if any."""
        if self._final_chunk_emitted:
            return None
        choice: dict[str, Any] = {"index": 0, "delta": {}}
        if self._pending_stop_reason:
            choice["finish_reason"] = self._pending_stop_reason
            self._pending_stop_reason = None
        # Anthropic-native terminal fields preserved for the wire format.
        if self._pending_stop_sequence is not None:
            choice["stop_sequence"] = self._pending_stop_sequence
            self._pending_stop_sequence = None
        if self._pending_stop_details is not None:
            choice["stop_details"] = self._pending_stop_details
            self._pending_stop_details = None
        if self._pending_container is not None:
            # Container info survives to the protocol transformer; emit even
            # without stop_reason/usage so it is never silently dropped.
            choice["container"] = self._pending_container
            self._pending_container = None
        chunk_data: dict[str, Any] = {"choices": [choice]}
        if self._pending_usage:
            chunk_data["usage"] = self._pending_usage
        if (
            not choice.get("finish_reason")
            and "container" not in choice
            and not chunk_data.get("usage")
        ):
            return None
        # Keep ``_pending_usage`` for post-stream ``get_usage()`` calls.
        self._final_chunk_emitted = True
        return _make_openai_chunk(self._response_id, self._model, self._created_at, **chunk_data)

    def _handle_message_stop(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """message_stop: emit final chunk with finish_reason and usage."""
        return self._build_final_chunk()

    def _handle_ping(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """ping: keepalive event, no client-visible chunk."""
        return None

    # ------------------------------------------------------------------
    # StreamingTransformer protocol (protocol-side, for completeness)
    # ------------------------------------------------------------------

    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Protocol-side transform — not used for provider-side conversion.

        This method exists only to satisfy the ``StreamingTransformer`` ABC.
        ``AnthropicChunkConverter`` is a provider-side converter; the
        protocol-side transform is handled by
        ``AnthropicStreamingTransformer``.
        """
        raise NotImplementedError(
            "AnthropicChunkConverter is a provider-side converter. "
            "Use AnthropicStreamingTransformer for protocol-side SSE formatting."
        )

    def finalize(self) -> str:
        """Stream end marker (protocol-side requirement).

        This method exists only to satisfy the ``StreamingTransformer`` ABC.
        The adapter yields ``[DONE]`` directly; this is a fallback that is
        never called in the provider-side role.
        """
        return "[DONE]"

    def finalize_chunks(self) -> list[dict[str, Any]]:
        """Return any pending chunks on premature stream end (no message_stop)."""
        chunk = self._build_final_chunk()
        return [chunk] if chunk else []


# ------------------------------------------------------------------
# Event handler dispatch table  (faster than a chain of if/elif)
# ------------------------------------------------------------------

_EVENT_HANDLERS: dict[str, Any] = {
    "message_start": AnthropicChunkConverter._handle_message_start,
    "content_block_start": AnthropicChunkConverter._handle_content_block_start,
    "content_block_delta": AnthropicChunkConverter._handle_content_block_delta,
    "content_block_stop": AnthropicChunkConverter._handle_content_block_stop,
    "message_delta": AnthropicChunkConverter._handle_message_delta,
    "message_stop": AnthropicChunkConverter._handle_message_stop,
    "ping": AnthropicChunkConverter._handle_ping,
    "error": AnthropicChunkConverter._handle_error,
}
