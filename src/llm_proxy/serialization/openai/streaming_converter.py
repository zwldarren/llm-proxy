"""OpenAI Responses provider-side streaming chunk converter.

Converts OpenAI Responses API SSE streaming events to canonical OpenAI
``chat.completion.chunk`` dict format.

The adapter is responsible for reading the HTTP stream and framing SSE
events; this converter handles the *event → chunk* mapping.

Migrated from ``providers/openai/streaming.py``.
"""

import time
from typing import Any

from llm_proxy.observability.logger import get_logger
from llm_proxy.serialization.responses_toolkit import generate_item_id
from llm_proxy.streaming.transformer import StreamingTransformer, StreamingUsage

logger = get_logger(__name__)


class OpenAIResponsesChunkConverter(StreamingTransformer):
    """Convert OpenAI Responses API SSE events to canonical OpenAI chunk dicts.

    Usage::

        converter = OpenAIResponsesChunkConverter(model="gpt-5", request_id="resp_1")
        async for sse_line in adapter._iter_stream_lines(response):
            event_type = ...  # extracted from the SSE "event:" line
            data = orjson.loads(sse_line)  # SSE "data:" payload (or None)
            if data is None:
                continue
            data["event_type"] = event_type
            chunk = converter.convert_chunk(data)
            if chunk is not None:
                yield chunk
        # After stream ends, flush final chunks:
        for chunk in converter.finalize_chunks():
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
        # Mutable state (was StreamingState in the old parser).
        self._response_id: str = ""
        self._model: str = model
        self._created_at: int = 0
        self._status: str = "in_progress"
        self._current_item_id: str = ""
        self._current_item_type: str = ""
        self._tool_calls: list[dict[str, Any]] = []
        self._active_tool_calls: dict[str, dict[str, Any]] = {}
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._reasoning_tokens: int = 0
        self._cached_tokens: int = 0
        self._audio_input_tokens: int = 0
        self._audio_output_tokens: int = 0
        # Native server-side web search call count (billed per request)
        self._web_search_call_count: int = 0
        self._pending_reasoning_encrypted: str | None = None
        # Final chunks to flush after stream ends.
        self._final_chunks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a single parsed SSE event to an OpenAI chunk dict.

        Args:
            chunk: A dict containing the event data.  For OpenAI Responses API
                   events, the dict should include an ``event_type`` key
                   (the SSE event type, e.g. ``"response.created"``) alongside
                   the event data fields.

        Returns:
            An OpenAI ``chat.completion.chunk`` dict or ``None``.
        """
        event_type = chunk.get("event_type")
        if not event_type:
            return None

        handler = _RESPONSES_EVENT_HANDLERS.get(event_type)
        if handler is not None:
            return handler(self, chunk)
        return None

    def finalize_chunks(self) -> list[dict[str, Any]]:
        """Return any pending chunks after the stream ends.

        Must be called after the SSE stream is exhausted.
        """
        chunks = list(self._final_chunks)
        self._final_chunks.clear()
        return chunks

    def _build_usage_details(self) -> tuple[dict[str, int] | None, dict[str, int] | None]:
        """Build prompt_tokens_details and completion_tokens_details from accumulated state."""
        prompt_details: dict[str, int] | None = None
        if self._cached_tokens or self._audio_input_tokens:
            prompt_details = {}
            if self._cached_tokens:
                prompt_details["cached_tokens"] = self._cached_tokens
            if self._audio_input_tokens:
                prompt_details["audio_tokens"] = self._audio_input_tokens
        completion_details: dict[str, int] | None = None
        if self._audio_output_tokens or self._reasoning_tokens:
            completion_details = {}
            if self._audio_output_tokens:
                completion_details["audio_tokens"] = self._audio_output_tokens
            if self._reasoning_tokens:
                completion_details["reasoning_tokens"] = self._reasoning_tokens
        return prompt_details, completion_details

    def get_usage(self) -> StreamingUsage | None:
        """Return accumulated usage information."""
        web_search_requests = self._web_search_call_count or None
        if not self._input_tokens and not self._output_tokens:
            if web_search_requests is None:
                return None
            # No token usage, but native web search calls are still billable.
            return StreamingUsage(web_search_requests=web_search_requests)
        prompt_details, completion_details = self._build_usage_details()
        return StreamingUsage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=self._input_tokens + self._output_tokens,
            prompt_tokens_details=prompt_details,
            completion_tokens_details=completion_details,
            web_search_requests=web_search_requests,
        )

    # ------------------------------------------------------------------
    # StreamingTransformer protocol (for completeness)
    # ------------------------------------------------------------------

    def transform(self, chunk: str | dict[str, Any]) -> str | None:  # noqa: ARG002
        """Protocol-side transform — not used for provider-side conversion.

        This method exists only to satisfy the ``StreamingTransformer`` ABC.
        """
        raise NotImplementedError("OpenAIResponsesChunkConverter is a provider-side converter.")

    def finalize(self) -> str:
        """Stream end marker (protocol-side requirement).

        This method exists only to satisfy the ``StreamingTransformer`` ABC.
        The adapter yields ``[DONE]`` directly; this is a fallback that is
        never called in the provider-side role.
        """
        return "[DONE]"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_content_chunk(self, delta: str) -> dict[str, Any]:
        """Build a canonical OpenAI chunk dict with a ``content`` delta."""
        return {
            "id": self._response_id or "chatcmpl-proxy",
            "object": "chat.completion.chunk",
            "created": self._created_at or time.time_ns() // 1_000_000_000,
            "model": self._model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": delta},
                    "finish_reason": None,
                }
            ],
        }

    def _resolve_tool_call(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Look up the active tool call for a delta/done event."""
        item_id = data.get("item_id")
        if item_id and item_id in self._active_tool_calls:
            return self._active_tool_calls[item_id]
        if self._active_tool_calls:
            return next(reversed(self._active_tool_calls.values()))
        return None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_response_created(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """response.created: capture response metadata."""
        response = data.get("response", {})
        self._response_id = response.get("id", "")
        self._created_at = response.get("created_at", time.time_ns() // 1_000_000_000)
        self._model = response.get("model", self._model)
        self._status = "in_progress"
        return None  # No client-visible chunk

    def _handle_output_item_added(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """response.output_item.added: track new item."""
        item = data.get("item", {})
        self._current_item_id = item.get("id", generate_item_id())
        self._current_item_type = item.get("type", "message")

        if self._current_item_type == "message":
            pass
        elif self._current_item_type == "function_call":
            self._active_tool_calls[self._current_item_id] = {
                "id": self._current_item_id,
                "call_id": item.get("call_id", ""),
                "name": item.get("name", ""),
                "arguments": "",
            }
        elif self._current_item_type == "reasoning":
            encrypted = item.get("encrypted_content")
            if encrypted:
                self._pending_reasoning_encrypted = encrypted
        elif self._current_item_type == "web_search_call":
            # Web search call metadata is tracked only via _current_item_id;
            # no client-visible chunk is emitted for it.
            self._web_search_call_count += 1
        return None

    def _handle_text_delta(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """response.output_text.delta: emit content delta chunk."""
        delta = data.get("delta", "")
        if not delta:
            return None
        return self._make_content_chunk(delta)

    def _handle_text_done(self, _data: dict[str, Any]) -> dict[str, Any] | None:
        """response.output_text.done: no client-visible chunk."""
        return None

    def _handle_reasoning_delta(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """response.reasoning.delta / response.reasoning_text.delta."""
        delta_text = data.get("delta", "")
        if not delta_text:
            return None

        reasoning_delta: dict[str, Any] = {"reasoning_content": delta_text}
        if self._pending_reasoning_encrypted:
            reasoning_delta["encrypted_content"] = self._pending_reasoning_encrypted
            self._pending_reasoning_encrypted = None

        return {
            "id": self._response_id or "chatcmpl-proxy",
            "object": "chat.completion.chunk",
            "created": self._created_at or time.time_ns() // 1_000_000_000,
            "model": self._model,
            "choices": [
                {
                    "index": 0,
                    "delta": reasoning_delta,
                    "finish_reason": None,
                }
            ],
        }

    def _handle_reasoning_done(self, _data: dict[str, Any]) -> dict[str, Any] | None:
        """response.reasoning.done / response.reasoning_text.done."""
        return None

    def _handle_refusal_delta(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """response.refusal.delta: emit refusal delta chunk."""
        delta = data.get("delta", "")
        if not delta:
            return None
        return {
            "id": self._response_id or "chatcmpl-proxy",
            "object": "chat.completion.chunk",
            "created": self._created_at or time.time_ns() // 1_000_000_000,
            "model": self._model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"refusal": delta},
                    "finish_reason": None,
                }
            ],
        }

    def _handle_refusal_done(self, _data: dict[str, Any]) -> dict[str, Any] | None:
        """response.refusal.done."""
        return None

    def _handle_function_call_delta(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """response.function_call_arguments.delta: accumulate arguments."""
        delta = data.get("delta", "")
        if not delta:
            return None
        tool_call = self._resolve_tool_call(data)
        if not tool_call:
            return None
        parts = tool_call.setdefault("_arguments_parts", [])
        if isinstance(parts, list):
            parts.append(delta)
        return None  # Accumulate; emitted on function_call_arguments.done

    def _handle_function_call_done(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """response.function_call_arguments.done: emit tool_calls chunk."""
        tool_call = self._resolve_tool_call(data)
        if not tool_call:
            return None

        item_id = tool_call["id"]
        self._active_tool_calls.pop(item_id, None)

        finalized = tool_call.copy()
        args_parts = finalized.pop("_arguments_parts", [])
        finalized["arguments"] = "".join(args_parts) if isinstance(args_parts, list) else ""
        self._tool_calls.append(finalized)

        return {
            "id": self._response_id or "chatcmpl-proxy",
            "object": "chat.completion.chunk",
            "created": self._created_at or time.time_ns() // 1_000_000_000,
            "model": self._model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": len(self._tool_calls) - 1,
                                "id": tool_call["call_id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call["name"],
                                    "arguments": finalized["arguments"],
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }

    def _handle_response_finished(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """response.completed / response.failed / response.incomplete."""
        response = data.get("response", {})
        self._status = response.get("status", "completed")

        usage = response.get("usage", {})
        if usage:
            self._input_tokens = usage.get("input_tokens", 0)
            self._output_tokens = usage.get("output_tokens", 0)
            if "output_tokens_details" in usage:
                self._reasoning_tokens = usage["output_tokens_details"].get("reasoning_tokens", 0)
            input_details = usage.get("input_tokens_details", {})
            if isinstance(input_details, dict):
                self._cached_tokens = input_details.get("cached_tokens", 0)
                self._audio_input_tokens = input_details.get("audio_tokens", 0)
            output_details = usage.get("output_tokens_details", {})
            if isinstance(output_details, dict):
                self._audio_output_tokens = output_details.get("audio_tokens", 0)

        finish_reason = "stop"
        if self._status == "completed":
            finish_reason = "stop" if not self._tool_calls else "tool_calls"
        elif self._status == "incomplete":
            finish_reason = "length"
        elif self._status == "failed":
            finish_reason = "error"

        # Best-effort: encrypted_content from final reasoning items.
        if self._status == "completed":
            for item in response.get("output", []):
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    encrypted = item.get("encrypted_content")
                    if encrypted:
                        enc_chunk = {
                            "id": self._response_id or "chatcmpl-proxy",
                            "object": "chat.completion.chunk",
                            "created": self._created_at or time.time_ns() // 1_000_000_000,
                            "model": self._model,
                            "encrypted_content": encrypted,
                        }
                        self._final_chunks.append(enc_chunk)
                    break

        chunk: dict[str, Any] = {
            "id": self._response_id or "chatcmpl-proxy",
            "object": "chat.completion.chunk",
            "created": self._created_at or time.time_ns() // 1_000_000_000,
            "model": self._model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }

        if usage:
            chunk_usage: dict[str, Any] = {
                "prompt_tokens": self._input_tokens,
                "completion_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
            }
            prompt_details, completion_details = self._build_usage_details()
            if prompt_details:
                chunk_usage["prompt_tokens_details"] = prompt_details
            if completion_details:
                chunk_usage["completion_tokens_details"] = completion_details
            chunk["usage"] = chunk_usage

        self._final_chunks.append(chunk)
        return None  # Chunks queued in _final_chunks; adapter flushes them.

    def _handle_error(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """error event: emit error chunk."""
        error = data.get("error", {})
        return {
            "error": {
                "type": error.get("type", "api_error"),
                "code": error.get("code"),
                "message": error.get("message", "An error occurred"),
                "param": error.get("param"),
            }
        }


# ------------------------------------------------------------------
# Event handler dispatch table
# ------------------------------------------------------------------

_RESPONSES_EVENT_HANDLERS: dict[str, Any] = {
    "response.created": OpenAIResponsesChunkConverter._handle_response_created,
    "response.output_item.added": (OpenAIResponsesChunkConverter._handle_output_item_added),
    "response.output_text.delta": OpenAIResponsesChunkConverter._handle_text_delta,
    "response.output_text.done": OpenAIResponsesChunkConverter._handle_text_done,
    "response.reasoning.delta": (OpenAIResponsesChunkConverter._handle_reasoning_delta),
    "response.reasoning_text.delta": (OpenAIResponsesChunkConverter._handle_reasoning_delta),
    "response.reasoning.done": (OpenAIResponsesChunkConverter._handle_reasoning_done),
    "response.reasoning_text.done": (OpenAIResponsesChunkConverter._handle_reasoning_done),
    "response.refusal.delta": (OpenAIResponsesChunkConverter._handle_refusal_delta),
    "response.refusal.done": OpenAIResponsesChunkConverter._handle_refusal_done,
    "response.function_call_arguments.delta": (
        OpenAIResponsesChunkConverter._handle_function_call_delta
    ),
    "response.function_call_arguments.done": (
        OpenAIResponsesChunkConverter._handle_function_call_done
    ),
    "response.completed": (OpenAIResponsesChunkConverter._handle_response_finished),
    "response.failed": OpenAIResponsesChunkConverter._handle_response_finished,
    "response.incomplete": OpenAIResponsesChunkConverter._handle_response_finished,
    "error": OpenAIResponsesChunkConverter._handle_error,
}
