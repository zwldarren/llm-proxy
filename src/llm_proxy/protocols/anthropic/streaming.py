"""Anthropic streaming transformer.

Converts canonical ``chat.completion.chunk`` dicts into Anthropic SSE wire
format. Protocol-side transformer — lives next to the Anthropic protocol
module; the provider-side chunk converter stays in
``serialization/anthropic/streaming_converter.py``.
"""

from typing import Any

import orjson

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    RawBlock,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llm_proxy.models.finish_reasons import OPENAI_TO_ANTHROPIC
from llm_proxy.serialization.anthropic import ANTHROPIC_USAGE_EXTENSION_KEYS
from llm_proxy.streaming.transformer import StreamingTransformer, StreamingUsage


def _message_delta_usage(usage: dict) -> dict:
    """Shape the terminal ``message_delta`` usage block.

    Everything except ``service_tier`` passes through: ``service_tier`` is
    only valid on the full ``Usage`` object (emitted in ``message_start``).
    The remaining keys — cache counters, ``output_tokens_details``, and the
    beta extensions (fast-mode ``speed``, compaction/fallback
    ``iterations``) — ride through verbatim: SDKs ignore unknown keys, and
    stripping the compaction counter would break per-iteration cost
    accounting, since top-level tokens EXCLUDE compaction iterations.
    """
    return {k: v for k, v in usage.items() if k != "service_tier"}


class AnthropicStreamingTransformer(StreamingTransformer):
    """Transform OpenAI SSE chunks to Anthropic SSE format."""

    def __init__(
        self,
        model: str = "",
        request_id: str | None = None,
        estimated_input_tokens: int = 0,
        intercept_web_search: bool = True,
    ):
        super().__init__(model, request_id)
        self._intercept_web_search = intercept_web_search
        self._input_tokens = estimated_input_tokens
        self._output_tokens = 0
        self._sent_message_start = False
        self._current_block_index = 0
        self._in_text_block = False
        self._in_thinking_block = False
        self._in_tool_block = False
        self._text_buffer = ""
        self._thinking_buffer = ""
        self._thinking_signature = ""
        self._thinking_is_redacted = False
        self._text_output_started = False
        self._tool_id = ""
        self._tool_name = ""
        self._tool_args = ""
        self._pending_stop_reason: str | None = None
        self._pending_stop_sequence: str | None = None
        self._pending_stop_details: dict[str, Any] | None = None
        self._pending_container: dict[str, Any] | None = None
        self._pending_usage: dict | None = None
        self._has_pending_usage = False
        # Cache-diagnostics beta: message-level object arriving on the
        # canonical channel from the provider converter; replayed inside
        # message_start.message.
        self._pending_diagnostics: dict[str, Any] | None = None

    @classmethod
    def continuation(
        cls, model: str, request_id: str, start_index: int
    ) -> AnthropicStreamingTransformer:
        """Create a transformer that continues an existing stream.

        Skips the message_start event (already emitted) and starts
        content block indices from start_index.
        """
        instance = cls(model=model, request_id=request_id)
        instance._sent_message_start = True
        instance._current_block_index = start_index
        return instance

    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Transform OpenAI SSE chunk to Anthropic format.

        Converts OpenAI-style chunks (with choices, delta, content, etc.)
        to Anthropic SSE events (message_start, content_block_start,
        content_block_delta, content_block_stop, message_delta, message_stop).

        Filters out non-standard fields and converts the format appropriately.
        """
        if isinstance(chunk, dict):
            return self._transform_openai_chunk(chunk)

        if not chunk or not chunk.strip():
            return None

        # Skip non-data lines and [DONE]
        if not chunk.startswith("data: "):
            return None
        if "[DONE]" in chunk:
            return None

        data = chunk[6:].strip()
        if not data:
            return None

        try:
            parsed = orjson.loads(data)
            return self._transform_openai_chunk(parsed)
        except orjson.JSONDecodeError:
            return None

    def _transform_openai_chunk(self, chunk: dict) -> str | None:
        """Transform a parsed OpenAI chunk to Anthropic SSE events."""
        result_chunks = []

        choices = chunk.get("choices", [])
        usage = chunk.get("usage")

        if usage:
            input_tokens = self._extract_input_tokens(usage)
            output_tokens = self._extract_output_tokens(usage)
            if input_tokens > 0 or self._input_tokens == 0:
                self._input_tokens = input_tokens
            if output_tokens > 0 or self._output_tokens == 0:
                self._output_tokens = output_tokens
            # Fold usage before emitting message_start so its ``usage`` block
            # can carry the full Anthropic shape (cache keys, server_tool_use).
            normalized = self._normalize_usage(usage)
            if self._pending_usage is not None:
                for key, value in normalized.items():
                    if value or key not in self._pending_usage:
                        self._pending_usage[key] = value
            else:
                self._pending_usage = normalized
                self._has_pending_usage = True
        # Cache-diagnostics beta: capture before the message_start emission so
        # the first message (emitted below) already carries it.
        diag = chunk.get("diagnostics")
        if diag is not None:
            self._pending_diagnostics = diag

        # Send message_start on first chunk with content
        if not self._sent_message_start and (choices or usage):
            self._sent_message_start = True
            result_chunks.append(
                self._message_start(
                    input_tokens=self._input_tokens,
                    output_tokens=self._output_tokens,
                )
            )

        # Process content from choices
        if choices:
            for choice in choices:
                if choice is None:
                    continue
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Handle text content
                content = delta.get("content")
                if content is not None and content != "":
                    if self._in_thinking_block:
                        # Providers can emit reasoning first and then normal text.
                        # Close thinking block before switching back to text.
                        result_chunks.append(self._content_block_stop(self._current_block_index))
                        self._current_block_index += 1
                        self._in_thinking_block = False

                    if not self._in_text_block:
                        self._in_text_block = True
                        result_chunks.append(
                            self._content_block_start(
                                self._current_block_index, {"type": "text", "text": ""}
                            )
                        )
                    if self._in_text_block:
                        self._text_output_started = True
                        self._text_buffer += content
                        result_chunks.append(
                            self._content_block_delta(
                                self._current_block_index, {"type": "text_delta", "text": content}
                            )
                        )

                # Handle reasoning/thinking content
                reasoning = delta.get("reasoning_content")
                if reasoning is None or reasoning == "":
                    reasoning = delta.get("reasoning")
                if reasoning is not None and reasoning != "":
                    if self._text_output_started:
                        continue

                    is_redacted = delta.get("reasoning_is_redacted", False)
                    if is_redacted and reasoning == "[redacted]":
                        if self._in_text_block:
                            result_chunks.append(
                                self._content_block_stop(self._current_block_index)
                            )
                            self._current_block_index += 1
                            self._in_text_block = False
                        result_chunks.append(
                            self._content_block_start(
                                self._current_block_index,
                                {"type": "redacted_thinking", "data": reasoning},
                            )
                        )
                        self._in_thinking_block = True
                        self._thinking_is_redacted = True
                        self._thinking_buffer = reasoning
                        continue

                    if is_redacted:
                        self._thinking_is_redacted = True

                    if not self._in_thinking_block:
                        if self._in_text_block:
                            result_chunks.append(
                                self._content_block_stop(self._current_block_index)
                            )
                            self._current_block_index += 1
                            self._in_text_block = False
                        self._in_thinking_block = True
                        result_chunks.append(
                            self._content_block_start(
                                self._current_block_index, {"type": "thinking", "thinking": ""}
                            )
                        )
                    self._thinking_buffer += reasoning
                    result_chunks.append(
                        self._content_block_delta(
                            self._current_block_index,
                            {"type": "thinking_delta", "thinking": reasoning},
                        )
                    )

                reasoning_sig = delta.get("reasoning_signature")
                if reasoning_sig is not None and reasoning_sig != "":
                    self._thinking_signature += reasoning_sig
                    if self._in_thinking_block:
                        result_chunks.append(
                            self._content_block_delta(
                                self._current_block_index,
                                {"type": "signature_delta", "signature": reasoning_sig},
                            )
                        )

                # Handle tool calls
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        # Tool call start
                        if tc.get("id") and tc.get("function", {}).get("name"):
                            if self._in_tool_block:
                                # Close previous tool block before starting the next one.
                                # Accumulate the previous tool before closing
                                if self._tool_id:
                                    try:
                                        tool_input = (
                                            orjson.loads(self._tool_args) if self._tool_args else {}
                                        )
                                    except orjson.JSONDecodeError:
                                        tool_input = {}

                                    normalized = (
                                        self._tool_name.lower().replace("_", "").replace("-", "")
                                    )
                                    if normalized == "websearch" and self._intercept_web_search:
                                        self._accumulated_output.append(
                                            ServerToolUseBlock(
                                                id=self._tool_id,
                                                name=self._tool_name,
                                                input=tool_input,
                                            )
                                        )
                                    else:
                                        self._accumulated_output.append(
                                            ToolUseBlock(
                                                id=self._tool_id,
                                                name=self._tool_name,
                                                input=tool_input,
                                            )
                                        )
                                result_chunks.append(
                                    self._content_block_stop(self._current_block_index)
                                )
                                self._current_block_index += 1
                                self._in_tool_block = False
                                self._tool_id = ""
                                self._tool_name = ""
                                self._tool_args = ""

                            if self._in_text_block or self._in_thinking_block:
                                # Close previous block
                                result_chunks.append(
                                    self._content_block_stop(self._current_block_index)
                                )
                                self._current_block_index += 1
                                self._in_text_block = False
                                self._in_thinking_block = False

                            self._in_tool_block = True
                            self._tool_id = tc["id"]
                            self._tool_name = tc["function"]["name"]
                            self._tool_args = ""  # Reset for new tool

                            normalized = self._tool_name.lower().replace("_", "").replace("-", "")
                            is_ws = normalized == "websearch" and self._intercept_web_search
                            block_type = "server_tool_use" if is_ws else "tool_use"
                            result_chunks.append(
                                self._content_block_start(
                                    self._current_block_index,
                                    {
                                        "type": block_type,
                                        "id": self._tool_id,
                                        "name": self._tool_name,
                                        "input": {},
                                    },
                                )
                            )

                        # Tool call arguments delta
                        args = tc.get("function", {}).get("arguments")
                        if args and self._in_tool_block:
                            self._tool_args += args  # Accumulate arguments
                            result_chunks.append(
                                self._content_block_delta(
                                    self._current_block_index,
                                    {"type": "input_json_delta", "partial_json": args},
                                )
                            )
                # Server-side tool result blocks arriving losslessly from the
                # converter (web_search_tool_result, web_fetch_tool_result,
                # container_upload, ...): replay as a start+stop pair so
                # native-shaped clients keep the full block.
                raw_block = delta.get("raw_content_block")
                if isinstance(raw_block, dict):
                    self._close_current_open_block(result_chunks)
                    block_events = [self._content_block_start(self._current_block_index, raw_block)]
                    block_events.append(self._content_block_stop(self._current_block_index))
                    result_chunks.extend(block_events)
                    self._current_block_index += 1
                    self._accumulated_output.append(
                        RawBlock(
                            provider_type=f"anthropic:{raw_block.get('type', 'unknown')}",
                            data=raw_block,
                        )
                    )

                # Citations attach only to text blocks: close any open
                # tool/thinking block first, then land the citation on a
                # (possibly fresh) text block.
                citation = delta.get("citations")
                if isinstance(citation, dict):
                    if self._in_tool_block or self._in_thinking_block:
                        self._close_current_open_block(result_chunks)
                    if not self._in_text_block:
                        self._in_text_block = True
                        result_chunks.append(
                            self._content_block_start(
                                self._current_block_index, {"type": "text", "text": ""}
                            )
                        )
                    result_chunks.append(
                        self._content_block_delta(
                            self._current_block_index,
                            {"type": "citations_delta", "citation": citation},
                        )
                    )

                if finish_reason:
                    self._close_current_open_block(result_chunks)
                    self._pending_stop_reason = self._map_finish_reason(finish_reason)
                # Anthropic-native terminal fields arriving through the
                # converter's canonical channel: choice["stop_sequence"] and
                # choice["stop_details"]. Read regardless of finish_reason so
                # finalize flushes without a stop_reason still replay them.
                stop_sequence = choice.get("stop_sequence")
                if stop_sequence is not None:
                    self._pending_stop_sequence = stop_sequence
                stop_details = choice.get("stop_details")
                if stop_details is not None:
                    self._pending_stop_details = stop_details
                container = choice.get("container")
                if container is not None:
                    self._pending_container = container

        return "".join(result_chunks) if result_chunks else None

    def _map_finish_reason(self, finish_reason: str) -> str:
        """Map OpenAI finish_reason to Anthropic stop_reason."""
        return OPENAI_TO_ANTHROPIC.get(finish_reason, finish_reason or "end_turn")

    def _extract_token_count(self, usage: dict | None, *keys: str) -> int:
        """Extract token count from usage while preserving explicit zero values."""
        if not usage:
            return 0

        for key in keys:
            if key in usage and usage[key] is not None:
                return int(usage[key])
        return 0

    def _extract_input_tokens(self, usage: dict | None) -> int:
        """Extract input token count from usage payload."""
        return self._extract_token_count(usage, "prompt_tokens", "input_tokens")

    def _extract_output_tokens(self, usage: dict | None) -> int:
        """Extract output token count from usage payload."""
        return self._extract_token_count(usage, "completion_tokens", "output_tokens")

    def _normalize_usage(self, usage: dict) -> dict[str, Any]:
        """Normalize provider usage keys to Anthropic-compatible usage keys."""
        normalized: dict[str, Any] = {}

        if "prompt_tokens" in usage and usage["prompt_tokens"] is not None:
            normalized["input_tokens"] = int(usage["prompt_tokens"])
        elif "input_tokens" in usage and usage["input_tokens"] is not None:
            normalized["input_tokens"] = int(usage["input_tokens"])

        if "completion_tokens" in usage and usage["completion_tokens"] is not None:
            normalized["output_tokens"] = int(usage["completion_tokens"])
        elif "output_tokens" in usage and usage["output_tokens"] is not None:
            normalized["output_tokens"] = int(usage["output_tokens"])

        for key in (
            *ANTHROPIC_USAGE_EXTENSION_KEYS,
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "server_tool_use",
            "prompt_tokens_details",
            "completion_tokens_details",
        ):
            if key in usage and usage[key] is not None:
                normalized[key] = usage[key]

        return normalized

    def _message_start(self, input_tokens: int = 0, output_tokens: int = 0) -> str:
        """Generate message_start event."""
        # Anthropic's message_start.usage carries the full usage shape; fold
        # in cache/server-tool keys already accumulated from the first chunk.
        start_usage: dict[str, Any] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        if self._pending_usage:
            for key in (
                *ANTHROPIC_USAGE_EXTENSION_KEYS,
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "server_tool_use",
            ):
                value = self._pending_usage.get(key)
                if value is not None:
                    start_usage[key] = value
        message: dict[str, Any] = {
            "id": self.response_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": start_usage,
        }
        if self._pending_diagnostics is not None:
            # Cache-diagnostics beta: official streams attach ``diagnostics``
            # to the message_start message.
            message["diagnostics"] = self._pending_diagnostics
            self._pending_diagnostics = None
        return self._sse_event("message_start", {"type": "message_start", "message": message})

    def _content_block_start(self, index: int, block: dict[str, Any]) -> str:
        """Generate content_block_start event."""
        return self._sse_event(
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": block},
        )

    def _content_block_delta(self, index: int, delta: dict[str, Any]) -> str:
        """Generate content_block_delta event."""
        return self._sse_event(
            "content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta}
        )

    def _content_block_stop(self, index: int) -> str:
        """Generate content_block_stop event."""
        return self._sse_event("content_block_stop", {"type": "content_block_stop", "index": index})

    def _message_delta_with_stop_reason_and_usage(
        self, stop_reason: str, usage: dict[str, Any]
    ) -> str:
        """Generate message_delta event with stop_reason and usage combined."""
        delta: dict[str, Any] = {"stop_reason": stop_reason}
        if self._pending_stop_sequence is not None:
            delta["stop_sequence"] = self._pending_stop_sequence
            self._pending_stop_sequence = None
        if self._pending_stop_details is not None:
            delta["stop_details"] = self._pending_stop_details
            self._pending_stop_details = None
        if self._pending_container is not None:
            delta["container"] = self._pending_container
            self._pending_container = None
        return self._sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": delta,
                "usage": usage,
            },
        )

    def _message_delta_with_usage(self, usage: dict[str, Any]) -> str:
        """Generate message_delta event with usage only."""
        return self._sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {},
                "usage": usage,
            },
        )

    def _message_stop(self) -> str:
        """Generate message_stop event."""
        return self._sse_event("message_stop", {"type": "message_stop"})

    def _web_search_result_block(
        self,
        index: int,
        tool_use_id: str,
        results: list[dict[str, Any]] | list[str],
        is_error: bool = False,
        query: str = "",
    ) -> str:
        """Generate complete web_search_tool_result content block events.

        Generates: content_block_start, content_block_stop with results

        Args:
            index: Content block index
            tool_use_id: The ID of the corresponding server_tool_use
            results: List of web search result dicts
            is_error: Whether this is an error result
            query: The search query (ignored for Anthropic protocol)

        Returns:
            SSE events string for the complete web_search_tool_result block
        """
        events = []
        if is_error:
            default_error = '{"type": "web_search_tool_result_error", "error_code": "unavailable"}'
            content = results[0] if results else default_error
            events.append(
                self._content_block_start(
                    index,
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                    },
                )
            )
        else:
            events.append(
                self._content_block_start(
                    index,
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": tool_use_id,
                        "content": results,
                    },
                )
            )
        events.append(self._content_block_stop(index))
        return "".join(events)

    def _close_current_open_block(self, result_chunks: list[str]) -> None:
        """Close the currently open content block if one exists."""
        if not (self._in_text_block or self._in_thinking_block or self._in_tool_block):
            return

        # If closing a text block, accumulate the TextBlock
        if self._in_text_block and self._text_buffer:
            self._accumulated_output.append(TextBlock(text=self._text_buffer))

        # If closing a thinking block, accumulate the ThinkingBlock or RedactedThinkingBlock
        if self._in_thinking_block and self._thinking_buffer:
            if self._thinking_is_redacted:
                from llm_proxy.models import RedactedThinkingBlock

                self._accumulated_output.append(RedactedThinkingBlock(data=self._thinking_buffer))
            else:
                self._accumulated_output.append(
                    ThinkingBlock(
                        thinking=self._thinking_buffer,
                        signature=self._thinking_signature or None,
                    )
                )
            self._thinking_signature = ""
            self._thinking_is_redacted = False

        # If closing a tool block, accumulate the ToolUseBlock
        if self._in_tool_block and self._tool_id:
            try:
                tool_input = orjson.loads(self._tool_args) if self._tool_args else {}
            except orjson.JSONDecodeError:
                tool_input = {}

            self._accumulated_output.append(
                ToolUseBlock(
                    id=self._tool_id,
                    name=self._tool_name,
                    input=tool_input,
                )
            )

        result_chunks.append(self._content_block_stop(self._current_block_index))
        self._current_block_index += 1
        self._in_text_block = False
        self._in_thinking_block = False
        self._in_tool_block = False
        self._tool_id = ""
        self._tool_name = ""
        self._tool_args = ""
        self._text_buffer = ""
        self._thinking_buffer = ""

    def _sse_event(self, event_type: str, data: dict[str, Any]) -> str:
        """Create an SSE event string."""
        return f"event: {event_type}\ndata: {orjson.dumps(data).decode()}\n\n"

    def finalize(self) -> str:
        """Generate stream end marker.

        Sends one message_delta with latest cached usage/stop_reason and message_stop.
        """
        result_chunks = []

        self._close_current_open_block(result_chunks)

        if self._pending_stop_reason is not None:
            usage = self._pending_usage if self._pending_usage is not None else {"output_tokens": 0}
            result_chunks.append(
                self._message_delta_with_stop_reason_and_usage(
                    self._pending_stop_reason,
                    _message_delta_usage(usage),
                )
            )
            self._pending_stop_reason = None
            self._has_pending_usage = False
        elif self._has_pending_usage and self._pending_usage is not None:
            # Official message_delta events always carry a stop_reason alongside
            # usage; fall back to "end_turn" when the upstream never supplied
            # one (degenerate truncation path).
            result_chunks.append(
                self._message_delta_with_stop_reason_and_usage(
                    "end_turn",
                    _message_delta_usage(self._pending_usage),
                )
            )
            self._has_pending_usage = False

        result_chunks.append(self._message_stop())
        return "".join(result_chunks)

    def error_frames(self, exc: Exception) -> list[str]:
        """Anthropic SSE errors are named events (``event: error``).

        A bare data frame would be ignored by the SDK's event dispatcher. No
        [DONE] marker: the Anthropic wire format terminates on connection close.
        """
        error_type = getattr(exc, "error_type", None) if isinstance(exc, ProviderError) else None
        error_data = {
            "type": "error",
            "error": {"type": error_type or "api_error", "message": str(exc)},
        }
        return [f"event: error\ndata: {orjson.dumps(error_data).decode()}\n\n"]

    def get_usage(self) -> StreamingUsage | None:
        """Get accumulated usage information from streaming response.

        Returns:
            StreamingUsage object if usage data is available from provider.
        """
        # Fallback to _pending_usage if instance variables weren't set.
        # This handles cases where usage is provided but with zero values
        # (which wouldn't trigger the > 0 check in _transform_openai_chunk).
        if (
            not self._input_tokens
            and not self._output_tokens
            and self._pending_usage
            and isinstance(self._pending_usage, dict)
        ):
            self._input_tokens = self._pending_usage.get("input_tokens", 0)
            self._output_tokens = self._pending_usage.get("output_tokens", 0)

        if self._input_tokens or self._output_tokens:
            pending_usage = self._pending_usage
            # Extract nested details (OpenAI-style cached_tokens, audio_tokens)
            prompt_details = None
            if pending_usage:
                ptd = pending_usage.get("prompt_tokens_details")
                if isinstance(ptd, dict) and any(v is not None for v in ptd.values()):
                    prompt_details = {k: v for k, v in ptd.items() if v is not None}
                ctd = pending_usage.get("completion_tokens_details")
                completion_details = (
                    {k: v for k, v in ctd.items() if v is not None}
                    if isinstance(ctd, dict) and any(v is not None for v in ctd.values())
                    else None
                )

            cache_read = (
                pending_usage.get("cache_read_input_tokens") if pending_usage else None
            ) or 0
            cache_create = (
                pending_usage.get("cache_creation_input_tokens") if pending_usage else None
            ) or 0
            return StreamingUsage(
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                total_tokens=self._input_tokens + self._output_tokens + cache_read + cache_create,
                cache_read_input_tokens=cache_read if cache_read else None,
                cache_creation_input_tokens=cache_create if cache_create else None,
                prompt_tokens_details=prompt_details,
                completion_tokens_details=completion_details,
            )
        return None
