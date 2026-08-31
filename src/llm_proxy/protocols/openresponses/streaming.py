"""Streaming transformer for OpenResponses format.

Converts OpenAI streaming chunks to OpenResponses semantic events.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import orjson
from orjson import JSONDecodeError

from llm_proxy.models import ServerToolUseBlock, TextBlock, ThinkingBlock, ToolUseBlock
from llm_proxy.models.tools import is_web_search_tool_name
from llm_proxy.observability.logger import get_logger
from llm_proxy.protocols.openresponses.handler import get_format_context
from llm_proxy.protocols.openresponses.serializer import (
    conversation_to_input_items,
    incomplete_reason_from_finish,
)
from llm_proxy.protocols.openresponses.streaming_emitter import StreamingContentEmitter
from llm_proxy.protocols.openresponses.streaming_events import StreamingEventFactory
from llm_proxy.serialization.responses_toolkit import generate_item_id
from llm_proxy.streaming.transformer import StreamingTransformer, StreamingUsage

logger = get_logger(__name__)

# Raw reasoning delta/done events are emitted as the summary event family
# (``response.reasoning_summary_text.delta|done``), whose names are identical in
# the OpenResponses spec and the OpenAI Responses API.
_RAW_REASONING_EVENT_TYPES: dict[str, str] = {
    "response.reasoning.delta": "response.reasoning_summary_text.delta",
    "response.reasoning_text.delta": "response.reasoning_summary_text.delta",
    "response.reasoning.done": "response.reasoning_summary_text.done",
    "response.reasoning_text.done": "response.reasoning_summary_text.done",
}


def _normalize_reasoning_event(data: dict[str, Any], chunk_type: str) -> str:
    """Normalize a raw reasoning event to the summary event family.

    Returns the emitted event type. Raw reasoning delta/done events are
    converted in place to ``response.reasoning_summary_text.delta|done``
    (adding ``summary_index`` and dropping the content-part ``content_index``,
    which the summary events do not carry).
    """
    summary_type = _RAW_REASONING_EVENT_TYPES.get(chunk_type)
    if summary_type is None:
        return chunk_type
    data["type"] = summary_type
    data["summary_index"] = data.pop("content_index", 0)
    return summary_type


@dataclass
class OpenResponsesStreamingState:
    """Stateful context for OpenResponses streaming transformations."""

    response_created: bool = False
    current_item_index: int = 0
    current_content_index: dict[int, int] = field(default_factory=dict)
    accumulated_text: dict[tuple[int, int], str] = field(default_factory=dict)
    accumulated_tool_args: dict[int, str] = field(default_factory=dict)
    has_tool_calls: bool = False
    tool_call_ids: dict[int, str] = field(default_factory=dict)
    tool_call_names: dict[int, str] = field(default_factory=dict)
    tool_call_thought_signatures: dict[int, str] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    reasoning_tokens: int = 0
    is_done: bool = False
    pending_items: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Track current content type to detect transitions between reasoning/text
    current_content_type: str | None = None  # "reasoning_text", "output_text", etc.
    # Track content types per item (keyed by (item_idx, content_idx))
    content_types: dict[tuple[int, int], str] = field(default_factory=dict)
    # Reasoning summary text accumulated from response.reasoning_summary_text.delta
    # (keyed by (item_idx, content_idx))
    reasoning_summary_text: dict[tuple[int, int], str] = field(default_factory=dict)
    # Whether stream_options.include_obfuscation was requested
    include_obfuscation: bool | None = None
    # Sequence number for OpenResponses events (starts from -1, first event → 0)
    sequence_number: int = -1
    # Track items that have been closed (emitted output_item.done)
    closed_items: set[int] = field(default_factory=set)
    # Map provider tool_call.index values to output item indices so tool calls
    # don't collide with closed/pending content items (reasoning/message).
    tool_call_index_map: dict[int, int] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: time.time_ns() // 1_000_000_000)
    # Reasoning config from the original request
    reasoning: dict[str, Any] | None = None
    # Encrypted reasoning content (from include: reasoning.encrypted_content)
    reasoning_encrypted_content: str | None = None
    # Per-item encrypted content (captured when each reasoning item is created)
    reasoning_encrypted_contents: dict[int, str] = field(default_factory=dict)
    # Whether reasoning.encrypted_content was requested in include
    include_reasoning_encrypted: bool = False
    # Anthropic thinking signatures (bridged as reasoning-item
    # ``encrypted_content``), tracked per item so multi-block streams keep the
    # signature attached to the item that produced it.
    reasoning_signatures: dict[int, str] = field(default_factory=dict)
    # Track tool call indices that are server-side web_search (not emitted to client)
    web_search_tool_indices: set[int] = field(default_factory=set)
    # Count of native (provider-executed) web search calls, billed per request.
    # Only incremented when intercept_web_search is False (no proxy interceptor).
    native_web_search_call_count: int = 0
    # Whether a web search interceptor is active (server-side execution).
    # When False, web_search calls are emitted to the client as web_search_call items.
    intercept_web_search: bool = True
    # Whether the client declared web_search as a client-executed function
    # tool ({"type": "function", "name": "web_search"}, e.g. Hermes Agent)
    # rather than the server-side builtin {"type": "web_search"}. Client-side
    # declarations receive the call back as a function_call item they execute
    # themselves; builtin declarations receive a web_search_call report.
    web_search_as_function: bool = False
    # Custom tool names from the original request.
    # When a function_call result matches one of these names, the streaming
    # transformer emits a ``custom_tool_call`` item with ``input`` (unwrapped
    # from the function-call JSON wrapper) instead of a ``function_call``.
    custom_tool_names: set[str] = field(default_factory=set)
    # Namespace mapping for restoring flattened tool names in streaming output.
    # Maps flat name (e.g. "mcp__github__list_issues") to its namespace path
    # (e.g. ["github", "list_issues"]), where the last element is the original name.
    namespace_map: dict[str, list[str]] | None = None
    # Effective ``store`` value from the request (OpenAI default: true).
    # Echoed in response snapshots and gates post-stream persistence.
    store: bool = False
    # ── Request-field echo for response snapshots (spec: ResponseResource
    # carries the effective request configuration). Populated from the
    # FormatContext so streaming snapshots echo the same values the
    # non-streaming formatter emits.
    previous_response_id: str | None = None
    instructions: str | None = None
    metadata: dict[str, str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    truncation: str | None = None
    parallel_tool_calls: bool | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    service_tier: str | None = None
    background: bool | None = None
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    top_logprobs: int | None = None
    text: Any | None = None
    tool_choice: Any | None = None
    raw_tools: list[dict[str, Any]] = field(default_factory=list)
    # Final response snapshot from response.completed / response.incomplete,
    # reused by the streaming processor to persist store=true responses.
    final_response_payload: dict[str, Any] | None = None


class OpenResponsesStreamingTransformer(StreamingTransformer):
    """Transformer for converting OpenAI streaming to OpenResponses format.

    Generates semantic events per Open Responses spec:
    - response.created
    - response.in_progress
    - response.output_item.added
    - response.content_part.added
    - response.output_text.delta
    - response.output_text.done
    - response.content_part.done
    - response.output_item.done
    - response.completed
    - response.failed
    - response.incomplete
    """

    def __init__(
        self,
        model: str = "",
        request_id: str | None = None,
        intercept_web_search: bool = True,
        include_obfuscation: bool | None = None,
    ) -> None:
        super().__init__(
            model=model,
            request_id=request_id,
        )
        self.state = OpenResponsesStreamingState(
            intercept_web_search=intercept_web_search,
            include_obfuscation=include_obfuscation,
        )
        self._factory = StreamingEventFactory(
            state=self.state,
            response_id=self.response_id,
            model=self.model,
        )
        self._emitter = StreamingContentEmitter(
            state=self.state,
            factory=self._factory,
        )

        ctx = get_format_context()
        if ctx is not None:
            # ``store`` defaults to true per the OpenAI API (normalized in
            # set_format_context); explicit store=false opts out.
            self.state.store = ctx.store if ctx.store is not None else True
            # Echo fields: streaming response snapshots must carry the same
            # effective request configuration the non-streaming formatter
            # emits (spec ResponseResource required fields).
            for echo_field in (
                "previous_response_id",
                "instructions",
                "metadata",
                "temperature",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
                "truncation",
                "parallel_tool_calls",
                "max_output_tokens",
                "max_tool_calls",
                "service_tier",
                "background",
                "safety_identifier",
                "prompt_cache_key",
                "top_logprobs",
                "text",
                "tool_choice",
            ):
                echo_val = getattr(ctx, echo_field, None)
                if echo_val is not None:
                    setattr(self.state, echo_field, echo_val)
            if ctx.reasoning is not None:
                self.state.reasoning = ctx.reasoning
            if ctx.include is not None:
                self.state.include_reasoning_encrypted = (
                    "reasoning.encrypted_content" in ctx.include
                )
            if ctx.tools:
                from llm_proxy.protocols.openresponses.serializer import (
                    extract_custom_tool_names,
                    web_search_declared_as_function,
                )

                self.state.custom_tool_names = extract_custom_tool_names(ctx.tools)
                self.state.web_search_as_function = web_search_declared_as_function(ctx.tools)
                self.state.raw_tools = [t for t in ctx.tools if isinstance(t, dict)]
            if ctx.namespace_map is not None:
                self.state.namespace_map = ctx.namespace_map

        # Attributes used by streaming_processor._merge_transformer_usage
        self._pending_stop_reason: str | None = None
        self._pending_usage: dict | None = None
        self._has_pending_usage: bool = False

    @classmethod
    def continuation(
        cls,
        model: str,
        request_id: str,
        start_index: int,
        include: list[str] | None = None,
        web_search_tool_indices: set[int] | None = None,
        intercept_web_search: bool = True,
    ) -> OpenResponsesStreamingTransformer:
        """Create a transformer that continues an existing stream.

        Skips the response.created and response.in_progress events (already emitted)
        and starts item indices from start_index.

        Args:
            model: The model name
            request_id: The request ID
            start_index: Starting output item index
            include: Optional include fields from the original request
            web_search_tool_indices: Optional set of web_search tool indices to propagate
        """
        instance = cls(
            model=model,
            request_id=request_id,
            intercept_web_search=intercept_web_search,
        )
        instance.state.response_created = True
        instance.state.current_item_index = start_index
        if include is not None:
            instance.state.include_reasoning_encrypted = "reasoning.encrypted_content" in include
        if web_search_tool_indices is not None:
            instance.state.web_search_tool_indices = set(web_search_tool_indices)
        return instance

    def _has_pending_web_search_continuation(self) -> bool:
        """Whether an intercepted web search continuation follows this finish.

        True when an intercepted web_search tool call is present and the model
        ended its turn on it (no text emitted after the call), meaning the proxy
        must execute the search and stream the results back before the response
        can be considered complete.
        """
        if not self.state.intercept_web_search or not self.state.web_search_tool_indices:
            return False

        accumulated = self.get_accumulated_output()
        # Walk backwards to find the last web search block
        for i in range(len(accumulated) - 1, -1, -1):
            block = accumulated[i]
            if isinstance(block, (ToolUseBlock, ServerToolUseBlock)) and (
                is_web_search_tool_name(block.name)
            ):
                return not any(
                    isinstance(b, TextBlock) and b.text.strip() for b in accumulated[i + 1 :]
                )
        return False

    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Transform an OpenAI streaming chunk to OpenResponses events.

        Args:
            chunk: The SSE chunk in OpenAI format (string or dict)

        Returns:
            OpenResponses SSE events string, or None to skip
        """
        if isinstance(chunk, dict):
            return self._transform_chunk(chunk)

        if not isinstance(chunk, str) or not chunk.strip() or "[DONE]" in chunk:
            return None

        data_str = chunk.removeprefix("data: ").strip()
        if not data_str or data_str == "[DONE]":
            return None

        try:
            data = orjson.loads(data_str)
        except JSONDecodeError:
            logger.warning(
                f"Failed to parse chunk as JSON: {data_str[:200]}. Request ID: {self.response_id}"
            )
            return None

        return self._transform_chunk(data)

    def finalize(self) -> str:
        """Generate stream end marker.

        Returns the [DONE] terminal event as required by the OpenResponses spec.
        The spec states: "The terminal event MUST be the literal string [DONE]."
        """
        return "data: [DONE]\n\n"

    def error_frames(self, exc: Exception) -> list[str]:
        """Emit the spec-required terminal failure sequence.

        The OpenResponses spec requires that any error incurred while streaming
        is followed by a ``response.failed`` event, and that the terminal event
        is the literal string [DONE]. The generic chat-completions error frame
        is not part of the OpenResponses wire format, so it is not emitted.
        """
        from llm_proxy.protocols.openresponses.errors import openresponses_error_code

        code = openresponses_error_code(getattr(exc, "error_type", None))
        message = str(exc) or "An error occurred during response generation"
        return [
            self._factory._create_response_failed_event(error_code=code, error_message=message),
            "data: [DONE]\n\n",
        ]

    async def finalize_persistence(
        self,
        unified_request: Any,
        response_store: Any,
        event_context: Any,
    ) -> None:
        """Persist a completed streamed response when store is in effect.

        Non-streaming responses are persisted by RequestExecutionStage; streamed
        ones previously were not, which broke follow-up ``previous_response_id``
        continuations and ``GET /v1/responses/{id}`` for streaming clients (and
        for the WebSocket transport, which always streams). The final response
        snapshot emitted with ``response.completed`` / ``response.incomplete``
        is reused verbatim so the stored body matches what the client saw; the
        materialized conversation is attached as ``input`` so continuations and
        ``/v1/responses/compact`` can replay the full chain.
        """
        payload = self.state.final_response_payload
        if not isinstance(payload, dict):
            return
        store_flag = payload.get("store")
        if store_flag is None:
            # Native upstream snapshots echo the upstream-side store value; when
            # it is absent, fall back to the request-side effective value.
            store_flag = self.state.store
        if not store_flag:
            return
        response_id = payload.get("id")
        api_key_name = event_context.api_key_name if event_context is not None else None
        if not response_id or not api_key_name:
            return

        body = dict(payload)
        conversation = getattr(unified_request, "conversation", None)
        if conversation is not None and getattr(conversation, "messages", None):
            try:
                # The instructions echo is restored from the response's own
                # ``instructions`` field on continuation, so the matching system
                # message is excluded from the serialized input.
                items = conversation_to_input_items(
                    conversation, exclude_system_text=payload.get("instructions")
                )
                if items:
                    body["input"] = items
            except Exception:
                logger.debug("Failed to materialize streamed input items", exc_info=True)
        if "input" not in body:
            raw_data = unified_request._raw_protocol_data
            if isinstance(raw_data, dict) and raw_data.get("input") is not None:
                body["input"] = raw_data["input"]

        await response_store.store(api_key_name, response_id, body)

    def _transform_chunk(self, data: dict[str, Any]) -> str:
        """Transform parsed OpenAI chunk to OpenResponses events.

        Handles both Chat Completions format (``choices[0].delta``) and
        OpenAI Responses API format (``type``-based events). Responses API
        chunks are already in the target format and are passed through
        while accumulating text/usage for downstream consumers (e.g. Langfuse).

        Args:
            data: Parsed OpenAI chunk dictionary

        Returns:
            OpenResponses SSE events string
        """
        chunk_type = data.get("type")
        if chunk_type is not None:
            # ── Responses API format ──
            return self._transform_responses_api_chunk(data)

        # ── Chat Completions format ──
        return self._transform_chat_completions_chunk(data)

    def _transform_responses_api_chunk(self, data: dict[str, Any]) -> str:
        """Handle an OpenAI Responses API streaming chunk.

        Responses API chunks are already in the target event format.  We pass
        them through unchanged while accumulating text deltas and usage so
        that ``get_accumulated_output()`` and cost calculation still work.
        """
        chunk_type = data.get("type", "")

        # Track response lifecycle from the upstream events (avoids duplicate
        # response.created / response.in_progress that the Chat Completions
        # path emits).
        if chunk_type == "response.created":
            self.state.response_created = True
        elif chunk_type == "response.completed":
            response_data = data.get("response", {})
            usage = response_data.get("usage")
            if usage:
                self._update_responses_api_usage(usage)
            self.state.is_done = True
            # Keep the upstream completed snapshot for store=true persistence.
            if isinstance(response_data, dict) and response_data:
                self.state.final_response_payload = response_data
        elif chunk_type == "response.incomplete":
            response_data = data.get("response", {})
            if isinstance(response_data, dict) and response_data:
                usage = response_data.get("usage")
                if usage:
                    self._update_responses_api_usage(usage)
                self.state.final_response_payload = response_data
        elif chunk_type in (
            "response.output_text.delta",
            "response.reasoning.delta",
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        ):
            delta_text = data.get("delta", "")
            if delta_text:
                item_idx = data.get("output_index", self.state.current_item_index)
                content_idx = data.get("content_index", 0)
                key = (item_idx, content_idx)
                if chunk_type == "response.reasoning_summary_text.delta":
                    # Accumulate reasoning summaries so the final output items
                    # carry them (spec: reasoning items expose summary via
                    # summary_text).
                    self.state.reasoning_summary_text[key] = (
                        self.state.reasoning_summary_text.get(key, "") + delta_text
                    )
                else:
                    content_type = (
                        "reasoning_text"
                        if chunk_type != "response.output_text.delta"
                        else "output_text"
                    )
                    self.state.accumulated_text[key] = (
                        self.state.accumulated_text.get(key, "") + delta_text
                    )
                    self.state.content_types[key] = content_type
                    self.state.current_content_type = content_type

        # Re-serialize and emit the upstream event, normalizing raw reasoning
        # events (``response.reasoning.delta|done`` / OpenAI's
        # ``response.reasoning_text.delta|done``) to the summary event family
        # (``response.reasoning_summary_text.delta|done``) whose names are
        # identical in both specs.
        emitted_type = _normalize_reasoning_event(data, chunk_type)
        serialized = orjson.dumps(data).decode()
        # Spec: the event field MUST match the type in the event body.
        return f"event: {emitted_type}\ndata: {serialized}\n\n"

    def _update_responses_api_usage(self, usage: dict[str, Any]) -> None:
        """Update token counts from a Responses API usage dict."""
        self.state.input_tokens = usage.get("input_tokens", self.state.input_tokens)
        self.state.output_tokens = usage.get("output_tokens", self.state.output_tokens)

        input_details = usage.get("input_tokens_details") or {}
        self.state.cached_tokens = input_details.get("cached_tokens", self.state.cached_tokens) or 0

        output_details = usage.get("output_tokens_details") or {}
        self.state.reasoning_tokens = (
            output_details.get("reasoning_tokens", self.state.reasoning_tokens) or 0
        )

    def _transform_chat_completions_chunk(self, data: dict[str, Any]) -> str:
        """Handle a Chat Completions format chunk."""
        events = ""

        self._maybe_capture_encrypted_content(data)

        choices = data.get("choices", [])
        usage = data.get("usage")
        if usage:
            self._update_usage(usage)

        if not self.state.response_created:
            events += self._factory._create_response_created_event()
            events += self._factory._create_response_in_progress_event()
            self.state.response_created = True

        finish_reason = None
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict):
                return ""
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            if delta:
                events += self._process_delta(delta)

        if finish_reason and not self.state.is_done:
            events += self._process_finish(finish_reason)

        return events

    def _maybe_capture_encrypted_content(self, data: dict[str, Any]) -> None:
        """Capture encrypted reasoning content from the chunk if requested."""
        if not self.state.include_reasoning_encrypted or self.state.reasoning_encrypted_content:
            return

        encrypted = data.get("encrypted_content")
        if encrypted:
            self.state.reasoning_encrypted_content = encrypted
            return

        choices = data.get("choices", [])
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta", {})
            if isinstance(delta, dict):
                encrypted = delta.get("encrypted_content")
                if encrypted:
                    self.state.reasoning_encrypted_content = encrypted

    def _update_usage(self, usage: Any) -> None:
        """Update streaming token counts from a usage dict."""
        if not isinstance(usage, dict):
            return

        self.state.input_tokens = usage.get("prompt_tokens", self.state.input_tokens)
        self.state.output_tokens = usage.get("completion_tokens", self.state.output_tokens)

        ptd = usage.get("prompt_tokens_details")
        if isinstance(ptd, dict):
            self.state.cached_tokens = ptd.get("cached_tokens", self.state.cached_tokens) or 0
            self.state.audio_input_tokens = (
                ptd.get("audio_tokens", self.state.audio_input_tokens) or 0
            )
        # Provider converters normalize provider-native counters into the
        # OpenAI-dialect details objects, so only the dialect keys are read here.
        ctd = usage.get("completion_tokens_details")
        if isinstance(ctd, dict):
            self.state.audio_output_tokens = (
                ctd.get("audio_tokens", self.state.audio_output_tokens) or 0
            )
            self.state.reasoning_tokens = (
                ctd.get("reasoning_tokens", self.state.reasoning_tokens) or 0
            )

    def _process_delta(self, delta: dict[str, Any]) -> str:
        """Process delta content from OpenAI chunk.

        Args:
            delta: The delta object from OpenAI chunk

        Returns:
            SSE events string
        """
        events = ""

        signature = delta.get("reasoning_signature")
        if signature:
            # Provider-emitted thinking signature (Anthropic upstream via the
            # canonical chunk channel).
            self.state.reasoning_signatures.setdefault(self.state.current_item_index, signature)

        reasoning_content = delta.get("reasoning_content")
        if reasoning_content:
            events += self._emitter._emit_reasoning_content(reasoning_content)
        content = delta.get("content")
        if content:
            events += self._emitter._emit_text_content(content)

        refusal = delta.get("refusal")
        if refusal:
            events += self._emitter._emit_refusal_content(refusal)

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            events += self._emitter._emit_tool_calls(tool_calls)

        return events

    def _process_finish(self, finish_reason: str) -> str:
        """Process finish reason and generate completion events.

        Args:
            finish_reason: The finish reason from OpenAI

        Returns:
            SSE events string
        """
        events = ""

        for item_idx, item in sorted(self.state.pending_items.items()):
            if item_idx in self.state.closed_items:
                continue

            item_id = item["id"]
            item_type = item["type"]

            content_idx = self.state.current_content_index.get(item_idx, 0)
            key = (item_idx, content_idx)

            if item_type == "function_call":
                if item_idx in self.state.web_search_tool_indices:
                    continue
                arguments = self.state.accumulated_tool_args.get(item_idx, "")
                events += self._factory._create_function_call_arguments_done_event(
                    item_index=item_idx,
                    arguments=arguments,
                    item_id=item_id,
                )
                events += self._factory._create_output_item_done_event(
                    item_id=item_id,
                    item_index=item_idx,
                    item_type=item_type,
                    status="completed",
                )
            elif item_type in ("custom_tool_call", "tool_search_call"):
                if item_idx in self.state.web_search_tool_indices:
                    continue
                events += self._factory._create_output_item_done_event(
                    item_id=item_id,
                    item_index=item_idx,
                    item_type=item_type,
                    status="completed",
                )
            elif item_type == "web_search_call":
                if "action" not in item:
                    action = self._build_web_search_action(item_idx)
                    item["action"] = action
                    saved_index = self.state.current_item_index
                    self.state.current_item_index = item_idx
                    events += self._factory._create_output_item_added_event(
                        item_id=item_id,
                        item_type="web_search_call",
                        action=action,
                    )
                    self.state.current_item_index = saved_index

                events += self._factory._create_output_item_done_event(
                    item_id=item_id,
                    item_index=item_idx,
                    item_type=item_type,
                    status="completed",
                )
            else:
                accumulated = self.state.accumulated_text.get(key, "")
                content_type = self.state.content_types.get(key, "output_text")
                if accumulated:
                    events += self._factory._create_content_done_event(
                        item_index=item_idx,
                        content_index=content_idx,
                        text=accumulated,
                        item_id=item_id,
                        content_type=content_type,
                        item_type=item_type,
                    )

                events += self._factory._create_content_part_done_event(
                    item_index=item_idx,
                    content_index=content_idx,
                    item_id=item_id,
                    content_type=content_type,
                    text=accumulated,
                )
                events += self._factory._create_output_item_done_event(
                    item_id=item_id,
                    item_index=item_idx,
                    item_type=item_type,
                    status="completed",
                    content_type=content_type,
                    text=accumulated,
                )

        if self._has_pending_web_search_continuation():
            return events

        reason = incomplete_reason_from_finish(finish_reason)
        if reason is not None:
            events += self._factory._create_response_incomplete_event(
                input_tokens=self.state.input_tokens,
                output_tokens=self.state.output_tokens,
                reason=reason,
            )
        else:
            events += self._factory._create_response_completed_event(
                status="completed",
                input_tokens=self.state.input_tokens,
                output_tokens=self.state.output_tokens,
            )

        self.state.is_done = True

        return events

    def _build_web_search_action(self, item_idx: int) -> dict[str, Any]:
        """Build the web_search_call action from accumulated tool arguments."""
        args_str = self.state.accumulated_tool_args.get(item_idx, "")
        try:
            query = orjson.loads(args_str).get("query", "") if args_str else ""
        except JSONDecodeError:
            query = args_str

        return {
            "type": "search",
            "query": query,
            "queries": [query] if query else [],
        }

    def get_accumulated_output(self) -> list:
        """Get accumulated output content for final response.

        Returns:
            List of ContentBlocks accumulated during streaming.
        """
        output = []

        for key, text in self.state.accumulated_text.items():
            if text:
                content_type = self.state.content_types.get(key, "output_text")
                if content_type == "output_text":
                    output.append(TextBlock(text=text))
                elif content_type == "reasoning_text":
                    output.append(
                        ThinkingBlock(
                            thinking=text, signature=self.state.reasoning_signatures.get(key[0])
                        )
                    )

        for idx, tool_id in self.state.tool_call_ids.items():
            tool_name = self.state.tool_call_names.get(idx, "")
            tool_args = self.state.accumulated_tool_args.get(idx, "")

            try:
                tool_input = orjson.loads(tool_args) if tool_args else {}
            except JSONDecodeError:
                tool_input = {}

            output.append(
                ToolUseBlock(
                    id=tool_id,
                    name=tool_name,
                    input=tool_input,
                )
            )

        return output

    def get_usage(self) -> StreamingUsage | None:
        """Get accumulated usage information from streaming response.

        Returns:
            StreamingUsage object with token counts from the streaming state.
        """
        web_search_requests = self.state.native_web_search_call_count or None
        if not self.state.input_tokens and not self.state.output_tokens:
            if web_search_requests is None:
                return None
            # No token usage, but native web search calls are still billable.
            return StreamingUsage(web_search_requests=web_search_requests)
        if self.state.input_tokens or self.state.output_tokens:
            prompt_details = None
            if self.state.cached_tokens or self.state.audio_input_tokens:
                prompt_details = {}
                if self.state.cached_tokens:
                    prompt_details["cached_tokens"] = self.state.cached_tokens
                if self.state.audio_input_tokens:
                    prompt_details["audio_tokens"] = self.state.audio_input_tokens
            completion_details = None
            if self.state.audio_output_tokens or self.state.reasoning_tokens:
                completion_details = {}
                if self.state.audio_output_tokens:
                    completion_details["audio_tokens"] = self.state.audio_output_tokens
                if self.state.reasoning_tokens:
                    completion_details["reasoning_tokens"] = self.state.reasoning_tokens

            return StreamingUsage(
                input_tokens=self.state.input_tokens,
                output_tokens=self.state.output_tokens,
                total_tokens=self.state.input_tokens + self.state.output_tokens,
                prompt_tokens_details=prompt_details,
                completion_tokens_details=completion_details,
                web_search_requests=web_search_requests,
            )
        return None

    def _emit_web_search_call(
        self,
        index: int,
        action: dict[str, Any],
        status: str = "completed",
        mark_closed: bool = False,
    ) -> str:
        """Emit web_search_call output item events.

        Args:
            index: Output item index
            action: The web search action dict
            status: Item status ("completed" or "failed")
            mark_closed: Whether to mark the item as closed
                (prevents duplicate done in _process_finish)

        Returns:
            SSE events string
        """
        events = ""
        item_id = generate_item_id()

        self.state.current_item_index = index
        events += self._factory._create_output_item_added_event(
            item_id=item_id,
            item_type="web_search_call",
            action=action,
        )

        self.state.pending_items[index] = {
            "id": item_id,
            "type": "web_search_call",
            "action": action,
        }

        events += self._factory._create_output_item_done_event(
            item_id=item_id,
            item_index=index,
            item_type="web_search_call",
            status=status,
        )

        if mark_closed:
            self.state.closed_items.add(index)

        return events

    def _web_search_result_block(
        self,
        index: int,
        _tool_use_id: str,
        results: list[dict[str, Any] | str],
        is_error: bool = False,
        query: str = "",
    ) -> str:
        """Generate SSE events for a web search result block.

        For OpenResponses protocol, this generates a web_search_call output item
        with the search results. This method is called by the WebSearchStreamProcessor
        during streaming continuation.

        Args:
            index: Output item index
            tool_use_id: The ID of the corresponding tool use
            results: List of web search result dicts or error content
            is_error: Whether this is an error result
            query: The search query that produced these results

        Returns:
            SSE events string
        """
        queries = [query] if query else []
        action: dict[str, Any] = {
            "type": "search",
            "query": query,
            "queries": queries,
        }

        if is_error:
            action["error_code"] = "unavailable"
            action["error_message"] = "Web search failed"
        else:
            sources = []
            for r in results:
                if isinstance(r, dict):
                    sources.append({"url": r.get("url", ""), "title": r.get("title", "")})
                elif isinstance(r, str):
                    sources.append({"url": r, "title": ""})
            if sources:
                action["sources"] = sources

        status = "completed" if not is_error else "failed"
        return self._emit_web_search_call(index, action, status=status, mark_closed=True)

    def _message_delta_with_usage(self, usage: dict[str, Any]) -> str:
        """Generate a response.in_progress event with updated usage.

        For OpenResponses protocol, this emits a response.in_progress event
        with the current usage information, similar to how Anthropic emits
        message_delta with usage.

        Args:
            usage: Usage dict with output_tokens and server_tool_use info

        Returns:
            SSE events string
        """
        self._pending_usage = usage
        self._has_pending_usage = True
        return self._factory._create_response_in_progress_event()


__all__ = ["OpenResponsesStreamingTransformer", "OpenResponsesStreamingState"]
