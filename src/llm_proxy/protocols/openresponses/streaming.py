"""Streaming transformer for OpenResponses format.

Converts OpenAI streaming chunks to OpenResponses semantic events.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import orjson
from orjson import JSONDecodeError

from llm_proxy.models import ServerToolUseBlock, TextBlock, ToolUseBlock
from llm_proxy.models.content_blocks import ContentBlock
from llm_proxy.models.tools import is_web_search_tool_name
from llm_proxy.models.types import Usage
from llm_proxy.observability.logger import get_logger
from llm_proxy.protocols.openresponses.handler import get_format_context
from llm_proxy.protocols.openresponses.serializer import (
    WEB_SEARCH_ERROR_FALLBACK,
    conversation_to_input_items,
    incomplete_reason_from_finish,
    parse_web_search_error,
)
from llm_proxy.protocols.openresponses.streaming_emitter import StreamingContentEmitter
from llm_proxy.protocols.openresponses.streaming_events import StreamingEventFactory
from llm_proxy.serialization.responses_toolkit import generate_item_id
from llm_proxy.serialization.usage import UsageAccumulator, merge_latest
from llm_proxy.streaming.accumulation import CanonicalDelta, SegmentAccumulator
from llm_proxy.streaming.sse_parse import iter_sse_data_events
from llm_proxy.streaming.transformer import (
    PendingTerminalState,
    StreamingTransformer,
)

logger = get_logger(__name__)

# Raw reasoning delta/done events are emitted as the summary event family
# (``response.reasoning_summary_text.delta|done``), whose names are identical in
# the OpenResponses spec and the OpenAI Responses API.
_RAW_REASONING_EVENT_TYPES: frozendict[str, str] = frozendict(
    {
        "response.reasoning.delta": "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta": "response.reasoning_summary_text.delta",
        "response.reasoning.done": "response.reasoning_summary_text.done",
        "response.reasoning_text.done": "response.reasoning_summary_text.done",
    }
)


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


def _web_search_error_fields(results: list[dict[str, Any] | str]) -> tuple[str, str]:
    """Pick the ``(error_code, error_message)`` to report for a failed search.

    The interceptor encodes a failed search as the Anthropic-compatible
    ``web_search_tool_result_error`` object; OpenResponses carries the same
    information in the ``web_search_call.action`` fields, so the client sees the
    real reason (e.g. ``invalid_api_key``) instead of a generic code. Falls back
    to :data:`WEB_SEARCH_ERROR_FALLBACK` when no candidate carries a reason.
    """
    for candidate in results:
        parsed = parse_web_search_error(candidate)
        if parsed is not None:
            return parsed
    return WEB_SEARCH_ERROR_FALLBACK


#: Input-detail keys the snapshot emits from their own canonical scalar on the
#: state. Every other key a provider reports is carried verbatim, because
#: ``serializer._build_usage`` hands the provider's whole details block to the
#: non-streaming client body and ``build_usage_payload`` merges it as-is.
_CANONICAL_INPUT_DETAILS: frozenset[str] = frozenset({"cached_tokens", "cache_write_tokens"})
_CANONICAL_OUTPUT_DETAILS: frozenset[str] = frozenset({"reasoning_tokens"})


def _reported_count(value: Any) -> int | None:
    """A count the wire actually reported; a bool is not a count.

    The non-streaming body gets its details from ``asdict(PromptTokensDetails)``
    (typed ints), so a wire field carrying ``true`` must not become the count 1
    on the streaming side alone.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _extra_details(block: Any, canonical: frozenset[str]) -> dict[str, int]:
    """The detail keys ``block`` reports that have no canonical scalar of their own.

    ``audio_tokens``, ``image_tokens``, ``text_tokens``, ``video_tokens`` and the
    prediction-token counters are facts the non-streaming formatter emits inside
    the client's details object; they have no scalar on the canonical ``Usage``
    record, so nothing else carries them to the streaming snapshot. Dropping them
    here is the stream/non-stream drift this state exists to prevent.
    """
    if not isinstance(block, Mapping):
        return {}
    extra: dict[str, int] = {}
    for key, value in block.items():
        count = _reported_count(value)
        if key not in canonical and count is not None:
            extra[key] = count
    return extra


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
    cache_write_tokens: int = 0
    # Provider detail keys with no canonical scalar (the audio/image/text/video
    # token splits, the prediction-token counters). Merged into the snapshot's
    # ``input_tokens_details`` / ``output_tokens_details`` so both paths report
    # the same breakdown (``build_usage_payload``).
    input_details: dict[str, int] = field(default_factory=dict)
    output_details: dict[str, int] = field(default_factory=dict)
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
    # url_citation/file_citation annotations accumulated for an output_text
    # part (keyed by (item_idx, content_idx)). Fed by annotation deltas surfaced
    # from the upstream ``response.output_text.annotation.added`` event; echoed
    # in the part's ``annotations`` array at content_part.done / output_item.done
    # and in the terminal snapshot.
    accumulated_annotations: dict[tuple[int, int], list[dict[str, Any]]] = field(
        default_factory=dict
    )
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
    # Captured encrypted reasoning blobs not yet attributed to an output item,
    # in arrival order. Consumed FIFO as reasoning items open (delta-carried
    # blobs) or close (blobs an upstream reveals only at item/response
    # completion), so interleaved multi-item streams keep each blob on the
    # item that produced it.
    pending_reasoning_encrypted: list[str] = field(default_factory=list)
    # Every blob ever captured, for dedupe: an upstream may repeat the same
    # payload on a reasoning delta and again in the terminal response.
    seen_reasoning_encrypted: set[str] = field(default_factory=set)
    # Per-item encrypted content (attributed when each reasoning item is
    # created or closed).
    reasoning_encrypted_contents: dict[int, str] = field(default_factory=dict)
    # Whether encrypted reasoning should be emitted: requested in include, or
    # implied by a stateless response (store=false).
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
    # ── Native passthrough accumulation for the request log ──
    # Native frames never reach the transformer, so the response skeleton from
    # response.created plus the items closed by response.output_item.done are
    # all a stream cut off before its terminal event leaves behind (see
    # OpenResponsesStreamingTransformer.accumulate_native_frame).
    native_created_payload: dict[str, Any] | None = None
    native_output_items: dict[int, dict[str, Any]] = field(default_factory=dict)


class OpenResponsesStreamingTransformer(PendingTerminalState, StreamingTransformer):
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

    # Native Responses frames carry the whole response on their terminal event
    # (and a complete skeleton plus finished items before it), so the
    # native-passthrough tier is logged as a reassembled body too (see
    # accumulate_native_frame / native_log_body).
    native_frame_accumulation = True

    # OpenResponses renders the searches as web_search_call items and reports
    # the count on a response.in_progress event.
    supports_web_search_continuation = True

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
            # Genuine encrypted reasoning is emitted when the client asks via
            # ``include`` or when the response is stateless (``store: false``),
            # matching the current OpenAI reasoning docs; ``include`` remains
            # accepted as the legacy explicit request.
            self.state.include_reasoning_encrypted = self.state.store is False or (
                bool(ctx.include) and "reasoning.encrypted_content" in ctx.include
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

        # Pending terminal usage is owned by ``PendingTerminalState``; the
        # only writer is ``usage_delta`` (web-search usage updates), and the
        # continuation merge reaches it through the public
        # ``merge_terminal_state`` verb instead of poking private fields.
        #
        # What the proxy bills from (ADR-0025): two dialects reach this
        # transformer — a converted upstream's canonical chat chunks and a
        # Responses-shaped event — so each accumulates separately and
        # ``get_usage`` merges the two.
        self._usage_chat = UsageAccumulator("chat")
        self._usage_responses = UsageAccumulator("responses")
        # The segment engine assembles the canonical blocks (ADR-0031). Both
        # input dialects feed it — chat deltas through ``feed``, Responses
        # events through ``feed_delta`` — while the emitter keeps its own
        # per-item accumulation for the wire's content_part/item done events.
        # The default policy fits: tool calls pool by provider index, the
        # item cursor is wire state, and tool variants classify structurally.
        self._segments = SegmentAccumulator()
        # The last (output_index, content_index) fed to the engine as a
        # text/reasoning delta; a change is a new canonical block boundary.
        self._last_segment_key: tuple[int | None, int] | None = None
        # The finish_reason whose terminal snapshot is owed to the client, held
        # until ``finalize`` (see ``_process_finish``).
        self._pending_terminal_finish: str | None = None

    def accumulate_native_frame(self, frame: str | dict[str, Any]) -> None:
        """Accumulate one native Responses frame for the request log.

        Native passthrough forwards upstream events verbatim. The terminal
        ``response.completed`` / ``response.incomplete`` / ``response.failed``
        event carries the whole response and is stashed by the passthrough
        handler into :attr:`state.final_response_payload`; the events captured
        here are the fallback for a stream cut off before it — the
        ``response.created`` skeleton (a complete ``ResponseResource`` with
        ``status: in_progress`` and no output) plus every item closed by
        ``response.output_item.done``, which carries the finished item.

        Args:
            frame: One native SSE frame (``event: ...`` + ``data: {...}``,
                possibly carrying several events) or an already-parsed event
                payload.
        """
        if isinstance(frame, dict):
            self._accumulate_native_event("", frame)
            return
        if not isinstance(frame, str) or not frame.strip():
            return
        for event_type, payload in iter_sse_data_events(frame):
            if isinstance(payload, dict):
                self._accumulate_native_event(event_type, payload)

    def native_log_body(self) -> dict[str, Any] | None:
        """Return the native stream's response body for the request log.

        The terminal snapshot is authoritative (it is the exact body a
        non-streaming call would have returned). When the stream never reached
        it, the ``response.created`` skeleton with the items that did close is
        logged instead, marked ``incomplete`` — the same outcome the client
        observed.
        """
        snapshot = self.state.final_response_payload
        if snapshot is not None:
            return snapshot
        skeleton = self.state.native_created_payload
        if skeleton is None:
            return None
        body = dict(skeleton)
        body["output"] = [item for _, item in sorted(self.state.native_output_items.items())]
        body["status"] = "incomplete"
        if self.model:
            # The client-requested alias (the passthrough handler rewrites the
            # terminal snapshot's model the same way).
            body["model"] = self.model
        return body

    def _accumulate_native_event(self, event_type: str | None, payload: dict[str, Any]) -> None:
        """Route one parsed Responses event into the request-log accumulator."""
        event = payload.get("type") or event_type
        if event in ("response.created", "response.in_progress"):
            response = payload.get("response")
            if isinstance(response, dict) and self.state.native_created_payload is None:
                self.state.native_created_payload = response
        elif event == "response.output_item.done":
            item = payload.get("item")
            index = payload.get("output_index")
            if isinstance(item, dict) and isinstance(index, int):
                self.state.native_output_items[index] = item

    @classmethod
    def continuation(
        cls,
        model: str,
        request_id: str,
        start_index: int,
        *,
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
            intercept_web_search: Whether the proxy is still executing searches for
                this request (the continuation hop matches the original transformer)
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

    def adopt_continuation_state(self, previous: Any) -> None:
        """Take over the previous hop's reasoning decision and open search items.

        The continuation is a second upstream call whose client-visible
        response is the same one, so the encrypted-reasoning decision (which
        depends on include/store, not on the continuation kwargs) must survive
        the hop, and the web_search_call items the first hop opened but did not
        close have to be carried over rather than re-opened.
        """
        previous_state = getattr(previous, "state", None)
        if previous_state is None:
            return
        self.state.include_reasoning_encrypted = getattr(
            previous_state, "include_reasoning_encrypted", False
        )
        for idx, item in previous_state.pending_items.items():
            if item.get("type") == "web_search_call":
                self.state.pending_items[idx] = dict(item)
                self.state.closed_items.add(idx)

    def block_cursor(self) -> int:
        """Absolute output-item cursor.

        Unlike Anthropic's next-free-index cursor, ``current_item_index`` rests
        on the last emitted item (the emitters advance it themselves).
        """
        return self.state.current_item_index

    def continuation_start_index(self, result_count: int, fallback: int) -> int:
        """First free item index after the last emitted web-search result item."""
        return self.state.current_item_index + 1

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
        """Flush the deferred terminal snapshot, then the [DONE] marker.

        The ``response.completed`` / ``response.incomplete`` snapshot cannot be
        built on the ``finish_reason`` chunk: Chat Completions upstreams send
        their usage in a trailing chunk *after* it (forced
        ``stream_options.include_usage``), so the snapshot would carry all-zero
        counts — which is exactly what Codex's context meter reads. Holding the
        snapshot to the end of the stream folds that chunk in first, the way the
        Anthropic transformer holds its ``message_delta``.

        Returns the terminal events plus the spec-mandated literal ``[DONE]``.
        """
        events = ""
        finish_reason = self._pending_terminal_finish
        self._pending_terminal_finish = None
        if finish_reason is not None:
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
        return events + "data: [DONE]\n\n"

    def error_frames(self, exc: Exception) -> list[str]:
        """Emit the spec-required terminal failure sequence.

        The OpenResponses spec requires that any error incurred while streaming
        is followed by a ``response.failed`` event, and that the terminal event
        is the literal string [DONE]. The generic chat-completions error frame
        is not part of the OpenResponses wire format, so it is not emitted.
        """
        from llm_proxy.protocols.openresponses.errors import (
            client_error_message,
            openresponses_error_code,
        )

        code = openresponses_error_code(getattr(exc, "error_type", None))
        message = client_error_message(exc)
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

        # The event's own usage, before the dispatch: the terminal snapshot
        # carries it, and the snapshot's output items carry the billable
        # web-search calls this branch never counted.
        self._usage_responses.observe(data)

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
            self._segments.finish()
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
            self.state.is_done = True
            self._segments.finish()
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
                    # The engine assembles the canonical blocks (ADR-0031).
                    # A change of (item, content part) is a new block: the
                    # deleted state-dict rebuild keyed blocks by this pair, so
                    # state the boundary and let same-part deltas keep merging.
                    boundary = key != self._last_segment_key
                    self._last_segment_key = key
                    self._segments.feed_delta(
                        CanonicalDelta(
                            kind="text" if content_type == "output_text" else "reasoning",
                            text=delta_text,
                            close=boundary,
                        )
                    )

        # Re-serialize and emit the upstream event, normalizing raw reasoning
        # events (``response.reasoning.delta|done`` / OpenAI's
        # ``response.reasoning_text.delta|done``) to the summary event family
        # (``response.reasoning_summary_text.delta|done``) whose names are
        # identical in both specs.
        # A malformed ``type`` (array or object) is unhashable: looking it up in
        # the frozen mapping would raise before the drop rule below can apply.
        if not isinstance(chunk_type, str):
            logger.warning("Dropping upstream stream chunk with invalid event type")
            return ""
        emitted_type = _normalize_reasoning_event(data, chunk_type)
        # The event name is spliced into the SSE frame, not JSON-encoded: an
        # upstream-controlled type containing CR/LF could inject a forged
        # ``data:``/``event:`` line. Fall back to a protocol-known error event.
        if (
            not isinstance(emitted_type, str)
            or not emitted_type
            or "\r" in emitted_type
            or "\n" in emitted_type
        ):
            # The event name is spliced into the SSE frame, not JSON-encoded.
            # Do not substitute a different name while leaving the body's
            # ``type`` untouched (that would break the spec's
            # event/body-consistency rule and emit a non-event name). Drop the
            # malformed chunk; valid chunks continue the stream.
            logger.warning("Dropping upstream stream chunk with invalid event type")
            return ""
        serialized = orjson.dumps(data).decode()
        # Spec: the event field MUST match the type in the event body.
        return f"event: {emitted_type}\ndata: {serialized}\n\n"

    def _update_responses_api_usage(self, usage: dict[str, Any]) -> None:
        """Update token counts from a Responses API usage dict."""
        self.state.input_tokens = usage.get("input_tokens", self.state.input_tokens)
        self.state.output_tokens = usage.get("output_tokens", self.state.output_tokens)

        input_details = usage.get("input_tokens_details") or {}
        self.state.cached_tokens = input_details.get("cached_tokens", self.state.cached_tokens) or 0
        self.state.cache_write_tokens = (
            input_details.get("cache_write_tokens", self.state.cache_write_tokens) or 0
        )
        self.state.input_details.update(_extra_details(input_details, _CANONICAL_INPUT_DETAILS))

        output_details = usage.get("output_tokens_details") or {}
        self.state.reasoning_tokens = (
            output_details.get("reasoning_tokens", self.state.reasoning_tokens) or 0
        )
        self.state.output_details.update(_extra_details(output_details, _CANONICAL_OUTPUT_DETAILS))

    def _transform_chat_completions_chunk(self, data: dict[str, Any]) -> str:
        """Handle a Chat Completions format chunk."""
        events = ""

        self._maybe_capture_encrypted_content(data)

        choices = data.get("choices", [])
        usage = data.get("usage")
        if usage:
            self._update_usage(usage)
            self._usage_chat.observe(data)

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
            # The terminal chunk closes the engine's lanes: the web-search
            # interceptor reads the canonical blocks right after this chunk,
            # and an open tool lane would hide the call it intercepts.
            # finish() is idempotent and non-latching (ADR-0031).
            self._segments.finish()
            events += self._process_finish(finish_reason)

        return events

    def _maybe_capture_encrypted_content(self, data: dict[str, Any]) -> None:
        """Capture encrypted reasoning content from the chunk if requested.

        Genuine OpenAI encrypted reasoning is captured only when the client
        requested ``reasoning.encrypted_content`` via ``include``.
        Redacted-thinking payloads bypass that gate: like Anthropic thinking
        signatures they are integrity data the client must be able to echo
        back verbatim, not spec-gated encrypted reasoning (the non-streaming
        formatter emits them ungated for the same reason).
        """
        blob = data.get("encrypted_content")
        redacted = False
        if not blob:
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta", {})
                if isinstance(delta, dict):
                    blob = delta.get("encrypted_content")
                    redacted = bool(delta.get("reasoning_is_redacted"))
        if not blob or not isinstance(blob, str):
            return
        if not (self.state.include_reasoning_encrypted or redacted):
            return
        if blob in self.state.seen_reasoning_encrypted:
            return
        self.state.seen_reasoning_encrypted.add(blob)
        self.state.pending_reasoning_encrypted.append(blob)

    def _update_usage(self, usage: Any) -> None:
        """Update streaming token counts from a usage dict."""
        if not isinstance(usage, dict):
            return

        self.state.input_tokens = usage.get("prompt_tokens", self.state.input_tokens)
        self.state.output_tokens = usage.get("completion_tokens", self.state.output_tokens)

        ptd = usage.get("prompt_tokens_details")
        ptd = ptd if isinstance(ptd, dict) else {}
        # Provider converters put provider-native counters into the canonical
        # record under flat aliases (Gemini, Cohere, Interactions, Ollama all
        # ride ``cache_read_input_tokens`` / ``reasoning_tokens``); the dialect
        # details objects below remain the precedence when a converter emits
        # both (Anthropic normalizes to both). Without the fallback the client's
        # ``response.completed`` reported zero cached/reasoning tokens on every
        # one of those providers.
        cached = ptd.get("cached_tokens")
        if cached is None:
            cached = usage.get("cache_read_input_tokens")
        if cached is not None:
            self.state.cached_tokens = cached or 0
        cache_write = ptd.get("cache_write_tokens")
        if cache_write is None:
            cache_write = usage.get("cache_creation_input_tokens")
        if cache_write is not None:
            self.state.cache_write_tokens = cache_write or 0
        # Detail facts with no canonical scalar (audio/image/text/video splits)
        # are emitted by the non-streaming body, so they must reach the snapshot.
        self.state.input_details.update(_extra_details(ptd, _CANONICAL_INPUT_DETAILS))

        ctd = usage.get("completion_tokens_details")
        ctd = ctd if isinstance(ctd, dict) else {}
        self.state.output_details.update(_extra_details(ctd, _CANONICAL_OUTPUT_DETAILS))
        reasoning = ctd.get("reasoning_tokens")
        if reasoning is None:
            reasoning = usage.get("reasoning_tokens")
        if reasoning is not None:
            self.state.reasoning_tokens = reasoning or 0

    def _process_delta(self, delta: dict[str, Any]) -> str:
        """Process delta content from OpenAI chunk.

        Args:
            delta: The delta object from OpenAI chunk

        Returns:
            SSE events string
        """
        events = ""

        # The engine assembles the canonical blocks (ADR-0031); the returned
        # segment events are ignored — the emitter below renders the wire from
        # the delta itself, with the per-item bookkeeping it already owns.
        self._segments.feed(delta)

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

        annotations = delta.get("annotations")
        if annotations:
            events += self._emitter._emit_annotations(annotations)

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

        # A truncated turn's last open item ends in the terminal ``incomplete``
        # state; the spec requires such an item to be the last item emitted and
        # the response to be incomplete. The non-streaming formatter applies the
        # same rule, and the two bodies must agree for the same response
        # (tests/protocols/test_openresponses_streaming_parity.py).
        truncation_reason = incomplete_reason_from_finish(finish_reason)
        open_indices = [
            idx for idx in self.state.pending_items if idx not in self.state.closed_items
        ]
        last_open_index = max(open_indices) if open_indices else None
        done_status = "completed"

        for item_idx, item in sorted(self.state.pending_items.items()):
            if item_idx in self.state.closed_items:
                continue

            done_status = (
                "incomplete"
                if truncation_reason is not None and item_idx == last_open_index
                else "completed"
            )
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
                    status=done_status,
                )
            elif item_type in ("custom_tool_call", "tool_search_call"):
                if item_idx in self.state.web_search_tool_indices:
                    continue
                events += self._factory._create_output_item_done_event(
                    item_id=item_id,
                    item_index=item_idx,
                    item_type=item_type,
                    status=done_status,
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
                    status=done_status,
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
                    annotations=self.state.accumulated_annotations.get(key),
                )
                events += self._factory._create_output_item_done_event(
                    item_id=item_id,
                    item_index=item_idx,
                    item_type=item_type,
                    status=done_status,
                    content_type=content_type,
                    text=accumulated,
                    annotations=self.state.accumulated_annotations.get(key),
                )

        if self._has_pending_web_search_continuation():
            return events

        # Hold the terminal snapshot for ``finalize``: the trailing usage chunk
        # has not arrived yet (see ``finalize``).
        self._pending_terminal_finish = finish_reason
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

    def flush_pending_accumulation(self) -> None:
        """Close the engine's lanes when the stream ended before its terminal event.

        Called before the log body is reassembled: a client abort or an
        upstream cut ends the stream without a terminal event, and the
        engine's tool and media lanes only materialize on ``finish()``.
        Without this a completed tool call is missing from the log entirely.
        """
        self._segments.finish()

    def get_accumulated_output(self) -> list:
        """The canonical blocks, from the segment engine (ADR-0031).

        The rebuild the old machine performed from the emitter's per-item
        state dicts ordered every text/reasoning block before every tool
        call; the engine keeps arrival order, keeps signature-only reasoning
        segments, refusal deltas and custom-tool freeform input the rebuild
        dropped, and attaches tool ``thought_signature`` values as block extras.
        """
        return self._segments.blocks()

    def supersede(self, index: int, block: ContentBlock) -> None:
        """Replace a closed block in the engine's canonical list (ADR-0031).

        The web-search interception's ``ToolUseBlock → ServerToolUseBlock``
        substitution; the base verb, delegating to the same engine this
        transformer reads through.
        """
        self._segments.supersede(index, block)

    def _own_usage(self) -> Usage | None:
        """Get accumulated usage information from streaming response.

        The counters on ``state`` above are emission state for the client's own
        ``response.completed`` usage block; what the proxy bills from is the
        canonical record the intake adapter produced (ADR-0025).
        """
        usage = self._usage_responses.result()
        chat_usage = self._usage_chat.result()
        if usage is None:
            usage = chat_usage
        elif chat_usage is not None:
            # Both shapes of one turn (a continuation that changed shape): the
            # later observation adds facts without withdrawing any.
            merge_latest(usage, chat_usage)

        # A provider-executed search the upstream announced as a tool call is a
        # conversion-level fact, not a usage-block one, so it does not come from
        # the seam. An upstream that reports both signals describes the same
        # calls, so the larger wins — the monotonic rule the EventContext applies
        # across usage writes.
        announced = self.state.native_web_search_call_count
        if announced:
            if usage is None:
                usage = Usage(web_search_requests=announced)
            else:
                usage.web_search_requests = max(usage.web_search_requests or 0, announced)

        return usage

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

    def web_search_result_block(
        self,
        index: int,
        tool_use_id: str,  # noqa: ARG002 - the item id is not part of a search action
        results: list[dict[str, Any] | str],
        is_error: bool = False,
        query: str = "",
        error_message: str | None = None,
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
            error_message: Failure reason reported by the search provider; wins
                over the message embedded in the error payload

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
            error_code, payload_message = _web_search_error_fields(results)
            action["error_code"] = error_code
            action["error_message"] = error_message or payload_message
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

    def usage_delta(self, usage: dict[str, Any]) -> str:
        """Generate a response.in_progress event with updated usage.

        For OpenResponses protocol, this emits a response.in_progress event
        with the current usage information, similar to how Anthropic emits
        message_delta with usage.

        Args:
            usage: Usage dict with output_tokens and server_tool_use info

        Returns:
            SSE events string
        """
        self.capture_usage(usage)
        return self._factory._create_response_in_progress_event()


__all__ = ["OpenResponsesStreamingTransformer", "OpenResponsesStreamingState"]
