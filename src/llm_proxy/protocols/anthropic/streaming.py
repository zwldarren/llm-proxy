"""Anthropic streaming transformer.

Converts canonical ``chat.completion.chunk`` dicts into Anthropic SSE wire
format. Protocol-side transformer — lives next to the Anthropic protocol
module; the provider-side chunk converter stays in
``serialization/anthropic/streaming_converter.py``.
"""

from typing import Any

import orjson

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import RawBlock
from llm_proxy.models.content_blocks import ContentBlock
from llm_proxy.models.finish_reasons import to_anthropic_stop_reason
from llm_proxy.models.tools import is_web_search_tool_name
from llm_proxy.models.types import Usage
from llm_proxy.serialization.anthropic import (
    ANTHROPIC_USAGE_EXTENSION_KEYS,
    normalize_service_tier_for_anthropic,
    parse_usage_and_provider_extras,
)
from llm_proxy.serialization.content_parsers import REDACTED_THINKING_TEXT
from llm_proxy.serialization.usage import UsageAccumulator
from llm_proxy.serialization.usage.folds import fold_deepseek_cache_hits
from llm_proxy.streaming.accumulation import (
    CanonicalDelta,
    Coexistence,
    CursorPolicy,
    SegmentAccumulator,
    SegmentClosed,
    SegmentEvent,
    SegmentOpened,
    SegmentPolicy,
    ToolVariant,
)
from llm_proxy.streaming.sse_parse import iter_sse_data_events
from llm_proxy.streaming.transformer import (
    PendingTerminalState,
    StreamingTransformer,
)


def _cache_counters(usage: dict[str, Any]) -> tuple[int | None, int | None]:
    """Read the canonical cache counters, tolerating the nested OpenAI dialect.

    Canonical usage carries cache reads in the flat ``cache_read_input_tokens``;
    OpenAI-family providers express the same fact as
    ``prompt_tokens_details.cached_tokens`` (DeepSeek additionally uses the
    top-level ``prompt_cache_hit_tokens``). Either expression has to become the
    Anthropic wire's flat ``cache_read_input_tokens`` so ``input_tokens`` can be
    emitted exclusive of it (see :func:`_exclusive_input_tokens`).
    """
    cache_read = usage.get("cache_read_input_tokens")
    if cache_read is None:
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        cache_read = fold_deepseek_cache_hits(cached, usage.get("prompt_cache_hit_tokens"))
    cache_creation = usage.get("cache_creation_input_tokens")
    return cache_read, cache_creation


#: Keys the official ``MessageDeltaUsage`` object defines (plus the beta
#: counters that ride it). Anything else a canonical usage dict may carry — the
#: OpenAI-dialect ``prompt_tokens_details``/``completion_tokens_details`` nested
#: objects, ``total_tokens``, ``prompt_tokens``, … — has no field on the
#: Anthropic wire and is dropped rather than leaked to strict clients.
_MESSAGE_DELTA_USAGE_KEYS: frozenset[str] = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens_details",
        "server_tool_use",
        # Beta counters: fast-mode ``speed`` and compaction/server-side-fallback
        # ``iterations``. Top-level token counts exclude compaction iterations,
        # so dropping them would break per-iteration cost accounting.
        "speed",
        "iterations",
    }
)


def _thinking_block_start() -> dict[str, Any]:
    """The official ``thinking`` block-start shape.

    ``signature`` is a required field of the ``thinking`` block and the official
    stream opens the block with an empty one; omitting it makes SDK clients
    reject the event outright (``ThinkingBlock.signature`` is required).
    """
    return {"type": "thinking", "thinking": "", "signature": ""}


def _exclusive_input_tokens(usage: dict[str, Any]) -> int:
    """Canonical inclusive ``input_tokens`` → Anthropic's exclusive count.

    The internal invariant makes ``input_tokens`` INCLUDE cache reads/writes
    (the Anthropic wire reports them separately). Emitting the inclusive value
    next to the cache counters makes any client that sums the three — Claude
    Code does — count the cached prompt twice, inflating its usage display far
    beyond the real total. Clamped at 0 so a provider violating the invariant
    yields 0 instead of a negative count.
    """
    cache_read, cache_creation = _cache_counters(usage)
    return max(0, (usage.get("input_tokens") or 0) - (cache_read or 0) - (cache_creation or 0))


def _message_delta_usage(usage: dict) -> dict:
    """Shape the terminal ``message_delta`` usage block.

    Only the keys the official ``MessageDeltaUsage`` defines are emitted
    (``_MESSAGE_DELTA_USAGE_KEYS``): ``service_tier`` is valid on the full
    ``Usage`` object alone (emitted in ``message_start``), and the canonical
    channel's OpenAI-dialect members (``prompt_tokens_details``,
    ``completion_tokens_details``, ``total_tokens``) have no field on the
    Anthropic wire.

    ``input_tokens`` is rewritten to the exclusive count: the delta usage
    repeats the cumulative prompt accounting, and the wire invariant for that
    field excludes the cache counters reported beside it.
    """
    result = {k: v for k, v in usage.items() if k in _MESSAGE_DELTA_USAGE_KEYS}
    cache_read, cache_creation = _cache_counters(usage)
    if cache_read is not None:
        result["cache_read_input_tokens"] = cache_read
    if cache_creation is not None:
        result["cache_creation_input_tokens"] = cache_creation
    if "input_tokens" in result:
        result["input_tokens"] = _exclusive_input_tokens(usage)
    return result


class AnthropicStreamingTransformer(PendingTerminalState, StreamingTransformer):
    """Transform OpenAI SSE chunks to Anthropic SSE format."""

    # Native Anthropic frames carry the same content the canonical path builds
    # blocks from, so the native-passthrough tier can be logged as a
    # reassembled body too (see accumulate_native_frame).
    native_frame_accumulation = True

    # Anthropic renders the search results as web_search_tool_result blocks and
    # reports the search count on a message_delta event.
    supports_web_search_continuation = True

    def __init__(
        self,
        model: str = "",
        request_id: str | None = None,
        estimated_input_tokens: int = 0,
        intercept_web_search: bool = True,
    ):
        super().__init__(model, request_id)
        self._intercept_web_search = intercept_web_search
        # Emission state: what the client's message_start/message_delta carry.
        self._input_tokens = estimated_input_tokens
        self._output_tokens = 0
        # What the proxy bills from: the canonical parse of the same usage blocks
        # (ADR-0025). This transformer is fed canonical chunks, so its dialect is
        # the chat one.
        self._usage = UsageAccumulator("chat")
        self._sent_message_start = False
        self._current_block_index = 0
        # The kind of block currently open on the wire. The engine tracks its
        # own lanes; what the wire has open is this adapter's state — the two
        # views coincide except where the wire has blocks the canonical model
        # does not (the estimated-tokens thinking corner, the citations text
        # block) or the reverse (thinking suppressed after text).
        self._wire_open_kind: str | None = None
        # Why the wire-open block exists: ``engine`` — a segment lane backs
        # it; ``closed`` — the lane closed but the wire block lingers (the
        # stop is withheld until a visible boundary needs the slot, so a
        # reasoning tail after text does not split the client's text block);
        # ``wire_only`` — no lane ever backed it (estimated-tokens thinking,
        # citations text) and a later segment of the same kind adopts it.
        self._wire_block_state: str = "engine"
        # Whether the wire-open block is a ``redacted_thinking`` one: its
        # payload is complete in the start event's ``data``, so the canonical
        # ``[redacted]`` placeholder must never be rendered as a
        # ``thinking_delta`` inside it.
        self._wire_is_redacted = False
        # The wire keeps Anthropic's rule that thinking never resumes after
        # text; the canonical model no longer enforces it (ADR-0031).
        self._text_output_started = False
        # The segment engine assembles the canonical blocks (ADR-0031). This
        # protocol is the engine's first event consumer: one block on the wire
        # at a time (sequential tool calls), tool variants classified at open,
        # and the cursor counting closed blocks.
        self._segments = SegmentAccumulator(
            SegmentPolicy(
                tool_coexistence=Coexistence.SEQUENTIAL,
                cursor=CursorPolicy.NEXT_FREE,
                classify_tool=self._classify_tool,
            )
        )
        # Which tier this instance serves — flipped by the first native frame;
        # a converted-path instance never sees one. The routing behind
        # ``get_accumulated_output``/``flush_pending_accumulation`` must be
        # stable for the instance's whole life, not re-derived per call.
        self._native_tier = False
        # Pending terminal state (stop_reason / stop_sequence / stop_details /
        # container / usage) is owned by ``PendingTerminalState``; captured in
        # ``_transform_openai_chunk`` and flushed in ``finalize``.
        # Cache-diagnostics beta: message-level object arriving on the
        # canonical channel from the provider converter; replayed inside
        # message_start.message.
        self._pending_diagnostics: dict[str, Any] | None = None
        # Cache-diagnostics beta for the request log, kept apart from
        # ``_pending_diagnostics`` (``message_start`` consumes and clears that
        # one): an explicit ``null`` is a meaningful state, so presence is
        # tracked rather than truthiness.
        self._has_log_diagnostics = False
        self._log_diagnostics: Any = None
        # Native-frame accumulation for the request log (see
        # accumulate_native_frame): the content block being built, keyed by its
        # wire ``index``, plus its buffered ``input_json_delta`` payload.
        self._native_blocks: dict[int, dict[str, Any]] = {}
        self._native_partial_json: dict[int, str] = {}
        # Native blocks already closed, keyed by wire index: a flush re-queues
        # them so the logged body stays in index order even when the stream
        # ended with several blocks open (see flush_pending_accumulation).
        self._native_log_blocks: dict[int, RawBlock] = {}
        # Native passthrough keeps its own wire-rebuilt block list — the
        # RawBlocks replayed verbatim into the log body (ADR-0015). This is
        # the native tier's state, not a canonical buffer: the converted
        # tier's canonical blocks live in the segment engine (ADR-0031).
        self._accumulated_output: list[ContentBlock] = []
        # Where the native blocks start in ``_accumulated_output``; ``None``
        # until the first one is queued.
        self._native_output_start: int | None = None
        # Raw usage payloads from native ``message_start``/``message_delta``
        # frames, kept for the usage-level provider extras the non-streaming
        # parser collects (see _provider_extras).
        self._native_usage: dict[str, Any] = {}
        # Top-level ``service_tier`` off the ``message_start`` message (the
        # non-streaming parse reads it there, with usage as the primary source).
        self._native_service_tier: str | None = None

    def accumulate_native_frame(self, frame: str | dict[str, Any]) -> None:
        """Accumulate one Anthropic-native frame for the request log.

        Native passthrough forwards upstream events verbatim, so the
        accumulation :meth:`transform` performs as a side effect never runs for
        those requests. The events reconstruct each content block exactly as
        the client received it; a closed block is stored as a ``RawBlock``
        holding its Anthropic payload, which the protocol serializer emits
        verbatim — including beta blocks the internal model does not represent
        (server tool results, citations, cache control).

        Args:
            frame: One native SSE frame (``event: ...`` + ``data: {...}``,
                possibly carrying several events) or an already-parsed event
                payload.
        """
        if isinstance(frame, dict):
            self._native_tier = True
            self._accumulate_native_event("", frame)
            return
        if not isinstance(frame, str) or not frame.strip():
            return
        self._native_tier = True
        for event_type, payload in iter_sse_data_events(frame):
            if isinstance(payload, dict):
                self._accumulate_native_event(event_type, payload)

    def flush_pending_accumulation(self) -> None:
        """Queue blocks left open when the stream ended before ``content_block_stop``.

        Called before the response log is reassembled: a client abort or an
        upstream cut ends the stream without the closing events, so the partial
        block would otherwise be missing from the log entirely. The queued blocks
        are re-emitted in wire ``index`` order — a block that closed after a
        higher-index one would otherwise be logged out of order.

        The converted tier flushes the segment engine instead; the
        native-passthrough tier's RawBlock rebuild is out of the engine's
        scope (ADR-0031) and keeps its own queue.
        """
        if self._native_tier:
            for index in sorted(self._native_blocks):
                self._close_native_block(index)
            if self._native_output_start is None:
                return
            self._accumulated_output[self._native_output_start :] = [
                self._native_log_blocks[index] for index in sorted(self._native_log_blocks)
            ]
            self._native_log_blocks.clear()
            self._native_output_start = None
            return
        self._segments.finish()

    def get_accumulated_output(self) -> list[ContentBlock]:
        """The canonical blocks of the stream.

        The native-passthrough tier accumulates ``RawBlock``s on the base
        list; the converted tier reads the segment engine (ADR-0031). One
        instance serves one tier, so the routing is stable per request.
        """
        if self._native_tier:
            return self._accumulated_output
        return self._segments.blocks()

    def supersede(self, index: int, block: ContentBlock) -> None:
        """Replace a closed canonical block — the web-search interception's verb.

        The interceptor substitutes the ``ToolUseBlock`` it fulfills with the
        ``ServerToolUseBlock``. One instance serves one tier, so the verb
        routes by the same flag as :meth:`get_accumulated_output`: the engine
        holds the converted tier's list, the native tier rebuilds its own.
        """
        if self._native_tier:
            self._accumulated_output[index] = block
            return
        self._segments.supersede(index, block)

    def adopt(self, block: ContentBlock) -> tuple[SegmentEvent, ...]:
        """Close every open lane, then append a pre-built block as closed.

        The converted path's one window for blocks that exist outside the
        segment vocabulary (``raw_content_block`` replays, and tests injecting
        a closed block); see :meth:`SegmentAccumulator.adopt`.
        """
        return self._segments.adopt(block)

    def get_terminal_provider_info(self) -> dict[str, Any] | None:
        """Beta terminal extras for the reassembled request-log body.

        Mirrors the non-streaming Anthropic response's ``provider_info`` (see
        ``AnthropicProviderSerializer.parse_provider_response``) so a streamed
        and a non-streamed call log the same fields — the terminal extras and
        the usage-level ones (``server_tool_use``, ``service_tier``, …) that the
        formatter re-emits inside ``usage``. Read before ``finalize`` flushes the
        pending terminal state.
        """
        info: dict[str, Any] = self._provider_extras()
        if self._pending_stop_sequence:
            info["stop_sequence"] = self._pending_stop_sequence
        if self._pending_stop_details:
            info["stop_details"] = self._pending_stop_details
        if self._pending_container:
            info["container"] = self._pending_container
        if self._has_log_diagnostics:
            info["diagnostics"] = self._log_diagnostics
        if self._pending_safeguard_results is not None:
            info["safeguard_results"] = self._pending_safeguard_results
        return info or None

    def _provider_extras(self) -> dict[str, Any]:
        """Usage-level extras the non-streaming parse puts in ``provider_info``.

        ``parse_usage_and_provider_extras`` is the single source of truth for
        which usage keys belong there, so feeding it the accumulated usage keeps
        the streamed and non-streamed bodies equal. ``_pending_usage`` is the
        converted path's normalized usage; ``_native_usage`` the raw payloads off
        the native frames. Only one of them is ever filled.
        """
        _, extras = parse_usage_and_provider_extras(
            {"usage": {**(self._pending_usage or {}), **self._native_usage}}
        )
        if self._native_service_tier is not None:
            # Same precedence as the non-streaming parse: usage wins, the
            # top-level field fills in when usage carries none.
            extras.setdefault("service_tier", self._native_service_tier)
        return extras

    def _merge_native_usage(self, usage: Any) -> None:
        """Fold one native ``usage`` payload into the logged body's extras."""
        if not isinstance(usage, dict):
            return
        for key, value in usage.items():
            if value is not None:
                self._native_usage[key] = value

    def _accumulate_native_event(self, event_type: str | None, payload: dict[str, Any]) -> None:
        """Route one parsed Anthropic event into the request-log accumulator."""
        event = payload.get("type") or event_type
        if event == "message_start":
            message = payload.get("message")
            if isinstance(message, dict):
                message_id = message.get("id")
                if isinstance(message_id, str) and message_id:
                    # Native frames are forwarded verbatim, so the client saw
                    # the upstream's id; the logged body echoes it rather than
                    # the proxy's generated one.
                    self.response_id = message_id
                if "diagnostics" in message:
                    self._has_log_diagnostics = True
                    self._log_diagnostics = message["diagnostics"]
                # ``message`` is the full response shape, so the terminal extras
                # a non-streaming parse reads off it are captured here too.
                self._merge_native_usage(message.get("usage"))
                self.capture_stop_sequence(message.get("stop_sequence"))
                self.capture_stop_details(message.get("stop_details"))
                self.capture_container(message.get("container"))
                service_tier = message.get("service_tier")
                if isinstance(service_tier, str) and service_tier:
                    self._native_service_tier = service_tier
        elif event == "content_block_start":
            index = payload.get("index")
            block = payload.get("content_block")
            if isinstance(index, int) and isinstance(block, dict):
                self._native_blocks[index] = dict(block)
        elif event == "content_block_delta":
            self._accumulate_native_delta(payload)
        elif event == "content_block_stop":
            self._close_native_block(payload.get("index"))
        elif event == "message_delta":
            delta = payload.get("delta")
            if isinstance(delta, dict):
                # Kept for the reassembled log body: the formatter would
                # otherwise default every stream to ``end_turn``, hiding
                # truncation (``max_tokens``) and tool turns.
                self.capture_stop_reason(delta.get("stop_reason"))
                self.capture_stop_sequence(delta.get("stop_sequence"))
                self.capture_stop_details(delta.get("stop_details"))
                self.capture_container(delta.get("container"))
            self._merge_native_usage(payload.get("usage"))

    def _accumulate_native_delta(self, payload: dict[str, Any]) -> None:
        """Apply a ``content_block_delta`` to the block it belongs to."""
        index = payload.get("index")
        delta = payload.get("delta")
        if not isinstance(index, int) or not isinstance(delta, dict):
            return
        block = self._native_blocks.get(index)
        if block is None:
            return
        delta_type = delta.get("type", "")
        if delta_type == "input_json_delta":
            partial = delta.get("partial_json")
            if isinstance(partial, str):
                self._native_partial_json[index] = (
                    self._native_partial_json.get(index, "") + partial
                )
            return
        if delta_type == "citations_delta":
            citation = delta.get("citation")
            if isinstance(citation, dict):
                citations = block.setdefault("citations", [])
                if isinstance(citations, list):
                    citations.append(citation)
            return
        # ``text_delta``, ``thinking_delta``, ``signature_delta`` and any
        # future delta append onto the matching key of the block opened by
        # ``content_block_start``.
        for key, value in delta.items():
            if key == "type":
                continue
            existing = block.get(key)
            if isinstance(existing, str) and isinstance(value, str):
                block[key] = existing + value
            else:
                block[key] = value

    def _close_native_block(self, index: Any) -> None:
        """Finish a native content block and queue it for the logged body."""
        if not isinstance(index, int):
            return
        block = self._native_blocks.pop(index, None)
        if block is None:
            return
        partial = self._native_partial_json.pop(index, "")
        if partial:
            try:
                block["input"] = orjson.loads(partial)
            except orjson.JSONDecodeError:
                block["input"] = {}
        raw = RawBlock(provider_type=f"anthropic:{block.get('type', 'unknown')}", data=block)
        if self._native_output_start is None:
            # First native block: everything queued from here on is native, so a
            # flush can re-queue the tail in wire order.
            self._native_output_start = len(self._accumulated_output)
        self._native_log_blocks[index] = raw
        self._accumulated_output.append(raw)

    @classmethod
    def continuation(
        cls,
        model: str,
        request_id: str,
        start_index: int,
        *,
        intercept_web_search: bool = True,
    ) -> AnthropicStreamingTransformer:
        """Create a transformer that continues an existing stream.

        Skips the message_start event (already emitted) and starts
        content block indices from start_index. ``intercept_web_search`` is
        forwarded so the continuation hop agrees with the original transformer
        about whether the proxy is executing searches (the flag decides how a
        ``websearch`` tool call is handled).
        """
        instance = cls(
            model=model, request_id=request_id, intercept_web_search=intercept_web_search
        )
        instance._sent_message_start = True
        instance._current_block_index = start_index
        return instance

    def block_cursor(self) -> int:
        """Absolute next content-block index (result blocks don't advance it)."""
        return self._current_block_index

    def continuation_start_index(self, result_count: int, fallback: int) -> int:
        """Next free block index after ``result_count`` emitted result blocks."""
        return self._current_block_index + result_count

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
            # Billing reads the canonical record; the counters above are emission
            # state for the client's own Anthropic usage blocks.
            self._usage.observe(chunk)
            # Fold usage before emitting message_start so its ``usage`` block
            # can carry the full Anthropic shape (cache keys, server_tool_use).
            self.fold_usage(self._normalize_usage(usage))
        # Cache-diagnostics beta: capture before the message_start emission so
        # the first message (emitted below) already carries it.
        if "diagnostics" in chunk:
            diag = chunk["diagnostics"]
            if diag is not None:
                self._pending_diagnostics = diag
            self._has_log_diagnostics = True
            self._log_diagnostics = diag

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
                if not isinstance(delta, dict):
                    delta = {}
                finish_reason = choice.get("finish_reason")

                # The engine assembles the canonical blocks (ADR-0031); this
                # adapter renders the wire from the segment events it returns.
                if delta:
                    self._render_segments(
                        self._segments.feed(delta),
                        result_chunks,
                        estimated=delta.get("reasoning_estimated_tokens"),
                    )

                # Server-side tool result blocks arriving losslessly from the
                # converter (web_search_tool_result, web_fetch_tool_result,
                # container_upload, ...): replay as a start+stop pair so
                # native-shaped clients keep the full block. The engine adopts
                # the RawBlock so the logged body keeps the wire order.
                raw_block = delta.get("raw_content_block")
                if isinstance(raw_block, dict):
                    self._render_segments(
                        self._segments.adopt(
                            RawBlock(
                                provider_type=f"anthropic:{raw_block.get('type', 'unknown')}",
                                data=raw_block,
                            )
                        ),
                        result_chunks,
                    )
                    # adopt() closes the engine's lanes, but their wire stops
                    # are lazy: free the open wire block's slot first so the
                    # replayed block starts at a fresh index (what the old
                    # ``_close_current_open_block`` call did here).
                    self._flush_wire_block(result_chunks)
                    result_chunks.append(
                        self._content_block_start(self._current_block_index, raw_block)
                    )
                    result_chunks.append(self._content_block_stop(self._current_block_index))
                    self._current_block_index += 1

                # Citations attach only to text blocks: close any open
                # tool/thinking block first, then land the citation on a
                # (possibly fresh) text block. They are wire-only: the
                # canonical TextBlock never carried them and still does not.
                citation = delta.get("citations")
                if isinstance(citation, dict):
                    if self._wire_open_kind in ("tool", "reasoning"):
                        self._flush_wire_block(result_chunks)
                    if self._wire_open_kind != "text":
                        result_chunks.append(
                            self._content_block_start(
                                self._current_block_index, {"type": "text", "text": ""}
                            )
                        )
                        self._wire_open_kind = "text"
                        self._wire_block_state = "wire_only"
                    result_chunks.append(
                        self._content_block_delta(
                            self._current_block_index,
                            {"type": "citations_delta", "citation": citation},
                        )
                    )

                if finish_reason:
                    # The terminal chunk closes the engine's lanes as well: the
                    # web-search interceptor reads the canonical blocks right
                    # after this chunk, and an open tool lane would hide the
                    # call it intercepts. finish() is idempotent and
                    # non-latching (ADR-0031), so finalize's own finish() is a
                    # no-op after this.
                    self._render_segments(self._segments.finish(), result_chunks)
                    self._flush_wire_block(result_chunks)
                    self.capture_stop_reason(self._map_finish_reason(finish_reason))
                # Anthropic-native terminal fields arriving through the
                # converter's canonical channel: choice["stop_sequence"] and
                # choice["stop_details"]. Read regardless of finish_reason so
                # finalize flushes without a stop_reason still replay them.
                self.capture_stop_sequence(choice.get("stop_sequence"))
                self.capture_stop_details(choice.get("stop_details"))
                self.capture_container(choice.get("container"))
                # Server-side auto mode: classifier results arrive on the
                # canonical terminal chunk (from the provider converter) and
                # are replayed inside the terminal message_delta delta.
                self.capture_safeguard_results(choice.get("safeguard_results"))

        return "".join(result_chunks) if result_chunks else None

    def _map_finish_reason(self, finish_reason: str) -> str:
        """Map OpenAI finish_reason to an official Anthropic stop_reason.

        Unknown upstream values (OpenRouter ``eos``, legacy OpenAI
        ``function_call``, the internal ``error`` sentinel) collapse to
        ``end_turn`` instead of leaking an illegal enum member.
        """
        return to_anthropic_stop_reason(finish_reason)

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
        pending = self._pending_usage or {}
        start_usage: dict[str, Any] = {"output_tokens": output_tokens}
        for key in (
            *ANTHROPIC_USAGE_EXTENSION_KEYS,
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "server_tool_use",
        ):
            value = pending.get(key)
            if key == "service_tier":
                # Official response enum only; OpenAI tiers fold or drop.
                value = normalize_service_tier_for_anthropic(value)
            if value is not None:
                start_usage[key] = value
        # Surface the nested OpenAI-dialect cache read on the wire too, then
        # emit ``input_tokens`` exclusive of the cache counters (the internal
        # count includes them; see _exclusive_input_tokens).
        cache_read, cache_creation = _cache_counters(pending)
        if cache_read is not None:
            start_usage["cache_read_input_tokens"] = cache_read
        if cache_creation is not None:
            start_usage["cache_creation_input_tokens"] = cache_creation
        start_usage["input_tokens"] = max(
            0, input_tokens - (cache_read or 0) - (cache_creation or 0)
        )
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
        # Official terminal deltas always carry ``stop_sequence`` (``null`` when
        # no custom sequence matched), so emit it on every turn rather than only
        # when one matched.
        delta["stop_sequence"] = self._pending_stop_sequence
        self._pending_stop_sequence = None
        if self._pending_stop_details is not None:
            delta["stop_details"] = self._pending_stop_details
            self._pending_stop_details = None
        if self._pending_container is not None:
            delta["container"] = self._pending_container
            self._pending_container = None
        if self._pending_safeguard_results is not None:
            # Server-side auto mode: Claude Code reads the classifier results
            # from the terminal message_delta's delta.
            delta["safeguard_results"] = self._pending_safeguard_results
            self._pending_safeguard_results = None
        return self._sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": delta,
                "usage": usage,
            },
        )

    def usage_delta(self, usage: dict[str, Any]) -> str:
        """Generate message_delta event with usage only."""
        return self._sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {},
                "usage": usage,
            },
        )

    def _message_delta_with_safeguard_results(self) -> str:
        """Generate a message_delta carrying only auto-mode classifier results.

        Degenerate fallback: the upstream sent ``safeguard_results`` with no
        stop_reason and no usage to fold, so the terminal delta would otherwise
        be dropped and the session would fall back to its own classifier.
        """
        delta: dict[str, Any] = {"safeguard_results": self._pending_safeguard_results}
        self._pending_safeguard_results = None
        return self._sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": delta,
                "usage": {"output_tokens": 0},
            },
        )

    def _message_stop(self) -> str:
        """Generate message_stop event."""
        return self._sse_event("message_stop", {"type": "message_stop"})

    def web_search_result_block(
        self,
        index: int,
        tool_use_id: str,
        results: list[dict[str, Any]] | list[str],
        is_error: bool = False,
        query: str = "",
        error_message: str | None = None,
    ) -> str:
        """Generate complete web_search_tool_result content block events.

        Generates: content_block_start, content_block_stop with results

        Args:
            index: Content block index
            tool_use_id: The ID of the corresponding server_tool_use
            results: List of web search result dicts
            is_error: Whether this is an error result
            query: The search query (ignored for Anthropic protocol)
            error_message: Failure reason (ignored: the Anthropic wire error
                object defines only ``error_code`` and clients echo it back
                verbatim, so an extra field would break the round trip)

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

    def _classify_tool(self, delta: CanonicalDelta) -> ToolVariant:
        """Decide a tool call's block variant at open (ADR-0031).

        The upstream ``server_tool_use`` marker and the proxy-intercepted
        ``web_search`` name both make the block a ``ServerToolUseBlock`` —
        decided once, at open, so the wire's ``content_block_start`` and the
        canonical block can never disagree. The old machine decided at close,
        and only on the tool-start boundary, leaving a ``web_search`` call
        closed by any other boundary a plain ``ToolUseBlock`` even though the
        wire had already announced ``server_tool_use``.
        """
        if delta.server_tool or (
            is_web_search_tool_name(delta.tool_name or "") and self._intercept_web_search
        ):
            return "server_tool_use"
        return delta.tool_variant or "function"

    def _open_is_suppressed(self, kind: str, frag: CanonicalDelta) -> bool:
        """Whether a segment open renders nothing on the Anthropic wire.

        ``refusal`` and ``media`` have no wire representation at all;
        thinking does not resume after text (the canonical model keeps it,
        ADR-0031). A signature-only thinking fragment still renders — the
        signature is the integrity payload clients echo back.
        """
        if kind in ("refusal", "media"):
            return True
        if kind == "reasoning" and self._text_output_started:
            return bool(frag.text or frag.is_redacted)
        return False

    def _flush_wire_block(self, chunks: list[str]) -> None:
        """Emit ``content_block_stop`` for the wire-open block and free its slot.

        The engine's lane closes arrive as ``SegmentClosed`` events and are
        rendered by :meth:`_render_segments`; a block only the wire opened —
        the estimated-tokens thinking corner, the citations text block — has
        no engine lane, so the adapter closes it itself at boundaries.
        """
        if self._wire_open_kind is None:
            return
        chunks.append(self._content_block_stop(self._current_block_index))
        self._current_block_index += 1
        self._wire_open_kind = None
        self._wire_block_state = "engine"
        self._wire_is_redacted = False

    def _open_wire_block(self, kind: str, frag: CanonicalDelta, chunks: list[str]) -> None:
        """Start the wire block for an opened segment, unless already open.

        ``refusal`` and ``media`` have no Anthropic wire representation — the
        canonical model carries them (ADR-0031's drift rulings). Thinking is
        suppressed once text has started, except a signature-only fragment:
        the signature is the integrity payload clients echo back, so it still
        gets its block. A tool call without id/name stays unopened, as before.
        """
        if kind in ("refusal", "media"):
            return
        if self._open_is_suppressed(kind, frag):
            return
        if self._wire_open_kind == kind and self._wire_block_state in (
            "wire_only",
            "adoptable",
        ):
            # A block the wire keeps up without a live lane — a wire-only
            # block (citations text, estimated-tokens thinking) or a text
            # block that yielded to a suppressed segment — is adopted by the
            # segment that just opened; its deltas keep flowing into it.
            self._wire_block_state = "engine"
            return
        if self._wire_open_kind is not None:
            # A fresh segment of a kind whose block is still up (sequential
            # tool calls, a closed lane): close it, start the new one.
            self._flush_wire_block(chunks)
        self._wire_is_redacted = False
        if kind == "text":
            chunks.append(
                self._content_block_start(self._current_block_index, {"type": "text", "text": ""})
            )
        elif kind == "reasoning":
            if frag.is_redacted and (frag.encrypted_content or frag.text == REDACTED_THINKING_TEXT):
                # The provider converter keeps "[redacted]" as a display
                # placeholder in ``reasoning_content`` and carries the real
                # ``data`` in ``encrypted_content``; clients must echo the
                # real payload back.
                self._wire_is_redacted = True
                chunks.append(
                    self._content_block_start(
                        self._current_block_index,
                        {
                            "type": "redacted_thinking",
                            "data": frag.encrypted_content or frag.text,
                        },
                    )
                )
            else:
                chunks.append(
                    self._content_block_start(self._current_block_index, _thinking_block_start())
                )
        elif kind == "tool":
            if not (frag.tool_id and frag.tool_name):
                return
            block_type = (
                "server_tool_use" if self._classify_tool(frag) == "server_tool_use" else "tool_use"
            )
            chunks.append(
                self._content_block_start(
                    self._current_block_index,
                    {
                        "type": block_type,
                        "id": frag.tool_id,
                        "name": frag.tool_name or "",
                        "input": {},
                    },
                )
            )
        self._wire_open_kind = kind

    def _render_segments(
        self,
        events: tuple[SegmentEvent, ...],
        chunks: list[str],
        estimated: Any = None,
    ) -> None:
        """Render segment events onto the Anthropic wire.

        The engine assembles the canonical blocks (ADR-0031); this adapter is
        its first event consumer. ``estimated`` is the delta's
        ``reasoning_estimated_tokens`` beta value: it rides the first thinking
        delta the batch renders, or — when no reasoning text arrives — opens
        a (possibly empty) thinking block of its own, as the
        thinking-token-count wire contract requires.
        """
        pending_estimated = estimated
        for pos, event in enumerate(events):
            if isinstance(event, SegmentClosed):
                # Closes are lazy on this wire: the stop is emitted when a
                # visible boundary needs the slot. One exception stays live: a
                # TEXT block whose lane yielded to a suppressed segment (a
                # reasoning tail after text) — the client's text block
                # continues, so the next text segment adopts it instead of
                # splitting it. Tool calls never adopt: two calls are two
                # blocks, whatever the wire suppressed between them.
                if self._wire_open_kind == event.kind:
                    nxt = events[pos + 1] if pos + 1 < len(events) else None
                    if (
                        event.kind == "text"
                        and isinstance(nxt, SegmentOpened)
                        and self._open_is_suppressed(nxt.kind, nxt.delta)
                    ):
                        self._wire_block_state = "adoptable"
                    else:
                        self._wire_block_state = "closed"
                continue
            if isinstance(event, SegmentOpened):
                self._open_wire_block(event.kind, event.delta, chunks)
                continue
            kind, frag = event.kind, event.delta
            if kind == "text":
                self._text_output_started = True
                if frag.text:
                    chunks.append(
                        self._content_block_delta(
                            self._current_block_index,
                            {"type": "text_delta", "text": frag.text},
                        )
                    )
                if frag.markdown:
                    chunks.append(
                        self._content_block_delta(
                            self._current_block_index,
                            {"type": "text_delta", "text": frag.markdown},
                        )
                    )
            elif kind == "reasoning":
                if frag.text and not self._text_output_started and not self._wire_is_redacted:
                    thinking_delta: dict[str, Any] = {
                        "type": "thinking_delta",
                        "thinking": frag.text,
                    }
                    if pending_estimated is not None:
                        thinking_delta["estimated_tokens"] = pending_estimated
                        pending_estimated = None
                    chunks.append(
                        self._content_block_delta(self._current_block_index, thinking_delta)
                    )
                if frag.signature:
                    if self._wire_open_kind != "reasoning":
                        if self._wire_open_kind is not None:
                            self._flush_wire_block(chunks)
                        chunks.append(
                            self._content_block_start(
                                self._current_block_index, _thinking_block_start()
                            )
                        )
                        self._wire_open_kind = "reasoning"
                        self._wire_block_state = "engine"
                    chunks.append(
                        self._content_block_delta(
                            self._current_block_index,
                            {"type": "signature_delta", "signature": frag.signature},
                        )
                    )
            elif kind == "tool":
                if self._wire_open_kind == "tool" and frag.text:
                    chunks.append(
                        self._content_block_delta(
                            self._current_block_index,
                            {"type": "input_json_delta", "partial_json": frag.text},
                        )
                    )
            # refusal and media appends render nothing: no wire representation.

        if pending_estimated is not None and not self._text_output_started:
            if self._wire_open_kind != "reasoning":
                if self._wire_open_kind is not None:
                    self._flush_wire_block(chunks)
                chunks.append(
                    self._content_block_start(self._current_block_index, _thinking_block_start())
                )
                self._wire_open_kind = "reasoning"
                self._wire_block_state = "wire_only"
            chunks.append(
                self._content_block_delta(
                    self._current_block_index,
                    {
                        "type": "thinking_delta",
                        "thinking": "",
                        "estimated_tokens": pending_estimated,
                    },
                )
            )

    def _sse_event(self, event_type: str, data: dict[str, Any]) -> str:
        """Create an SSE event string."""
        return f"event: {event_type}\ndata: {orjson.dumps(data).decode()}\n\n"

    def finalize(self) -> str:
        """Generate stream end marker.

        Sends one message_delta with latest cached usage/stop_reason and message_stop.
        """
        result_chunks = []

        self._render_segments(self._segments.finish(), result_chunks)
        self._flush_wire_block(result_chunks)

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
        elif self._pending_safeguard_results is not None:
            result_chunks.append(self._message_delta_with_safeguard_results())

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

    def _own_usage(self) -> Usage | None:
        """Get accumulated usage information from streaming response.

        The canonical chunks this transformer is fed carry their usage in the chat
        dialect, which the intake adapter parses (ADR-0025); the ``_input_tokens`` /
        ``_output_tokens`` counters below are emission state for the client's
        Anthropic ``usage`` blocks, not the billing record.

        Returns:
            The canonical usage record, or None when the stream reported none.
        """
        return self._usage.result()
