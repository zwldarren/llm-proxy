"""Streaming response lifecycle classes.

Extracted from ``core/processing/streaming_processor.py`` (issue LLMP-3):
the streaming response lifecycle was a ~265-line closure with 20
default-parameter captures, so heartbeat gating, disconnect marking,
native passthrough, web-search continuation, finalize and persistence
could only be tested by driving the full SSE harness. The closures are
now classes whose captured variables are named constructor fields and
whose concerns are methods:

- :class:`StreamLifecycle` — the chat-completions lifecycle (first-chunk
  replay, heartbeat-gated pump, native passthrough, web-search
  continuation, finalize + reasoning cache, persistence, error framing,
  teardown).
- :class:`GenericStreamLifecycle` — the same lifecycle skeleton for the
  generic streams (image, speech, transcription): raw provider chunks
  pass through unchanged, with an optional per-chunk observer for
  billing capture.

Each instance drives exactly one response; :meth:`events` is the SSE
async generator handed to the ``StreamingHandler``.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from llm_proxy.core.constants import DEFAULT_DISCONNECT_CHECK_INTERVAL
from llm_proxy.core.conversion import NativePassthroughHandler
from llm_proxy.core.exceptions import ClientDisconnectedError
from llm_proxy.core.processing.stream_assembly import (
    accumulate_native_frame,
    assemble_stream_response_body,
    can_accumulate_native_frames,
    collect_accumulated_output,
    first_native_log_body,
    flush_pending_accumulation,
    terminal_finish_reason,
    terminal_provider_info,
)
from llm_proxy.core.processing.web_search_streaming import (
    ContinuationState,
    WebSearchStreamProcessor,
    merge_continuation_usage,
)
from llm_proxy.core.reasoning_cache import try_cache_reasoning_from_blocks
from llm_proxy.core.utils import quiet_aclose
from llm_proxy.observability.cost import finalize_event_cost
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.logger import get_logger
from llm_proxy.streaming.handler import ClientDisconnectWatcher
from llm_proxy.streaming.sse_parse import contains_sse_event

logger = get_logger(__name__)

# SSE comment frame emitted (and ignored by every SSE parser per the WHATWG
# spec) when the upstream stream falls silent, so fronting CDNs such as
# Cloudflare keep the connection open while a model ponders its next token.
SSE_KEEPALIVE_COMMENT = ": keep-alive\n\n"

# Default heartbeat interval, overridden by the UI-managed keepalive config.
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15.0

# How long the pump teardown waits for a just-finishing provider iteration to
# record its exhaustion before cancelling it (see StreamPumpExhaustion).
_PUMP_SETTLE_GRACE_SECONDS = 0.05


async def safe_cleanup(coro: Any, description: str) -> None:
    """Safely run a cleanup coroutine, logging exceptions but never raising."""
    try:
        await coro
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning(f"{description}: {e}")


async def close_stream_quietly(stream: Any) -> None:
    """Close a provider stream, swallowing close-time errors."""
    if stream is None:
        return
    with suppress(Exception, asyncio.CancelledError):
        await asyncio.shield(quiet_aclose(stream))


def _experience_counts_as_success(error: Exception | None) -> bool:
    """Whether a finished stream counts as a success for model-experience EWMA.

    The hook measures provider health, and a client walking away mid-response
    is not the model failing — so a recorded abandonment still counts as one.
    """
    return error is None or isinstance(error, ClientDisconnectedError)


async def check_client_disconnect(
    req: Any,
    chunk_count: int,
    cancel_token: asyncio.Event | None,
    interval: int = DEFAULT_DISCONNECT_CHECK_INTERVAL,
    watcher: ClientDisconnectWatcher | None = None,
) -> bool:
    """Poll the client connection every *interval* chunks (always on the first).

    On disconnect, set *cancel_token* so the provider-side stream loop stops
    pushing data into a dead connection.

    When *watcher* is supplied the check is a synchronous flag read on a
    background receive task — see :class:`ClientDisconnectWatcher`. Without
    it, this falls back to the legacy timed poll, which blocks the response
    pump for ``_DISCONNECT_RECEIVE_WAIT_SECONDS`` on every healthy stream.
    """
    # Always check the very first chunk: a client that vanished while the
    # (potentially minutes-long) pre-response pipeline ran would otherwise
    # serve an entire stream into the void before the interval check fires.
    if req is None or (chunk_count % interval != 0 and chunk_count != 1):
        return False
    try:
        if watcher is not None:
            disconnected = await watcher.poll()
        else:
            from llm_proxy.streaming.handler import check_client_disconnected

            disconnected = await check_client_disconnected(req)
    except Exception:
        logger.debug("Failed to check client disconnect", exc_info=True)
        return False
    if not disconnected:
        return False
    logger.debug("Client disconnected during stream, signalling cancel_token to stop provider")
    if cancel_token:
        cancel_token.set()
    return True


@dataclass
class StreamPumpExhaustion:
    """Whether the pump behind :func:`iterate_chunks_with_comments` is done.

    The pump records its own exhaustion — the moment the upstream iterator has
    pumped every chunk it holds — because state owned by the pump *task* (its
    done flag) is already gone by the time a client close is processed: the
    wrapper cancels the task while unwinding. Streams whose wire format carries
    no recognizable terminal marker (binary payloads) need exactly this to tell
    a clean client close from a mid-generation abandonment.
    """

    provider_done: bool = False
    # Chunks the pump has queued that the consumer has not handed on yet.
    pending: int = 0

    @property
    def exhausted(self) -> bool:
        """True once the provider stream is fully pumped and fully drained."""
        return self.provider_done and self.pending == 0


async def iterate_chunks_with_comments(
    stream: AsyncIterator[Any],
    *,
    interval: float,
    comment: str | None,
    exhaustion: StreamPumpExhaustion | None = None,
):
    """Iterate stream chunks, emitting an SSE comment during silent gaps.

    A single pump task owns the upstream iterator and feeds a bounded queue,
    so the per-chunk cost is a queue handoff rather than a fresh task plus
    ``asyncio.wait`` per chunk. The upstream iterator itself is never
    cancelled mid-stream: the pump is only cancelled here in the teardown
    path (equivalent to the stream close the lifecycle performs anyway), and
    heartbeat timeouts only interrupt the consumer's ``queue.get``, which is
    cancellation-safe and loses no data. The bounded queue preserves
    backpressure for slow clients.

    ``exhaustion`` is optional: a caller that does not track liveness gets a
    private copy, which keeps the pump and drain loops free of ``None`` checks.
    """
    if interval <= 0:
        interval = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    if exhaustion is None:
        exhaustion = StreamPumpExhaustion()

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=256)

    async def _pump() -> None:
        try:
            async for chunk in stream:
                exhaustion.pending += 1
                await queue.put(("chunk", chunk))
            # Recorded BEFORE the end marker: the pump task's own state is not
            # observable after a teardown cancel, but this latch is.
            exhaustion.provider_done = True
            await queue.put(("end", None))
        except Exception as e:
            await queue.put(("error", e))

    pump_task = asyncio.ensure_future(_pump())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=interval)
            except TimeoutError:
                if pump_task.done() and queue.empty():
                    # The pump died without delivering end/error (only
                    # possible via external cancellation): stop instead of
                    # heartbeating forever. A done pump with a non-empty
                    # queue is just a consumer lagging behind — drain on.
                    return
                if comment is not None:
                    # Comments bypass the chunk transformer: they are not
                    # protocol data, just raw SSE comment frames.
                    yield ("comment", comment)
                continue
            if kind == "end":
                return
            if kind == "error":
                raise payload
            exhaustion.pending -= 1
            yield ("chunk", payload)
    finally:
        if not pump_task.done():
            # The provider may be one event-loop turn from recording its
            # exhaustion (StopAsyncIteration only lands once its own close
            # reaches us), so give it that window before cancelling: a clean
            # client close that raced the provider's EOF must not be misread as
            # an abandonment (see StreamPumpExhaustion).
            if not exhaustion.provider_done:
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(
                        asyncio.shield(pump_task), timeout=_PUMP_SETTLE_GRACE_SECONDS
                    )
            if not pump_task.done():
                pump_task.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await pump_task


class StreamLifecycle:
    """Lifecycle of a streaming chat-completion response.

    Extracted from the ``stream_generator`` closure in
    ``StreamingProcessor._create_streaming_response`` (issue LLMP-3):
    every captured variable is a named constructor field, and every
    branch of the old closure is a method — first-chunk replay
    (:meth:`_replay_first_chunks`), heartbeat-gated provider pump
    (:meth:`_pump_provider_stream`), native passthrough chunk shaping
    (:meth:`_shape_native_chunk`), web-search continuation
    (:meth:`_pump_web_search_continuation`), finalize + reasoning cache
    (:meth:`_finalize_and_cache`), persistence (:meth:`_persist_response`),
    error framing (:meth:`_handle_stream_error`), and best-effort teardown
    (:meth:`_teardown`).
    """

    def __init__(
        self,
        *,
        first_chunks: list[str],
        stream: Any,
        transformer: Any,
        stream_request: Any,
        event_context: EventContext | None,
        tracing_registry: Any,
        exit_stack: Any,
        req: Any = None,
        cancel_token: asyncio.Event | None = None,
        native_streaming: bool = False,
        protocol_name: str | None = None,
        config_manager: Any = None,
        response_store: Any = None,
        on_request_completed: Any = None,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        heartbeat_comment: str | None = None,
        web_search_interceptor: Any = None,
        web_search_tool_config: Any = None,
        should_intercept_web_search: bool = False,
        proxy_web_search_active: bool = False,
        current_adapter: Any = None,
        web_search_processor: WebSearchStreamProcessor | None = None,
        native_passthrough_handler: NativePassthroughHandler | None = None,
    ) -> None:
        self.first_chunks = first_chunks
        self.stream = stream
        self.transformer = transformer
        self.stream_request = stream_request
        self.event_context = event_context
        self.tracing_registry = tracing_registry
        self.exit_stack = exit_stack
        self.req = req
        self.cancel_token = cancel_token
        # Watches the ASGI receive channel on a background task so the chunk
        # pump never blocks waiting for a disconnect that is not coming.
        self.disconnect_watcher = ClientDisconnectWatcher(req) if req is not None else None
        self.native_streaming = native_streaming
        self.protocol_name = protocol_name
        self.config_manager = config_manager
        self.response_store = response_store
        self.on_request_completed = on_request_completed
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_comment = heartbeat_comment
        self.web_search_interceptor = web_search_interceptor
        self.web_search_tool_config = web_search_tool_config
        self.should_intercept_web_search = should_intercept_web_search
        self.proxy_web_search_active = proxy_web_search_active
        self.current_adapter = current_adapter
        self.web_search_processor = web_search_processor or WebSearchStreamProcessor()
        self.native_passthrough_handler = native_passthrough_handler or NativePassthroughHandler()

        # Mutable lifecycle state (one instance drives exactly one response).
        self.stream_error: Exception | None = None
        self.client_disconnected = False
        self.chunk_count = 0
        self.first_chunk_time: datetime | None = None
        # Set once every frame the protocol emits on success has been yielded:
        # a client close from then on is a clean close, so it must not be
        # recorded as a 499 (see _note_terminal_frame/_markable_abandonment).
        self._terminal_frames_delivered = False
        # Terminal stop/finish reason read before ``finalize()`` clears it
        # (protocols flush and reset their pending reason there).
        self._finish_reason: str | None = None
        # Provider extras (stop_sequence, stop_details, container,
        # diagnostics) read at the same moment, for the same reason.
        self._terminal_provider_info: dict[str, Any] | None = None
        # The pump's liveness, read at teardown: a provider whose stream was
        # fully pumped and drained when the close lands delivered its complete
        # output (see _markable_abandonment).
        self._pump_exhaustion = StreamPumpExhaustion()
        self.state = ContinuationState(transformer=transformer, stream_request=stream_request)

    async def events(self) -> AsyncIterator[str]:
        """Yield the SSE frames for this response."""
        try:
            if self.event_context is not None:
                self.event_context.is_streaming = True
                self.event_context.transformer = self.transformer
                # Native passthrough frames bypass the transformer entirely.
                # Protocols whose transformer rebuilds a body from those frames
                # still get a reassembled log body; the rest keep the raw SSE
                # rather than losing it.
                if (
                    self.native_streaming
                    and self._native_raw_capture_needed()
                    and self.event_context.should_capture_full_body
                ):
                    self.event_context.should_capture_raw_stream = True
            await self.tracing_registry.on_stream_start(self.stream_request, self.event_context)

            async for frame in self._replay_first_chunks():
                yield frame

            async for frame in self._pump_provider_stream():
                yield frame

            async for frame in self._pump_web_search_continuation():
                yield frame

            if (
                self.native_streaming
                and self.protocol_name == "openresponses"
                and not self._cancel_requested()
                and not self.client_disconnected
            ):
                # Spec: the terminal event MUST be the literal string
                # [DONE]. The native Responses upstream ends its stream
                # after response.completed without one; append it so the
                # passthrough path terminates exactly like the
                # transformer path.
                yield "data: [DONE]\n\n"

            if not self.native_streaming and not self.client_disconnected:
                # The finalize frame carries the protocol's terminal marker
                # (``data: [DONE]``) — the frame plenty of SSE SDKs act on by
                # dropping the TCP connection on the spot, whose racing FIN
                # cancels this task while the proxy is still finalizing.
                # Everything that frame depends on is already complete, so arm
                # the verdict BEFORE yielding it: a cancellation from here on
                # is a clean close and the request logs as a success.
                self._terminal_frames_delivered = True
                async for frame in self._finalize_and_cache():
                    yield frame
            elif self.native_streaming and not self.client_disconnected:
                # Native tiers skip _finalize_and_cache, but the transformer
                # still accumulated the forwarded frames for the request log;
                # cache the real reasoning from them too.
                self._cache_native_reasoning()

            # Persist store=true streamed responses so follow-up
            # previous_response_id continuations and GET /v1/responses/{id}
            # work, matching the non-streaming path. The persistence rules
            # are protocol knowledge owned by the transformer; protocols
            # without response storage no-op. Skipped when the client
            # disconnected mid-stream (the snapshot would be partial) and
            # best-effort otherwise.
            await self._persist_response()
        except asyncio.CancelledError, GeneratorExit:
            # The response task was cancelled, or the response was closed
            # before it ended. Record a client abandonment (499) only when the
            # client could actually have lost output: a teardown that lands
            # after the terminal frame is the SDK that dropped the socket on
            # [DONE] while the proxy was still persisting — a successful
            # request, logged as one (see _markable_abandonment).
            if self._markable_abandonment():
                self._mark_disconnected()
            raise
        except Exception as e:
            async for frame in self._handle_stream_error(e):
                yield frame
        finally:
            await self._teardown()

    # -- state helpers ------------------------------------------------

    _TERMINAL_FRAME_MARKERS = ("data: [DONE]", "message_stop", '"response.completed"')

    def _note_terminal_frame(self, frame: Any) -> None:
        """Arm the delivered verdict when the frame about to yield is terminal.

        The arm happens BEFORE the frame is yielded: with the terminal frame in
        the response pipe the client holds the whole response, so a socket it
        drops afterwards (the SDK acting on [DONE]/message_stop, whose FIN
        cancels this task) is a clean close, not a 499 abandonment.
        """
        if self._terminal_frames_delivered or self.client_disconnected:
            return
        if isinstance(frame, str) and any(
            marker in frame for marker in self._TERMINAL_FRAME_MARKERS
        ):
            self._terminal_frames_delivered = True

    def _markable_abandonment(self) -> bool:
        """Whether a premature close/cancellation is a real abandonment.

        False once every frame the protocol emits on success has been handed to
        the streaming response, and false once the provider pump ran dry with
        everything it queued delivered: either way the client has the full
        response and the teardown cancellation its own close caused must not
        be blamed on it as a 499.
        """
        if self.client_disconnected or self._terminal_frames_delivered:
            return False
        return not self._pump_exhaustion.exhausted

    def _mark_disconnected(self) -> None:
        """Record the abandonment (client_disconnected + stream error) once."""
        self.client_disconnected = True
        if self.stream_error is None:
            self.stream_error = ClientDisconnectedError()

    def _cancel_requested(self) -> bool:
        return self.cancel_token is not None and self.cancel_token.is_set()

    # -- lifecycle phases ---------------------------------------------

    async def _replay_first_chunks(self) -> AsyncIterator[str]:
        for chunk in self.first_chunks:
            if self.native_streaming:
                # Peeked native blocks must get the same bookkeeping as
                # pumped ones (model-alias rewrite, usage capture).
                chunk = self._shape_native_chunk(chunk)
                if chunk is None:
                    continue
            if self.first_chunk_time is None:
                self.first_chunk_time = datetime.now(UTC)
            if self.event_context is not None:
                self.event_context.first_chunk_time = self.first_chunk_time
            await self.tracing_registry.on_stream_chunk(
                self.stream_request, chunk, self.event_context
            )
            yield chunk

    async def _pump_provider_stream(self) -> AsyncIterator[str]:
        async for kind, payload in iterate_chunks_with_comments(
            self.stream,
            interval=self.heartbeat_interval,
            comment=self.heartbeat_comment,
            exhaustion=self._pump_exhaustion,
        ):
            if kind == "comment":
                if not self.client_disconnected and not self._cancel_requested():
                    yield payload
                continue
            chunk = payload
            if not isinstance(chunk, (str, dict)):
                continue
            if self._cancel_requested():
                logger.debug("Stream cancelled by cancel_token, stopping chunk iteration")
                return

            if self.native_streaming:
                chunk = self._shape_native_chunk(chunk)
                if chunk is None:
                    # Bookkeeping-only frame (e.g. a usage frame the client
                    # did not ask for): captured, not emitted.
                    continue
                self._note_terminal_frame(chunk)
                await self.tracing_registry.on_stream_chunk(
                    self.stream_request, chunk, self.event_context
                )
                yield chunk
                self.chunk_count += 1
                if await self._poll_disconnect():
                    return
                continue

            transformed = self.transformer.transform(chunk)
            if transformed:
                self._note_terminal_frame(transformed)
                await self.tracing_registry.on_stream_chunk(
                    self.stream_request, transformed, self.event_context
                )
                yield transformed
                self.chunk_count += 1
                if await self._poll_disconnect():
                    return

        # Only reached when the provider stream ran to its natural end: every
        # early exit above returned, so no frames were lost. Providers whose
        # terminal event is free-form may never match the marker check in
        # _note_terminal_frame, but with the pump drained the client has
        # everything either way. Transformed protocols still owe the finalize()
        # frame after this pump — that call site arms the verdict itself — and
        # a web-search continuation round to come manages the verdict on its
        # own.
        if self.native_streaming and not self.should_intercept_web_search:
            self._terminal_frames_delivered = True

    def _shape_native_chunk(self, chunk: Any) -> Any:
        """Rewrite a native passthrough chunk per protocol before emitting.

        Returns the frame to emit, or None to swallow it (bookkeeping-only
        frames).
        """
        handler = self.native_passthrough_handler
        if isinstance(chunk, str) and contains_sse_event(chunk, "message_start"):
            # Mask the upstream's internal model name with the
            # client-requested alias (see InternalRequest.echo_model).
            chunk = handler.inject_model_into_anthropic_message_start(
                chunk, self.stream_request.echo_model
            )
        # Feed the request-log accumulator for every native tier; each
        # transformer decides what it can rebuild from the frames (see
        # ``native_frame_accumulation``).
        self._accumulate_native_chunk(chunk)
        if self.protocol_name == "anthropic":
            handler.maybe_capture_native_streaming_usage(chunk, self.event_context)
        elif self.protocol_name == "openai":
            chunk = handler.handle_native_openai_chunk(
                chunk, self.stream_request, self.event_context, self.current_adapter
            )
        elif self.protocol_name == "openresponses":
            # The snapshot's model is rewritten to the client-requested
            # alias (see InternalRequest.echo_model) so the native stream
            # echoes the same name as the transformer path.
            rewritten = handler.maybe_capture_native_openresponses(
                chunk,
                self.transformer,
                self.event_context,
                model=self.stream_request.echo_model,
            )
            if rewritten is not None:
                chunk = rewritten
        return chunk

    def _accumulate_native_chunk(self, chunk: Any) -> None:
        """Feed a native frame to the transformer's accumulator.

        Runs for the request log (only when full-body capture is on and raw SSE
        capture is off — reassembly would be wasted work otherwise) and always
        for the reasoning cache on delta protocols, whose accumulation is the
        only reasoning source at stream end (``_cache_native_reasoning``). The
        cache must not depend on the log-sampling decision, or a lower
        ``sampling_rate`` silently stops teaching it. Snapshot protocols
        (``openresponses``) write the cache from their terminal snapshot, so
        their accumulation stays gated on logging.
        """
        context = self.event_context
        if context is None:
            return
        logging_needs_accumulation = (
            context.should_capture_full_body and not context.should_capture_raw_stream
        )
        if not logging_needs_accumulation and self.protocol_name not in (
            "openai",
            "anthropic",
        ):
            return
        accumulate_native_frame(self.transformer, chunk)

    def _native_raw_capture_needed(self) -> bool:
        """Whether a native stream has to buffer its raw SSE for the log.

        Only when the protocol's transformer cannot rebuild a body from the
        native frames; otherwise the reassembled body is stored and per-chunk
        buffering is skipped entirely.
        """
        return not can_accumulate_native_frames(self.transformer)

    async def _poll_disconnect(self) -> bool:
        """Check for client disconnect; on hit, record the abandonment.

        Screened by the delivered verdict: once the terminal frame is out, a FIN
        is a clean close — the pump stops pumping without labelling the request
        a 499.
        """
        disconnected = await check_client_disconnect(
            self.req, self.chunk_count, self.cancel_token, watcher=self.disconnect_watcher
        )
        if disconnected and not self._terminal_frames_delivered:
            self._mark_disconnected()
        return disconnected

    async def _pump_web_search_continuation(self) -> AsyncIterator[str]:
        if not self.should_intercept_web_search or self.client_disconnected:
            return
        # A continuation round generates MORE provider tokens: a client close
        # from here on is a real abandonment again, until the round's own
        # frames are all out.
        self._terminal_frames_delivered = False
        self._pump_exhaustion = StreamPumpExhaustion()
        async for kind, payload in iterate_chunks_with_comments(
            self.web_search_processor.generate_continuation(
                self.state,
                web_search_interceptor=self.web_search_interceptor,
                web_search_tool_config=self.web_search_tool_config,
                proxy_web_search_active=self.proxy_web_search_active,
                current_adapter=self.current_adapter,
                tracing_registry=self.tracing_registry,
                event_context=self.event_context,
                cancel_token=self.cancel_token,
            ),
            interval=self.heartbeat_interval,
            comment=self.heartbeat_comment,
            exhaustion=self._pump_exhaustion,
        ):
            if kind == "comment":
                # Same gating as the main loop: comments stop
                # flowing once the client abandoned the stream.
                if not self.client_disconnected and not self._cancel_requested():
                    yield payload
                continue
            if self._cancel_requested():
                return
            yield payload
        # The continuation round finished pumping: its frames are all out, the
        # client has the full response again.
        self._terminal_frames_delivered = True

    async def _finalize_and_cache(self) -> AsyncIterator[str]:
        state = self.state
        if state.depth > 0 and self.transformer is not state.transformer:
            merge_continuation_usage(self.transformer, state.transformer)

        # Read before finalize(): protocols flush and clear their pending stop
        # reason there, and the reassembled log body needs the real reason
        # (``length``, ``max_tokens``) rather than the formatter's default.
        self._finish_reason = terminal_finish_reason(state.transformer)
        self._terminal_provider_info = terminal_provider_info(state.transformer)
        final_chunk = state.transformer.finalize()
        if final_chunk:
            await self.tracing_registry.on_stream_chunk(
                state.stream_request, final_chunk, self.event_context
            )
            yield final_chunk

        # The accumulated output is complete after finalize(). Cache real
        # reasoning paired with its tool calls so subsequent turns can
        # restore it (DeepSeek-style echo) even when the client strips
        # reasoning or another provider served an intermediate turn.
        # Web-search continuations keep the final output split across two
        # transformers, so cache both.
        self._cache_accumulated_reasoning(state.transformer, self.transformer)

    def _cache_native_reasoning(self) -> None:
        """Cache reasoning accumulated from native-passthrough frames.

        Native tiers skip ``_finalize_and_cache``, so streamed reasoning would
        never reach the reasoning cache; but the transformer still accumulates
        the forwarded frames for the request log. Finalize that accumulation
        (idempotent) and cache the real reasoning paired with its tool calls, so
        a later turn can restore it when the client strips reasoning or another
        provider served the intermediate turn.
        """
        transformers = [self.state.transformer, self.transformer]
        flush_pending_accumulation(transformers)
        self._cache_accumulated_reasoning(*transformers)

    def _cache_accumulated_reasoning(self, *transformers: Any) -> None:
        """Cache reasoning from each transformer's accumulated output.

        The final output can be split across two transformers (web-search
        continuations), and the same transformer may be passed more than once;
        cache each one exactly once. Never fatal.
        """
        seen: set[int] = set()
        for target in transformers:
            if id(target) in seen:
                continue
            seen.add(id(target))
            try_cache_reasoning_from_blocks(
                target.get_accumulated_output(),
                response_id=getattr(target, "response_id", "") or "",
            )

    async def _persist_response(self) -> None:
        if (
            self.response_store is not None
            and not self.client_disconnected
            and not self._cancel_requested()
        ):
            await safe_cleanup(
                self.transformer.finalize_persistence(
                    self.stream_request,
                    self.response_store,
                    self.event_context,
                ),
                "Failed to persist streamed response",
            )

    async def _handle_stream_error(self, e: Exception) -> AsyncIterator[str]:
        self.stream_error = e
        if self.event_context is not None:
            self.event_context.error_message = str(e)
        await self.tracing_registry.on_error(self.state.stream_request, e, self.event_context)
        # Error wire shaping is protocol knowledge owned by each
        # protocol-side transformer (OpenResponses: response.failed +
        # [DONE]; Anthropic: named ``event: error``; default: generic
        # chat-completions error frame + [DONE]).
        for frame in self.transformer.error_frames(e):
            yield frame
        if self.should_intercept_web_search:
            await self.web_search_processor.process_streaming_web_search(
                self.transformer, self.web_search_interceptor, self.web_search_tool_config
            )

    def _assemble_logged_response_body(self) -> None:
        """Reassemble the streamed response for the request log.

        Stored in place of the raw SSE text unless the operator opted into raw
        capture (``logging.log_raw_stream``), the client forced it with
        ``x-log-full``, or the request took a native-passthrough tier whose
        transformer cannot rebuild the body from native frames. Best-effort:
        when reassembly fails the audit handler stores a marker rather than
        dropping the log row.
        """
        context = self.event_context
        if context is None or not context.should_capture_full_body:
            return
        if context.should_capture_raw_stream:
            # Per-chunk capture already holds the raw SSE text.
            return
        transformers = [self.transformer, self.state.transformer]
        # Content buffered by a stream that ended before its terminal event
        # (client abort, upstream cut) has to be flushed on every tier: the
        # converted path skips ``_finalize_and_cache`` — and with it
        # ``finalize()`` — on exactly those paths. No-op once ``finalize()``
        # ran (it clears the buffers) and where a transformer buffers nothing.
        flush_pending_accumulation(transformers)
        if self.native_streaming:
            # A protocol that carries the whole response on its terminal event
            # needs no reassembly at all. A native tier whose transformer cannot
            # rebuild the body from frames buffered the raw SSE instead, which
            # ``events`` recorded by arming ``should_capture_raw_stream`` (and
            # this method returned on above).
            snapshot = first_native_log_body(transformers)
            if snapshot is not None:
                context.assembled_response_body = snapshot
                return
        context.assembled_response_body = assemble_stream_response_body(
            context,
            protocol_name=self.protocol_name,
            response_id=getattr(self.transformer, "response_id", "") or "",
            model=getattr(self.stream_request, "user_facing_model", None) or context.model or "",
            output=collect_accumulated_output(transformers),
            finish_reason=self._finish_reason or terminal_finish_reason(self.state.transformer),
            provider_info=self._terminal_provider_info
            or terminal_provider_info(self.state.transformer),
        )

    async def _teardown(self) -> None:
        if self.disconnect_watcher is not None:
            await safe_cleanup(
                self.disconnect_watcher.aclose(),
                "Failed to stop client disconnect watcher",
            )
        if self.stream is not None:
            await safe_cleanup(
                asyncio.shield(quiet_aclose(self.stream)),
                f"Failed to close provider stream for {self.state.stream_request.model}",
            )
        await safe_cleanup(
            asyncio.shield(self.exit_stack.aclose()),
            "Failed to close adapter exit stack",
        )
        if self.event_context is not None and self.config_manager is not None:
            await safe_cleanup(
                finalize_event_cost(self.event_context, self.config_manager),
                "Failed to finalize event cost",
            )
        # Fire the post-request hook (model-experience observation) once
        # the stream has actually finished; the early return in process()
        # never reaches the hook call that covers setup failures.
        if self.event_context is not None and self.on_request_completed is not None:
            await safe_cleanup(
                self.on_request_completed(
                    self.event_context, _experience_counts_as_success(self.stream_error)
                ),
                "Failed to call on_request_completed",
            )
        # Reassemble the streamed body for the request log. No-op when raw
        # capture is enabled or the body was sampled out; runs after cost
        # finalization so token usage is already on the context.
        self._assemble_logged_response_body()
        await safe_cleanup(
            self.tracing_registry.on_stream_end(
                self.state.stream_request,
                self.event_context,
                error=self.stream_error,
            ),
            "Failed to call on_stream_end for tracing registry",
        )


class GenericStreamLifecycle:
    """Lifecycle for the generic streams: image, speech, transcription.

    Reuses the same lifecycle skeleton as :class:`StreamLifecycle` (issue
    LLMP-3) — tracing start/chunk/end, best-effort teardown, and the
    post-request hook — but pumps raw provider chunks unchanged: generic
    streams have no transformer (they may carry binary audio payloads) and
    emit no heartbeat comment, so the shared queue pump is a pass-through with
    an optional per-chunk observer used for billing capture.

    Logging is raw where a protocol handler asked for it: with
    ``trace_chunks`` the lifecycle arms ``should_capture_raw_stream`` (nothing
    here can reassemble a body), so the frames the client received are what the
    request log stores.
    """

    def __init__(
        self,
        *,
        stream: Any,
        stream_request: Any,
        event_context: EventContext | None,
        tracing_registry: Any,
        streaming_stack: Any,
        config_manager: Any = None,
        on_request_completed: Any = None,
        trace_chunks: bool = False,
        chunk_observer: Any = None,
    ) -> None:
        self.stream = stream
        self.stream_request = stream_request
        self.event_context = event_context
        self.tracing_registry = tracing_registry
        self.streaming_stack = streaming_stack
        self.config_manager = config_manager
        self.on_request_completed = on_request_completed
        self.trace_chunks = trace_chunks
        self.chunk_observer = chunk_observer
        # Capture the stream error so it can be forwarded to on_stream_end.
        self.stream_error: Exception | None = None
        # The provider payload was fully handed to the client once the pump
        # delivered its last chunk: from that point a client close is a clean
        # close, not an abandonment — the same verdict StreamLifecycle reaches
        # from its terminal frames (see _record_abandonment).
        self._response_fully_delivered = False
        # The pump's liveness, read at teardown; one stream per instance.
        self._pump_exhaustion = StreamPumpExhaustion()

    def _record_abandonment(self) -> None:
        """Record a client abandonment (499) when the payload was cut short.

        The exhaustion state is read at cancellation/teardown time, after the
        close itself already went through the event loop: a provider whose
        stream was fully pumped and drained (every chunk it queued was handed
        to the response) delivered its complete output — the close is clean.
        Only a still-open or under-drained pump means the generation was
        actually cut short and becomes a ClientDisconnectedError.
        """
        if self._response_fully_delivered or self._pump_exhaustion.exhausted:
            return
        self.stream_error = ClientDisconnectedError()
        if self.event_context is not None:
            self.event_context.error_message = str(self.stream_error)

    async def events(self) -> AsyncIterator[Any]:
        if self.event_context is not None and self.trace_chunks:
            # Generic streams have no transformer and no reassembly step, so the
            # SSE frames are the only body the request log can store: raw
            # capture is what makes them reach it, exactly as a native tier
            # whose transformer cannot rebuild the body arms it. ``trace_chunks``
            # is set by the protocol handler that asked for per-chunk capture;
            # the sampling decision has already gated it on full-body capture.
            self.event_context.should_capture_raw_stream = (
                self.event_context.should_capture_full_body
            )
        await self.tracing_registry.on_stream_start(self.stream_request, self.event_context)
        try:
            async for kind, payload in iterate_chunks_with_comments(
                self.stream,
                # Generic streams emit no heartbeat comment (binary payloads):
                # this interval only paces the queue read, and the pump exists
                # to report the stream's liveness (see _record_abandonment).
                interval=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
                comment=None,
                exhaustion=self._pump_exhaustion,
            ):
                if kind != "chunk":
                    continue
                if self.trace_chunks:
                    await self.tracing_registry.on_stream_chunk(
                        self.stream_request, payload, self.event_context
                    )
                if self.chunk_observer is not None:
                    with suppress(Exception):
                        self.chunk_observer(payload)
                yield payload
            # The provider stream was fully pumped and every frame it sent was
            # handed to the response: the client has the complete output.
            self._response_fully_delivered = True
        except asyncio.CancelledError, GeneratorExit:
            # The response task was cancelled, or the response was closed
            # before it ended. A generation the client actually lost is recorded
            # so the log shows a 499 instead of a phantom success; a close that
            # lands after the full payload was handed over stays a clean
            # success.
            self._record_abandonment()
            raise
        except Exception as e:
            self.stream_error = e
            if self.event_context:
                self.event_context.error_message = str(e)
            await self.tracing_registry.on_error(self.stream_request, e, self.event_context)
            raise
        finally:
            with suppress(Exception, asyncio.CancelledError):
                if self.stream is not None:
                    await asyncio.shield(quiet_aclose(self.stream))
                if self.event_context is not None:
                    await finalize_event_cost(self.event_context, self.config_manager)
                # Post-request hook (model-experience observation) for
                # generic streams; see StreamLifecycle._teardown.
                if self.event_context is not None and self.on_request_completed is not None:
                    await self.on_request_completed(
                        self.event_context, _experience_counts_as_success(self.stream_error)
                    )
                await self.tracing_registry.on_stream_end(
                    self.stream_request, self.event_context, error=self.stream_error
                )
                await asyncio.shield(self.streaming_stack.aclose())
