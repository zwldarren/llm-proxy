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
from datetime import UTC, datetime
from typing import Any

from llm_proxy.core.constants import DEFAULT_DISCONNECT_CHECK_INTERVAL
from llm_proxy.core.conversion import NativePassthroughHandler
from llm_proxy.core.exceptions import ClientDisconnectedError
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
from llm_proxy.streaming.sse_parse import contains_sse_event

logger = get_logger(__name__)

# SSE comment frame emitted (and ignored by every SSE parser per the WHATWG
# spec) when the upstream stream falls silent, so fronting CDNs such as
# Cloudflare keep the connection open while a model ponders its next token.
SSE_KEEPALIVE_COMMENT = ": keep-alive\n\n"

# Default heartbeat interval, overridden by the UI-managed keepalive config.
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15.0


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


async def check_client_disconnect(
    req: Any,
    chunk_count: int,
    cancel_token: asyncio.Event | None,
    interval: int = DEFAULT_DISCONNECT_CHECK_INTERVAL,
) -> bool:
    """Poll the client connection every *interval* chunks (always on the first).

    On disconnect, set *cancel_token* so the provider-side stream loop stops
    pushing data into a dead connection.
    """
    # Always check the very first chunk: a client that vanished while the
    # (potentially minutes-long) pre-response pipeline ran would otherwise
    # serve an entire stream into the void before the interval check fires.
    if req is None or (chunk_count % interval != 0 and chunk_count != 1):
        return False
    try:
        from llm_proxy.streaming.handler import check_client_disconnected

        if await check_client_disconnected(req):
            logger.debug(
                "Client disconnected during stream, signalling cancel_token to stop provider"
            )
            if cancel_token:
                cancel_token.set()
            return True
    except Exception:
        logger.debug("Failed to check client disconnect", exc_info=True)
    return False


async def iterate_chunks_with_comments(
    stream: AsyncIterator[Any],
    *,
    interval: float,
    comment: str | None,
):
    """Iterate stream chunks, emitting an SSE comment during silent gaps.

    The upstream iterator itself is never cancelled: a cancelled ``anext``
    would inject a ``CancelledError`` into the provider stream and
    truncate it. One racing task is instead awaited repeatedly between
    comment emissions, so silence yields comments and data resumes from
    the same in-flight read.
    """
    if interval <= 0:
        interval = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    iterator = aiter(stream)
    while True:
        chunk_task: asyncio.Task = asyncio.ensure_future(anext(iterator))
        try:
            while True:
                done, _ = await asyncio.wait({chunk_task}, timeout=interval)
                if done:
                    try:
                        chunk = chunk_task.result()
                    except StopAsyncIteration:
                        return
                    yield ("chunk", chunk)
                    break
                if comment is not None:
                    # Comments bypass the chunk transformer: they are not
                    # protocol data, just raw SSE comment frames.
                    yield ("comment", comment)
        finally:
            if not chunk_task.done():
                chunk_task.cancel()
                with suppress(Exception):
                    await chunk_task


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
        self.state = ContinuationState(transformer=transformer, stream_request=stream_request)

    async def events(self) -> AsyncIterator[str]:
        """Yield the SSE frames for this response."""
        try:
            if self.event_context is not None:
                self.event_context.is_streaming = True
                self.event_context.transformer = self.transformer
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
                async for frame in self._finalize_and_cache():
                    yield frame

            # Persist store=true streamed responses so follow-up
            # previous_response_id continuations and GET /v1/responses/{id}
            # work, matching the non-streaming path. The persistence rules
            # are protocol knowledge owned by the transformer; protocols
            # without response storage no-op. Skipped when the client
            # disconnected mid-stream (the snapshot would be partial) and
            # best-effort otherwise.
            await self._persist_response()
        except asyncio.CancelledError:
            # The response task was cancelled — for a client disconnect
            # this is the origin-side 524 moment. Mark the log entry as a
            # client abandonment (499) instead of a successful request.
            self._mark_disconnected()
            raise
        except GeneratorExit:
            # The response was closed without reaching the end (client
            # disconnect mid-response, server teardown). Recorded so the
            # abandonment is visible in the logs.
            self._mark_disconnected()
            raise
        except Exception as e:
            async for frame in self._handle_stream_error(e):
                yield frame
        finally:
            await self._teardown()

    # -- state helpers ------------------------------------------------

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
                break

            if self.native_streaming:
                chunk = self._shape_native_chunk(chunk)
                await self.tracing_registry.on_stream_chunk(
                    self.stream_request, chunk, self.event_context
                )
                yield chunk
                self.chunk_count += 1
                if await self._poll_disconnect():
                    break
                continue

            transformed = self.transformer.transform(chunk)
            if transformed:
                await self.tracing_registry.on_stream_chunk(
                    self.stream_request, transformed, self.event_context
                )
                yield transformed
                self.chunk_count += 1
                if await self._poll_disconnect():
                    break

    def _shape_native_chunk(self, chunk: Any) -> Any:
        """Rewrite a native passthrough chunk per protocol before emitting."""
        handler = self.native_passthrough_handler
        if isinstance(chunk, str) and contains_sse_event(chunk, "message_start"):
            # Mask the upstream's internal model name with the
            # client-requested alias (see InternalRequest.echo_model).
            chunk = handler.inject_model_into_anthropic_message_start(
                chunk, self.stream_request.echo_model
            )
        if self.protocol_name == "anthropic":
            handler.maybe_capture_native_streaming_usage(chunk, self.event_context)
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

    async def _poll_disconnect(self) -> bool:
        """Check for client disconnect; on hit, record the abandonment."""
        if await check_client_disconnect(self.req, self.chunk_count, self.cancel_token):
            self._mark_disconnected()
            return True
        return False

    async def _pump_web_search_continuation(self) -> AsyncIterator[str]:
        if not self.should_intercept_web_search or self.client_disconnected:
            return
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
        ):
            if kind == "comment":
                # Same gating as the main loop: comments stop
                # flowing once the client abandoned the stream.
                if not self.client_disconnected and not self._cancel_requested():
                    yield payload
                continue
            if self._cancel_requested():
                break
            yield payload

    async def _finalize_and_cache(self) -> AsyncIterator[str]:
        state = self.state
        if state.depth > 0 and self.transformer is not state.transformer:
            merge_continuation_usage(self.transformer, state.transformer)

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
        # transformers, so cache both. Never fatal.
        cache_targets = [state.transformer]
        if self.transformer is not state.transformer:
            cache_targets.append(self.transformer)
        for target in cache_targets:
            try_cache_reasoning_from_blocks(
                target.get_accumulated_output(),
                response_id=target.response_id,
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

    async def _teardown(self) -> None:
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
        # never reaches the hook call that covers setup failures. A
        # client disconnect is not a model failure (EWMA measures
        # provider health), so it still counts as success here.
        if self.event_context is not None and self.on_request_completed is not None:
            experience_success = self.stream_error is None or isinstance(
                self.stream_error, ClientDisconnectedError
            )
            await safe_cleanup(
                self.on_request_completed(self.event_context, experience_success),
                "Failed to call on_request_completed",
            )
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
    no heartbeat, so the pump is a plain pass-through with an optional
    per-chunk observer used for billing capture.
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

    async def events(self) -> AsyncIterator[Any]:
        await self.tracing_registry.on_stream_start(self.stream_request, self.event_context)
        try:
            async for chunk in self.stream:
                if self.trace_chunks:
                    await self.tracing_registry.on_stream_chunk(
                        self.stream_request, chunk, self.event_context
                    )
                if self.chunk_observer is not None:
                    with suppress(Exception):
                        self.chunk_observer(chunk)
                yield chunk
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
                    await self.on_request_completed(self.event_context, self.stream_error is None)
                await self.tracing_registry.on_stream_end(
                    self.stream_request, self.event_context, error=self.stream_error
                )
                await asyncio.shield(self.streaming_stack.aclose())
