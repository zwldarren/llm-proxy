"""Streaming request processor with provider fallback support."""

import asyncio
import inspect
import uuid
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.protocols.base import ProtocolEndpoint

from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from llm_proxy.billing.image_stream_usage import ImageStreamUsageTracker
from llm_proxy.billing.transcription_stream_usage import TranscriptionStreamUsageTracker
from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.conversion import NativePassthroughHandler, plan_conversion
from llm_proxy.core.errors import (
    is_context_length_finish_reason,
    is_retryable_stream_finish_reason,
)
from llm_proxy.core.errors.handler import ErrorHandler
from llm_proxy.core.exceptions import (
    ConfigurationError,
    ProviderError,
)
from llm_proxy.core.processing.base import RequestContext, mirror_conversion_tier
from llm_proxy.core.processing.stages.fallback_handler import FallbackHandler
from llm_proxy.core.processing.stages.parameter_override import ParameterOverrideService
from llm_proxy.core.processing.strategies import StreamingResponseMarker
from llm_proxy.core.processing.strategies.chunk_parser import OpenAIStreamChunkParser
from llm_proxy.core.processing.stream_lifecycle import (
    DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    SSE_KEEPALIVE_COMMENT,
    GenericStreamLifecycle,
    StreamLifecycle,
)
from llm_proxy.core.processing.web_search_streaming import (
    WebSearchStreamProcessor,
)
from llm_proxy.core.utils import quiet_aclose
from llm_proxy.models import (
    ConversionTier,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalRequest,
    InternalSpeechRequest,
    InternalTranscriptionRequest,
)
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.logger import get_logger
from llm_proxy.streaming.handler import StreamingHandler

logger = get_logger(__name__)


@dataclass
class _PrefetchResult:
    """Result of prefetching the first chunks from a stream."""

    first_chunks: list[str]
    stream_started: bool
    context_exceeded: bool
    context_exceeded_reason: str | None = None
    retryable_stream_finish_reason: str | None = None


# ------------------------------------------------------------------
# Stream usage trackers (image/transcription billing) live in
# ``llm_proxy.billing`` as public modules; the processor only wires them
# into the generic stream loop.
# ------------------------------------------------------------------


class StreamingProcessor:
    """Process streaming chat completion requests with provider fallback support."""

    def __init__(
        self,
        protocol_endpoint: ProtocolEndpoint,
        streaming_handler: StreamingHandler,
        error_handler: ErrorHandler,
        param_override_service: ParameterOverrideService,
        chunk_parser: OpenAIStreamChunkParser | None = None,
        web_search_processor: WebSearchStreamProcessor | None = None,
    ):
        self.protocol_endpoint = protocol_endpoint
        self.streaming_handler = streaming_handler
        self._error_handler = error_handler
        self._param_override_service = param_override_service
        self._chunk_parser = chunk_parser or OpenAIStreamChunkParser()
        self._web_search_processor = web_search_processor or WebSearchStreamProcessor()
        self._fallback_handler = FallbackHandler(error_handler, param_override_service)
        self._native_passthrough_handler = NativePassthroughHandler()

    async def process(
        self,
        streaming_marker: StreamingResponseMarker,
        raw_request_data: dict[str, Any],
        req: Request,
        context: RequestContext,
        trace_id: str,
        event_context: EventContext | None = None,
        exit_stack: AsyncExitStack | None = None,
    ) -> Response:
        """Process a streaming chat completion request with provider fallback support.

        This method handles streaming requests with automatic fallback to other providers
        when the current provider fails during stream initialization or returns empty
        responses. Once streaming starts (first chunk received), we commit to that
        provider since streaming is stateful and cannot be safely retried mid-stream.

        Args:
            streaming_marker: The streaming response marker with request and adapter
            raw_request_data: Raw request data dict for parameter overrides
            req: The FastAPI request
            context: Request context containing orchestrator and dependencies
            trace_id: The trace ID for request tracking and cleanup
            event_context: Optional EventContext for unified data capture
            exit_stack: AsyncExitStack managing adapter lifecycle

        Returns:
            StreamingResponse with appropriate headers
        """
        unified_request = streaming_marker.request
        current_adapter = streaming_marker.adapter

        if isinstance(unified_request, (InternalImageEditRequest, InternalImageRequest)):
            return await self._process_image_streaming(
                streaming_marker, unified_request, context, exit_stack
            )
        if isinstance(unified_request, InternalSpeechRequest):
            return await self._process_speech_streaming(
                streaming_marker, unified_request, context, exit_stack
            )
        if isinstance(unified_request, InternalTranscriptionRequest):
            return await self._process_transcription_streaming(
                streaming_marker, unified_request, context, exit_stack
            )

        _exit_stack = exit_stack or AsyncExitStack()
        response_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        # The client-visible model name echoed in stream chunks: the
        # client-requested alias (event_context.model, set by
        # ProviderSelectionStage), never the resolved provider model name.
        model = (
            event_context.model if event_context and event_context.model else unified_request.model
        )
        tracing_registry = context.tracing_registry or self._get_tracing_registry()

        stream_cancel_token = asyncio.Event()

        result_response: Response | None = None
        final_error: Exception | None = None

        stream = None
        should_clean_stream = True
        should_clean_exit_stack = True

        protocol_name = context.protocol_name

        try:
            while True:
                transformer_cls = self.protocol_endpoint.get_streaming_transformer()
                if transformer_cls is None:
                    raise ConfigurationError(
                        f"Protocol {self.protocol_endpoint.name} does not support streaming"
                    )

                # Read the interception flag fresh on every attempt: provider
                # fallback re-runs WebSearchStage, which recomputes it per
                # provider (native_web_search).
                proxy_web_search_active = context.proxy_web_search_active

                stream_plan = plan_conversion(current_adapter, unified_request)
                native = stream_plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH
                # Stamp the response-side tier for observability: streaming
                # responses never pass through _parse_response /
                # _build_passthrough_response (the non-stream chokepoints),
                # so mirror_conversion_tier below reads the stream mode
                # instead. Re-stamped per attempt like conversion_tier.
                unified_request.response_tier = stream_plan.stream_mode

                include_obfuscation = None
                if hasattr(unified_request, "stream_options") and unified_request.stream_options:
                    include_obfuscation = unified_request.stream_options.include_obfuscation

                transformer_kwargs = {
                    "model": model,
                    "request_id": response_id,
                }
                if include_obfuscation is not None:
                    sig = inspect.signature(transformer_cls.__init__)
                    if "include_obfuscation" in sig.parameters:
                        transformer_kwargs["include_obfuscation"] = include_obfuscation

                # Pass web_search interceptor presence so the streaming transformer
                # knows whether to intercept (server-side) or emit to client.
                sig = inspect.signature(transformer_cls.__init__)
                if "intercept_web_search" in sig.parameters:
                    transformer_kwargs["intercept_web_search"] = proxy_web_search_active

                transformer = transformer_cls(**transformer_kwargs)

                stream = None
                try:
                    if native:
                        stream = await current_adapter.stream_chat_completion_native(
                            unified_request, cancel_token=stream_cancel_token
                        )
                        first_chunks: list[str] = []
                        stream_started = True
                        context_exceeded = False
                        retryable_stream_finish_reason = None
                    else:
                        stream = await current_adapter.stream_chat_completion(
                            unified_request, cancel_token=stream_cancel_token
                        )

                        prefetch = await self._prefetch_stream_chunks(stream, transformer)
                        first_chunks = prefetch.first_chunks
                        stream_started = prefetch.stream_started
                        context_exceeded = prefetch.context_exceeded
                        context_exceeded_reason = prefetch.context_exceeded_reason
                        retryable_stream_finish_reason = prefetch.retryable_stream_finish_reason

                        if context_exceeded and context_exceeded_reason:
                            await self._close_stream(stream)
                            stream = None
                            should_continue = await self._fallback_handler.handle_context_exceeded(
                                context_exceeded_reason,
                                current_adapter,
                                context,
                                unified_request,
                                raw_request_data,
                                req,
                            )
                            if should_continue:
                                await self._fallback_handler.switch_adapter(
                                    _exit_stack, current_adapter, should_continue[0]
                                )
                                current_adapter = should_continue[0]
                                unified_request = should_continue[1]
                                continue
                            final_error = self._error_handler.create_context_length_error(
                                current_adapter.provider_name, context_exceeded_reason
                            )
                            result_response = self._error_handler.format_response(final_error)
                            break

                        if retryable_stream_finish_reason:
                            await self._close_stream(stream)
                            stream = None
                            should_continue = (
                                await self._fallback_handler.handle_retryable_finish_reason(
                                    retryable_stream_finish_reason,
                                    current_adapter,
                                    context,
                                    unified_request,
                                    raw_request_data,
                                    req,
                                )
                            )
                            if should_continue:
                                await self._fallback_handler.switch_adapter(
                                    _exit_stack, current_adapter, should_continue[0]
                                )
                                current_adapter = should_continue[0]
                                unified_request = should_continue[1]
                                continue
                            final_error = self._error_handler.create_retryable_stream_error(
                                current_adapter.provider_name, retryable_stream_finish_reason
                            )
                            result_response = self._error_handler.format_response(final_error)
                            break

                        if not stream_started:
                            await self._close_stream(stream)
                            stream = None
                            should_continue = await self._fallback_handler.handle_empty_stream(
                                current_adapter,
                                context,
                                unified_request,
                                raw_request_data,
                                req,
                            )
                            if should_continue:
                                await self._fallback_handler.switch_adapter(
                                    _exit_stack, current_adapter, should_continue[0]
                                )
                                current_adapter = should_continue[0]
                                unified_request = should_continue[1]
                                continue
                            final_error = self._error_handler.create_empty_stream_error(
                                current_adapter.provider_name
                            )
                            result_response = self._error_handler.format_response(final_error)
                            break

                    should_clean_stream = False
                    should_clean_exit_stack = False

                    # Mirror the serving attempt's conversion tier into the
                    # EventContext for logs/audit (stamped when the adapter
                    # built the outbound body just above).
                    mirror_conversion_tier(
                        unified_request,
                        event_context,
                        current_adapter.provider_name,
                        stream=True,
                    )

                    # At this point the first chunks just arrived, so the
                    # elapsed time ≈ TTFT — feed it to the latency stats store.
                    context.orchestrator.record_last_success(
                        event_context.ttft_ms if event_context is not None else None
                    )

                    return await self._create_streaming_response(
                        first_chunks=first_chunks,
                        stream=stream,
                        transformer=transformer,
                        web_search_interceptor=context.web_search_interceptor,
                        web_search_tool_config=context.web_search_tool_config,
                        proxy_web_search_active=proxy_web_search_active,
                        current_adapter=current_adapter,
                        trace_id=trace_id,
                        tracing_registry=tracing_registry,
                        unified_request=unified_request,
                        event_context=event_context,
                        protocol_name=protocol_name,
                        exit_stack=_exit_stack,
                        req=req,
                        cancel_token=stream_cancel_token,
                        native_streaming=native,
                        config_manager=context.config_manager,
                        response_store=context.response_store,
                        on_request_completed=context.on_request_completed,
                    )

                except Exception as e:
                    await self._close_stream(stream)
                    stream = None
                    result = await self._fallback_handler.handle_stream_error(
                        e,
                        current_adapter,
                        context,
                        unified_request,
                        raw_request_data,
                        req,
                        event_context,
                        _exit_stack,
                    )
                    if result is not None:
                        if isinstance(result, Response):
                            final_error = e
                            result_response = result
                            break
                        current_adapter, unified_request = result
                        continue
                    final_error = e
                    if not isinstance(e, ProviderError):
                        e = ProviderError(
                            message=f"Streaming error: {e}",
                            error_type="api_error",
                            status_code=500,
                        )
                    result_response = self._error_handler.format_response(e)
                    break

            if event_context is not None and context.on_request_completed is not None:
                await context.on_request_completed(event_context, final_error is None)

            if final_error is not None and event_context is not None:
                event_context.error_message = str(final_error)
                await tracing_registry.on_error(unified_request, final_error, event_context)

            assert result_response is not None
            return result_response

        finally:

            async def run_cleanup():
                if should_clean_stream and stream is not None:
                    await self._close_stream(stream)
                if should_clean_exit_stack and _exit_stack is not None:
                    await _exit_stack.aclose()

            await asyncio.shield(run_cleanup())

    def _get_tracing_registry(self):
        from llm_proxy.observability.tracing.handlers import get_tracing_registry

        return get_tracing_registry()

    async def _close_stream(self, stream) -> None:
        if stream is None:
            return
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(quiet_aclose(stream))

    async def _prefetch_stream_chunks(
        self,
        stream,
        transformer,
    ) -> _PrefetchResult:
        first_chunks: list[str] = []
        stream_started = False
        context_exceeded = False
        context_exceeded_reason: str | None = None
        retryable_stream_finish_reason: str | None = None

        try:
            async for chunk in stream:
                if not isinstance(chunk, (str, dict)):
                    continue

                parsed = self._chunk_parser.parse_chunk(chunk)

                if parsed is not None:
                    choices = parsed.get("choices", [])
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        finish_reason = choice.get("finish_reason")
                        if is_context_length_finish_reason(finish_reason):
                            context_exceeded = True
                            context_exceeded_reason = finish_reason
                            break

                        if is_retryable_stream_finish_reason(
                            finish_reason
                        ) and not self._chunk_parser.choice_has_non_role_output(choice):
                            retryable_stream_finish_reason = finish_reason
                            break

                if context_exceeded:
                    break
                if retryable_stream_finish_reason:
                    break

                transformed = transformer.transform(chunk)
                if transformed:
                    first_chunks.append(transformed)
                    if parsed is not None and self._chunk_parser.chunk_has_meaningful_content(
                        parsed
                    ):
                        stream_started = True
                        break
        except Exception:
            await self._close_stream(stream)
            raise

        return _PrefetchResult(
            first_chunks=first_chunks,
            stream_started=stream_started,
            context_exceeded=context_exceeded,
            context_exceeded_reason=context_exceeded_reason,
            retryable_stream_finish_reason=retryable_stream_finish_reason,
        )

    async def _process_image_streaming(
        self,
        streaming_marker: StreamingResponseMarker,
        unified_request: InternalImageRequest | InternalImageEditRequest,
        context: RequestContext,
        exit_stack: AsyncExitStack | None,
    ) -> Response:
        adapter = streaming_marker.adapter
        tracing_registry = context.tracing_registry or self._get_tracing_registry()

        async def stream_coro():
            if isinstance(unified_request, InternalImageEditRequest):
                return await adapter.stream_image_edit(unified_request)
            return await adapter.stream_image_generation(
                cast(InternalImageRequest, unified_request)
            )

        # Parse provider SSE chunks to capture actual image counts and token usage
        # during streaming. The request-side n-based fallback is set first and the
        # tracker overwrites with actual counts as completed events arrive.
        tracker = ImageStreamUsageTracker()
        ctx = context.event_context
        n_fallback = getattr(unified_request, "n", 1) or 1
        if ctx is not None:
            ctx.images_generated = n_fallback

        def _img_observer(chunk: Any) -> None:
            tracker.observe(chunk)
            if ctx is not None:
                tracker.apply_to(ctx)

        response = await self._run_generic_stream(
            adapter,
            unified_request,
            context,
            tracing_registry,
            exit_stack,
            stream_coro,
            self.streaming_handler.create_response,
            trace_chunks=True,
            chunk_observer=_img_observer,
        )

        return response

    async def _process_speech_streaming(
        self,
        streaming_marker: StreamingResponseMarker,
        unified_request: InternalSpeechRequest,
        context: RequestContext,
        exit_stack: AsyncExitStack | None,
    ) -> Response:
        adapter = streaming_marker.adapter
        tracing_registry = context.tracing_registry or self._get_tracing_registry()

        def _build_response(gen):
            media_type = adapter.speech_stream_media_type(unified_request)
            if not isinstance(media_type, str) or not media_type:
                media_type = f"audio/{unified_request.response_format}"
                if unified_request.response_format == "mp3":
                    media_type = "audio/mpeg"
            return StreamingResponse(gen, media_type=media_type)

        async def stream_coro():
            return await adapter.stream_speech(unified_request)

        return await self._run_generic_stream(
            adapter,
            unified_request,
            context,
            tracing_registry,
            exit_stack,
            stream_coro,
            _build_response,
        )

    async def _process_transcription_streaming(
        self,
        streaming_marker: StreamingResponseMarker,
        unified_request: InternalTranscriptionRequest,
        context: RequestContext,
        exit_stack: AsyncExitStack | None,
    ) -> Response:
        adapter = streaming_marker.adapter
        tracing_registry = context.tracing_registry or self._get_tracing_registry()

        async def stream_coro():
            return await adapter.stream_transcription(unified_request)

        # Parse provider SSE chunks for usage data (tokens / duration).
        tracker = TranscriptionStreamUsageTracker()
        ctx = context.event_context

        def _stt_observer(chunk: Any) -> None:
            tracker.observe(chunk)
            if ctx is not None:
                tracker.apply_to(ctx, adapter)

        return await self._run_generic_stream(
            adapter,
            unified_request,
            context,
            tracing_registry,
            exit_stack,
            stream_coro,
            self.streaming_handler.create_response,
            chunk_observer=_stt_observer,
        )

    async def _run_generic_stream(
        self,
        adapter: BaseAdapter,
        unified_request: Any,
        context: RequestContext,
        tracing_registry: Any,
        exit_stack: AsyncExitStack | None,
        stream_coro,
        build_response,
        trace_chunks: bool = False,
        chunk_observer: Any | None = None,
    ) -> Response:
        _exit_stack = exit_stack or AsyncExitStack()
        event_context = context.event_context
        try:
            stream = await stream_coro()
            streaming_stack = _exit_stack.pop_all()
            _exit_stack = None

            lifecycle = GenericStreamLifecycle(
                stream=stream,
                stream_request=unified_request,
                event_context=event_context,
                tracing_registry=tracing_registry,
                streaming_stack=streaming_stack,
                config_manager=context.config_manager,
                on_request_completed=context.on_request_completed,
                trace_chunks=trace_chunks,
                chunk_observer=chunk_observer,
            )
            return build_response(lifecycle.events())
        except Exception as e:
            if event_context:
                event_context.error_message = str(e)
            await tracing_registry.on_error(unified_request, e, event_context)
            error = (
                e
                if isinstance(e, ProviderError)
                else ProviderError(
                    message=f"Streaming error: {e}",
                    error_type="api_error",
                    provider_name=adapter.provider_name,
                    status_code=500,
                )
            )
            return self._error_handler.format_response(error)
        finally:
            if _exit_stack is not None:
                await asyncio.shield(_exit_stack.aclose())

    async def _create_streaming_response(
        self,
        first_chunks: list[str],
        stream,
        transformer,
        web_search_interceptor,
        web_search_tool_config,
        current_adapter: BaseAdapter,
        trace_id: str,
        tracing_registry,
        unified_request: InternalRequest,
        event_context: EventContext | None,
        exit_stack: AsyncExitStack,
        protocol_name: str | None = None,
        req: Request | None = None,
        cancel_token: asyncio.Event | None = None,
        native_streaming: bool = False,
        config_manager: DatabaseConfigManager | None = None,
        proxy_web_search_active: bool = False,
        response_store: Any | None = None,
        on_request_completed: Any | None = None,
    ) -> Response:
        _should_intercept_web_search = proxy_web_search_active

        # Streaming-side of the CDN keepalive: while the upstream stream
        # falls silent (model thinking, slow provider), emit SSE comment
        # frames so the CDN's time budget does not expire mid-stream.
        # Comments are ignored by SSE parsers by definition, so this is safe
        # for every protocol and not gated on the non-streaming keepalive
        # toggle. Interval follows the same operator-facing knob.
        heartbeat_interval = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        if config_manager is not None:
            try:
                from llm_proxy.config.manager import resolve_keepalive_params

                heartbeat_interval = max(
                    resolve_keepalive_params(config_manager).interval_seconds, 0.5
                )
            except Exception:  # noqa: BLE001 - heartbeat tuning must never break streaming
                logger.debug(
                    "Failed to resolve keepalive params for stream heartbeat", exc_info=True
                )
        heartbeat_comment = (
            SSE_KEEPALIVE_COMMENT
            if self.streaming_handler.config.media_type == "text/event-stream"
            else None
        )

        lifecycle = StreamLifecycle(
            first_chunks=first_chunks,
            stream=stream,
            transformer=transformer,
            stream_request=unified_request,
            event_context=event_context,
            tracing_registry=tracing_registry,
            exit_stack=exit_stack,
            req=req,
            cancel_token=cancel_token,
            native_streaming=native_streaming,
            protocol_name=protocol_name,
            config_manager=config_manager,
            response_store=response_store,
            on_request_completed=on_request_completed,
            heartbeat_interval=heartbeat_interval,
            heartbeat_comment=heartbeat_comment,
            web_search_interceptor=web_search_interceptor,
            web_search_tool_config=web_search_tool_config,
            should_intercept_web_search=_should_intercept_web_search,
            proxy_web_search_active=proxy_web_search_active,
            current_adapter=current_adapter,
            web_search_processor=self._web_search_processor,
            native_passthrough_handler=self._native_passthrough_handler,
        )

        response = self.streaming_handler.create_response(lifecycle.events())

        # Forward upstream response headers captured by the adapter when the
        # stream started (x-request-id, openai-version, rate-limit headers).
        # setdefault so the proxy's own headers (Cache-Control, ...) win.
        upstream_headers = getattr(current_adapter, "_last_stream_response_headers", None)
        if upstream_headers:
            for key, value in upstream_headers.items():
                response.headers.setdefault(key, value)

        trace_id = tracing_registry.get_trace_id()
        trace_header_name = tracing_registry.get_trace_header_name()
        if trace_id is not None:
            response.headers[trace_header_name] = trace_id

        if event_context is not None:
            event_context.response_headers = dict(response.headers)

        return response
