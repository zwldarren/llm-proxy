"""Streaming request processor with provider fallback support."""

import asyncio
import inspect
import uuid
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.protocols.base import ProtocolEndpoint

import orjson
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from orjson import JSONDecodeError

from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.conversion import NativePassthroughHandler, plan_conversion
from llm_proxy.core.errors import (
    is_context_length_finish_reason,
    is_retryable_stream_finish_reason,
)
from llm_proxy.core.errors.handler import ErrorHandler
from llm_proxy.core.exceptions import (
    ClientDisconnectedError,
    ConfigurationError,
    ProviderError,
)
from llm_proxy.core.processing.base import RequestContext, mirror_conversion_tier
from llm_proxy.core.processing.stages.fallback_handler import FallbackHandler
from llm_proxy.core.processing.stages.parameter_override import ParameterOverrideService
from llm_proxy.core.processing.strategies import StreamingResponseMarker
from llm_proxy.core.processing.strategies.chunk_parser import OpenAIStreamChunkParser
from llm_proxy.core.processing.web_search_streaming import (
    ContinuationState,
    WebSearchStreamProcessor,
    merge_continuation_usage,
)
from llm_proxy.core.reasoning_cache import try_cache_reasoning_from_blocks
from llm_proxy.core.utils import quiet_aclose
from llm_proxy.models import (
    ConversionTier,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalRequest,
    InternalSpeechRequest,
    InternalTranscriptionRequest,
)
from llm_proxy.observability.cost import finalize_event_cost
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.logger import get_logger
from llm_proxy.streaming.handler import StreamingHandler

logger = get_logger(__name__)

# SSE comment frame emitted (and ignored by every SSE parser per the WHATWG
# spec) when the upstream stream falls silent, so fronting CDNs such as
# Cloudflare keep the connection open while a model ponders its next token.
_SSE_KEEPALIVE_COMMENT = ": keep-alive\n\n"

# Default heartbeat interval, overridden by the UI-managed keepalive config.
_DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15.0


@dataclass
class _PrefetchResult:
    """Result of prefetching the first chunks from a stream."""

    first_chunks: list[str]
    stream_started: bool
    context_exceeded: bool
    context_exceeded_reason: str | None = None
    retryable_stream_finish_reason: str | None = None


# ------------------------------------------------------------------
# Stream usage trackers — parse provider SSE chunks for billing data
# during image generation and transcription streaming.
# ------------------------------------------------------------------


class _ImageStreamUsageTracker:
    """Parse provider SSE chunks for image-generation billing data.

    OpenAI: ``data: {"type":"image_generation.completed","usage":{...}}``
    (and ``image_edit.completed`` for edit streaming)
    Gemini (generateContent): ``data: {...,"usageMetadata":{...}}``
    (inlineData images counted)
    Gemini (Interactions): ``data: {"type":"step.delta","delta":{"type":"image",…}}``
    partial images + ``data: {"type":"interaction.completed",
    "interaction":{"usage":{…}}}`` with the new usage vocabulary.
    """

    def __init__(self) -> None:
        self._images_completed = 0
        self._usage: dict[str, Any] | None = None
        self._gemini_usage: dict[str, Any] | None = None
        # Interactions-API usage (new vocabulary: total_input_tokens, …)
        self._interactions_usage: dict[str, Any] | None = None

    @property
    def images_completed(self) -> int:
        return self._images_completed

    @property
    def captured_usage(self) -> dict[str, Any] | None:
        return self._usage

    @property
    def gemini_usage(self) -> dict[str, Any] | None:
        return self._gemini_usage

    @property
    def interactions_usage(self) -> dict[str, Any] | None:
        return self._interactions_usage

    def observe(self, chunk: Any) -> None:
        if not isinstance(chunk, str):
            return
        for line in chunk.split("\n"):
            line = line.strip()
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if not payload or payload == "[DONE]":
                continue
            try:
                data: dict[str, Any] = orjson.loads(payload)
            except JSONDecodeError:
                continue
            self._observe_json(data)

    def _observe_json(self, data: dict[str, Any]) -> None:
        # OpenAI Images API streaming: type field in payload
        event_type = data.get("type")
        if event_type in {"image_generation.completed", "image_edit.completed"}:
            self._images_completed += 1
            usage = data.get("usage")
            if isinstance(usage, dict):
                self._usage = usage
            return

        # Gemini Interactions: step.delta image content + interaction.completed
        # usage. Accept both the "type" and "event_type" discriminator keys
        # (the API reference uses event_type; the migration guide uses type).
        gemini_event = data.get("type") or data.get("event_type")
        if gemini_event == "step.delta":
            delta = data.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "image" and delta.get("data"):
                self._images_completed += 1
            return
        if gemini_event == "interaction.completed":
            interaction = data.get("interaction")
            usage = interaction.get("usage") if isinstance(interaction, dict) else None
            if isinstance(usage, dict):
                self._interactions_usage = usage
            return

        # Gemini (generateContent): usageMetadata + candidates with inlineData images
        gemini_usage = data.get("usageMetadata")
        if isinstance(gemini_usage, dict):
            self._gemini_usage = gemini_usage
        candidates = data.get("candidates")
        if isinstance(candidates, list):
            for c in candidates:
                if isinstance(c, dict):
                    parts = c.get("content", {}).get("parts", [])
                    if isinstance(parts, list):
                        for p in parts:
                            if isinstance(p, dict):
                                inline = p.get("inlineData")
                                if (
                                    isinstance(inline, dict)
                                    and isinstance(inline.get("mimeType", ""), str)
                                    and inline["mimeType"].startswith("image/")
                                ):
                                    self._images_completed += 1

    def apply_to(self, ctx: EventContext) -> None:
        """Write captured billing data into an EventContext."""
        if self._images_completed > 0:
            # Overwrite request-side n-based fallback with actual count.
            ctx.images_generated = self._images_completed

        usage = self._usage
        if isinstance(usage, dict):
            # OpenAI gpt-image usage: input_tokens, output_tokens,
            # input_tokens_details.image_tokens
            if usage.get("input_tokens") is not None:
                ctx.prompt_tokens = usage["input_tokens"]
            if usage.get("output_tokens") is not None:
                ctx.completion_tokens = usage["output_tokens"]
            if usage.get("total_tokens") is not None:
                ctx.total_tokens = usage["total_tokens"]
            itd = usage.get("input_tokens_details")
            if isinstance(itd, dict) and itd.get("image_tokens") is not None:
                ctx.image_input_tokens = itd["image_tokens"]
        elif self._interactions_usage is not None:
            # Gemini Interactions usage: total_input_tokens, total_output_tokens,
            # total_thought_tokens, total_tool_use_tokens, total_tokens.
            iu = self._interactions_usage
            from llm_proxy.serialization.gemini_interactions.usage import (
                interactions_billable_token_counts,
                interactions_web_search_requests,
            )

            has_search = interactions_web_search_requests(iu) > 0
            input_tokens, output_tokens = interactions_billable_token_counts(
                iu, has_search_grounding=has_search
            )
            ctx.prompt_tokens = input_tokens
            ctx.completion_tokens = output_tokens
            if "total_tokens" in iu:
                ctx.total_tokens = iu["total_tokens"]
            else:
                ctx.total_tokens = (input_tokens + output_tokens) or None
        elif self._gemini_usage is not None:
            # Gemini usageMetadata: promptTokenCount, candidatesTokenCount, etc.
            gu = self._gemini_usage
            ctx.prompt_tokens = gu.get("promptTokenCount", 0) or 0
            ctx.completion_tokens = gu.get("candidatesTokenCount", 0) or 0
            if "totalTokenCount" in gu:
                ctx.total_tokens = gu["totalTokenCount"]
            else:
                ctx.total_tokens = (ctx.prompt_tokens or 0) + (ctx.completion_tokens or 0) or None


class _TranscriptionStreamUsageTracker:
    """Parse provider SSE chunks for transcription billing data.

    OpenAI streaming transcription: ``data: {...,"usage":{...}}`` in
    ``transcript.text.done`` events (when ``include[]=usage`` is used).
    The usage dict may be token-based (gpt-4o-transcribe) or duration-based
    (whisper: ``{"type":"duration","seconds":N}``).
    """

    def __init__(self) -> None:
        self._usage: dict[str, Any] | None = None

    @property
    def captured_usage(self) -> dict[str, Any] | None:
        return self._usage

    def observe(self, chunk: Any) -> None:
        if not isinstance(chunk, str):
            return
        for line in chunk.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if not payload or payload == "[DONE]":
                continue
            try:
                data: dict[str, Any] = orjson.loads(payload)
            except JSONDecodeError:
                continue
            usage = data.get("usage")
            if isinstance(usage, dict):
                self._usage = usage

    def apply_to(self, ctx: EventContext, adapter: Any) -> None:
        """Write captured billing data into an EventContext using the adapter's
        ``_parse_usage`` for proper Usage object construction."""
        if self._usage is None:
            return
        parsed = adapter._parse_usage(self._usage)
        if parsed is not None:
            ctx.update_usage(parsed)


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

    @staticmethod
    async def _check_disconnect(
        req: Request | None,
        chunk_count: int,
        cancel_token: asyncio.Event | None,
        interval: int = 10,
    ) -> bool:
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

    @staticmethod
    async def _iterate_chunks_with_comments(
        stream,
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
            interval = _DEFAULT_HEARTBEAT_INTERVAL_SECONDS
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

    @staticmethod
    async def _safe_cleanup(coro, description: str) -> None:
        """Safely run a cleanup coroutine, logging exceptions but never raising."""
        try:
            await coro
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"{description}: {e}")

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
        tracker = _ImageStreamUsageTracker()
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
        tracker = _TranscriptionStreamUsageTracker()
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

            async def stream_generator():
                # Capture the stream error so it can be forwarded to on_stream_end.
                stream_error: Exception | None = None
                await tracing_registry.on_stream_start(unified_request, event_context)
                try:
                    async for chunk in stream:
                        if trace_chunks:
                            await tracing_registry.on_stream_chunk(
                                unified_request, chunk, event_context
                            )
                        if chunk_observer is not None:
                            with suppress(Exception):
                                chunk_observer(chunk)
                        yield chunk
                except Exception as e:
                    stream_error = e
                    if event_context:
                        event_context.error_message = str(e)
                    await tracing_registry.on_error(unified_request, e, event_context)
                    raise
                finally:
                    with suppress(Exception, asyncio.CancelledError):
                        if stream is not None:
                            await asyncio.shield(quiet_aclose(stream))
                        if event_context is not None:
                            await finalize_event_cost(event_context, context.config_manager)
                        # Post-request hook (model-experience observation) for
                        # generic streams; see _create_streaming_response.
                        if event_context is not None and context.on_request_completed is not None:
                            await context.on_request_completed(event_context, stream_error is None)
                        await tracing_registry.on_stream_end(
                            unified_request, event_context, error=stream_error
                        )
                        await asyncio.shield(streaming_stack.aclose())

            return build_response(stream_generator())
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
        heartbeat_interval = _DEFAULT_HEARTBEAT_INTERVAL_SECONDS
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
            _SSE_KEEPALIVE_COMMENT
            if self.streaming_handler.config.media_type == "text/event-stream"
            else None
        )

        async def stream_generator(
            _first_chunks=first_chunks,
            _stream=stream,
            _transformer=transformer,
            _web_search_interceptor=web_search_interceptor,
            _web_search_tool_config=web_search_tool_config,
            _should_intercept_web_search=_should_intercept_web_search,
            _proxy_web_search_active=proxy_web_search_active,
            _tracing_registry=tracing_registry,
            _stream_request=unified_request,
            _event_context=event_context,
            _exit_stack=exit_stack,
            _req=req,
            _cancel_token=cancel_token,
            _native_streaming=native_streaming,
            _protocol_name=protocol_name,
            _config_manager=config_manager,
            _response_store=response_store,
            _on_request_completed=on_request_completed,
            _heartbeat_interval=heartbeat_interval,
            _heartbeat_comment=heartbeat_comment,
        ):
            stream_error: Exception | None = None
            first_chunk_time: datetime | None = None
            chunk_count = 0
            client_disconnected = False
            state = ContinuationState(transformer=_transformer, stream_request=_stream_request)
            try:
                if _event_context is not None:
                    _event_context.is_streaming = True
                    _event_context.transformer = _transformer
                await _tracing_registry.on_stream_start(_stream_request, _event_context)

                for chunk in _first_chunks:
                    if first_chunk_time is None:
                        first_chunk_time = datetime.now(UTC)
                    if _event_context is not None:
                        _event_context.first_chunk_time = first_chunk_time
                    await _tracing_registry.on_stream_chunk(_stream_request, chunk, _event_context)
                    yield chunk

                async for kind, payload in self._iterate_chunks_with_comments(
                    _stream,
                    interval=_heartbeat_interval,
                    comment=_heartbeat_comment,
                ):
                    if kind == "comment":
                        if not client_disconnected and not (
                            _cancel_token and _cancel_token.is_set()
                        ):
                            yield payload
                        continue
                    chunk = payload
                    if not isinstance(chunk, (str, dict)):
                        continue
                    if _cancel_token and _cancel_token.is_set():
                        logger.debug("Stream cancelled by cancel_token, stopping chunk iteration")
                        break

                    if _native_streaming:
                        if isinstance(chunk, str) and "event: message_start" in chunk:
                            handler = self._native_passthrough_handler
                            # Mask the upstream's internal model name with the
                            # client-requested alias (see InternalRequest.echo_model).
                            chunk = handler.inject_model_into_anthropic_message_start(
                                chunk, _stream_request.echo_model
                            )
                        if _protocol_name == "anthropic":
                            self._native_passthrough_handler.maybe_capture_native_streaming_usage(
                                chunk, _event_context
                            )
                        elif _protocol_name == "openresponses":
                            # The snapshot's model is rewritten to the
                            # client-requested alias (see InternalRequest.echo_model)
                            # so the native stream echoes the same name as the
                            # transformer path.
                            rewritten = (
                                self._native_passthrough_handler.maybe_capture_native_openresponses(
                                    chunk,
                                    _transformer,
                                    _event_context,
                                    model=_stream_request.echo_model,
                                )
                            )
                            if rewritten is not None:
                                chunk = rewritten
                        await _tracing_registry.on_stream_chunk(
                            _stream_request, chunk, _event_context
                        )
                        yield chunk
                        chunk_count += 1
                        if await self._check_disconnect(_req, chunk_count, _cancel_token):
                            client_disconnected = True
                            if stream_error is None:
                                stream_error = ClientDisconnectedError()
                            break
                        continue

                    transformed = _transformer.transform(chunk)
                    if transformed:
                        await _tracing_registry.on_stream_chunk(
                            _stream_request, transformed, _event_context
                        )
                        yield transformed
                        chunk_count += 1
                        if await self._check_disconnect(_req, chunk_count, _cancel_token):
                            client_disconnected = True
                            if stream_error is None:
                                stream_error = ClientDisconnectedError()
                            break

                if _should_intercept_web_search and not client_disconnected:
                    async for _kind, payload in self._iterate_chunks_with_comments(
                        self._web_search_processor.generate_continuation(
                            state,
                            web_search_interceptor=_web_search_interceptor,
                            web_search_tool_config=_web_search_tool_config,
                            proxy_web_search_active=_proxy_web_search_active,
                            current_adapter=current_adapter,
                            tracing_registry=_tracing_registry,
                            event_context=_event_context,
                            cancel_token=_cancel_token,
                        ),
                        interval=_heartbeat_interval,
                        comment=_heartbeat_comment,
                    ):
                        yield payload

                if (
                    _native_streaming
                    and _protocol_name == "openresponses"
                    and not (_cancel_token and _cancel_token.is_set())
                    and not client_disconnected
                ):
                    # Spec: the terminal event MUST be the literal string
                    # [DONE]. The native Responses upstream ends its stream
                    # after response.completed without one; append it so the
                    # passthrough path terminates exactly like the
                    # transformer path.
                    yield "data: [DONE]\n\n"

                if not _native_streaming and not client_disconnected:
                    if state.depth > 0 and _transformer is not state.transformer:
                        merge_continuation_usage(_transformer, state.transformer)

                    final_chunk = state.transformer.finalize()
                    if final_chunk:
                        await _tracing_registry.on_stream_chunk(
                            state.stream_request, final_chunk, _event_context
                        )
                        yield final_chunk

                    # The accumulated output is complete after finalize(). Cache
                    # real reasoning paired with its tool calls so subsequent
                    # turns can restore it (DeepSeek-style echo) even when the
                    # client strips reasoning or another provider served an
                    # intermediate turn. Web-search continuations keep the final
                    # output split across two transformers, so cache both.
                    # Never fatal.
                    _cache_targets = [state.transformer]
                    if _transformer is not state.transformer:
                        _cache_targets.append(_transformer)
                    for _t in _cache_targets:
                        try_cache_reasoning_from_blocks(
                            _t.get_accumulated_output(),
                            response_id=_t.response_id,
                        )

                # Persist store=true streamed responses so follow-up
                # previous_response_id continuations and GET /v1/responses/{id}
                # work, matching the non-streaming path. The persistence rules
                # are protocol knowledge owned by the transformer; protocols
                # without response storage no-op. Skipped when the client
                # disconnected mid-stream (the snapshot would be partial) and
                # best-effort otherwise.
                if (
                    _response_store is not None
                    and not client_disconnected
                    and not (_cancel_token and _cancel_token.is_set())
                ):
                    await self._safe_cleanup(
                        _transformer.finalize_persistence(
                            _stream_request,
                            _response_store,
                            _event_context,
                        ),
                        "Failed to persist streamed response",
                    )
            except asyncio.CancelledError:
                # The response task was cancelled — for a client disconnect
                # this is the origin-side 524 moment. Mark the log entry as a
                # client abandonment (499) instead of a successful request.
                client_disconnected = True
                if stream_error is None:
                    stream_error = ClientDisconnectedError()
                raise
            except GeneratorExit:
                # The response was closed without reaching the end (client
                # disconnect mid-response, server teardown). Recorded so the
                # abandonment is visible in the logs.
                client_disconnected = True
                if stream_error is None:
                    stream_error = ClientDisconnectedError()
                raise
            except Exception as e:
                stream_error = e
                if _event_context is not None:
                    _event_context.error_message = str(e)
                await _tracing_registry.on_error(state.stream_request, e, _event_context)
                # Error wire shaping is protocol knowledge owned by each
                # protocol-side transformer (OpenResponses: response.failed +
                # [DONE]; Anthropic: named ``event: error``; default: generic
                # chat-completions error frame + [DONE]).
                for frame in _transformer.error_frames(e):
                    yield frame
                if _should_intercept_web_search:
                    await self._web_search_processor.process_streaming_web_search(
                        _transformer, _web_search_interceptor, _web_search_tool_config
                    )
            finally:
                if _stream is not None:
                    await self._safe_cleanup(
                        asyncio.shield(quiet_aclose(_stream)),
                        f"Failed to close provider stream for {state.stream_request.model}",
                    )
                await self._safe_cleanup(
                    asyncio.shield(_exit_stack.aclose()),
                    "Failed to close adapter exit stack",
                )
                if _event_context is not None and _config_manager is not None:
                    await self._safe_cleanup(
                        finalize_event_cost(_event_context, _config_manager),
                        "Failed to finalize event cost",
                    )
                # Fire the post-request hook (model-experience observation) once
                # the stream has actually finished; the early return in process()
                # never reaches the hook call that covers setup failures. A
                # client disconnect is not a model failure (EWMA measures
                # provider health), so it still counts as success here.
                if _event_context is not None and _on_request_completed is not None:
                    experience_success = stream_error is None or isinstance(
                        stream_error, ClientDisconnectedError
                    )
                    await self._safe_cleanup(
                        _on_request_completed(_event_context, experience_success),
                        "Failed to call on_request_completed",
                    )
                await self._safe_cleanup(
                    _tracing_registry.on_stream_end(
                        state.stream_request,
                        _event_context,
                        error=stream_error,
                    ),
                    "Failed to call on_stream_end for tracing registry",
                )

        response = self.streaming_handler.create_response(stream_generator())

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
