"""Request execution stage with retry/fallback and streaming support."""

from typing import TYPE_CHECKING, Any

import orjson
from fastapi import Response

from llm_proxy.core.errors import get_error_handler
from llm_proxy.core.errors.protocols import protocol_for_name
from llm_proxy.core.exceptions import (
    ConfigurationError,
    LLMProxyError,
    ProviderError,
)
from llm_proxy.core.processing.base import RequestContext, mirror_conversion_tier
from llm_proxy.core.processing.fallback import (
    FallbackAction,
    execute_fallback,
    plan_fallback,
)
from llm_proxy.core.processing.stages.base import PipelineStage, PipelineState
from llm_proxy.core.processing.stages.parameter_override import ParameterOverrideService
from llm_proxy.core.processing.stages.role_normalization import normalize_developer_roles
from llm_proxy.core.processing.strategies import StreamingResponseMarker
from llm_proxy.core.processing.streaming_processor import StreamingProcessor
from llm_proxy.core.processing.web_search_streaming import (
    MAX_CONTINUATION_DEPTH,
    WebSearchStreamProcessor,
    sum_usage,
)
from llm_proxy.core.request_type import RequestType
from llm_proxy.models import InternalResponse
from llm_proxy.observability.cost import calculate_event_cost
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.tracing.handlers import get_tracing_registry

if TYPE_CHECKING:
    from llm_proxy.protocols.base import ProtocolEndpoint
    from llm_proxy.protocols.serializer_base import ProtocolSerializer

logger = get_logger(__name__)


def _format_attempt_summary(attempt: dict, last_error: ProviderError) -> str:
    """Format a single fallback attempt into a human-readable summary string."""
    provider = attempt.get("provider", "unknown")
    provider_type = attempt.get("provider_type")
    label = f"{provider} [{provider_type}]" if provider_type else provider
    error_type = attempt.get("error_type", type(last_error).__name__)
    return f"{label} ({error_type}: {attempt.get('error_message', '')})"


class RetryExecutor:
    """Execute a strategy with retry/fallback loop.

    Encapsulates the provider-fallback and role-transform logic so that
    RequestExecutionStage only has to deal with the happy-path response
    formatting.
    """

    def __init__(self, param_override_service: ParameterOverrideService) -> None:
        self._param_override_service = param_override_service

    async def execute(
        self,
        state: PipelineState,
        context: RequestContext,
    ) -> tuple[Any, Exception | None]:
        """Execute strategy with retry/fallback loop.

        Returns (result, error). Exactly one will be None.
        """
        current_adapter = state.adapter
        current_request = state.unified_request

        while True:
            try:
                result = await state.strategy.execute(current_request, current_adapter, context)
                context.orchestrator.record_last_success(state.event_context.latency_ms)
                if (
                    context.web_search_interceptor is not None
                    and context.proxy_web_search_active
                    and hasattr(result, "usage")
                    and isinstance(result, InternalResponse)
                ):
                    search_state: dict[str, int] = {"count": 0}
                    (
                        result,
                        search_results,
                    ) = await context.web_search_interceptor.inject_results_into_response(
                        response=result,
                        tool_config=context.web_search_tool_config,
                        request_id=state.trace_id,
                        search_state=search_state,
                    )
                    if search_state["count"] > 0:
                        state.event_context.web_search_requests = search_state["count"]
                    # Non-streaming counterpart of the streaming continuation
                    # loop: re-call the provider with the injected search
                    # results so the model produces a final answer instead of
                    # the raw server_tool_use/web_search_result blocks being
                    # degraded to text in the client response.
                    if search_results:
                        result = await self._continue_web_search(
                            response=result,
                            search_results=search_results,
                            original_request=current_request,
                            adapter=current_adapter,
                            context=context,
                            state=state,
                        )

                return result, None

            except ConfigurationError:
                raise
            except Exception as e:
                decision = plan_fallback(e, context.orchestrator, current_adapter.provider_name)

                if decision.action is FallbackAction.ABORT:
                    return None, decision.error

                if decision.action is FallbackAction.RETRY_ROLE_TRANSFORM:
                    logger.warning(
                        f"Role error from {current_adapter.provider_name}, "
                        "applying developer -> system role transformation and retrying"
                    )
                    normalize_developer_roles(current_request)
                    context.orchestrator.mark_role_transformed()
                    continue

                logger.debug(
                    f"Provider {current_adapter.provider_name} failed, trying next provider"
                )
                swapped = await execute_fallback(
                    decision,
                    context.orchestrator,
                    event_context=state.event_context,
                    provider_type=current_adapter.provider_name,
                    exit_stack=state.exit_stack,
                    current_adapter=current_adapter,
                    req=state.req,
                    unified_request=current_request,
                    raw_request_data=state.fallback_raw_data,
                    context=context,
                    param_override_service=self._param_override_service,
                    process_request=context.process_request,
                )
                if swapped is not None:
                    current_adapter, current_request = swapped
                    # Keep the pipeline state pointing at the live attempt so
                    # downstream consumers (response echo, tracing, response
                    # store) see the provider that actually served the request.
                    state.adapter = current_adapter
                    state.unified_request = current_request
                    # raw_data must track the live attempt too. When the
                    # fallback request carries no raw protocol data (e.g. a
                    # request rebuilt by the process_request middleware), fall
                    # back to the pristine client body rather than leaving the
                    # failed provider's override-modified body in place.
                    current_raw = getattr(current_request, "_raw_protocol_data", None)
                    state.raw_data = (
                        current_raw if current_raw is not None else state.fallback_raw_data
                    )
                    continue

                return None, decision.error

    async def _continue_web_search(
        self,
        response: InternalResponse,
        search_results: list[Any],
        original_request: Any,
        adapter: Any,
        context: RequestContext,
        state: PipelineState,
    ) -> InternalResponse:
        """Re-call the provider with injected search results until the model
        produces a final answer.

        Non-streaming counterpart of
        ``WebSearchStreamProcessor.generate_continuation``: while the model
        ended its turn waiting for search results, build a continuation
        request carrying the assistant's web_search tool call plus the tool
        results, re-call the provider (non-streaming), and repeat if the
        follow-up turn issues another web_search call.

        The final response replaces the intermediate one, so the client sees
        the model's answer instead of raw server_tool_use/web_search_result
        blocks (which the OpenAI protocol would degrade to text). Usage from
        every upstream call is summed into the final response (each call is
        billed separately).
        """
        depth = 0
        total_usage = response.usage
        total_search_count = state.event_context.web_search_requests or 0
        # The caller only invokes this when the interceptor is active.
        interceptor = context.web_search_interceptor
        assert interceptor is not None

        while search_results and depth < MAX_CONTINUATION_DEPTH:
            if not WebSearchStreamProcessor.needs_continuation(response.output):
                break

            continuation_req = WebSearchStreamProcessor.build_continuation_request(
                original_request=original_request,
                accumulated_output=response.output,
                search_results=search_results,
                web_search_interceptor=interceptor,
                stream=False,
            )
            response = await adapter.chat_completion(continuation_req)

            # The follow-up turn may itself call web_search again; inject
            # those results and continue if the model is still waiting.
            search_state: dict[str, int] = {"count": 0}
            (
                response,
                search_results,
            ) = await interceptor.inject_results_into_response(
                response=response,
                tool_config=context.web_search_tool_config,
                request_id=state.trace_id,
                search_state=search_state,
            )
            if search_state["count"] > 0:
                total_search_count += search_state["count"]
                state.event_context.web_search_requests = total_search_count

            # Sum usage across the independent upstream calls.
            if response.usage is not None:
                if total_usage is not None:
                    sum_usage(response.usage, total_usage)
                total_usage = response.usage
            elif total_usage is not None:
                response.usage = total_usage
            depth += 1

        # Report the accumulated web-search count on the final response so
        # billing consumers reading provider_info see the total.
        if total_search_count > 0:
            server_tool_use = response.provider_info.get("server_tool_use", {})
            server_tool_use["web_search_requests"] = total_search_count
            response.provider_info["server_tool_use"] = server_tool_use

        return response


class RequestExecutionStage(PipelineStage):
    """Execute the strategy with retry/fallback, handle streaming, format response."""

    def __init__(
        self,
        protocol_name: str,
        protocol_endpoint: ProtocolEndpoint,
        serializer: ProtocolSerializer,
        streaming_processor: StreamingProcessor,
        param_override_service: ParameterOverrideService,
    ):
        self.protocol_name = protocol_name
        self._protocol_endpoint = protocol_endpoint
        self._serializer = serializer
        self._streaming_processor = streaming_processor
        self._error_handler = get_error_handler()
        self._retry_executor = RetryExecutor(param_override_service)

    async def process(self, state: PipelineState, context: RequestContext) -> None:
        tracing_registry = context.tracing_registry or get_tracing_registry()

        try:
            await tracing_registry.on_request_start(state.unified_request, state.event_context)

            self._populate_unit_billing_dimensions(state)

            # Protocol-specific adjustments now that the upstream is known
            # (e.g. stripping Claude Code's billing header for non-Anthropic providers).
            if self._protocol_endpoint.on_provider_selected is not None:
                self._protocol_endpoint.on_provider_selected(
                    state.unified_request, state.adapter.provider_name, state.req
                )

            result, final_error = await self._retry_executor.execute(state, context)

            # Mirror the live attempt's conversion tier (stamped when the
            # outbound body was built; state.unified_request tracks the
            # serving attempt) into logs/audit. The streaming branch stamps
            # later, inside StreamingProcessor, once its body is built.
            mirror_conversion_tier(
                state.unified_request, state.event_context, state.adapter.provider_name
            )

            if final_error is not None:
                if isinstance(final_error, ProviderError):
                    enhanced = self._build_exhausted_error(final_error, state.event_context)
                    state.event_context.error_message = str(enhanced)
                    await tracing_registry.on_error(
                        state.unified_request, enhanced, state.event_context
                    )
                    state.response = self._create_error_response(enhanced)
                else:
                    wrapped = ProviderError(
                        message=(f"Unexpected error: {type(final_error).__name__}: {final_error}"),
                        error_type="api_error",
                        status_code=500,
                    )
                    state.event_context.error_message = str(wrapped)
                    await tracing_registry.on_error(
                        state.unified_request, wrapped, state.event_context
                    )
                    state.response = self._create_error_response(wrapped)

                if context.on_request_completed is not None:
                    await context.on_request_completed(state.event_context, False)

                return

            if result is None:
                raise LLMProxyError(
                    message="Request processing produced no result",
                    code="internal_error",
                )

            if isinstance(result, StreamingResponseMarker):
                streaming_stack = state.exit_stack.pop_all()
                state.response = await self._streaming_processor.process(
                    streaming_marker=result,
                    raw_request_data=state.fallback_raw_data,
                    req=state.req,
                    context=context,
                    trace_id=state.trace_id,
                    event_context=state.event_context,
                    exit_stack=streaming_stack,
                )
                return

            if context.process_response is not None:
                result = await context.process_response(
                    result, state.adapter, context.config_manager
                )

            # Echo the client-requested alias so stream and non-stream
            # responses agree. Passthrough responses are excluded: they carry
            # the raw upstream body, rewritten in _build_passthrough_response.
            user_facing_model = state.unified_request.user_facing_model
            if (
                user_facing_model
                and hasattr(result, "model")
                and not result.provider_info.get("_raw_response_body")
            ):
                result.model = user_facing_model

            if hasattr(result, "usage") and result.usage:
                state.event_context.update_usage(result.usage)
            if hasattr(result, "provider_info") and result.provider_info:
                state.event_context.update_provider_info(result.provider_info)
            self._populate_response_billing_dimensions(state, result)
            state.event_context.response_status_code = 200

            response = await state.strategy.format_response(
                result, self._serializer, self.protocol_name
            )

            # Forward upstream rate-limit headers captured by the adapter.
            rate_limit_headers = getattr(result, "provider_info", {}).get("_rate_limit_headers")
            if rate_limit_headers:
                for key, value in rate_limit_headers.items():
                    response.headers[key] = value

            if self._protocol_endpoint.on_format_done is not None:
                self._protocol_endpoint.on_format_done()

            if context.response_store is not None and hasattr(response, "body") and response.body:
                try:
                    response_data = orjson.loads(response.body)
                    # Background responses are persisted by the background task
                    # itself under the pollable id it returns up front; storing
                    # here too would create an orphaned entry under a different,
                    # unreachable id.
                    if response_data.get("store") and not response_data.get("background"):
                        response_id = response_data.get("id")
                        api_key_name = state.event_context.api_key_name
                        if response_id and api_key_name:
                            # Protocol-specific body enrichment before
                            # persistence (e.g. OpenResponses attaches the
                            # materialized conversation as the stored input).
                            if self._protocol_endpoint.on_response_store is not None:
                                response_data = self._protocol_endpoint.on_response_store(
                                    response_data, state.unified_request, state.raw_data
                                )
                            await context.response_store.store(
                                api_key_name, response_id, response_data
                            )
                        elif response_id and not api_key_name:
                            logger.debug("Skipping response store: api_key_name is not available")
                except Exception:
                    logger.debug("Failed to store response", exc_info=True)

            state.event_context.response_headers = dict(response.headers)
            if hasattr(response, "body") and response.body:
                try:
                    state.event_context.response_body = orjson.loads(response.body)
                except Exception:
                    logger.debug("Failed to parse response body as JSON", exc_info=True)
                    state.event_context.response_body = {
                        "raw": True,
                        "size": len(response.body),
                    }

            try:
                await calculate_event_cost(state.event_context, context.config_manager)
            except Exception:
                logger.debug("Failed to calculate event cost", exc_info=True)

            if context.on_request_completed is not None:
                await context.on_request_completed(state.event_context, True)

            await tracing_registry.on_request_end(
                state.unified_request, result, state.event_context
            )

            trace_id = tracing_registry.get_trace_id()
            if trace_id:
                response.headers[tracing_registry.get_trace_header_name()] = trace_id

            state.response = response

        except ProviderError as e:
            state.event_context.error_message = str(e)
            await tracing_registry.on_error(state.unified_request, e, state.event_context)
            state.response = self._create_error_response(e)
        except ConfigurationError:
            raise

    @staticmethod
    def _populate_unit_billing_dimensions(state: PipelineState) -> None:
        """Populate request-side unit billing dimensions on the event context.

        TTS has no usage in the response — providers bill per input character,
        so the character count is taken from the request text.
        """
        request = state.unified_request
        if getattr(request, "request_type", None) == RequestType.SPEECH:
            input_text = getattr(request, "input", None)
            if input_text:
                state.event_context.tts_characters = len(input_text)

    @staticmethod
    def _populate_response_billing_dimensions(state: PipelineState, result: Any) -> None:
        """Populate response-side unit billing dimensions on the event context.

        Image generation/edit responses carry the generated images in ``data``;
        providers without token usage bill per image.
        """
        if getattr(state.unified_request, "request_type", None) in (
            RequestType.IMAGE_GENERATION,
            RequestType.IMAGE_EDIT,
        ):
            data = getattr(result, "data", None)
            if data:
                state.event_context.images_generated = len(data)

    def _create_error_response(self, e: ProviderError) -> Response:
        return self._error_handler.format_response(
            e, protocol=protocol_for_name(self.protocol_name)
        )

    @staticmethod
    def _build_exhausted_error(
        last_error: ProviderError, event_context: EventContext
    ) -> ProviderError:
        if not event_context.fallback_attempts:
            return last_error

        summary_parts = [
            _format_attempt_summary(a, last_error) for a in event_context.fallback_attempts
        ]
        composite_message = (
            f"All {len(event_context.fallback_attempts)} providers failed: "
            + ", ".join(summary_parts)
        )

        return ProviderError(
            message=composite_message,
            error_type=last_error.error_type,
            provider_name=last_error.provider_name,
            status_code=last_error.status_code,
            original_error=last_error.original_error,
        )
