# src/llm_proxy/core/processing/unified.py
"""Unified request processor for ProtocolEndpoint classes.

This processor handles InternalRequest from protocol endpoints,
passing them directly to adapters that now accept InternalRequest.

The processor uses a pipeline of stages to process requests:
ProviderSelection -> ParameterOverride -> PreviousResponseResolution ->
WebSearch -> RequestExecution. RoleNormalization is applied on-demand during retry.
"""

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, NoReturn

from fastapi import Request, Response

from llm_proxy.config.manager import load_logging_config
from llm_proxy.core.constants import CLIENT_DISCONNECTED_STATE_KEY
from llm_proxy.core.context import RequestUserContext, set_request_user_context
from llm_proxy.core.errors import get_error_handler
from llm_proxy.core.errors.protocols import protocol_for_name
from llm_proxy.core.exceptions import (
    ClientDisconnectedError,
    LLMProxyError,
    ValidationError,
)
from llm_proxy.core.identity import get_request_identity
from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.stages import (
    ParameterOverrideService,
    ParameterOverrideStage,
    PipelineState,
    PreviousResponseResolutionStage,
    ProviderSelectionStage,
    RequestExecutionStage,
    WebSearchStage,
)
from llm_proxy.core.processing.strategies import ProcessingStrategy, get_strategy
from llm_proxy.core.processing.streaming_processor import StreamingProcessor
from llm_proxy.core.request_type import RequestType
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.models import InternalRequest
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.sampling import make_sampling_decision
from llm_proxy.observability.tracing.handlers import get_tracing_registry
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.streaming.handler import StreamingHandler, StreamingResponseConfig

if TYPE_CHECKING:
    from llm_proxy.protocols.base import ProtocolEndpoint

logger = get_logger(__name__)


class UnifiedProcessor:
    """Processor for ProtocolEndpoint instances.

    Handles InternalRequest from protocol endpoints, passing them to adapters
    and converting responses back to protocol-specific format.

    Uses a pipeline of stages (ProviderSelection, ParameterOverride,
    PreviousResponseResolution, WebSearch, RequestExecution) to process
    requests. Each stage is independently testable.
    """

    def __init__(
        self,
        protocol_endpoint: ProtocolEndpoint,
        streaming_handler: StreamingHandler | None = None,
    ):
        self.protocol_endpoint = protocol_endpoint
        self.protocol_name = protocol_endpoint.name
        self._serializer = get_protocol_serializer(self.protocol_name)
        self.streaming_handler = streaming_handler or StreamingHandler()
        self._error_handler = get_error_handler()

        # Create parameter override service (needed by both pipeline and streaming processor)
        self._param_override_service = ParameterOverrideService(self._serializer)
        self._param_override_stage = ParameterOverrideStage(self._param_override_service)

        self._streaming_processor = StreamingProcessor(
            protocol_endpoint=protocol_endpoint,
            streaming_handler=self.streaming_handler,
            error_handler=self._error_handler,
            param_override_service=self._param_override_service,
        )

        # Build pipeline stages
        self._stages = [
            ProviderSelectionStage(),
            self._param_override_stage,
            PreviousResponseResolutionStage(),
            WebSearchStage(),
            RequestExecutionStage(
                protocol_name=self.protocol_name,
                protocol_endpoint=protocol_endpoint,
                serializer=self._serializer,
                streaming_processor=self._streaming_processor,
                param_override_service=self._param_override_service,
            ),
        ]

    def _handle_unexpected_error(self, e: Exception, req: Request) -> NoReturn:
        """Handle unexpected errors."""
        request_id = getattr(req.state, "request_id", None)
        provider = getattr(req.state, "provider", None)
        model = getattr(req.state, "model", None)

        logger.error(
            f"Unexpected error [request_id={request_id}] "
            f"[provider={provider}] [model={model}]: {e}",
            exc_info=e,
        )

        raise LLMProxyError(
            message=f"An internal error occurred: {e!s}",
            code="internal_error",
        ) from e

    async def _create_event_context(
        self,
        request: Request,
        unified_request: InternalRequest,
        raw_request_data: dict[str, Any],
        trace_id: str,
        context: RequestContext,
    ) -> EventContext:
        """Create EventContext with sampling decision and request metadata."""
        config_manager = context.config_manager
        if config_manager is not None:
            logging_config = (await config_manager.get_config()).server_params.logging
        else:
            logging_config = load_logging_config()

        sampling = make_sampling_decision(
            config=logging_config,
            request=request,
            path=request.url.path,
        )

        should_log_input_output = logging_config.enable_database_logging

        request_headers = dict(request.headers)
        identity = get_request_identity(request)

        event_context = EventContext(
            request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
            trace_id=trace_id,
            model=unified_request.model,
            internal_model=unified_request.model,
            provider=None,
            session_id=context.session_id,
            user_id=context.user_id or identity.user,
            auth_user_id=identity.user_id,
            request_type=RequestType(unified_request.request_type),
            log_type=sampling.log_type.value,
            is_api_endpoint=request.url.path.startswith("/api/"),
            should_capture_full_body=sampling.should_capture_full_body,
            should_log_input_output=should_log_input_output,
            start_time=time.perf_counter(),
            start_timestamp=time.time(),
            request_headers=request_headers,
            request_body=raw_request_data,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            api_key_name=identity.api_key_name,
            auth_method=identity.auth_method,
            metadata={
                "endpoint": request.url.path,
                "method": request.method,
                "request_type": str(unified_request.request_type),
            },
        )

        # Set request user context for downstream logging (e.g., web search logs)
        set_request_user_context(
            RequestUserContext(
                user_id=identity.user_id,
                user_identity=context.user_id or identity.user,
                api_key_name=identity.api_key_name,
                auth_method=identity.auth_method,
            )
        )

        return event_context

    async def process(
        self,
        protocol_request: Any,
        req: Request,
        context: RequestContext,
    ) -> Response:
        """Process a unified request through the pipeline.

        1. Extract raw data and parse to unified request
        2. Determine processing strategy
        3. Run pipeline stages: ProviderSelection -> ParameterOverride ->
           PreviousResponseResolution -> WebSearch -> RequestExecution
        """
        if hasattr(protocol_request, "model_dump"):
            raw_request_data = protocol_request.model_dump(exclude_none=True)
        else:
            raw_request_data = {k: v for k, v in dict(protocol_request).items() if v is not None}

        # Inject file bytes from wrapper objects into request data for audio protocols
        file = getattr(protocol_request, "file", None)
        filename = getattr(protocol_request, "filename", None)
        if file is not None and isinstance(file, (bytes, bytearray)):
            raw_request_data["file"] = file
            if filename:
                raw_request_data["filename"] = filename

        # Set format context before parsing so that parse_request can
        # enrich it (e.g., with namespace_map from tool conversion).
        if self.protocol_endpoint.on_parse_request is not None:
            self.protocol_endpoint.on_parse_request(raw_request_data)

        unified_request = self._serializer.parse_request(raw_request_data)

        unified_request.metadata.protocol_name = self.protocol_name
        unified_request._raw_protocol_data = raw_request_data

        strategy = get_strategy(unified_request.request_type)
        if strategy is None:
            raise ValidationError(message=f"Unknown request type: {unified_request.request_type}")

        return await self._run_pipeline(
            strategy=strategy,
            unified_request=unified_request,
            raw_request_data=raw_request_data,
            req=req,
            context=context,
        )

    async def _run_pipeline(
        self,
        strategy: ProcessingStrategy,
        unified_request: Any,
        raw_request_data: dict[str, Any],
        req: Request,
        context: RequestContext,
    ) -> Response:
        """Run the processing pipeline and return the response."""
        trace_id = context.trace_id or str(uuid.uuid4())

        event_context = await self._create_event_context(
            request=req,
            unified_request=unified_request,
            raw_request_data=raw_request_data,
            trace_id=trace_id,
            context=context,
        )
        context.event_context = event_context

        state = PipelineState(
            raw_data=raw_request_data,
            unified_request=unified_request,
            req=req,
            strategy=strategy,
            trace_id=trace_id,
            event_context=event_context,
            original_raw_data=raw_request_data,
        )

        try:
            for stage in self._stages:
                await stage.process(state, context)
                if state.response is not None or state.error is not None:
                    break

            if state.error is not None:
                return self._error_handler.format_response(
                    state.error, protocol=protocol_for_name(self.protocol_name)
                )

            if state.response is not None:
                return state.response

            raise LLMProxyError(
                message="Pipeline completed without producing a response",
                code="internal_error",
            )
        except LLMProxyError:
            raise
        except asyncio.CancelledError:
            # The pipeline was cancelled — for a client disconnect (see
            # await_with_disconnect_monitor / keepalive in api/keepalive.py)
            # this is the "invisible 524" moment: the origin keeps generating
            # for a client that is already gone. Record the abandonment so it
            # shows up as a failed (499) request instead of vanishing.
            if getattr(req.state, CLIENT_DISCONNECTED_STATE_KEY, False):
                error = ClientDisconnectedError()
                tracing_registry = context.tracing_registry or get_tracing_registry()
                await asyncio.shield(
                    tracing_registry.on_error(state.unified_request, error, event_context)
                )
            raise
        except Exception as e:
            tracing_registry = context.tracing_registry or get_tracing_registry()
            await tracing_registry.on_error(state.unified_request, e, event_context)
            event_context.error_message = str(e)
            req.state.audit_log_written = True
            self._handle_unexpected_error(e, req)
        finally:
            await asyncio.shield(state.exit_stack.aclose())


def create_unified_processor(
    protocol_endpoint: ProtocolEndpoint,
    streaming_config: StreamingResponseConfig | None = None,
) -> UnifiedProcessor:
    """Factory function to create a UnifiedProcessor."""
    streaming_handler = StreamingHandler(streaming_config) if streaming_config else None

    return UnifiedProcessor(
        protocol_endpoint=protocol_endpoint,
        streaming_handler=streaming_handler,
    )
