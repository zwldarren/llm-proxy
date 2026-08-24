"""Logging handler for request tracing."""

from typing import TYPE_CHECKING

from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.tracing.handlers.base import TracingHandler

if TYPE_CHECKING:
    from llm_proxy.models import InternalRequest, InternalResponse
    from llm_proxy.observability.event_context import EventContext

logger = get_logger(__name__)


class LoggingHandler(TracingHandler):
    """Handler that logs request lifecycle events."""

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)

    async def on_request_start(
        self,
        request: InternalRequest,
        context: EventContext,
    ) -> None:
        if not self._enabled:
            return
        context_parts = [f"model={request.model}", f"stream={request.stream}"]
        if context.trace_id:
            context_parts.append(f"trace_id={context.trace_id}")
        if context.provider:
            context_parts.append(f"provider={context.provider}")
        if context.session_id:
            context_parts.append(f"session_id={context.session_id}")
        if context.user_id:
            context_parts.append(f"user_id={context.user_id}")
        if context.api_key_name:
            context_parts.append(f"api_key={context.api_key_name}")
        logger.debug(f"Request started: {', '.join(context_parts)}")

    async def on_request_end(
        self,
        request: InternalRequest,
        response: InternalResponse,
        context: EventContext,
    ) -> None:
        if not self._enabled:
            return
        logger.debug(
            f"Request completed: model={request.model}, "
            f"tokens={response.usage.total_tokens if response.usage else 'N/A'}, "
            f"latency={context.latency_ms:.1f}ms"
        )

    async def on_error(
        self,
        request: InternalRequest,
        error: Exception,
        context: EventContext,
    ) -> None:
        if not self._enabled:
            return
        context_parts = [f"model={request.model}"]
        if context.provider:
            context_parts.append(f"provider={context.provider}")
        if context.trace_id:
            context_parts.append(f"trace_id={context.trace_id}")
        if context.api_key_name:
            context_parts.append(f"api_key={context.api_key_name}")
        context_parts.append(f"error={error}")
        logger.error(f"Request failed: {', '.join(context_parts)}")

    async def on_stream_end(
        self,
        request: InternalRequest,
        context: EventContext,
        error: Exception | None = None,
    ) -> None:
        if not self._enabled:
            return
        if error:
            logger.error(f"Stream failed: model={request.model}, error={error}")
        else:
            logger.debug(f"Stream completed: model={request.model}")
