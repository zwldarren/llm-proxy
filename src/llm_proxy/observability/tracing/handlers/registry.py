"""Tracing handler registry with thread-safe registration."""

from threading import RLock
from typing import TYPE_CHECKING

from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.models import InternalRequest, InternalResponse
    from llm_proxy.observability.event_context import EventContext

from llm_proxy.observability.tracing.handlers.base import TracingHandler

logger = get_logger(__name__)

# Handler type registry for provider discovery and handler creation
_HANDLER_TYPES: dict[str, type[TracingHandler]] = {}
_HANDLER_TYPES_LOCK = RLock()


def register_handler_type(name: str, handler_cls: type[TracingHandler]) -> None:
    """Register a handler type by provider name.

    Args:
        name: Provider name (e.g., 'otlp')
        handler_cls: The handler class to register
    """
    with _HANDLER_TYPES_LOCK:
        _HANDLER_TYPES[name] = handler_cls


def get_handler_type(name: str) -> type[TracingHandler] | None:
    """Get a handler type by provider name.

    Args:
        name: Provider name (e.g., 'otlp')

    Returns:
        The handler class or None if not found
    """
    with _HANDLER_TYPES_LOCK:
        return _HANDLER_TYPES.get(name)


def list_handler_types() -> list[type[TracingHandler]]:
    """List all registered handler types.

    Returns:
        List of registered handler classes
    """
    with _HANDLER_TYPES_LOCK:
        return list(_HANDLER_TYPES.values())


class TracingRegistry:
    """Thread-safe registry for tracing handlers.

    Supports registering multiple handlers that will be called in order.
    """

    MAX_HANDLERS = 10

    def __init__(self) -> None:
        self._handlers: list[TracingHandler] = []
        self._lock = RLock()

    def _get_handler_key(self, handler: TracingHandler) -> str:
        """Get a unique key for deduplication based on class + config."""
        parts = [type(handler).__name__]
        for attr, value in vars(handler).items():
            if not attr.startswith("_") and isinstance(value, (str, bool, int, float)):
                parts.append(f"{attr}={value}")
        return "-".join(parts)

    def register(self, handler: TracingHandler) -> None:
        """Register a tracing handler with deduplication.

        Args:
            handler: The tracing handler to register
        """
        with self._lock:
            if len(self._handlers) >= self.MAX_HANDLERS:
                logger.warning(f"Max handlers ({self.MAX_HANDLERS}) reached, skipping registration")
                return

            handler_key = self._get_handler_key(handler)
            for existing in self._handlers:
                if self._get_handler_key(existing) == handler_key:
                    logger.debug(f"Handler {handler_key} already registered, skipping")
                    return

            self._handlers.append(handler)

    def clear(self) -> None:
        """Remove all registered handlers."""
        with self._lock:
            self._handlers.clear()

    @property
    def handlers(self) -> list[TracingHandler]:
        """Get a copy of registered handlers."""
        with self._lock:
            return list(self._handlers)

    async def on_request_start(
        self,
        request: InternalRequest,
        context: EventContext,
    ) -> None:
        """Notify all handlers of request start.

        Args:
            request: The unified request
            context: Event context with sampling decision and accumulated data
        """
        for handler in self.handlers:
            try:
                await handler.on_request_start(request, context)
            except Exception as e:
                logger.error(
                    f"Handler {type(handler).__name__} on_request_start failed: {e}",
                    exc_info=True,
                )

    async def on_request_end(
        self,
        request: InternalRequest,
        response: InternalResponse,
        context: EventContext,
    ) -> None:
        """Notify all handlers of successful request end.

        Args:
            request: The unified request
            response: The unified response
            context: Event context with accumulated token/cost data
        """
        for handler in self.handlers:
            try:
                await handler.on_request_end(request, response, context)
            except Exception as e:
                logger.error(
                    f"Handler {type(handler).__name__} on_request_end failed: {e}",
                    exc_info=True,
                )

    async def on_error(
        self,
        request: InternalRequest,
        error: Exception,
        context: EventContext | None = None,
    ) -> None:
        """Notify all handlers of request error.

        Args:
            request: The unified request
            error: The exception that occurred
            context: Event context with error details (optional)
        """
        for handler in self.handlers:
            try:
                if context is not None:
                    await handler.on_error(request, error, context)
            except Exception as e:
                logger.error(
                    f"Handler {type(handler).__name__} on_error failed: {e}",
                    exc_info=True,
                )

    async def on_stream_start(
        self,
        request: InternalRequest,
        context: EventContext | None = None,
    ) -> None:
        """Notify all handlers of stream start.

        Args:
            request: The unified request
            context: Event context for streaming state (optional)
        """
        for handler in self.handlers:
            try:
                if context is not None:
                    await handler.on_stream_start(request, context)
            except Exception as e:
                logger.error(
                    f"Handler {type(handler).__name__} on_stream_start failed: {e}",
                    exc_info=True,
                )

    async def on_stream_chunk(
        self,
        request: InternalRequest,
        chunk: str,
        context: EventContext | None = None,
    ) -> None:
        """Notify all handlers of stream chunk.

        Args:
            request: The unified request
            chunk: The SSE chunk data
            context: Event context with first_chunk_time for TTFT (optional)
        """
        for handler in self.handlers:
            try:
                if context is not None:
                    await handler.on_stream_chunk(request, chunk, context)
            except Exception as e:
                logger.debug(f"Handler {type(handler).__name__} on_stream_chunk failed: {e}")

    async def on_stream_end(
        self,
        request: InternalRequest,
        context: EventContext | None = None,
        error: Exception | None = None,
    ) -> None:
        """Notify all handlers of stream end.

        Args:
            request: The unified request
            context: Event context with transformer and accumulated data (optional)
            error: Exception if stream ended with error
        """
        for handler in self.handlers:
            try:
                if context is not None:
                    await handler.on_stream_end(request, context, error=error)
            except Exception as e:
                logger.error(
                    f"Handler {type(handler).__name__} on_stream_end failed: {e}",
                    exc_info=True,
                )

    async def shutdown(self) -> None:
        """Shut down all handlers, flushing pending data and releasing resources."""
        for handler in self.handlers:
            try:
                await handler.shutdown()
            except Exception as e:
                logger.error(
                    f"Handler {type(handler).__name__} shutdown failed: {e}",
                    exc_info=True,
                )

    def get_observation_id(self) -> str | None:
        """Get the observation ID from the first handler that has one.

        Returns:
            The observation ID or None if no handler has one
        """
        for handler in self.handlers:
            observation_id = handler.get_observation_id()
            if observation_id is not None:
                return observation_id
        return None

    def get_trace_id(self) -> str | None:
        """Get the trace ID from the first handler that has one.

        Returns:
            The trace ID or None if no handler has one
        """
        for handler in self.handlers:
            trace_id = handler.get_trace_id()
            if trace_id is not None:
                return trace_id
        return None

    def get_trace_header_name(self) -> str:
        """Get the trace header name from the first handler.

        Returns:
            The header name, defaults to 'x-trace-id' if no handlers registered
        """
        for handler in self.handlers:
            return handler.get_trace_header_name()
        return "x-trace-id"


_global_registry: TracingRegistry | None = None
_registry_lock = RLock()


def get_tracing_registry() -> TracingRegistry:
    """Get the global tracing registry singleton."""
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = TracingRegistry()
        return _global_registry


def register_tracing_handler(handler: TracingHandler) -> None:
    """Register a tracing handler with the global registry."""
    get_tracing_registry().register(handler)
