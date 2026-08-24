"""Base class for tracing handlers."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.models import InternalRequest, InternalResponse
    from llm_proxy.observability.event_context import EventContext


class TracingHandler:
    """Base class for request lifecycle tracing.

    Subclasses can override any method to hook into the request processing
    lifecycle. All methods have default no-op implementations.

    Lifecycle:
        on_request_start -> [processing] -> on_request_end | on_error
    """

    # Provider metadata — subclasses override these
    provider_name: str = ""
    required_settings: list[str] = []
    optional_settings: list[str] = []
    description: str | None = None
    field_metadata: list[dict[str, Any]] = []
    name: str = ""

    def __init__(self, enabled: bool = True) -> None:
        """Initialize tracing handler.

        Args:
            enabled: Whether this handler is enabled. Defaults to True.
        """
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Whether this handler is enabled."""
        return self._enabled

    @classmethod
    def validate_config(cls, settings: dict[str, Any]) -> bool:
        """Validate that required settings are present and valid.

        Args:
            settings: Provider-specific settings dictionary

        Returns:
            True if configuration is valid, False otherwise
        """
        return all(settings.get(key) for key in cls.required_settings)

    @classmethod
    def create_handler(
        cls,
        settings: dict[str, Any],
        config_manager: DatabaseConfigManager | None = None,
    ) -> TracingHandler:
        """Create a handler instance from configuration.

        Default implementation creates a basic handler with enabled status.
        Subclasses should override this to pass settings to __init__.

        Args:
            settings: Provider-specific settings dictionary
            config_manager: Optional config manager for cost calculation

        Returns:
            Configured TracingHandler instance
        """
        return cls(enabled=True)

    async def on_request_start(
        self,
        request: InternalRequest,
        context: EventContext,
    ) -> None:
        """Called before processing a request.

        Args:
            request: The unified request being processed
            context: Event context with sampling decision and accumulated data
        """
        pass

    async def on_request_end(
        self,
        request: InternalRequest,
        response: InternalResponse,
        context: EventContext,
    ) -> None:
        """Called after successfully processing a request.

        Args:
            request: The unified request that was processed
            response: The unified response
            context: Event context with accumulated token/cost data
        """
        pass

    async def on_error(
        self,
        request: InternalRequest,
        error: Exception,
        context: EventContext,
    ) -> None:
        """Called when an error occurs during request processing.

        Args:
            request: The unified request that failed
            error: The exception that was raised
            context: Event context with error details populated
        """
        pass

    async def on_stream_start(
        self,
        request: InternalRequest,
        context: EventContext,
    ) -> None:
        """Called when a streaming response starts.

        Args:
            request: The unified request being streamed
            context: Event context for streaming state
        """
        pass

    async def on_stream_chunk(
        self,
        request: InternalRequest,
        chunk: str,
        context: EventContext,
    ) -> None:
        """Called for each chunk in a streaming response.

        Args:
            request: The unified request being streamed
            chunk: The SSE chunk data
            context: Event context with first_chunk_time for TTFT
        """
        pass

    async def on_stream_end(
        self,
        request: InternalRequest,
        context: EventContext,
        error: Exception | None = None,
    ) -> None:
        """Called when a streaming response ends.

        Args:
            request: The unified request that was streamed
            context: Event context with transformer and accumulated data
            error: Exception if the stream ended due to an error, None otherwise
        """
        pass

    async def shutdown(self) -> None:
        """Called during application shutdown to flush pending data and release resources.

        Subclasses should override this to perform cleanup such as flushing
        pending trace data to external services.
        """
        pass

    def get_observation_id(self) -> str | None:
        """Get the current observation ID for trace correlation.

        Returns:
            The observation ID (e.g., trace ID) or None if not available
        """
        return None

    def get_trace_id(self) -> str | None:
        """Get the current trace ID for response header correlation.

        Returns:
            The trace ID or None if not available
        """
        return None

    def get_trace_header_name(self) -> str:
        """Get the HTTP header name for trace correlation.

        Returns:
            The header name to use for trace correlation in responses.
            Defaults to 'x-trace-id'.
        """
        return "x-trace-id"
