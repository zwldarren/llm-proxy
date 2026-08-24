"""SSE (Server-Sent Events) builder utilities.

This module provides the base SSE builder class that handles generic SSE formatting.
Protocol-specific builders (OpenAI, Anthropic) should extend this base class.

Architecture:
    SSEBuilder (base) - Generic SSE formatting (data:, event:, comments)
        └── AnthropicSSEBuilder - Anthropic-specific events (in protocols/anthropic/)
        └── (OpenAI uses base SSEBuilder directly with done_marker="[DONE]")
"""

from typing import Any

import orjson


class SSEBuilder:
    """Base builder class for creating SSE (Server-Sent Events) formatted strings.

    This class provides the foundation for SSE event formatting. It handles:
    - Basic SSE formatting (data-only events, named events)
    - Done markers for stream termination
    - Error events in OpenAI format
    - Comments and keepalive signals

    Protocol-specific builders should extend this class to add their own
    event types while reusing the base formatting logic.

    Example:
        ```python
        builder = SSEBuilder()

        # Create a data event
        event = builder.data({"content": "Hello"})
        # Output: "data: {"content":"Hello"}\n\n"

        # Create a named event
        event = builder.event("message_start", {"type": "message_start"})
        # Output: "event: message_start\ndata: {"type":"message_start"}\n\n"

        # Create a done marker
        done = builder.done()
        # Output: "data: [DONE]\n\n"
        ```
    """

    def __init__(self, done_marker: str = "[DONE]"):
        """Initialize the SSE builder.

        Args:
            done_marker: The marker to use for the done event (default: "[DONE]")
        """
        self.done_marker = done_marker

    def _serialize_payload(self, payload: Any) -> str:
        """Serialize a payload to JSON string.

        This method can be overridden by subclasses to customize serialization,
        e.g., to use Pydantic's model_dump_json() for better performance.

        Args:
            payload: The data payload (dict, Pydantic model, or string)

        Returns:
            JSON string representation
        """
        if isinstance(payload, str):
            return payload
        if hasattr(payload, "model_dump_json"):
            return payload.model_dump_json()
        return orjson.dumps(payload).decode()

    def data(self, payload: dict[str, Any] | str) -> str:
        """Create a data-only SSE event.

        Args:
            payload: The data payload (dict will be JSON-encoded)

        Returns:
            SSE formatted string
        """
        data_str = self._serialize_payload(payload)
        return f"data: {data_str}\n\n"

    def event(self, event_type: str, payload: Any) -> str:
        """Create a named SSE event.

        Args:
            event_type: The event type name
            payload: The data payload (dict, Pydantic model, or string)

        Returns:
            SSE formatted string with event type
        """
        data_str = self._serialize_payload(payload)
        return f"event: {event_type}\ndata: {data_str}\n\n"

    def done(self) -> str:
        """Create a done marker event.

        Returns:
            SSE formatted done marker
        """
        return f"data: {self.done_marker}\n\n"

    def error(
        self,
        error_data: dict[str, Any],
        include_done: bool = True,
    ) -> str:
        """Create an error event from pre-formatted error data.

        This method only wraps the error data in SSE format. The actual error
        formatting should be done by ErrorResponseBuilder to ensure consistency.

        Args:
            error_data: Pre-formatted error dictionary
            include_done: Whether to include done marker after error

        Returns:
            SSE formatted error event

        Example:
            ```python
            from llm_proxy.api.error_responses import ErrorResponseBuilder

            error_dict = ErrorResponseBuilder.create_openai_error(
                message="Invalid request",
                error_type="invalid_request_error"
            )
            sse_error = builder.error(error_dict)
            ```
        """
        result = self.data(error_data)
        if include_done:
            result += self.done()
        return result


# Default SSE builder instance
_default_builder = SSEBuilder()


def create_sse_error(
    error_data: dict[str, Any],
    include_done: bool = True,
) -> str:
    """Create an error event using the default builder.

    This function wraps pre-formatted error data in SSE format.
    Use ErrorResponseBuilder to create the error_data.

    Args:
        error_data: Pre-formatted error dictionary
        include_done: Whether to include done marker

    Returns:
        SSE formatted error event

    Example:
        ```python
        from llm_proxy.api.error_responses import ErrorResponseBuilder
        from llm_proxy.streaming.sse_builder import create_sse_error

        error_dict = ErrorResponseBuilder.create_openai_error(
            message="Invalid request",
            error_type="invalid_request_error"
        )
        sse_error = create_sse_error(error_dict)
        ```
    """
    return _default_builder.error(error_data=error_data, include_done=include_done)


__all__ = [
    "SSEBuilder",
    "create_sse_error",
]
