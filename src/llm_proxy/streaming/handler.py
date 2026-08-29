"""Streaming handler for HTTP streaming responses.

This module provides HTTP-level streaming functionality:
- Client disconnect detection
- Streaming response configuration (headers, media type)
- Response creation utilities

Note: SSE event formatting is handled by SSEBuilder (streaming/sse_builder.py).
This module focuses only on HTTP streaming concerns.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import cast

from fastapi import Request
from fastapi.responses import StreamingResponse

from llm_proxy.core.constants import DEFAULT_DISCONNECT_CHECK_INTERVAL

# How long a disconnect poll waits on the receive channel before concluding
# the client is still connected. Short enough not to delay detection, long
# enough to avoid busy-looping on message-event wake-ups.
_DISCONNECT_RECEIVE_WAIT_SECONDS = 0.5


async def check_client_disconnected(req: Request) -> bool:
    """Return True when the client connection is gone.

    Cannot rely on ``Request.is_disconnected()``: it only observes messages
    that arrive without an await checkpoint (anyio pre-cancelled scope), which
    under BaseHTTPMiddleware never happens — the wrapped receive blocks until
    a real message arrives. Instead, race the receive against a short wait so
    ``http.disconnect`` is seen when the client is actually gone.

    Safe to call repeatedly: ASGI servers keep answering with disconnect
    messages after the connection is lost, and a timed-out receive is simply
    cancelled (nothing was delivered, so nothing is lost).
    """
    receive = getattr(req, "_receive", None)
    if receive is None:
        return False
    try:
        message = await asyncio.wait_for(receive(), timeout=_DISCONNECT_RECEIVE_WAIT_SECONDS)
    except TimeoutError:
        # No message in time — the connection is still open.
        return False
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a failed receive check must not kill the request
        return False
    return isinstance(message, dict) and message.get("type") == "http.disconnect"


@dataclass
class StreamingResponseConfig:
    """Configuration for streaming HTTP response behavior.

    This configures the HTTP response aspects of streaming, not the
    SSE event format (see streaming/sse_builder.py for SSE formatting).
    """

    disconnect_check_interval: int = DEFAULT_DISCONNECT_CHECK_INTERVAL
    media_type: str = "text/event-stream"
    cache_control: str = "no-cache"
    connection: str = "keep-alive"
    allow_origin: str = "*"


class StreamingHandler:
    """Handler for HTTP streaming responses.

    Provides HTTP-level streaming functionality:
    - Client disconnect detection with configurable check interval
    - Standard SSE headers generation
    - Streaming response creation

    For SSE event formatting (data events, errors, done markers),
    use SSEBuilder from streaming/sse_builder.py.

    Example:
        ```python
        from llm_proxy.streaming.handler import StreamingHandler
        from llm_proxy.streaming.sse_builder import SSEBuilder

        handler = StreamingHandler()
        sse = SSEBuilder()

        async def generate():
            chunk_count = 0
            async for data in stream:
                chunk_count += 1
                if await handler.is_disconnected(request, chunk_count):
                    break
                yield sse.data(data)
            yield sse.done()

        response = handler.create_response(generate)
        ```
    """

    def __init__(self, config: StreamingResponseConfig | None = None):
        """Initialize the streaming handler.

        Args:
            config: Optional streaming configuration
        """
        self.config = config or StreamingResponseConfig()

    def get_headers(self) -> dict[str, str]:
        """Get standard SSE headers for streaming responses.

        Returns:
            Dictionary of headers
        """
        return {
            "Cache-Control": self.config.cache_control,
            "Connection": self.config.connection,
            "Access-Control-Allow-Origin": self.config.allow_origin,
        }

    async def is_disconnected(self, req: Request, chunk_count: int) -> bool:
        """Check if the client has disconnected.

        Only performs the check at intervals defined by disconnect_check_interval
        to avoid performance overhead.

        Args:
            req: The FastAPI request object
            chunk_count: Current chunk count

        Returns:
            True if client is disconnected
        """
        if chunk_count % self.config.disconnect_check_interval == 0:
            return await req.is_disconnected()
        return False

    def create_response(
        self,
        generator: AsyncIterator[str] | Callable[[], AsyncIterator[str]],
    ) -> StreamingResponse:
        """Create a streaming response with standard headers.

        Args:
            generator: Async generator or callable returning async generator

        Returns:
            StreamingResponse with appropriate headers
        """
        if callable(generator):  # noqa: SIM108
            content = cast(Callable[[], AsyncIterator[str]], generator)()
        else:
            content = generator
        return StreamingResponse(
            content,
            media_type=self.config.media_type,
            headers=self.get_headers(),
        )


__all__ = [
    "StreamingHandler",
    "StreamingResponseConfig",
    "check_client_disconnected",
]
