"""Streaming handler for HTTP streaming responses.

This module provides HTTP-level streaming functionality:
- Client disconnect detection
- Streaming response configuration (headers, media type)
- Response creation utilities

Note: SSE event formatting is handled by SSEBuilder (streaming/sse_builder.py).
This module focuses only on HTTP streaming concerns.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import cast

from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.types import Receive

# How long the legacy timed poll waits on the receive channel before concluding
# the client is still connected. Only used when a request exposes no ASGI
# receive channel (functional/unit tests); production paths use
# :class:`ClientDisconnectWatcher`, which never blocks the response pump.
_DISCONNECT_RECEIVE_WAIT_SECONDS = 0.5


async def check_client_disconnected(req: Request) -> bool:
    """Return True when the client connection is gone.

    Cannot rely on ``Request.is_disconnected()``: it only observes messages
    that arrive without an await checkpoint (anyio pre-cancelled scope).
    Instead, race the receive against a short wait so ``http.disconnect`` is
    seen when the client is actually gone.

    Blocking for up to ``_DISCONNECT_RECEIVE_WAIT_SECONDS`` before concluding
    "still connected": prefer :class:`ClientDisconnectWatcher` on any path that
    runs while a response body is being produced.
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


class ClientDisconnectWatcher:
    """Observe ``http.disconnect`` without ever blocking the response pump.

    ``receive()`` blocks until the server delivers a message, and uvicorn only
    wakes it on new client data or a disconnect. Polling it with
    ``asyncio.wait_for(timeout=...)`` therefore costs the *full* timeout on
    every healthy stream — the old 0.5s poll added half a second to the first
    chunk (and every 10th chunk) of every streaming response.

    Instead a single background task awaits the channel once and records the
    verdict in an :class:`asyncio.Event`. Callers read
    :attr:`disconnected` synchronously, so detection is immediate when the
    client really goes away and free when it does not.

    The watcher is inert (and :meth:`wait` degrades to the legacy timed poll)
    when the request exposes no ASGI receive channel, which is the case for
    functional middleware tests that build requests by hand.
    """

    def __init__(self, req: Request) -> None:
        self._req = req
        self._receive = getattr(req, "_receive", None)
        self._disconnected = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def disconnected(self) -> bool:
        """True once ``http.disconnect`` has been observed."""
        return self._disconnected.is_set()

    def start(self) -> None:
        """Begin watching. Idempotent, and a no-op without a receive channel."""
        receive = self._receive
        if receive is None or self._task is not None:
            return
        self._task = asyncio.ensure_future(self._watch(receive))

    async def _watch(self, receive: Receive) -> None:
        while True:
            try:
                message = await receive()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a failed watch must not kill the request
                return
            if isinstance(message, dict) and message.get("type") == "http.disconnect":
                self._disconnected.set()
                return

    async def wait(self) -> None:
        """Block until the client disconnects (cancellable)."""
        if self._receive is None:
            while not await check_client_disconnected(self._req):
                await asyncio.sleep(_DISCONNECT_RECEIVE_WAIT_SECONDS)
            return
        self.start()
        await self._disconnected.wait()

    async def poll(self) -> bool:
        """Report whether the client has disconnected, yielding a scheduling turn.

        The single ``sleep(0)`` lets the watcher observe a disconnect that is
        *already* pending (e.g. the client vanished while the pre-response
        pipeline ran) before the first chunk is emitted. Unlike the timed poll
        it costs one event-loop tick, not the full receive timeout, and it
        never blocks on a healthy connection.
        """
        if self._receive is None:
            return await check_client_disconnected(self._req)
        self.start()
        if not self._disconnected.is_set():
            await asyncio.sleep(0)
        return self._disconnected.is_set()

    async def aclose(self) -> None:
        """Cancel the watcher task. Safe to call more than once."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@dataclass
class StreamingResponseConfig:
    """Configuration for streaming HTTP response behavior.

    This configures the HTTP response aspects of streaming, not the
    SSE event format (see streaming/sse_builder.py for SSE formatting).
    """

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
            async for data in stream:
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
