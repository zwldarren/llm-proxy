"""Bidirectional WebSocket relay for the Realtime API.

The relay pumps messages between the client WebSocket and the upstream
provider WebSocket in both directions, forwarding text and binary frames
verbatim. It is protocol-agnostic: protocol knowledge (error envelopes,
usage observation) lives in the router and the usage observer, which hook
in through callbacks.

Close propagation: when one side closes, the other is closed with the same
close code — except RFC 6455 reserved codes (1005/1006/1015), which are
mapped to 1011 since they must never appear in a sent close frame.
Connection governance mirrors the OpenResponses WebSocket transport: a
64 MiB per-message cap on client messages (oversized messages are rejected
with an error event but do not kill the connection) and a 60-minute
connection age cap.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

from websockets import ConnectionClosed
from websockets.asyncio.client import ClientConnection

from llm_proxy.core.ws_common import (
    WS_MAX_CONNECTION_SECONDS,
    WS_MAX_MESSAGE_BYTES,
    WebSocketConnectionLimitError,
    receive_with_connection_cap,
)
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

# Message kinds used by the relay pump (mirrors the ASGI receive dict keys).
KIND_TEXT: Literal["text"] = "text"
KIND_BINARY: Literal["binary"] = "binary"

# Close codes that must never appear in a sent close frame (RFC 6455 §7.4.1
# reserves them for the local endpoint: 1005/1006/1015). Received codes in
# this set are mapped to 1011 before being propagated to the peer.
_UNSENDABLE_CLOSE_CODES = frozenset({1005, 1006, 1015})


@dataclass
class RealtimeRelayConfig:
    """Governance limits for a relayed Realtime connection.

    The defaults match the OpenResponses WebSocket transport (see
    :mod:`llm_proxy.core.ws_common`): a 64 MiB per-message cap on client
    messages (oversized messages are rejected with an error event but do not
    kill the connection) and a 60-minute connection age cap.
    """

    max_connection_seconds: float = WS_MAX_CONNECTION_SECONDS
    max_message_bytes: int = WS_MAX_MESSAGE_BYTES


class StarletteWebSocketAdapter:
    """Relay adapter over a Starlette WebSocket (the client side)."""

    def __init__(self, websocket: Any):
        self._ws = websocket
        # Close code received from the peer, set once the socket is closed.
        self.close_code: int | None = None

    async def receive(self) -> tuple[Literal["text", "binary"], str | bytes] | None:
        """Receive one message; returns (kind, data) or None when closed."""
        message = await self._ws.receive()
        mtype = message.get("type")
        if mtype == "websocket.disconnect":
            self.close_code = message.get("code")
            return None
        if mtype == "websocket.receive":
            if "text" in message:
                return (KIND_TEXT, message["text"])
            if "bytes" in message:
                return (KIND_BINARY, message["bytes"])
        return None

    async def send(self, kind: Literal["text", "binary"], data: str | bytes) -> None:
        if kind == KIND_TEXT:
            await self._ws.send_text(data)
        else:
            await self._ws.send_bytes(data)

    async def close(self, code: int) -> None:
        await self._ws.close(code=code)


class WebsocketsClientAdapter:
    """Relay adapter over a websockets ClientConnection (the upstream side)."""

    def __init__(self, connection: ClientConnection):
        self._conn = connection
        self.close_code: int | None = None

    async def receive(self) -> tuple[Literal["text", "binary"], str | bytes] | None:
        """Receive one message; returns (kind, data) or None when closed."""
        try:
            message = await self._conn.recv()
        except ConnectionClosed as exc:
            # ``exc.code`` is deprecated in websockets 17; read the received
            # close frame directly (None when the close frame was not
            # received, e.g. abnormal closure). Abnormal closures map to
            # 1011: 1006 is reserved by RFC 6455 and must never be sent in a
            # close frame.
            self.close_code = exc.rcvd.code if exc.rcvd is not None else 1011
            return None
        if isinstance(message, str):
            return (KIND_TEXT, message)
        return (KIND_BINARY, message)

    async def send(self, kind: Literal["text", "binary"], data: str | bytes) -> None:
        await self._conn.send(data)

    async def close(self, code: int) -> None:
        await self._conn.close(code=code)


class RealtimeRelay:
    """Pump messages between the client and upstream WebSockets.

    Args:
        client: Client-side socket adapter (StarletteWebSocketAdapter)
        upstream: Upstream socket adapter (WebsocketsClientAdapter)
        config: Governance limits
        on_upstream_message: Optional async hook called for every
            upstream→client message (used for usage observation). Exceptions
            are logged and swallowed — observation must never break the relay.
        on_client_error: Optional async hook called with an error
            code and message when the relay rejects a client message
            (oversized) or the connection age cap is hit. The router sends
            the Realtime error envelope.
    """

    def __init__(
        self,
        client: StarletteWebSocketAdapter,
        upstream: WebsocketsClientAdapter,
        *,
        config: RealtimeRelayConfig | None = None,
        on_upstream_message: (
            Callable[[Literal["text", "binary"], str | bytes], Awaitable[None]] | None
        ) = None,
        on_client_error: Callable[[str, str], Awaitable[None]] | None = None,
    ):
        self._client = client
        self._upstream = upstream
        self._config = config or RealtimeRelayConfig()
        self._on_upstream_message = on_upstream_message
        self._on_client_error = on_client_error
        self._started_at = time.monotonic()

    async def run(self) -> None:
        """Pump until either side closes, the age cap is hit, or a pump fails.

        On return both sockets are closed (or already closed by the peer).
        """
        tasks = [
            asyncio.create_task(self._pump_client_to_upstream()),
            asyncio.create_task(self._pump_upstream_to_client()),
        ]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    logger.warning(f"Realtime relay pump failed: {exc}")
        finally:
            # Final close: a pump that finished because its source closed
            # already closed the destination; this covers the remaining
            # cases (expiry, pump failure) with a no-op on closed sockets.
            code = self._sendable_close_code(self._client.close_code or self._upstream.close_code)
            await self._safe_close(self._client, code)
            await self._safe_close(self._upstream, code)

    async def _pump_client_to_upstream(self) -> None:
        try:
            while True:
                try:
                    message = await receive_with_connection_cap(
                        self._client.receive,
                        connected_at=self._started_at,
                        max_seconds=self._config.max_connection_seconds,
                    )
                except WebSocketConnectionLimitError:
                    await self._notify_client_error(
                        "websocket_connection_limit_reached",
                        "WebSocket connection limit of 60 minutes reached.",
                    )
                    break
                if message is None:
                    break  # client disconnected
                kind, data = message
                if self._exceeds_limit(kind, data):
                    await self._notify_client_error(
                        "invalid_request",
                        "WebSocket message exceeds the 64 MiB size limit.",
                    )
                    continue
                await self._upstream.send(kind, data)
        except Exception as exc:
            logger.warning(f"Realtime client→upstream pump error: {exc}")
        finally:
            # Propagate the client's close code to the upstream.
            await self._safe_close(
                self._upstream, self._sendable_close_code(self._client.close_code)
            )

    async def _pump_upstream_to_client(self) -> None:
        try:
            while True:
                message = await self._upstream.receive()
                if message is None:
                    break  # upstream closed
                kind, data = message
                if self._on_upstream_message is not None:
                    try:
                        await self._on_upstream_message(kind, data)
                    except Exception as exc:  # noqa: BLE001 - observation must not break the relay
                        logger.warning(f"Realtime observation hook failed: {exc}")
                await self._client.send(kind, data)
        except Exception as exc:
            logger.warning(f"Realtime upstream→client pump error: {exc}")
        finally:
            # Propagate the upstream's close code to the client.
            await self._safe_close(
                self._client, self._sendable_close_code(self._upstream.close_code)
            )

    def _exceeds_limit(self, kind: Literal["text", "binary"], data: str | bytes) -> bool:
        if kind == KIND_TEXT and isinstance(data, str):
            return len(data.encode("utf-8")) > self._config.max_message_bytes
        return len(data) > self._config.max_message_bytes

    async def _notify_client_error(self, code: str, message: str) -> None:
        if self._on_client_error is not None:
            try:
                await self._on_client_error(code, message)
            except Exception as exc:  # noqa: BLE001 - error delivery must not break the relay
                logger.warning(f"Realtime client error hook failed: {exc}")

    @staticmethod
    def _sendable_close_code(code: int | None) -> int:
        """Map a received close code to one that may be sent in a close frame.

        RFC 6455 reserves 1005/1006/1015 for the local endpoint and forbids
        sending them; an abnormal peer drop surfaces as 1006 (ASGI reports it
        for client disconnects, ``websockets`` yields it when no close frame
        arrived), which must be translated to 1011 before propagation.
        """
        if code in _UNSENDABLE_CLOSE_CODES:
            return 1011
        return code if code is not None else 1000

    @staticmethod
    async def _safe_close(socket: Any, code: int) -> None:
        # Closing an already-closed socket is a no-op for both libraries.
        with suppress(Exception):
            await socket.close(code)


__all__ = [
    "KIND_BINARY",
    "KIND_TEXT",
    "RealtimeRelay",
    "RealtimeRelayConfig",
    "StarletteWebSocketAdapter",
    "WebsocketsClientAdapter",
]
