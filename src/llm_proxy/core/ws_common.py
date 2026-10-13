"""Shared helpers for the proxy's WebSocket transports.

Both WebSocket transports (OpenResponses ``/v1/responses`` and the Realtime
relay ``/v1/realtime``) authenticate through the API-key gate
(:mod:`llm_proxy.api.gate`, with each router's own extraction policy and
rejection rendering), enforce the same 64 MiB per-message cap and 60-minute
connection cap, and speak one close-code language (4401 auth failure, 4403
forbidden, 1011 upstream/provider failure). The shared pieces live here so
the two routers cannot drift apart.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.requests import Request as StarletteRequest

from llm_proxy.core.request_facts import facts_for, mint_request_id

# Connection age cap shared by both WebSocket transports (60 minutes).
WS_MAX_CONNECTION_SECONDS = 60 * 60

# Per-message cap for client messages shared by both WebSocket transports
# (64 MiB; conversation items can carry full audio).
WS_MAX_MESSAGE_BYTES = 64 * 1024 * 1024

# Proxy close-code language shared by both WebSocket transports (RFC 6455
# private-use range, matching the OpenResponses WebSocket transport).
WS_CLOSE_AUTH_FAILED = 4401
WS_CLOSE_FORBIDDEN = 4403
WS_CLOSE_UPSTREAM_FAILURE = 1011


class WebSocketConnectionLimitError(TimeoutError):
    """The connection age cap was reached before a message arrived."""


def build_ws_request(websocket: Any) -> StarletteRequest:
    """Build an HTTP Request from a WebSocket scope for shared accessors.

    Config/dependency accessors expect an HTTP request; this fakes one from
    the websocket scope (a GET on the same path) and mints a per-connection
    request id so downstream code (model resolution, usage logging) works
    unchanged. The identity is *not* stamped here: the API-key gate stamps it
    into the request's facts after verification, like on every transport.
    The state dict is shared with the websocket's own scope, so the record
    stays reachable for the connection's lifetime.
    """
    scope = dict(websocket.scope)
    scope["type"] = "http"
    scope["method"] = "GET"
    scope.setdefault("http_version", "1.1")
    request = StarletteRequest(scope)
    facts_for(request).request_id = mint_request_id("ws_")
    return request


async def receive_with_connection_cap[T](
    receive: Callable[[], Awaitable[T]],
    *,
    connected_at: float,
    max_seconds: float = WS_MAX_CONNECTION_SECONDS,
) -> T:
    """Receive one message, failing fast when the connection age cap is hit.

    A non-positive remaining budget raises immediately, so an idle socket can
    never outlive the cap. Raises :class:`WebSocketConnectionLimitError` when
    the cap is reached before a message arrives.
    """
    remaining = max_seconds - (time.monotonic() - connected_at)
    if remaining <= 0:
        raise WebSocketConnectionLimitError
    try:
        return await asyncio.wait_for(receive(), timeout=remaining)
    except TimeoutError as exc:
        # ``wait_for`` raises the builtin ``TimeoutError``; normalize it so
        # callers can catch a single type.
        raise WebSocketConnectionLimitError from exc


__all__ = [
    "WS_CLOSE_AUTH_FAILED",
    "WS_CLOSE_FORBIDDEN",
    "WS_CLOSE_UPSTREAM_FAILURE",
    "WS_MAX_CONNECTION_SECONDS",
    "WS_MAX_MESSAGE_BYTES",
    "WebSocketConnectionLimitError",
    "build_ws_request",
    "receive_with_connection_cap",
]
