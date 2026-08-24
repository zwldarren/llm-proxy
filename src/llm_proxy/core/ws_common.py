"""Shared helpers for the proxy's WebSocket transports.

Both WebSocket transports (OpenResponses ``/v1/responses`` and the Realtime
relay ``/v1/realtime``) authenticate with proxy API keys the same way,
enforce the same 64 MiB per-message cap and 60-minute connection cap, and
speak one close-code language (4401 auth failure, 4403 forbidden, 1011
upstream/provider failure). The shared pieces live here so the two routers
cannot drift apart.
"""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.requests import Request as StarletteRequest

from llm_proxy.core.identity import RequestIdentity, set_request_identity

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


def extract_ws_api_key(
    websocket: Any, *, insecure_key_subprotocol_prefix: str | None = None
) -> str | None:
    """Extract the API key from headers, subprotocols, or query params.

    Header-based auth (``Authorization`` / ``x-api-key``) is preferred. The
    ``api_key`` query parameter is accepted because browser WebSocket clients
    cannot set headers — be aware that query strings can end up in reverse
    proxy access logs, so operators fronting the proxy should strip or hash
    them in log pipelines. When ``insecure_key_subprotocol_prefix`` is given,
    a subprotocol starting with that prefix (e.g. OpenAI's
    ``openai-insecure-api-key.`` convention for browser clients) is also
    accepted.
    """
    auth_header = websocket.headers.get("Authorization")
    if auth_header:
        if auth_header.startswith("Bearer "):
            return auth_header[7:].strip() or None
        return auth_header.strip() or None
    x_api_key = websocket.headers.get("x-api-key")
    if x_api_key:
        return x_api_key.strip() or None
    if insecure_key_subprotocol_prefix is not None:
        for subprotocol in websocket.scope.get("subprotocols") or []:
            if subprotocol.startswith(insecure_key_subprotocol_prefix):
                key = subprotocol[len(insecure_key_subprotocol_prefix) :]
                if key:
                    return key
    query_key = websocket.query_params.get("api_key")
    if query_key:
        return query_key.strip() or None
    return None


async def authenticate_ws(
    websocket: Any, *, insecure_key_subprotocol_prefix: str | None = None
) -> tuple[RequestIdentity, dict[str, Any]] | None:
    """Authenticate a WebSocket connection with a proxy API key.

    Returns ``(identity, auth_info)`` on success; ``auth_info`` carries the
    key's ``allowed_models`` (model-restriction check) and budget
    configuration (spending-cap check). ``insecure_key_subprotocol_prefix``
    is forwarded to :func:`extract_ws_api_key`. The import is deferred to
    avoid a circular import with the middleware package.
    """
    from llm_proxy.api.middleware.mcp_proxy import verify_api_key_for_mcp

    api_key = extract_ws_api_key(
        websocket, insecure_key_subprotocol_prefix=insecure_key_subprotocol_prefix
    )
    if not api_key:
        return None
    auth_info = await verify_api_key_for_mcp(api_key)
    if auth_info is None:
        return None
    identity = RequestIdentity(
        api_key_name=auth_info["principal_id"],
        auth_method="api_key",
        user_id=auth_info.get("user_id"),
    )
    return identity, auth_info


def build_ws_request(websocket: Any, identity: RequestIdentity) -> StarletteRequest:
    """Build an HTTP Request from a WebSocket scope for shared accessors.

    Config/dependency accessors expect an HTTP request; this fakes one from
    the websocket scope (a GET on the same path), stamps the authenticated
    identity and a fresh connection request id so downstream code (budget
    checks, model resolution, usage logging) works unchanged.
    """
    scope = dict(websocket.scope)
    scope["type"] = "http"
    scope["method"] = "GET"
    scope.setdefault("http_version", "1.1")
    request = StarletteRequest(scope)
    set_request_identity(request, identity)
    request.state.api_key_name = identity.api_key_name
    request.state.request_id = f"ws_{uuid.uuid4().hex[:24]}"
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
    "authenticate_ws",
    "build_ws_request",
    "extract_ws_api_key",
    "receive_with_connection_cap",
]
