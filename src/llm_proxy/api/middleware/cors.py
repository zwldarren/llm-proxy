"""CORS middleware with hot-reloadable, UI-managed allowed origins.

Unlike starlette's ``CORSMiddleware`` (which requires a static origin list at
startup), this middleware reads the allowed origins from the config manager's
cached ``ProxyConfig`` on every request, so changes made in the admin UI
(server_config ``cors_origins`` key) apply immediately without a restart.

When no origins are configured, no CORS headers are emitted at all — matching
the previous behaviour of not registering the middleware (CORS is only needed
when the admin frontend is served from a different origin).
"""

import re

from fastapi import Request
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_proxy.api.middleware.asgi_utils import (
    get_response_header,
    merge_response_headers,
)
from llm_proxy.observability.logger import get_logger
from llm_proxy.services import runtime_services

logger = get_logger(__name__)

_ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
#: Headers answered on a preflight that names none. A browser preflight always
#: carries ``Access-Control-Request-Headers`` when the request sets any header
#: beyond the safelisted set, so this list is the fallback, not the boundary:
#: :func:`_allowed_headers` echoes whatever else the client actually asks for.
_ALLOWED_HEADERS = [
    "Authorization",
    "Content-Type",
    "X-Request-ID",
    "X-Trace-ID",
    "X-Langfuse-Trace-ID",
    "X-CSRF-Token",
    "Accept",
    "Accept-Language",
    "Accept-Encoding",
    # The proxy's second credential form (``_extract_key`` reads either this
    # or ``Authorization``), so a browser client can authenticate without
    # synthesizing a Bearer header.
    "X-API-Key",
]

# Response headers browser JS is allowed to read on cross-origin requests.
# Trace ids are how a client correlates its own request with the console's log
# detail view, so they must be readable from the response. ``Retry-After`` is
# what a cross-origin client paces its retries with; without the exposure it
# is sent but unreadable.
_EXPOSED_HEADERS = ["X-Trace-ID", "X-Langfuse-Trace-ID", "X-Request-Id", "Retry-After"]
_PREFLIGHT_MAX_AGE_SECONDS = 600

#: RFC 9110 ``field-name``: what may appear in a reflected allow-headers list.
#: A requested value that is not one is dropped, never echoed into a response
#: header — the preflight still succeeds for the names that are valid.
_FIELD_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
#: Bounds on the reflected list: a preflight is unauthenticated and cheap, so a
#: hostile one must not be able to inflate the response without limit.
_MAX_REFLECTED_HEADERS = 64
_MAX_FIELD_NAME_LENGTH = 64


def _resolve_allowed_origins(request: Request) -> list[str]:
    """Read the current allowed origins from the cached ProxyConfig."""
    from llm_proxy.config.types import ProxyConfig

    config_manager = runtime_services(request).config_manager_or_none()
    cached = config_manager.get_cached_config() if config_manager is not None else None
    if isinstance(cached, ProxyConfig):
        return cached.server_params.cors_origins
    return []


def _has_origin_header(scope: Scope) -> bool:
    """Whether the raw ASGI headers carry an ``Origin`` (lowercased name)."""
    return any(key == b"origin" for key, _ in scope.get("headers", ()))


def _requested_headers(request: Request) -> list[str] | None:
    """The header names a preflight asks to send, in request order.

    Returns ``None`` when the preflight names no headers (the caller then
    answers with the built-in list) and an empty list when it names only
    invalid names — an empty answer no browser will accept, which is the right
    refusal. Names are matched case-insensitively on the built-in list so a
    client that spells ``x-api-key`` gets the canonical ``X-API-Key`` back.
    """
    raw = request.headers.get("access-control-request-headers")
    if raw is None or not raw.strip():
        # No header named: the built-in list is the answer. An empty value is
        # the same statement as an absent one.
        return None
    canonical = {name.lower(): name for name in _ALLOWED_HEADERS}
    names: list[str] = []
    for candidate in raw.split(","):
        name = candidate.strip()
        if not name or len(name) > _MAX_FIELD_NAME_LENGTH:
            continue
        if _FIELD_NAME.fullmatch(name) is None:
            continue
        names.append(canonical.get(name.lower(), name))
        if len(names) >= _MAX_REFLECTED_HEADERS:
            break
    return names


def _allowed_headers(request: Request) -> list[str]:
    """The ``Access-Control-Allow-Headers`` value for one preflight.

    The API's header surface is defined by the client's protocol, not by this
    middleware: SDKs send ``x-api-key``, ``anthropic-beta``, ``x-stainless-*``
    and provider-specific identity headers, and a fixed list rejects every one
    of them at the preflight — before auth runs, so the failure is a generic
    CORS error with no log row. Echoing the requested names (validated as
    field names, bounded, and only for an origin already allowed) keeps the
    allowlist from silently drifting behind the protocol surface.
    """
    requested = _requested_headers(request)
    if requested is None:
        return list(_ALLOWED_HEADERS)
    return requested


class CORSMiddleware:
    """Pure-ASGI CORS handling against the configured origins."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Fast path: without an Origin header the request can never need CORS
        # handling, so skip building a Request and resolving the config.
        # Non-browser clients (SDKs) never send Origin, so this is the common
        # path on the proxy's API traffic.
        if not _has_origin_header(scope):
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        allowed = _resolve_allowed_origins(request)
        origin = request.headers.get("origin")

        if not allowed or not origin:
            await self.app(scope, receive, send)
            return

        # Preflight request
        if request.method == "OPTIONS" and "access-control-request-method" in request.headers:
            if origin not in allowed:
                response = PlainTextResponse("Disallowed CORS origin", status_code=400)
            elif request.headers["access-control-request-method"].upper() not in _ALLOWED_METHODS:
                response = PlainTextResponse("Disallowed CORS method", status_code=400)
            else:
                response = PlainTextResponse(
                    "OK",
                    headers={
                        "Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Credentials": "true",
                        "Access-Control-Allow-Methods": ", ".join(_ALLOWED_METHODS),
                        "Access-Control-Allow-Headers": ", ".join(_allowed_headers(request)),
                        "Access-Control-Max-Age": str(_PREFLIGHT_MAX_AGE_SECONDS),
                        # The allow-headers answer now depends on the request,
                        # so caches must key on it too.
                        "Vary": "Origin, Access-Control-Request-Headers",
                    },
                )
            await response(scope, receive, send)
            return

        # Simple / actual request
        if origin not in allowed:
            await self.app(scope, receive, send)
            return

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                vary = get_response_header(message, "vary")
                if not vary:
                    vary = "Origin"
                elif "origin" not in {v.strip().lower() for v in vary.split(",")}:
                    vary = f"{vary}, Origin"
                merge_response_headers(
                    message,
                    {
                        "Access-Control-Allow-Origin": origin,
                        "Access-Control-Allow-Credentials": "true",
                        "Access-Control-Expose-Headers": ", ".join(_EXPOSED_HEADERS),
                        "Vary": vary,
                    },
                )
            await send(message)

        await self.app(scope, receive, send_with_cors)
