"""CORS middleware with hot-reloadable, UI-managed allowed origins.

Unlike starlette's ``CORSMiddleware`` (which requires a static origin list at
startup), this middleware reads the allowed origins from the config manager's
cached ``ProxyConfig`` on every request, so changes made in the admin UI
(server_config ``cors_origins`` key) apply immediately without a restart.

When no origins are configured, no CORS headers are emitted at all — matching
the previous behaviour of not registering the middleware (CORS is only needed
when the admin frontend is served from a different origin).
"""

from fastapi import Request
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_proxy.api.middleware.asgi_utils import (
    get_response_header,
    merge_response_headers,
)
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

_ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
_ALLOWED_HEADERS = [
    "Authorization",
    "Content-Type",
    "X-Request-ID",
    "X-Trace-ID",
    "X-Langfuse-Trace-ID",
    "Accept",
    "Accept-Language",
    "Accept-Encoding",
]

# Response headers browser JS is allowed to read on cross-origin requests.
# Trace ids are how a client correlates its own request with the console's log
# detail view, so they must be readable from the response.
_EXPOSED_HEADERS = ["X-Trace-ID", "X-Langfuse-Trace-ID"]
_PREFLIGHT_MAX_AGE_SECONDS = 600


def _resolve_allowed_origins(request: Request) -> list[str]:
    """Read the current allowed origins from the cached ProxyConfig."""
    from llm_proxy.config.types import ProxyConfig

    config_manager = getattr(request.app.state, "config_manager", None)
    cached = config_manager.get_cached_config() if config_manager is not None else None
    if isinstance(cached, ProxyConfig):
        return cached.server_params.cors_origins
    return []


class CORSMiddleware:
    """Pure-ASGI CORS handling against the configured origins."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
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
                        "Access-Control-Allow-Headers": ", ".join(_ALLOWED_HEADERS),
                        "Access-Control-Max-Age": str(_PREFLIGHT_MAX_AGE_SECONDS),
                        "Vary": "Origin",
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
