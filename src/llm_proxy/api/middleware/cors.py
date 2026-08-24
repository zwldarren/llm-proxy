"""CORS middleware with hot-reloadable, UI-managed allowed origins.

Unlike starlette's ``CORSMiddleware`` (which requires a static origin list at
startup), this middleware reads the allowed origins from the config manager's
cached ``ProxyConfig`` on every request, so changes made in the admin UI
(server_config ``cors_origins`` key) apply immediately without a restart.

When no origins are configured, no CORS headers are emitted at all — matching
the previous behaviour of not registering the middleware (CORS is only needed
when the admin frontend is served from a different origin).
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.responses import PlainTextResponse

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
_PREFLIGHT_MAX_AGE_SECONDS = 600


def _resolve_allowed_origins(request: Request) -> list[str]:
    """Read the current allowed origins from the cached ProxyConfig."""
    from llm_proxy.config.types import ProxyConfig

    config_manager = getattr(request.app.state, "config_manager", None)
    cached = config_manager.get_cached_config() if config_manager is not None else None
    if isinstance(cached, ProxyConfig):
        return cached.server_params.cors_origins
    return []


async def cors_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Handle CORS preflight and simple requests against the configured origins."""
    allowed = _resolve_allowed_origins(request)
    origin = request.headers.get("origin")

    if not allowed or not origin:
        return await call_next(request)

    # Preflight request
    if request.method == "OPTIONS" and "access-control-request-method" in request.headers:
        if origin not in allowed:
            return PlainTextResponse("Disallowed CORS origin", status_code=400)
        requested_method = request.headers["access-control-request-method"].upper()
        if requested_method not in _ALLOWED_METHODS:
            return PlainTextResponse("Disallowed CORS method", status_code=400)
        return PlainTextResponse(
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

    # Simple / actual request
    response = await call_next(request)
    if origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        vary = response.headers.get("Vary")
        if not vary:
            response.headers["Vary"] = "Origin"
        elif "origin" not in {v.strip().lower() for v in vary.split(",")}:
            response.headers["Vary"] = f"{vary}, Origin"
    return response
