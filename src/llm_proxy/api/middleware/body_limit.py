"""Request body size limit middleware.

Prevents memory exhaustion from unbounded request bodies by rejecting
requests whose Content-Length exceeds the configured maximum before the
body is read into memory.
"""

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


def _get_content_length(request: Request) -> int | None:
    """Parse Content-Length header, returning None if absent or invalid."""
    value = request.headers.get("content-length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


async def body_size_limit_middleware(request: Request, call_next):
    """Reject requests whose declared body size exceeds the configured limit.

    The limit is UI-managed (server_config ``security`` key, default 10 MiB)
    and hot-reloaded. A value of 0 disables the limit. Requests with
    ``Transfer-Encoding: chunked`` are rejected when a body size limit is
    active, because a Content-Length-based check alone is bypassable.
    """
    from llm_proxy.config.manager import resolve_security_params

    max_size = resolve_security_params(
        getattr(request.app.state, "config_manager", None)
    ).max_request_body_size_bytes
    if max_size <= 0:
        return await call_next(request)

    # Reject chunked transfer-encoding when body limit is active
    # (Content-Length based check alone is bypassable via chunked encoding).
    if request.headers.get("transfer-encoding", "").lower() == "chunked":
        logger.warning(
            "Chunked transfer encoding rejected",
            method=request.method,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "message": "Chunked transfer encoding not allowed with body size limit",
                    "type": "request_too_large",
                    "code": "body_size_exceeded",
                }
            },
        )

    content_length = _get_content_length(request)
    if content_length is not None and content_length < 0:
        logger.warning(
            "Invalid negative Content-Length",
            method=request.method,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Invalid Content-Length header",
                    "type": "bad_request",
                    "code": "invalid_content_length",
                }
            },
        )

    if content_length is not None and content_length > max_size:
        logger.warning(
            "Request body too large",
            content_length=content_length,
            max_size=max_size,
            method=request.method,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "message": f"Request body exceeds maximum size of {max_size} bytes",
                    "type": "request_too_large",
                    "code": "body_size_exceeded",
                }
            },
        )

    return await call_next(request)
