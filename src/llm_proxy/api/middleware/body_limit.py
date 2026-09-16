"""Request body size limit middleware.

Prevents memory exhaustion from unbounded request bodies. A trustworthy
``Content-Length`` is checked from the header before the body is read; a body
that does not declare one (chunked transfer encoding, or a compressed body
whose decompressed size is unknown) is measured as it streams and abandoned
the moment it crosses the cap, so the proxy never buffers an oversized body
just to reject it.
"""

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.api.middleware.asgi_utils import (
    BodyReader,
    BodyTooLargeError,
    CoreASGIMiddleware,
    adapt_http_middleware,
)
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


def _is_chunked(request: Request) -> bool:
    """Whether the request declares ``Transfer-Encoding: chunked``.

    A chunked body carries no trustworthy length — and per RFC 9112 a
    ``Content-Length`` sent alongside it must be ignored — so its size can only
    be enforced while it is read.
    """
    return "chunked" in request.headers.get("transfer-encoding", "").lower()


def _too_large_response(max_size: int) -> JSONResponse:
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


async def _dispatch(request: Request, body: BodyReader) -> JSONResponse | None:
    """Reject a request body that exceeds the configured limit.

    A trustworthy ``Content-Length`` is decided from the header alone, without
    touching the body. Otherwise — chunked transfer encoding, HTTP/2 without a
    declared length, or a body whose decompressed size is unknown — the bytes
    are measured as they arrive, so an oversized stream is abandoned mid-read
    rather than buffered in full.

    The limit is UI-managed (server_config ``security`` key) and hot-reloaded.
    A value of 0 disables the limit.
    """
    from llm_proxy.config.manager import resolve_security_params

    max_size = resolve_security_params(
        getattr(request.app.state, "config_manager", None)
    ).max_request_body_size_bytes
    if max_size <= 0:
        return None

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

    if content_length is not None and not _is_chunked(request):
        if content_length > max_size:
            logger.warning(
                "Request body too large",
                content_length=content_length,
                max_size=max_size,
                method=request.method,
                path=request.url.path,
            )
            return _too_large_response(max_size)
        return None

    # No trustworthy declared length: measure the bytes as they stream.
    try:
        await body.read_limited(max_size)
    except BodyTooLargeError:
        logger.warning(
            "Request body too large",
            max_size=max_size,
            method=request.method,
            path=request.url.path,
        )
        return _too_large_response(max_size)
    return None


class BodySizeLimitMiddleware(CoreASGIMiddleware):
    """Pure-ASGI middleware enforcing the configured request body limit."""

    async def dispatch(self, request: Request, body: BodyReader) -> JSONResponse | None:
        return await _dispatch(request, body)


#: ``(request, call_next)`` adapter kept for existing call sites and tests.
body_size_limit_middleware = adapt_http_middleware(_dispatch)
