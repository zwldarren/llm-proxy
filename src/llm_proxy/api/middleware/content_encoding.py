"""Middleware decompressing Content-Encoding request bodies.

Codex clients (especially Desktop with login state) may compress request
bodies (zstd/gzip/br/deflate, possibly stacked). Decode the body before the
JSON parser runs, strip the now-inaccurate entity headers
(content-encoding / content-length / transfer-encoding), and reject
unsupported encodings with a 400 ``invalid_request`` error instead of
letting the downstream parser fail with a confusing 422.

Decompression output is capped at the configured request body limit (the
same value :class:`~llm_proxy.api.middleware.body_limit.BodySizeLimitMiddleware`
enforces) as a zip-bomb guard; overflow is rejected with 413 in the same
envelope as the size limiter. The compressed *input* is read with the same cap
(``read_limited``), so a stream that declares no trustworthy length — chunked,
or a highly compressed body — is abandoned mid-read instead of being buffered
in full first.

Registered to run before ``BodySizeLimitMiddleware`` so the size check
applies to the decompressed body, and before ``FormEncodedMiddleware``
so a compressed form-encoded body is still converted.
"""

import asyncio
import zlib
from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import Scope

from llm_proxy.api.middleware.asgi_utils import (
    BodyReader,
    BodyTooLargeError,
    CoreASGIMiddleware,
    adapt_http_middleware,
    get_scope_header,
)
from llm_proxy.observability.logger import get_logger
from llm_proxy.services import runtime_services

logger = get_logger(__name__)

_IDENTITY = "identity"

# Fallback cap when the configured body limit is disabled (0); still guards
# against zip bombs.
_MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024  # 64 MiB

# The decoder runs before authentication (the auth gates only look at
# headers), so an unauthenticated client can otherwise start arbitrarily many
# simultaneous decompressions. A small per-worker admission limit bounds peak
# memory without rejecting legitimate requests: excess requests wait for a
# slot, they are not refused.
_MAX_CONCURRENT_DECOMPRESSIONS = 4
_decompress_limiter: asyncio.Semaphore | None = None
_decompress_limiter_loop: asyncio.AbstractEventLoop | None = None


def _get_decompress_limiter() -> asyncio.Semaphore:
    """Return the per-event-loop decompression semaphore."""
    global _decompress_limiter, _decompress_limiter_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _decompress_limiter is None or _decompress_limiter_loop is not loop:
        _decompress_limiter = asyncio.Semaphore(_MAX_CONCURRENT_DECOMPRESSIONS)
        _decompress_limiter_loop = loop
    return _decompress_limiter


# Entity headers made inaccurate by decompression; stripped from the scope so
# downstream (and any upstream forwarding) regenerates them from the plain body.
_STRIPPED_HEADERS = (b"content-encoding", b"content-length", b"transfer-encoding")


class _UnsupportedEncodingError(Exception):
    """The request declares a content-encoding the proxy cannot decode."""


class _DecompressionError(Exception):
    """The compressed body is corrupt or truncated."""


class _DecompressedTooLargeError(Exception):
    """The decompressed body exceeds the configured size cap."""


def _decode_zlib(data: bytes, max_output: int, wbits: int) -> bytes:
    """Decode zlib-wrapped data with an output cap (raises on overflow)."""
    try:
        obj = zlib.decompressobj(wbits=wbits)
        out = obj.decompress(data, max_output + 1)
        out += obj.flush()
    except zlib.error as exc:
        raise _DecompressionError(str(exc)) from exc
    if len(out) > max_output or obj.unconsumed_tail:
        raise _DecompressedTooLargeError
    return out


def _decode_gzip(data: bytes, max_output: int) -> bytes:
    return _decode_zlib(data, max_output, wbits=31)


def _decode_deflate(data: bytes, max_output: int) -> bytes:
    """Decode deflate, accepting both zlib-wrapped and raw deflate streams."""
    try:
        return _decode_zlib(data, max_output, wbits=zlib.MAX_WBITS)
    except _DecompressionError:
        # Some clients send raw (headerless) deflate; retry with raw wbits.
        return _decode_zlib(data, max_output, wbits=-zlib.MAX_WBITS)


def _decode_brotli(data: bytes, max_output: int) -> bytes:
    try:
        import brotli
    except ImportError as exc:  # pragma: no cover - brotli ships with the app
        raise _UnsupportedEncodingError("br") from exc
    try:
        decompressor = brotli.Decompressor()
        out = bytearray()
        # Feed in chunks so a brotli bomb is rejected at the cap instead of
        # expanding fully in memory first.
        view = memoryview(data)
        chunk_size = 1024 * 1024
        for pos in range(0, len(view), chunk_size):
            out += decompressor.process(view[pos : pos + chunk_size])
            if len(out) > max_output:
                raise _DecompressedTooLargeError
        if not decompressor.is_finished():
            raise _DecompressionError("truncated brotli stream")
        return bytes(out)
    except _DecompressedTooLargeError, _DecompressionError:
        raise
    except Exception as exc:
        raise _DecompressionError(str(exc)) from exc


def _decode_zstd(data: bytes, max_output: int) -> bytes:
    try:
        from compression import zstd
    except ImportError as exc:  # pragma: no cover - Python 3.14+ always has it
        raise _UnsupportedEncodingError("zstd") from exc
    try:
        decompressor = zstd.ZstdDecompressor()
        out = decompressor.decompress(data, max_output + 1)
    except zstd.ZstdError as exc:
        raise _DecompressionError(str(exc)) from exc
    if len(out) > max_output:
        raise _DecompressedTooLargeError
    return out


_DECODERS: frozendict[str, Callable[[bytes, int], bytes]] = frozendict(
    {
        "gzip": _decode_gzip,
        "x-gzip": _decode_gzip,
        "deflate": _decode_deflate,
        "br": _decode_brotli,
        "zstd": _decode_zstd,
        "zst": _decode_zstd,
    }
)


def _error_response(status: int, message: str, type_: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": code}},
    )


async def _run_decompression(limiter: asyncio.Semaphore, decode: Callable[[], bytes]) -> bytes:
    """Run a native decoder off-loop while holding its admission permit.

    ``asyncio.to_thread`` cancellation cannot stop a decoder already running
    in its worker thread. Releasing the semaphore in a ``finally`` would let a
    new decoder start while that thread still holds its input/output buffers.
    Hand release to the executor future's completion callback instead, and
    shield the future so cancellation of this coroutine does not cancel the
    future (which would release the permit immediately).
    """
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(None, decode)
    except BaseException:
        limiter.release()
        raise
    future.add_done_callback(lambda _: limiter.release())
    return await asyncio.shield(future)


async def _dispatch(request: Request, body: BodyReader) -> JSONResponse | None:
    """Decompress request bodies carrying a Content-Encoding header.

    Stacked encodings (e.g. ``gzip, zstd``) are listed in application order
    per RFC 9110 and are decoded in reverse.
    """
    if request.method not in ("POST", "PUT", "PATCH"):
        return None
    header = request.headers.get("content-encoding")
    if not header or header.lower().strip() == _IDENTITY:
        return None

    encodings = [e.strip().lower() for e in header.split(",") if e.strip()]
    for encoding in encodings:
        if encoding != _IDENTITY and encoding not in _DECODERS:
            return _error_response(
                400,
                f"Unsupported request content-encoding: {encoding}",
                "invalid_request",
                "unsupported_content_encoding",
            )

    # Enforce the configured body limit on both the compressed input and the
    # decompressed output (zip-bomb guard). A limit of 0 disables the
    # configured cap and falls back to a fixed safety ceiling.
    from llm_proxy.config.manager import resolve_security_params

    max_size = resolve_security_params(
        runtime_services(request).config_manager_or_none()
    ).max_request_body_size_bytes
    cap = max_size if max_size > 0 else _MAX_DECOMPRESSED_BYTES

    needs_decode = any(encoding != _IDENTITY for encoding in encodings)
    limiter: asyncio.Semaphore | None = None
    if needs_decode:
        # Admit before reading: otherwise every queued request first buffers
        # up to ``cap`` compressed bytes while waiting for a decode slot.
        limiter = _get_decompress_limiter()
        await limiter.acquire()

    try:
        try:
            # Bounded read: the compressed stream is dropped as soon as it
            # crosses the cap rather than being materialised in full first.
            body_bytes = await body.read_limited(cap)
        except BodyTooLargeError:
            return _error_response(
                413,
                f"Request body exceeds maximum size of {cap} bytes",
                "request_too_large",
                "body_size_exceeded",
            )
        except Exception:
            return _error_response(
                400, "Failed to read request body.", "invalid_request", "invalid_body"
            )

        def _decode_all() -> bytes:
            current = body_bytes
            for encoding in reversed(encodings):
                if encoding == _IDENTITY:
                    continue
                current = _DECODERS[encoding](current, cap)
            return current

        try:
            if limiter is not None:
                # Ownership of the permit moves to the worker-completion
                # callback; do not release it in the outer finally.
                owned_limiter = limiter
                limiter = None
                data = await _run_decompression(owned_limiter, _decode_all)
            else:
                data = body_bytes
        except _DecompressedTooLargeError:
            return _error_response(
                413,
                f"Decompressed request body exceeds maximum size of {cap} bytes",
                "request_too_large",
                "body_size_exceeded",
            )
        except _DecompressionError as exc:
            logger.warning(
                "Request body decompression failed",
                encodings=",".join(encodings),
                path=request.scope["path"],
                error=str(exc),
            )
            return _error_response(
                400,
                f"Failed to decompress request body ({', '.join(encodings)}): {exc}",
                "invalid_request",
                "invalid_content_encoding",
            )
    finally:
        if limiter is not None:
            limiter.release()

    logger.debug(
        "Decompressed request body",
        encodings=",".join(encodings),
        compressed=len(body_bytes),
        decompressed=len(data),
        path=request.scope["path"],
    )

    # Replace the body and rewrite the entity headers to match the plain body.
    body.replace(data)
    request.scope["headers"] = [
        (b"content-length", str(len(data)).encode()),
        *[h for h in request.scope.get("headers", []) if h[0].lower() not in _STRIPPED_HEADERS],
    ]
    return None


class ContentEncodingMiddleware(CoreASGIMiddleware):
    """Pure-ASGI request body decompression."""

    def should_handle(self, scope: Scope) -> bool:
        if scope.get("method") not in ("POST", "PUT", "PATCH"):
            return False
        header = get_scope_header(scope, "content-encoding")
        return bool(header) and header.lower().strip() != _IDENTITY

    async def dispatch(self, request: Request, body: BodyReader) -> JSONResponse | None:
        return await _dispatch(request, body)


#: ``(request, call_next)`` adapter kept for existing call sites and tests.
content_encoding_middleware = adapt_http_middleware(_dispatch)
