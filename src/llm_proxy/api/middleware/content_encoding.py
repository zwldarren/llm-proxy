"""Middleware decompressing Content-Encoding request bodies.

Codex clients (especially Desktop with login state) may compress request
bodies (zstd/gzip/br/deflate, possibly stacked). Decode the body before the
JSON parser runs, strip the now-inaccurate entity headers
(content-encoding / content-length / transfer-encoding), and reject
unsupported encodings with a 400 ``invalid_request`` error instead of
letting the downstream parser fail with a confusing 422.

Decompression output is capped at the configured request body limit (the
same value ``body_size_limit_middleware`` enforces) as a zip-bomb guard;
overflow is rejected with 413 in the same envelope as the size limiter.

Registered to run before ``body_size_limit_middleware`` so the size check
applies to the decompressed body, and before ``form_encoded_middleware``
so a compressed form-encoded body is still converted.
"""

import zlib

from starlette.responses import JSONResponse

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

_IDENTITY = "identity"

# Fallback cap when the configured body limit is disabled (0); still guards
# against zip bombs.
_MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024  # 64 MiB

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


_DECODERS = {
    "gzip": _decode_gzip,
    "x-gzip": _decode_gzip,
    "deflate": _decode_deflate,
    "br": _decode_brotli,
    "zstd": _decode_zstd,
    "zst": _decode_zstd,
}


def _error_response(status: int, message: str, type_: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": code}},
    )


async def content_encoding_middleware(request, call_next):
    """Decompress request bodies carrying a Content-Encoding header.

    Stacked encodings (e.g. ``gzip, zstd``) are listed in application order
    per RFC 9110 and are decoded in reverse.
    """
    if request.method not in ("POST", "PUT", "PATCH"):
        return await call_next(request)
    header = request.headers.get("content-encoding")
    if not header or header.lower().strip() == _IDENTITY:
        return await call_next(request)

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
        getattr(request.app.state, "config_manager", None)
    ).max_request_body_size_bytes
    cap = max_size if max_size > 0 else _MAX_DECOMPRESSED_BYTES

    try:
        body = await request.body()
    except Exception:
        return _error_response(
            400, "Failed to read request body.", "invalid_request", "invalid_body"
        )
    if len(body) > cap:
        return _error_response(
            413,
            f"Request body exceeds maximum size of {cap} bytes",
            "request_too_large",
            "body_size_exceeded",
        )

    data = body
    try:
        for encoding in reversed(encodings):
            if encoding == _IDENTITY:
                continue
            data = _DECODERS[encoding](data, cap)
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
            path=request.url.path,
            error=str(exc),
        )
        return _error_response(
            400,
            f"Failed to decompress request body ({', '.join(encodings)}): {exc}",
            "invalid_request",
            "invalid_content_encoding",
        )

    logger.debug(
        "Decompressed request body",
        encodings=",".join(encodings),
        compressed=len(body),
        decompressed=len(data),
        path=request.url.path,
    )

    # Replace the cached body in place (BaseHTTPMiddleware replays it to the
    # downstream app) and rewrite the entity headers to match the plain body.
    request._body = data
    request.scope["headers"] = [
        (b"content-length", str(len(data)).encode()),
        *[h for h in request.scope.get("headers", []) if h[0].lower() not in _STRIPPED_HEADERS],
    ]
    return await call_next(request)
