"""Read limits shared by the two outbound HTTP backends.

Both backends buffer upstream responses, so both need the same guard rails: a
cap on a buffered body (``HTTP_MAX_RESPONSE_BYTES``) and a cap on one line of a
stream. Duplicating the numbers would let the two drift apart — the httpx2
backend is the fallback, not a different policy — so the logic lives here and
each backend only decides how to read its transport.

The line cap matters because an SSE parser holds the bytes of an unterminated
line until the newline arrives: an endpoint that never sends one would grow
that buffer until the worker died.
"""

from typing import Final

from llm_proxy.config.settings import get_settings
from llm_proxy.http.errors import UpstreamResponseTooLarge

#: Longest single line a streamed upstream response may carry. One SSE event or
#: NDJSON record bigger than this is a broken upstream, not a payload; the cap
#: sits comfortably above the largest real event (an inline image chunk) while
#: keeping the read buffer bounded.
MAX_STREAM_LINE_BYTES: Final = 8 * 1024 * 1024


def max_response_bytes() -> int:
    """Largest upstream body the proxy will buffer (HTTP_MAX_RESPONSE_BYTES)."""
    return get_settings().http.max_response_bytes


async def read_capped(response, limit: int, url: str) -> bytes:
    """Read a response body, refusing to buffer more than ``limit`` bytes.

    ``response`` is any object exposing ``aiter_bytes()`` — either backend's
    response, or the raw httpx2 response.
    """
    total = 0
    chunks: list[bytes] = []
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > limit:
            raise UpstreamResponseTooLarge(
                f"Upstream response from '{url}' is larger than the {limit}-byte "
                "limit (HTTP_MAX_RESPONSE_BYTES)"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def split_lines(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Split complete lines off ``buffer`` on CR, LF or CRLF.

    Returns the lines and the unterminated tail, mirroring the SSE rule rather
    than ``bytes.splitlines`` so the caller can length-check the tail before it
    grows. A CR at the very end is kept: its LF may arrive in the next chunk.
    """
    lines: list[bytes] = []
    start = 0
    index = 0
    length = len(buffer)
    while index < length:
        byte = buffer[index]
        if byte == 0x0A:
            lines.append(buffer[start:index])
            index += 1
            start = index
        elif byte == 0x0D:
            if index + 1 == length:
                break
            lines.append(buffer[start:index])
            index += 2 if buffer[index + 1] == 0x0A else 1
            start = index
        else:
            index += 1
    return lines, buffer[start:]


def flush_lines(pending: bytes) -> list[bytes]:
    """The last line of a stream: strip a terminating CR, keep an empty line.

    ``split_lines`` holds a trailing CR because its LF may still arrive; at end
    of stream that CR was a terminator after all.
    """
    if not pending:
        return []
    if pending.endswith(b"\r"):
        pending = pending[:-1]
    return [pending]


def check_line_length(line: bytes, url: str) -> None:
    """Reject a stream line that would have to be buffered beyond the cap."""
    if len(line) > MAX_STREAM_LINE_BYTES:
        raise UpstreamResponseTooLarge(
            f"Upstream stream from '{url}' has a line longer than {MAX_STREAM_LINE_BYTES} bytes"
        )


__all__ = [
    "MAX_STREAM_LINE_BYTES",
    "check_line_length",
    "flush_lines",
    "max_response_bytes",
    "read_capped",
    "split_lines",
]
