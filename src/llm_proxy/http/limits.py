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

import re
from typing import Final

from llm_proxy.config.settings import get_settings
from llm_proxy.http.errors import UpstreamResponseTooLarge

#: Longest single line a streamed upstream response may carry. One SSE event or
#: NDJSON record bigger than this is a broken upstream, not a payload; the cap
#: sits comfortably above the largest real event (an inline image chunk) while
#: keeping the read buffer bounded.
MAX_STREAM_LINE_BYTES: Final = 8 * 1024 * 1024

#: Line terminators accepted in a stream (CRLF is matched before CR/LF).
_LINE_BREAK_RE: Final = re.compile(rb"\r\n|\r|\n")


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

    The scan runs in the regex engine (C) rather than byte-by-byte in Python;
    this function runs once per streamed chunk on the request path.
    """
    lines: list[bytes] = []
    start = 0
    length = len(buffer)
    for match in _LINE_BREAK_RE.finditer(buffer):
        if match.end() == length and match.group() == b"\r":
            # A trailing CR may be the first half of a CRLF crossing chunks.
            break
        lines.append(buffer[start : match.start()])
        start = match.end()
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
