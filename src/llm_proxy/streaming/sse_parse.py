"""Spec-tolerant SSE parsing for raw-upstream stream consumers.

The SSE spec defines a field's value as everything after the first colon
with at most one leading space stripped — so ``data:{...}`` and
``data: {...}`` (and likewise ``event:x`` / ``event: x``) are equivalent.
Upstreams disagree on the spelling: Kimi Code's Anthropic Messages endpoint
sends ``event:message_delta`` / ``data:{...}`` without the space.

Every parser that consumes **raw upstream** SSE frames must use these
helpers. Consumers that parse the proxy's *own* re-encoded SSE are exempt —
the proxy always emits the spaced form (``SSEBuilder``, transformer
``_sse_event``).
"""

from collections.abc import Iterator
from typing import Any

import orjson
from orjson import JSONDecodeError


def split_sse_field(line: str) -> tuple[str, str]:
    """Split one SSE line into ``(field, value)`` per the SSE spec.

    The value is everything after the first colon with at most one leading
    space stripped, so ``event: x`` and ``event:x`` parse identically.
    """
    field, _, value = line.partition(":")
    if value.startswith(" "):
        value = value[1:]
    return field, value


def parse_sse_data_line(line: str) -> str | None:
    """Return the payload of an SSE ``data:`` line, or ``None``.

    Accepts both ``data:{...}`` and ``data: {...}`` spellings; returns
    ``None`` for non-data lines and for empty payloads. Surrounding
    whitespace is stripped — JSON consumers tolerate it and the ``[DONE]``
    comparison stays robust.
    """
    field, value = split_sse_field(line)
    if field != "data":
        return None
    return value.strip() or None


def strip_sse_data_prefix(line: str) -> str:
    """Return the payload when ``line`` is an SSE ``data:`` line, else the line unchanged.

    Convenience for consumers that treat a data line's payload as the line
    itself (``data: {...}`` and ``data:{...}`` both become ``{...}``), while
    non-data lines (``event:``, comments, blanks) pass through untouched.
    """
    return parse_sse_data_line(line) or line


def contains_sse_event(chunk: str, event_name: str) -> bool:
    """True when a raw frame contains an ``event:`` line naming ``event_name``.

    Accepts both spellings (``event: x`` and ``event:x``) per the SSE spec.
    A substring pre-filter for hot paths — full event/data pairing lives in
    :func:`iter_sse_data_events`.
    """
    return f"event: {event_name}" in chunk or f"event:{event_name}" in chunk


def iter_sse_data_events(chunk: str) -> Iterator[tuple[str | None, Any]]:
    """Yield ``(event_type, payload)`` for each SSE data line in a frame.

    Field parsing follows the SSE spec: the value is everything after the
    first colon with at most one leading space stripped — so ``event: x`` /
    ``data: {...}`` and no-space frames as sent by e.g. Kimi Code
    (``event:message_delta`` / ``data:{...}``) parse identically. A malformed
    data line keeps the pending event type for the next data line; a
    successfully consumed data line resets it.
    """
    event_type: str | None = None
    for line in chunk.split("\n"):
        line = line.strip()
        if not line:
            continue
        field, value = split_sse_field(line)
        if field == "event":
            event_type = value or None
        elif field == "data":
            try:
                yield event_type, orjson.loads(value)
            except JSONDecodeError:
                continue
            event_type = None


__all__ = [
    "contains_sse_event",
    "iter_sse_data_events",
    "parse_sse_data_line",
    "split_sse_field",
    "strip_sse_data_prefix",
]
