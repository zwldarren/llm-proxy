"""Aggregate Responses API SSE streams into a single JSON response body.

Some upstreams (notably Codex OAuth gateways, and gateways with SSE-related
bugs) answer a non-streaming ``stream: false`` request with an SSE body —
sometimes even with a JSON Content-Type, so header-based detection misses it.
When ``response.json()`` fails, the caller can sniff the body and, if it looks
like SSE, aggregate the stream here into the response object the client should
have received.
"""

import re
from typing import Any

import orjson

from llm_proxy.core.exceptions import ProviderError

_SSE_LINE_PREFIXES = ("data:", "event:", "id:", "retry:", ":")


def body_looks_like_sse(text: str) -> bool:
    """Sniff whether a body is an SSE stream regardless of Content-Type."""
    for line in text.lstrip().splitlines():
        if not line.strip():
            continue
        return line.startswith(_SSE_LINE_PREFIXES)
    return False


def aggregate_sse_to_response_object(text: str) -> dict[str, Any]:
    """Aggregate a Responses SSE stream into a single response JSON object.

    Takes the terminal ``response.completed`` / ``response.incomplete`` event's
    ``response`` as the base object. When the stream carried
    ``response.output_item.done`` events, their items replace the base object's
    ``output`` (some upstreams only include partial output in the terminal
    event; the per-item terminal events are canonical). Raises ProviderError
    on ``response.failed`` or when no terminal event is present.
    """
    output_items: list[Any] = []
    final_response: dict[str, Any] | None = None

    # SSE blocks are separated by a blank line; tolerate both LF and CRLF
    # framing. The event type is read from the data payload's ``type`` field,
    # falling back to the ``event:`` line (mirrors the adapter's streaming
    # parser; some upstreams omit ``type``).
    for block in re.split(r"\r?\n\r?\n", text):
        event_line: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_line = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            continue
        data_str = "\n".join(data_lines).strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            event = orjson.loads(data_str)
        except orjson.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type") or event_line
        if event_type == "response.output_item.done":
            item = event.get("item")
            if item is not None:
                output_items.append(item)
        elif event_type in ("response.completed", "response.incomplete"):
            resp = event.get("response")
            if isinstance(resp, dict):
                final_response = resp
        elif event_type == "response.failed":
            resp = event.get("response")
            error = resp.get("error") if isinstance(resp, dict) else None
            error = error if isinstance(error, dict) else {}
            raise ProviderError(
                message=error.get("message") or "Upstream Responses stream failed",
                error_type=error.get("type") or "api_error",
                status_code=502,
            )

    if final_response is None:
        raise ProviderError(
            message="Failed to aggregate upstream SSE response: no response.completed event",
            error_type="api_error",
            status_code=502,
        )
    if output_items:
        final_response["output"] = output_items
    return final_response


def parse_json_or_sse(response: Any) -> Any:
    """Parse a non-streaming upstream response body, with an SSE fallback.

    First tries ``response.json()``. On a JSON decode failure, sniffs the raw
    body for SSE framing and aggregates it into a response object; re-raises
    the original decode error when the body does not look like SSE.
    """
    try:
        return response.json()
    except ValueError as decode_error:
        text = response.text
        if body_looks_like_sse(text):
            return aggregate_sse_to_response_object(text)
        raise decode_error


__all__ = [
    "body_looks_like_sse",
    "aggregate_sse_to_response_object",
    "parse_json_or_sse",
]
