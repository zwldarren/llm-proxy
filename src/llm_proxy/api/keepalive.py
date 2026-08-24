"""Keepalive (heartbeat) wrapper for slow non-streaming JSON responses.

CDN proxies such as Cloudflare (free/pro plans) abort requests whose origin
produces no bytes for ~100 seconds (error 524). Slow non-streaming LLM
requests — long-reasoning models with low token throughput — can exceed
that budget because the proxy only sends the response once the provider has
finished generating everything.

When enabled via ``NON_STREAM_KEEPALIVE_ENABLED``, this module wraps the
request-processing coroutine:

1. **Grace period** — the request is awaited normally. If it completes (or
   fails) within ``grace_seconds``, the original response is returned
   untouched: status codes, headers, and bodies are all preserved.
2. **Heartbeat mode** — once the grace period elapses, a ``200`` response
   with ``application/json`` is started immediately (satisfying the CDN's
   time-to-first-byte budget), and single-space bytes are emitted every
   ``interval_seconds``. A leading run of spaces is insignificant whitespace
   per RFC 8259, so standard JSON parsers (and every major LLM SDK) accept
   the final body unchanged.
3. **Completion** — when the processing task finishes, its response body is
   emitted and the stream ends.

Trade-offs of heartbeat mode (only reachable for requests slower than the
grace period):

* The status code is committed to ``200`` early. If processing fails after
  the switch, the error is delivered as a ``200`` response whose body is the
  proxy's usual error JSON instead of a 4xx/5xx status.
* Response headers from the inner response (e.g. usage hints) cannot be
  propagated, because headers are sent before processing finishes.

Client disconnects cancel the in-flight provider request in both phases.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable

import orjson
from fastapi import Response
from fastapi.responses import StreamingResponse

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

# A single space: insignificant whitespace before the JSON value (RFC 8259).
_HEARTBEAT_BYTE = b" "

# Protocols whose success payload is binary (e.g. synthesized audio) must
# never enter heartbeat mode — stray bytes would corrupt the payload.
_BINARY_PROTOCOLS: frozenset[str] = frozenset({"speech"})


def supports_keepalive(protocol_name: str, is_stream: bool) -> bool:
    """Return whether a request is eligible for the non-streaming keepalive path."""
    return not is_stream and protocol_name not in _BINARY_PROTOCOLS


def _error_body() -> bytes:
    """Build the sanitized error payload used when processing fails in heartbeat mode."""
    return orjson.dumps(
        {
            "error": {
                "message": "Internal server error",
                "type": "api_error",
                "code": "internal_error",
            }
        }
    )


def _extract_body(response: Response) -> bytes:
    """Extract the rendered body bytes from a completed response."""
    body = getattr(response, "body", None)
    if isinstance(body, (bytes, bytearray)):
        return bytes(body)
    raise TypeError(
        f"Expected a fully-rendered response, got {type(response).__name__}; "
        "heartbeat mode only supports non-streaming JSON responses"
    )


async def _heartbeat_body(
    task: asyncio.Task[Response], interval_seconds: float
) -> AsyncIterator[bytes]:
    """Yield heartbeat whitespace until the processing task completes, then its body."""
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=interval_seconds)
            if done:
                break
            yield _HEARTBEAT_BYTE

        yield _extract_body(task.result())
    except asyncio.CancelledError:
        # Client disconnected while we were heartbeating: stop the upstream
        # request as well so we don't keep paying for an abandoned generation.
        if not task.done():
            task.cancel()
        raise
    except Exception as exc:
        # Status 200 is already committed; deliver a sanitized error body.
        logger.error(f"Request processing failed after keepalive switch: {exc}", exc_info=exc)
        yield _error_body()


async def await_with_keepalive(
    coro: Awaitable[Response],
    *,
    grace_seconds: float,
    interval_seconds: float,
) -> Response:
    """Await a response-producing coroutine, switching to heartbeat mode if slow.

    Args:
        coro: Coroutine producing the final (fully rendered) response.
        grace_seconds: How long to wait for normal completion before
            committing to a 200 + heartbeat stream.
        interval_seconds: Heartbeat interval once in heartbeat mode.

    Returns:
        The original response when it completes within the grace period,
        otherwise a StreamingResponse that heartbeats until completion.
    """
    task = asyncio.ensure_future(coro)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=grace_seconds)
    except TimeoutError:
        # The shielded task keeps running; hand it to the heartbeat stream.
        logger.debug(
            f"Non-streaming request exceeded {grace_seconds}s grace period; "
            "switching to keepalive heartbeat mode"
        )
    except BaseException:
        # Real error or caller cancellation: don't leave the provider
        # request running unattended.
        if not task.done():
            task.cancel()
        raise

    return StreamingResponse(
        _heartbeat_body(task, interval_seconds),
        status_code=200,
        media_type="application/json",
    )


__all__ = ["await_with_keepalive", "supports_keepalive"]
