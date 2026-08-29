"""Keepalive (heartbeat) wrapper for slow non-streaming JSON responses.

CDN proxies such as Cloudflare (free/pro plans) abort requests whose origin
produces no bytes for ~100 seconds (error 524). Slow non-streaming LLM
requests — long-reasoning models with low token throughput — can exceed
that budget because the proxy only sends the response once the provider has
finished generating everything.

When enabled (on by default, UI-tunable, hot-reloaded), this module wraps the
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

In both phases the client connection is polled for disconnects: a gone
client cancels the in-flight provider request (stop paying for an abandoned
generation) and the request is logged as failed (499) by the pipeline.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import suppress
from dataclasses import dataclass

import orjson
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

# A single space: insignificant whitespace before the JSON value (RFC 8259).
_HEARTBEAT_BYTE = b" "

# How often the disconnect monitor polls the client connection while a
# request is in flight. Disconnects are detected asynchronously (uvicorn does
# not cancel handler tasks on client exit), so polling is what turns "client
# gave up" into "stop generating + log it".
_DISCONNECT_POLL_INTERVAL = 1.0

# 499 response returned to an already-disconnected client. The client never
# reads it, but it terminates the ASGI cycle cleanly through the middleware
# chain (X-Request-Id, audit middleware finalization, ...).
_DISCONNECT_RESPONSE_PAYLOAD: dict = {
    "error": {
        "message": "Client disconnected before the response completed",
        "type": "client_disconnected",
        "code": "client_disconnected",
    }
}


# Protocols whose success payload is binary (e.g. synthesized audio) must
# never enter heartbeat mode — stray bytes would corrupt the payload.
_BINARY_PROTOCOLS: frozenset[str] = frozenset({"speech"})


def supports_keepalive(protocol_name: str, is_stream: bool) -> bool:
    """Return whether a request is eligible for the non-streaming keepalive path."""
    return not is_stream and protocol_name not in _BINARY_PROTOCOLS


# Sentinel-like outcomes of a single wait_for_either cycle. Distinct types
# so type checkers can narrow ``Response | WaitingDisconnected | Waiting``.
@dataclass(frozen=True)
class _Waiting:
    """Neither completion nor disconnect happened yet; call again."""


@dataclass(frozen=True)
class _Disconnected:
    """Client went away; the task was cancelled and the pipeline logged it."""


async def wait_for_either(
    task: asyncio.Task[Response],
    request: Request,
    *,
    poll_interval: float = _DISCONNECT_POLL_INTERVAL,
) -> Response | _Waiting | _Disconnected:
    """Run ONE wait cycle: race the pipeline task against a disconnect check.

    Returns the task's response when it completes in this cycle, ``_Waiting``
    when neither happened (call again), or ``_Disconnected`` when the client
    went away (the task was cancelled and the pipeline logged the abandonment
    — callers return a terminal response).

    The disconnect check runs concurrently with the wait deadline: it never
    delays a cycle longer than ``poll_interval``, so callers can treat that as
    the upper bound of the cycle granularity.
    """
    from llm_proxy.streaming.handler import check_client_disconnected

    # One connection-check per cycle; it races the task, so a pending check is
    # cancelled rather than allowed to stretch the cycle.
    checks = asyncio.ensure_future(check_client_disconnected(request))
    try:
        done, _ = await asyncio.wait(
            {task, checks}, timeout=poll_interval, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        if not checks.done():
            checks.cancel()
            with suppress(asyncio.CancelledError):
                await checks

    if task in done:
        return task.result()
    if checks in done and checks.result():
        request.state.client_disconnected = True
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        return _Disconnected()
    if task.done():
        # Completed between the wait and this check.
        return task.result()
    return _Waiting()


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
    request: Request | None = None,
) -> Response:
    """Await a response-producing coroutine, switching to heartbeat mode if slow.

    Args:
        coro: Coroutine producing the final (fully rendered) response.
        grace_seconds: How long to wait for normal completion before
            committing to a 200 + heartbeat stream.
        interval_seconds: Heartbeat interval once in heartbeat mode.
        request: Optional request used for client-disconnect polling while
            waiting. When the client disconnects, the processing task is
            cancelled and a terminal 499 response is returned (the client is
            gone; the log entry carries the failure).

    Returns:
        The original response when it completes within the grace period,
        otherwise a StreamingResponse that heartbeats until completion.
    """
    task = asyncio.ensure_future(coro)

    if request is not None:
        # Wait out the grace period, watching for both completion and client
        # disconnects (uvicorn does not cancel handler tasks by itself).
        loop = asyncio.get_running_loop()
        deadline = loop.time() + grace_seconds
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            outcome = await wait_for_either(
                task, request, poll_interval=min(remaining, _DISCONNECT_POLL_INTERVAL)
            )
            if isinstance(outcome, _Waiting):
                continue
            if isinstance(outcome, _Disconnected):
                # Client disconnected: terminal response; the pipeline logged it.
                return JSONResponse(status_code=499, content=_DISCONNECT_RESPONSE_PAYLOAD)
            return outcome
    else:
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=grace_seconds)
        except TimeoutError:
            pass
        except BaseException:
            # Real error or caller cancellation: don't leave the provider
            # request running unattended.
            if not task.done():
                task.cancel()
            raise

    # The shielded task keeps running; hand it to the heartbeat stream.
    logger.debug(
        f"Non-streaming request exceeded {grace_seconds}s grace period; "
        "switching to keepalive heartbeat mode"
    )
    return StreamingResponse(
        _heartbeat_body(task, interval_seconds),
        status_code=200,
        media_type="application/json",
    )


async def await_with_disconnect_monitor(
    coro: Awaitable[Response],
    request: Request,
    *,
    poll_interval: float = _DISCONNECT_POLL_INTERVAL,
) -> Response:
    """Await the pipeline, cancelling it when the client goes away.

    Applies to every request that does not use the keepalive wrapper
    (streaming requests, disabled keepalive, binary protocols). While the
    pipeline has not yet produced a response, the client connection is polled:
    on disconnect the in-flight provider request is cancelled ("an abandoned
    generation costs money and serves nobody") and a terminal 499 response is
    returned so the ASGI cycle unwinds cleanly. The failure itself is logged
    by the pipeline's cancellation handler (see UnifiedProcessor._run_pipeline).
    """
    task = asyncio.ensure_future(coro)

    try:
        while True:
            outcome = await wait_for_either(task, request, poll_interval=poll_interval)
            if isinstance(outcome, _Waiting):
                continue
            break
    except BaseException:
        # Real error or caller cancellation: don't leave the provider request
        # running unattended.
        if not task.done():
            task.cancel()
        raise

    if isinstance(outcome, _Disconnected):
        # Client is gone: nobody will read this response. Return a terminal
        # response so the ASGI cycle unwinds cleanly.
        return JSONResponse(
            status_code=499,
            content=_DISCONNECT_RESPONSE_PAYLOAD,
        )
    return outcome


__all__ = [
    "await_with_disconnect_monitor",
    "await_with_keepalive",
    "supports_keepalive",
    "wait_for_either",
]
