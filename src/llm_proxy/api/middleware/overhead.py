"""Pure-ASGI middleware that reports proxy overhead via a response header.

``x-llm-proxy-overhead-duration-ms`` = wall time from request entry until the
response starts, minus the time spent waiting on upstream provider calls
(full body for non-streaming requests, time-to-headers for streaming ones).
This mirrors LiteLLM's ``x-litellm-overhead-duration-ms`` so gateway overhead
can be measured separately from end-to-end latency during load tests.

Implemented as plain ASGI (not BaseHTTPMiddleware) to stay off the streaming
re-pump path: it only touches the ``http.response.start`` message.
"""

import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_proxy.http.upstream_timing import UpstreamTimer, bind_upstream_timer

OVERHEAD_HEADER = b"x-llm-proxy-overhead-duration-ms"


class OverheadHeaderMiddleware:
    """Attach the overhead header to every HTTP response that passes through."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        timer = UpstreamTimer()
        bind_upstream_timer(timer)
        started = time.perf_counter()

        async def send_with_overhead(message: Message) -> None:
            if message["type"] == "http.response.start":
                overhead_ms = (time.perf_counter() - started) * 1000.0 - timer.total_ms
                headers = message.setdefault("headers", [])
                headers.append((OVERHEAD_HEADER, f"{max(0.0, overhead_ms):.2f}".encode()))
            await send(message)

        await self.app(scope, receive, send_with_overhead)
