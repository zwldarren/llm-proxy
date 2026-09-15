"""Pure-ASGI middleware that clears the per-request context after each request.

The proxy stores request-scoped state in context variables (``llm_proxy.core.context``).
Those are task-local, but a task reused across requests (e.g. by a test client or
a keep-alive connection) could otherwise observe a stale context, so the reset is
unconditional and happens even when the request raised.

Written as a plain ASGI callable: it does no work on the request path and only
wraps the call in a ``try/finally``, so paying ``BaseHTTPMiddleware``'s task
group and memory stream for it is pure overhead.
"""

from contextlib import suppress

from starlette.types import ASGIApp, Receive, Scope, Send

from llm_proxy.core.context import reset_context


class ContextResetMiddleware:
    """Reset the request context once the response has been produced."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self.app(scope, receive, send)
        finally:
            with suppress(Exception):
                reset_context()
