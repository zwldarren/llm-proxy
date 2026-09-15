"""Per-request upstream-wait accounting for the overhead response header.

A mutable ``UpstreamTimer`` is bound to a ContextVar by the outermost ASGI
middleware at request start. ContextVar semantics give every task spawned
within the request (BaseHTTPMiddleware task groups, detached tasks) a *copy*
of the binding — but the bound object is shared by reference, so the HTTP
client wrapper can accumulate upstream wait time from any descendant task and
the middleware still sees the total when the response starts.
"""

import contextvars


class UpstreamTimer:
    """Mutable accumulator for time spent waiting on upstream provider calls."""

    __slots__ = ("total_ms",)

    def __init__(self) -> None:
        self.total_ms = 0.0

    def add(self, elapsed_ms: float) -> None:
        self.total_ms += elapsed_ms


_current_timer: contextvars.ContextVar[UpstreamTimer | None] = contextvars.ContextVar(
    "llm_proxy_upstream_timer", default=None
)


def current_upstream_timer() -> UpstreamTimer | None:
    """Return the timer bound to the current request, or None off the request path."""
    return _current_timer.get()


def bind_upstream_timer(timer: UpstreamTimer) -> contextvars.Token[UpstreamTimer | None]:
    """Bind *timer* to the current context; returns the token for reset."""
    return _current_timer.set(timer)
