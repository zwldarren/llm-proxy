"""Helpers for writing pure-ASGI middleware.

Starlette's ``BaseHTTPMiddleware`` allocates a memory object stream, two task
groups and several wrapper objects on *every* request, even for a middleware
that only inspects a header and passes the request through. LiteLLM measured a
74% throughput gain (and 38% lower p50) from replacing a single such middleware
with a plain ASGI callable, so the proxy's function middlewares are written as
plain ASGI callables and registered with ``app.add_middleware``.

A middleware is implemented once as a ``dispatch(request, body)`` coroutine that
returns either a ``Response`` (to short-circuit) or ``None`` (to continue):

- :class:`CoreASGIMiddleware` drives it from ``__call__`` with a
  :class:`BodyBuffer` as the receive channel.
- :func:`adapt_http_middleware` exposes the same dispatch in the
  ``(request, call_next)`` shape used by existing call sites and tests.

Both share the dispatch function, so there is a single implementation.
"""

from collections.abc import Awaitable, Callable
from typing import Protocol

from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

NextHandler = Callable[[Request], Awaitable[Response]]


class BodyTooLargeError(Exception):
    """The request body crossed a caller's byte cap while it was being read.

    Raised by :meth:`BodyReader.read_limited` so a middleware can reject an
    oversized stream *without* first buffering it in full.
    """

    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"request body exceeds {max_bytes} bytes")
        self.max_bytes = max_bytes


class BodyReader(Protocol):
    """Read, and optionally replace, the request body."""

    async def read(self) -> bytes: ...

    async def read_limited(self, max_bytes: int) -> bytes: ...

    def replace(self, body: bytes) -> None: ...


class FunctionalBodyReader:
    """Body access for a ``BaseHTTPMiddleware``-style handler.

    ``BaseHTTPMiddleware`` replays the body cached on the request object, so
    replacing the body only means overwriting that cache.
    """

    def __init__(self, request: Request) -> None:
        self._request = request

    async def read(self) -> bytes:
        return await self._request.body()

    async def read_limited(self, max_bytes: int) -> bytes:
        # The body is already materialised by ``BaseHTTPMiddleware``; the cap
        # is checked after the fact rather than enforced during the read.
        body = await self.read()
        if len(body) > max_bytes:
            raise BodyTooLargeError(max_bytes)
        return body

    def replace(self, body: bytes) -> None:
        self._request._body = body


class BodyBuffer:
    """ASGI receive wrapper that buffers the request body on demand.

    The buffer stays transparent until ``read``/``read_limited``/``replace`` is
    used, so a streaming request body is never materialised for requests that
    do not need it. Each middleware buffers only on its own narrow predicate
    (e.g. a ``Content-Encoding`` header, or an audited admin path).
    """

    def __init__(self, receive: Receive) -> None:
        self._receive = receive
        self._body: bytes | None = None
        self._replacement: bytes | None = None
        self._replayed = False

    async def _drain(self, max_bytes: int | None = None) -> bytes:
        """Pull the body off the receive channel, enforcing ``max_bytes`` if set.

        Raises :class:`BodyTooLargeError` as soon as the accumulated body
        crosses the cap, so the remainder is never pulled off the channel.
        """
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await self._receive()
            if message["type"] == "http.disconnect":
                break
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise BodyTooLargeError(max_bytes)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        return b"".join(chunks)

    async def read(self) -> bytes:
        """Drain the body from the receive channel and cache it for replay.

        Unlike :meth:`read_limited` and :meth:`__call__`, this ignores
        :meth:`replace` and always returns the bytes the client actually sent,
        so a middleware inspecting the body (logging, model restriction, form
        parsing) sees the original. Callers that must observe the effective
        body should use :meth:`read_limited`.
        """
        if self._body is None:
            self._body = await self._drain()
        return self._body

    async def read_limited(self, max_bytes: int) -> bytes:
        """Drain the body, abandoning it as soon as it exceeds ``max_bytes``.

        Unlike :meth:`read`, this never accumulates more than ``max_bytes`` of
        body before raising :class:`BodyTooLargeError`, so a caller can put a
        hard memory bound on an untrusted stream. This matters for bodies
        whose size is not declared up front (chunked transfer encoding) or not
        trustworthy (a compressed body whose decompressed size is unknown).

        Once the body is already buffered, the cap is checked after the fact —
        those bytes are in memory by then — and a pending :meth:`replace` is
        preferred over the buffered body, since it is what downstream receives.
        """
        if self._body is not None:
            body = self._replacement if self._replacement is not None else self._body
            if len(body) > max_bytes:
                raise BodyTooLargeError(max_bytes)
            return body
        # A failed drain must not leave a partial body cached for replay.
        self._body = await self._drain(max_bytes)
        return self._body

    def replace(self, body: bytes) -> None:
        """Set the body that downstream receives."""
        self._replacement = body

    async def __call__(self) -> Message:
        if self._body is None:
            # Nothing buffered: stay out of the way and stream untouched.
            return await self._receive()
        if not self._replayed:
            self._replayed = True
            body = self._replacement if self._replacement is not None else self._body
            return {"type": "http.request", "body": body, "more_body": False}
        return await self._receive()


class CoreASGIMiddleware:
    """Drive a ``dispatch(request, body) -> Response | None`` as plain ASGI."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def dispatch(self, request: Request, body: BodyReader) -> Response | None:
        raise NotImplementedError

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        body = BodyBuffer(receive)
        request = Request(scope, body)
        response = await self.dispatch(request, body)
        if response is not None:
            await response(scope, receive, send)
            return
        await self.app(scope, body, send)


def adapt_http_middleware(
    dispatch: Callable[[Request, BodyReader], Awaitable[Response | None]],
) -> Callable[[Request, NextHandler], Awaitable[Response]]:
    """Wrap a ``dispatch`` function in the ``(request, call_next)`` shape."""

    async def middleware(request: Request, call_next: NextHandler) -> Response:
        response = await dispatch(request, FunctionalBodyReader(request))
        if response is not None:
            return response
        return await call_next(request)

    return middleware


def set_request_state(scope: Scope, key: str, value: object) -> None:
    """Set a value on the ASGI scope state that ``request.state`` reads."""
    scope.setdefault("state", {})[key] = value


def get_request_state(scope: Scope, key: str, default: object = None) -> object:
    return scope.setdefault("state", {}).get(key, default)


def merge_response_headers(message: Message, headers: dict[str, str]) -> None:
    """Set response headers on an ``http.response.start`` message.

    Mirrors ``MutableHeaders.__setitem__``: an existing header with the same
    case-insensitive name is replaced rather than duplicated.
    """
    replaced = {name.lower().encode("latin-1") for name in headers}
    kept = [h for h in message.get("headers", []) if h[0].lower() not in replaced]
    kept.extend((k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items())
    message["headers"] = kept


def get_response_header(message: Message, name: str) -> str | None:
    """Return a response header value from an ``http.response.start`` message."""
    wanted = name.lower().encode("latin-1")
    for key, value in message.get("headers", []):
        if key.lower() == wanted:
            return value.decode("latin-1")
    return None
