"""Exact-path ASGI fast path for protocol endpoints.

FastAPI's router walks every candidate route before it dispatches a request,
and FastAPI 0.14x adds per-candidate scope/version bookkeeping on top of that
(``_match``, ``matches``, ``_get_fastapi_scope``, ``_get_routes_version``). On
the proxy's hot surface — a handful of exact ``POST /v1/...`` protocol paths —
that scan is ~18% of per-request CPU, yet the routing result is a constant.

This module installs a dict-lookup dispatcher *inside* FastAPI's exception
middleware, by wrapping ``app.router.middleware_stack``. A matched request:

1. reads its body through a :class:`~llm_proxy.api.middleware.asgi_utils.BodyBuffer`;
2. validates it with the **same** Pydantic request model the FastAPI route
   declares (so validation/coercion and ``extra="allow"`` semantics match);
3. calls the **same** endpoint handler the FastAPI route wraps (so the pipeline,
   protocol middleware, keepalive/disconnect handling and error propagation are
   unchanged);
4. writes the handler's response back through the ASGI channel.

Every mismatch — unknown path, non-POST, non-JSON content type, undecodable
body, or a validation error — falls back to the original router with the body
replayed, so error responses (FastAPI's 422 and friends) are byte-for-byte the
ones FastAPI would have produced. Exceptions raised by the handler propagate to
FastAPI's ``ExceptionMiddleware`` exactly as before, because the dispatcher sits
inside it.

The routes stay registered on the FastAPI app, so OpenAPI generation, route
introspection and ``url_path_for`` are untouched.
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import orjson
from pydantic import ValidationError
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from llm_proxy.api.middleware.asgi_utils import BodyBuffer, get_scope_header
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

#: JSON media types for which the fast path parses the body itself. Anything
#: else (notably multipart) falls back to FastAPI.
_JSON_MEDIA_TYPES = frozenset({"application/json", "text/json"})
_JSON_MEDIA_SUFFIX = "+json"


@dataclass(frozen=True)
class FastPathEntry:
    """The request model and handler for one exact protocol path."""

    request_model: type[Any]
    handler: Callable[[Any, Request], Awaitable[Any]]


def _is_json_request(scope: Scope) -> bool:
    """Whether the request body should be parsed as JSON by the fast path.

    A missing content type is treated as JSON (FastAPI still parses the body);
    an explicit non-JSON type falls back so FastAPI's media-type handling runs.
    """
    content_type = get_scope_header(scope, "content-type")
    if not content_type:
        return True
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type in _JSON_MEDIA_TYPES or media_type.endswith(_JSON_MEDIA_SUFFIX)


class ProtocolFastPath:
    """Dispatch exact protocol paths without going through FastAPI routing."""

    def __init__(self, app: ASGIApp, entries: Mapping[str, FastPathEntry]) -> None:
        self._app = app
        self._entries = dict(entries)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self._app(scope, receive, send)
            return

        entry = self._entries.get(scope.get("path", ""))
        if entry is None:
            await self._app(scope, receive, send)
            return

        await self._dispatch(entry, scope, receive, send)

    async def _dispatch(
        self,
        entry: FastPathEntry,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        # Buffer the body so a validation/parse miss can hand the *same* bytes
        # to the original router for FastAPI's canonical error response.
        body = BodyBuffer(receive)
        try:
            raw = await body.read()
        except Exception:
            await self._app(scope, body, send)
            return

        if not _is_json_request(scope):
            await self._app(scope, body, send)
            return

        try:
            data = orjson.loads(raw)
        except ValueError:
            await self._app(scope, body, send)
            return

        if not isinstance(data, dict):
            await self._app(scope, body, send)
            return

        try:
            protocol_request = entry.request_model.model_validate(data)
        except ValidationError:
            await self._app(scope, body, send)
            return

        request = Request(scope, body)
        # Mirrors create_traced_handler: early-failure logging reads the parsed
        # body off request.state, so stash it here too.
        request.state.parsed_request_body = protocol_request

        # The FastAPI route declares `Depends(require_any_auth)`; auth normally
        # already ran in ApiKeyAuthMiddleware, but re-checking here keeps the
        # fast path behaviour-identical for any path the middleware does not gate.
        from llm_proxy.api.dependencies import require_api_key_auth
        from llm_proxy.core.identity import get_request_identity

        if not get_request_identity(request).is_authenticated:
            await require_api_key_auth(request)

        response = await entry.handler(protocol_request, request)
        await response(scope, body, send)


def install_protocol_fast_path(app: Any, entries: Mapping[str, FastPathEntry]) -> None:
    """Wrap the app router's entry point with the fast-path dispatcher.

    ``router.middleware_stack`` is what ``starlette.routing.Router.__call__``
    invokes, and it sits inside FastAPI's ``ExceptionMiddleware``, so wrapping
    it keeps exception handlers and the outer ASGI middleware stack in play.
    """
    if not entries:
        return
    router = app.router
    router.middleware_stack = ProtocolFastPath(router.middleware_stack, entries)
    logger.debug(f"Installed protocol fast path for {len(entries)} path(s)")
