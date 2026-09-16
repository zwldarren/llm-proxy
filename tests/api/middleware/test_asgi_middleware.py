"""Tests for the pure-ASGI middleware plumbing.

The function middlewares are registered as plain ASGI callables (see
``llm_proxy.api.middleware.asgi_utils``) instead of Starlette's
``BaseHTTPMiddleware``. The decision logic is covered by the per-middleware
test modules through the ``(request, call_next)`` adapters; these tests cover
the ASGI plumbing itself: request-body buffering/replay and the
``dispatch -> Response | None`` contract.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.types import Message, Receive, Scope

from llm_proxy.api.middleware.asgi_utils import (
    BodyBuffer,
    BodyTooLargeError,
    CoreASGIMiddleware,
    get_response_header,
    merge_response_headers,
)


def _scope(headers: list[tuple[bytes, bytes]] | None = None) -> Scope:
    return {
        "type": "http",
        "method": "POST",
        "path": "/echo",
        "query_string": b"",
        "headers": headers or [],
        "scheme": "http",
        "server": ("testserver", 80),
    }


def _receive_from(chunks: list[bytes], consumed: list[bytes] | None = None) -> Receive:
    pending = list(chunks)

    async def receive() -> Message:
        if pending:
            body = pending.pop(0)
            if consumed is not None:
                consumed.append(body)
            return {"type": "http.request", "body": body, "more_body": bool(pending)}
        return {"type": "http.disconnect"}

    return receive


class TestNoBaseHTTPMiddleware:
    """Guard against reintroducing Starlette's per-request re-pump.

    ``BaseHTTPMiddleware`` allocates a memory stream and two task groups on
    every request, even when the middleware only passes through. The proxy's
    function middlewares are plain ASGI callables; this is the static check
    that keeps a new one from silently regressing the hot path.
    """

    def test_function_middlewares_are_pure_asgi(self) -> None:
        from starlette.middleware.base import BaseHTTPMiddleware

        from llm_proxy.api import app

        offenders = [
            middleware.cls.__name__
            for middleware in app.user_middleware
            if isinstance(middleware.cls, type) and issubclass(middleware.cls, BaseHTTPMiddleware)
        ]
        assert offenders == [], f"BaseHTTPMiddleware in the request path: {offenders}"


class TestBodyBuffer:
    """The on-demand request body buffer."""

    @pytest.mark.asyncio
    async def test_transparent_when_untouched(self) -> None:
        """An unread buffer passes each chunk through unchanged."""
        buffer = BodyBuffer(_receive_from([b"one", b"two"]))

        assert (await buffer())["body"] == b"one"
        assert (await buffer())["body"] == b"two"
        assert (await buffer())["type"] == "http.disconnect"

    @pytest.mark.asyncio
    async def test_read_joins_chunks(self) -> None:
        buffer = BodyBuffer(_receive_from([b"one", b"two"]))

        assert await buffer.read() == b"onetwo"
        # Idempotent: the cached body is returned without touching the channel.
        assert await buffer.read() == b"onetwo"

    @pytest.mark.asyncio
    async def test_replays_buffered_body_downstream(self) -> None:
        """After read(), downstream receives the whole body as one message."""
        buffer = BodyBuffer(_receive_from([b"one", b"two"]))
        await buffer.read()

        message = await buffer()
        assert message["body"] == b"onetwo"
        assert message["more_body"] is False

    @pytest.mark.asyncio
    async def test_read_limited_returns_body_within_cap(self) -> None:
        buffer = BodyBuffer(_receive_from([b"one", b"two"]))

        assert await buffer.read_limited(100) == b"onetwo"
        # Cached for replay exactly like read().
        assert (await buffer())["body"] == b"onetwo"

    @pytest.mark.asyncio
    async def test_read_limited_stops_at_cap(self) -> None:
        """An oversized stream raises without draining the rest of the body."""
        consumed: list[bytes] = []
        buffer = BodyBuffer(_receive_from([b"a" * 8, b"b" * 8, b"c" * 8], consumed))

        with pytest.raises(BodyTooLargeError):
            await buffer.read_limited(10)

        # Reading stopped the moment the cap was crossed: the third chunk was
        # never pulled off the channel, so a huge body cannot be buffered.
        assert len(consumed) == 2

    @pytest.mark.asyncio
    async def test_read_limited_honours_replacement(self) -> None:
        buffer = BodyBuffer(_receive_from([b"original"]))
        await buffer.read()
        buffer.replace(b"replacement")

        assert await buffer.read_limited(100) == b"replacement"
        with pytest.raises(BodyTooLargeError):
            await buffer.read_limited(4)

    @pytest.mark.asyncio
    async def test_replace_substitutes_downstream_body(self) -> None:
        buffer = BodyBuffer(_receive_from([b"original"]))
        assert await buffer.read() == b"original"

        buffer.replace(b"replacement")
        assert (await buffer())["body"] == b"replacement"


class _ShortCircuitMiddleware(CoreASGIMiddleware):
    """Test double: rejects requests whose body is "blocked"."""

    async def dispatch(self, request: Request, body):
        if await body.read() == b"blocked":
            return JSONResponse(status_code=403, content={"error": "blocked"})
        return None


class TestCoreASGIMiddleware:
    """The ``dispatch -> Response | None`` adapter's ASGI wiring."""

    @staticmethod
    def _build_app() -> FastAPI:
        app = FastAPI()
        app.add_middleware(_ShortCircuitMiddleware)

        @app.post("/echo")
        async def echo(request: Request):
            return {"body": (await request.body()).decode()}

        return app

    def test_short_circuit_response(self) -> None:
        client = TestClient(self._build_app())
        response = client.post("/echo", content=b"blocked")
        assert response.status_code == 403
        assert response.json() == {"error": "blocked"}

    def test_body_reaches_downstream_after_inspection(self) -> None:
        """Reading the body in dispatch must not starve the route handler."""
        client = TestClient(self._build_app())
        response = client.post("/echo", content=b"allowed")
        assert response.status_code == 200
        assert response.json() == {"body": "allowed"}


class TestResponseHeaderHelpers:
    """ASGI response header merging mirrors MutableHeaders semantics."""

    def test_merge_replaces_existing_header(self) -> None:
        message: Message = {
            "type": "http.response.start",
            "headers": [(b"x-frame-options", b"ALLOW"), (b"x-other", b"1")],
        }

        merge_response_headers(message, {"X-Frame-Options": "DENY"})

        assert (b"x-frame-options", b"DENY") in message["headers"]
        assert (b"x-other", b"1") in message["headers"]
        assert len(message["headers"]) == 2

    def test_get_response_header_is_case_insensitive(self) -> None:
        message: Message = {"type": "http.response.start", "headers": [(b"Vary", b"Accept")]}
        assert get_response_header(message, "vary") == "Accept"


class TestBodyLimitRunsAsPureASGI:
    """The body limit middleware short-circuits from the ASGI __call__ path."""

    @staticmethod
    def _build_app() -> FastAPI:
        from llm_proxy.api.middleware.body_limit import BodySizeLimitMiddleware
        from llm_proxy.config.types import (
            ProxyAuthConfig,
            ProxyConfig,
            SecurityParams,
            ServerParams,
        )

        app = FastAPI()
        config_manager = MagicMock()
        config_manager.get_cached_config.return_value = ProxyConfig(
            server_params=ServerParams(
                auth=ProxyAuthConfig(jwt_secret="a" * 32),
                security=SecurityParams(max_request_body_size_bytes=16),
            )
        )
        app.state.config_manager = config_manager
        app.add_middleware(BodySizeLimitMiddleware)

        @app.post("/echo")
        async def echo(request: Request):
            return {"body": (await request.body()).decode()}

        return app

    def test_oversized_body_rejected(self) -> None:
        client = TestClient(self._build_app())
        response = client.post("/echo", content=b"x" * 64, headers={"Content-Length": "64"})
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "body_size_exceeded"

    def test_small_body_passes(self) -> None:
        client = TestClient(self._build_app())
        response = client.post("/echo", content=b"hi")
        assert response.status_code == 200
        assert response.json() == {"body": "hi"}

    def test_chunked_body_within_limit_passes(self) -> None:
        """A chunked body under the cap reaches the handler intact."""
        client = TestClient(self._build_app())
        response = client.post("/echo", content=iter([b"hi"]))
        assert response.status_code == 200
        assert response.json() == {"body": "hi"}

    def test_chunked_body_over_limit_rejected(self) -> None:
        """A chunked body over the cap is rejected without being buffered."""
        client = TestClient(self._build_app())
        response = client.post("/echo", content=iter([b"x" * 16, b"y" * 16]))
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "body_size_exceeded"


class TestFormEncodedRunsAsPureASGI:
    """Form-encoded conversion rewrites the body for the downstream parser."""

    @staticmethod
    def _build_app() -> FastAPI:
        from llm_proxy.api.middleware.form_encoded import FormEncodedMiddleware

        app = FastAPI()
        app.add_middleware(FormEncodedMiddleware)

        @app.post("/v1/responses")
        async def responses(request: Request):
            return json.loads(await request.body())

        return app

    def test_form_body_is_converted_to_json(self) -> None:
        client = TestClient(self._build_app())
        response = client.post(
            "/v1/responses",
            content=b"model=gpt-5&input=hello",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        assert response.json() == {"model": "gpt-5", "input": "hello"}


class TestSecurityHeadersRunsAsPureASGI:
    """Security headers are added by intercepting response.start."""

    def test_headers_present_on_short_circuit(self) -> None:
        from llm_proxy.api.middleware.security import SecurityHeadersMiddleware

        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/boom")
        async def boom():
            return PlainTextResponse("nope", status_code=401)

        client = TestClient(app)
        response = client.get("/boom")
        assert response.status_code == 401
        assert response.headers["X-Frame-Options"] == "DENY"


class TestLoggingRunsAsPureASGI:
    """X-Request-Id is stamped on every response."""

    def test_request_id_header_added(self) -> None:
        from llm_proxy.api.middleware.logging import HttpLoggingMiddleware

        app = FastAPI()
        app.add_middleware(HttpLoggingMiddleware)

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        client = TestClient(app)
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.headers["X-Request-Id"]


class TestModelRestrictionRunsAsPureASGI:
    """Model restriction short-circuits from the ASGI path."""

    @staticmethod
    def _build_app() -> FastAPI:
        from llm_proxy.api.middleware.model_restriction import ModelRestrictionMiddleware

        app = FastAPI()
        app.add_middleware(ModelRestrictionMiddleware)

        # Runs outermost (registered last): seeds the state api_key_auth sets.
        @app.middleware("http")
        async def seed_restrictions(request: Request, call_next):
            request.scope.setdefault("state", {}).update(
                {"allowed_models": ["gpt-4"], "api_key_name": "k"}
            )
            return await call_next(request)

        @app.post("/v1/chat/completions")
        async def handler(request: Request):
            return {"body": (await request.body()).decode()}

        return app

    def test_disallowed_model_rejected(self) -> None:
        client = TestClient(self._build_app())
        blocked = client.post("/v1/chat/completions", content=b'{"model": "claude-3"}')
        assert blocked.status_code == 403
        assert b"model_not_allowed" in blocked.content

    def test_allowed_model_reaches_handler_with_body_intact(self) -> None:
        client = TestClient(self._build_app())
        allowed = client.post("/v1/chat/completions", content=b'{"model": "gpt-4"}')
        assert allowed.status_code == 200
        assert allowed.json() == {"body": '{"model": "gpt-4"}'}
