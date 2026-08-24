"""Tests for the request body size limit middleware."""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.middleware.body_limit import body_size_limit_middleware
from llm_proxy.config.types import ProxyAuthConfig, ProxyConfig, SecurityParams, ServerParams

_TEST_AUTH = ProxyAuthConfig(jwt_secret="a" * 32)


def _build_app(max_bytes: int) -> FastAPI:
    """Build an app whose config manager serves the given body size limit."""
    app = FastAPI()
    config_manager = MagicMock()
    config_manager.get_cached_config.return_value = ProxyConfig(
        server_params=ServerParams(
            auth=_TEST_AUTH,
            security=SecurityParams(max_request_body_size_bytes=max_bytes),
        )
    )
    app.state.config_manager = config_manager
    app.middleware("http")(body_size_limit_middleware)

    @app.post("/echo")
    async def echo():
        return {"ok": True}

    return app


def test_body_size_limit_blocks_large_content_length():
    """Requests with Content-Length over the limit are rejected with 413."""
    client = TestClient(_build_app(1024))
    response = client.post("/echo", content=b"x" * 2048, headers={"Content-Length": "2048"})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "body_size_exceeded"


def test_body_size_limit_allows_small_content_length():
    """Requests within the limit are allowed through."""
    client = TestClient(_build_app(1024 * 1024))
    response = client.post("/echo", content=b"x" * 100)
    assert response.status_code == 200


def test_body_size_limit_allows_exact_limit():
    """Requests with Content-Length equal to the limit are allowed through."""
    client = TestClient(_build_app(1024))
    response = client.post("/echo", content=b"x" * 1024)
    assert response.status_code == 200


def test_body_size_limit_rejects_negative_content_length():
    """Negative Content-Length is invalid and rejected."""
    client = TestClient(_build_app(1024))
    response = client.post("/echo", headers={"Content-Length": "-1"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_content_length"


def test_body_size_limit_rejects_chunked_encoding():
    """Chunked transfer encoding is rejected when body size limit is active."""
    client = TestClient(_build_app(1024))
    response = client.post(
        "/echo",
        content=b"x" * 100,
        headers={"Transfer-Encoding": "chunked"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "body_size_exceeded"
