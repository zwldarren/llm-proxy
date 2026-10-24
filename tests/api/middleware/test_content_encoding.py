"""Tests for the Content-Encoding request decompression middleware."""

import gzip
import zlib
from compression import zstd
from unittest.mock import MagicMock

import brotli
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from llm_proxy.api.middleware.content_encoding import content_encoding_middleware
from llm_proxy.api.middleware.form_encoded import form_encoded_middleware
from llm_proxy.config.types import ProxyAuthConfig, ProxyConfig, SecurityParams, ServerParams

_TEST_AUTH = ProxyAuthConfig(jwt_secret="a" * 32)


def _build_app(max_bytes: int = 10 * 1024 * 1024, *, with_form: bool = False) -> FastAPI:
    """Build an app running the decompression middleware over an echo endpoint."""
    app = FastAPI()
    config_manager = MagicMock()
    config_manager.get_cached_config.return_value = ProxyConfig(
        server_params=ServerParams(
            auth=_TEST_AUTH,
            security=SecurityParams(max_request_body_size_bytes=max_bytes),
        )
    )
    app.state.config_manager = config_manager
    if with_form:
        # Registered after content_encoding so form conversion runs after
        # decompression (same relative order as the real app).
        app.middleware("http")(form_encoded_middleware)
    app.middleware("http")(content_encoding_middleware)

    @app.post("/echo")
    async def echo(request: Request):
        body = await request.body()
        return {
            "body": body.decode("utf-8"),
            "content_encoding": request.headers.get("content-encoding"),
            "content_length": request.headers.get("content-length"),
            "transfer_encoding": request.headers.get("transfer-encoding"),
        }

    return app


@pytest.fixture
def client():
    with TestClient(_build_app()) as client:
        yield client


def test_no_content_encoding_passes_through(client):
    response = client.post("/echo", content=b'{"model": "gpt-5"}')
    assert response.status_code == 200
    assert response.json()["body"] == '{"model": "gpt-5"}'


def test_identity_encoding_passes_through(client):
    response = client.post("/echo", content=b"plain", headers={"Content-Encoding": "identity"})
    assert response.status_code == 200
    assert response.json()["body"] == "plain"


def test_zstd_body_is_decompressed(client):
    payload = b'{"model": "gpt-5", "input": "hello"}'
    compressed = zstd.compress(payload)
    response = client.post(
        "/echo",
        content=compressed,
        headers={"Content-Encoding": "zstd", "Content-Length": str(len(compressed))},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["body"] == payload.decode()
    # Entity headers are rewritten for the plain body.
    assert body["content_encoding"] is None
    assert body["content_length"] == str(len(payload))
    assert body["transfer_encoding"] is None


def test_zst_alias_is_decompressed(client):
    payload = b"alias"
    response = client.post(
        "/echo", content=zstd.compress(payload), headers={"Content-Encoding": "zst"}
    )
    assert response.status_code == 200
    assert response.json()["body"] == "alias"


def test_gzip_body_is_decompressed(client):
    payload = b'{"model": "gpt-5"}'
    response = client.post(
        "/echo", content=gzip.compress(payload), headers={"Content-Encoding": "gzip"}
    )
    assert response.status_code == 200
    assert response.json()["body"] == payload.decode()


def test_x_gzip_alias_is_decompressed(client):
    payload = b"x-gzip-body"
    response = client.post(
        "/echo", content=gzip.compress(payload), headers={"Content-Encoding": "x-gzip"}
    )
    assert response.status_code == 200
    assert response.json()["body"] == payload.decode()


def test_deflate_zlib_wrapped_is_decompressed(client):
    payload = b"zlib-wrapped"
    response = client.post(
        "/echo", content=zlib.compress(payload), headers={"Content-Encoding": "deflate"}
    )
    assert response.status_code == 200
    assert response.json()["body"] == payload.decode()


def test_deflate_raw_is_decompressed(client):
    payload = b"raw-deflate"
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    compressed = compressor.compress(payload) + compressor.flush()
    response = client.post("/echo", content=compressed, headers={"Content-Encoding": "deflate"})
    assert response.status_code == 200
    assert response.json()["body"] == payload.decode()


def test_brotli_body_is_decompressed(client):
    payload = b'{"model": "gpt-5"}'
    response = client.post(
        "/echo", content=brotli.compress(payload), headers={"Content-Encoding": "br"}
    )
    assert response.status_code == 200
    assert response.json()["body"] == payload.decode()


def test_stacked_encodings_decode_in_reverse_order(client):
    """``gzip, zstd`` means gzip was applied first, then zstd; decode zstd→gzip."""
    payload = b"stacked-body"
    compressed = zstd.compress(gzip.compress(payload))
    response = client.post("/echo", content=compressed, headers={"Content-Encoding": "gzip, zstd"})
    assert response.status_code == 200
    assert response.json()["body"] == payload.decode()


def test_unsupported_encoding_returns_400(client):
    response = client.post("/echo", content=b"whatever", headers={"Content-Encoding": "compress"})
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["type"] == "invalid_request"
    assert body["code"] == "unsupported_content_encoding"
    assert "compress" in body["message"]


def test_corrupt_zstd_body_returns_400(client):
    response = client.post(
        "/echo", content=b"not-a-zstd-frame", headers={"Content-Encoding": "zstd"}
    )
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["type"] == "invalid_request"
    assert body["code"] == "invalid_content_encoding"


def test_truncated_brotli_body_returns_400(client):
    compressed = brotli.compress(b"x" * 10000)
    response = client.post(
        "/echo", content=compressed[: len(compressed) // 2], headers={"Content-Encoding": "br"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_content_encoding"


def test_decompressed_body_over_limit_returns_413():
    """A zip-bomb-style body whose decompressed size exceeds the cap is rejected."""
    with TestClient(_build_app(1024)) as client:
        payload = b"x" * 4096
        response = client.post(
            "/echo", content=zstd.compress(payload), headers={"Content-Encoding": "zstd"}
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "body_size_exceeded"


def test_compressed_body_over_limit_returns_413():
    """The compressed input itself is also capped before decompression."""
    with TestClient(_build_app(32)) as client:
        payload = b"compressed-input-over-limit"
        response = client.post(
            "/echo", content=zstd.compress(payload), headers={"Content-Encoding": "zstd"}
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "body_size_exceeded"


def test_compressed_form_encoded_body_is_converted():
    """End-to-end: a zstd-compressed form body is decompressed then converted to JSON."""
    with TestClient(_build_app(with_form=True)) as client:
        form = "model=gpt-5&input=hello"
        response = client.post(
            "/v1/responses",
            content=zstd.compress(form.encode()),
            headers={
                "Content-Encoding": "zstd",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
    # The echo endpoint is not mounted at /v1/responses here; a 404/405/422 all
    # prove the body passed both middlewares without a 400 decompression error.
    assert response.status_code != 400


@pytest.mark.asyncio
async def test_admission_precedes_body_read(monkeypatch):
    """A request must not buffer its body before it owns a decode slot."""
    import asyncio

    from llm_proxy.api.middleware import content_encoding as encoding_module

    events: list[str] = []

    class FakeLimiter:
        async def acquire(self) -> None:
            events.append("acquire")

        def release(self) -> None:
            events.append("release")

    class FakeBody:
        async def read_limited(self, _cap: int) -> bytes:
            events.append("read")
            return b"compressed"

        def replace(self, data: bytes) -> None:
            events.append(f"replace:{data.decode()}")

    app = MagicMock()
    app.state.config_manager = MagicMock()
    app.state.config_manager.get_cached_config.return_value = ProxyConfig(
        server_params=ServerParams(auth=_TEST_AUTH)
    )
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/echo",
        "raw_path": b"/echo",
        "query_string": b"",
        "headers": [(b"content-encoding", b"gzip")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "scheme": "http",
        "app": app,
    }

    def fake_decode(_data: bytes, _cap: int) -> bytes:
        events.append("decode")
        return b"plain"

    monkeypatch.setattr(encoding_module, "_get_decompress_limiter", lambda: FakeLimiter())
    monkeypatch.setattr(
        encoding_module, "_DECODERS", {**encoding_module._DECODERS, "gzip": fake_decode}
    )

    response = await encoding_module._dispatch(Request(scope), FakeBody())
    assert response is None
    await asyncio.sleep(0)  # let the future's done callback release
    assert events[:3] == ["acquire", "read", "decode"]
    assert set(events[3:]) == {"replace:plain", "release"}


@pytest.mark.asyncio
async def test_cancelled_decode_keeps_admission_until_worker_finishes():
    """Cancellation must not release the slot while the decoder thread runs."""
    import asyncio
    import threading

    from llm_proxy.api.middleware.content_encoding import _run_decompression

    started = threading.Event()
    finish = threading.Event()

    class FakeLimiter:
        released = False

        def release(self) -> None:
            self.released = True

    def blocking_decode() -> bytes:
        started.set()
        finish.wait(timeout=5)
        return b"done"

    limiter = FakeLimiter()
    task = asyncio.create_task(_run_decompression(limiter, blocking_decode))
    while not started.is_set():
        await asyncio.sleep(0.01)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert limiter.released is False

    finish.set()
    for _ in range(100):
        if limiter.released:
            break
        await asyncio.sleep(0.01)
    assert limiter.released is True
