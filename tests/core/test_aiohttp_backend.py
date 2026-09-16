"""Tests for the aiohttp outbound HTTP backend.

The session is exercised against an in-process aiohttp server so redirects,
streaming line iteration, multipart bodies and error translation are covered
end to end without leaving the machine.
"""

import socket
from types import SimpleNamespace

import aiohttp
import httpx2
import pytest
from aiohttp import web

from llm_proxy.http.aiohttp_backend import (
    AiohttpSession,
    _build_form,
    _split_timeout,
    _translate,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_split_timeout_normalizes_inputs() -> None:
    assert _split_timeout(None) == (10.0, 600.0)
    assert _split_timeout((3.0, 7.0)) == (3.0, 7.0)
    assert _split_timeout(httpx2.Timeout(connect=2.0, read=4.0, write=60.0, pool=60.0)) == (
        2.0,
        4.0,
    )


def test_build_form_accepts_httpx_shapes() -> None:
    form = _build_form({"file": ("clip.wav", b"abc", "audio/wav")}, {"model": "m"})
    assert isinstance(form, aiohttp.FormData)
    two_tuple = _build_form({"file": ("clip.wav", b"abc")}, None)
    assert isinstance(two_tuple, aiohttp.FormData)


def test_translate_maps_transport_errors_to_httpx_types() -> None:
    assert isinstance(_translate(TimeoutError("t")), httpx2.ReadTimeout)
    assert isinstance(_translate(aiohttp.ServerDisconnectedError()), httpx2.RemoteProtocolError)
    assert isinstance(_translate(aiohttp.ClientPayloadError()), httpx2.RemoteProtocolError)
    assert isinstance(_translate(aiohttp.ClientOSError()), httpx2.ReadError)
    assert isinstance(_translate(aiohttp.ClientError()), httpx2.RequestError)
    # Every mapped type must remain catchable as a retryable NetworkError/Timeout.
    assert isinstance(_translate(aiohttp.ClientOSError()), httpx2.NetworkError)


def test_dispatcher_selects_aiohttp_backend(monkeypatch) -> None:
    import llm_proxy.http.client as client_module

    settings = SimpleNamespace(http=SimpleNamespace(client_backend="aiohttp"))
    monkeypatch.setattr(client_module, "get_settings", lambda: settings)
    session = client_module.AsyncSession()
    assert isinstance(session._impl, AiohttpSession)


@pytest.fixture
async def server():
    async def unary(request: web.Request) -> web.Response:
        payload = await request.json()
        return web.json_response({"ok": True, "model": payload.get("model")})

    async def stream(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for index in range(3):
            await response.write(f'data: {{"i": {index}}}\n\n'.encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def missing(request: web.Request) -> web.Response:
        return web.Response(status=404, text="nope")

    async def form(request: web.Request) -> web.Response:
        reader = await request.multipart()
        fields: dict[str, str] = {}
        async for part in reader:
            fields[part.name or ""] = (await part.read()).decode()
        return web.json_response(fields)

    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound(location="/ok")

    async def redirect_private(request: web.Request) -> web.Response:
        raise web.HTTPFound(location="http://169.254.169.254/latest/meta-data/")

    async def ok(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", unary)
    app.router.add_get("/v1/chat/completions", unary)
    app.router.add_post("/stream", stream)
    app.router.add_get("/missing", missing)
    app.router.add_post("/form", form)
    app.router.add_get("/ok", ok)
    app.router.add_get("/redirect", redirect)
    app.router.add_get("/redirect-private", redirect_private)

    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


async def test_unary_json_request(server: str) -> None:
    session = AiohttpSession(timeout=(5.0, 10.0))
    try:
        response = await session.post(f"{server}/v1/chat/completions", json={"model": "fake-model"})
        assert response.status_code == 200
        assert response.json()["model"] == "fake-model"
        assert response.content is not None
    finally:
        await session.close()


async def test_streaming_iter_lines(server: str) -> None:
    session = AiohttpSession(timeout=(5.0, 10.0))
    try:
        response = await session.post(f"{server}/stream", json={"stream": True}, stream=True)
        lines = [line async for line in response.iter_lines()]
        await response.close()
        payload = [line for line in lines if line]
        assert payload[-1] == b"data: [DONE]"
        assert len(payload) == 4
    finally:
        await session.close()


async def test_raise_for_status_raises_httpx_error(server: str) -> None:
    session = AiohttpSession(timeout=(5.0, 10.0))
    try:
        response = await session.get(f"{server}/missing")
        with pytest.raises(httpx2.HTTPStatusError) as excinfo:
            response.raise_for_status()
        assert excinfo.value.response.status_code == 404
        assert excinfo.value.response.text == "nope"
    finally:
        await session.close()


async def test_multipart_form_is_encoded(server: str) -> None:
    session = AiohttpSession(timeout=(5.0, 10.0))
    try:
        response = await session.post(
            f"{server}/form",
            data={"model": "whisper"},
            files={"file": ("clip.wav", b"abc", "audio/wav")},
        )
        body = response.json()
        assert body["model"] == "whisper"
        assert body["file"] == "abc"
    finally:
        await session.close()


async def test_follows_redirect(server: str, monkeypatch) -> None:
    monkeypatch.setattr("llm_proxy.http.aiohttp_backend._validate_redirect", lambda url: None)
    session = AiohttpSession(timeout=(5.0, 10.0))
    try:
        response = await session.get(f"{server}/redirect")
        assert response.status_code == 200
    finally:
        await session.close()


async def test_rejects_redirect_to_private_host(server: str) -> None:
    from llm_proxy.core.exceptions import ValidationError

    session = AiohttpSession(timeout=(5.0, 10.0))
    try:
        with pytest.raises(ValidationError):
            await session.get(f"{server}/redirect-private")
    finally:
        await session.close()


async def test_connect_error_is_translated() -> None:
    session = AiohttpSession(timeout=(2.0, 2.0))
    try:
        with pytest.raises(httpx2.NetworkError):
            await session.post(f"http://127.0.0.1:{_free_port()}/v1", json={})
    finally:
        await session.close()


async def test_unsupported_kwarg_raises() -> None:
    session = AiohttpSession()
    try:
        with pytest.raises(TypeError, match="unsupported arguments"):
            await session.post("http://127.0.0.1:1/v1", nonsense=1)
    finally:
        await session.close()
