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

    settings = SimpleNamespace(http=SimpleNamespace(effective_client_backend=lambda: "aiohttp"))
    monkeypatch.setattr(client_module, "get_settings", lambda: settings)
    session = client_module.AsyncSession()
    assert isinstance(session._impl, AiohttpSession)


def test_dispatcher_selects_httpx_backend(monkeypatch) -> None:
    import llm_proxy.http.client as client_module

    settings = SimpleNamespace(
        http=SimpleNamespace(
            effective_client_backend=lambda: "httpx2",
            max_connections=200,
            max_keepalive=200,
        )
    )
    monkeypatch.setattr(client_module, "get_settings", lambda: settings)
    session = client_module.AsyncSession()
    assert not isinstance(session._impl, AiohttpSession)


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

    async def blob(request: web.Request) -> web.Response:
        return web.Response(body=b"x" * 1024, content_type="image/png")

    async def redirect_blob(request: web.Request) -> web.Response:
        raise web.HTTPFound(location="/blob")

    async def redirect_hostname(request: web.Request) -> web.Response:
        raise web.HTTPFound(location="http://private.example.com/")

    async def chunked(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "image/png"})
        await response.prepare(request)
        await response.write(b"x" * 300)
        await response.write(b"x" * 300)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", unary)
    app.router.add_get("/v1/chat/completions", unary)
    app.router.add_post("/stream", stream)
    app.router.add_get("/missing", missing)
    app.router.add_post("/form", form)
    app.router.add_get("/ok", ok)
    app.router.add_get("/redirect", redirect)
    app.router.add_get("/redirect-private", redirect_private)
    app.router.add_get("/blob", blob)
    app.router.add_get("/redirect-blob", redirect_blob)
    app.router.add_get("/chunked", chunked)
    app.router.add_get("/redirect-hostname", redirect_hostname)

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
    async def allow(url: str) -> None:
        return None

    monkeypatch.setattr("llm_proxy.http.aiohttp_backend._validate_redirect", allow)
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


async def test_aiter_bytes_reads_buffered_and_streamed_bodies(server: str) -> None:
    """Both request modes expose the body as chunks for the download cap."""
    session = AiohttpSession(timeout=(5.0, 10.0))
    try:
        buffered = await session.get(f"{server}/blob")
        chunks = [chunk async for chunk in buffered.aiter_bytes()]
        assert b"".join(chunks) == b"x" * 1024

        streamed = await session.get(f"{server}/blob", stream=True)
        chunks = [chunk async for chunk in streamed.aiter_bytes()]
        await streamed.close()
        assert b"".join(chunks) == b"x" * 1024
    finally:
        await session.close()


async def test_download_helper_caps_and_follows_redirects(server: str, monkeypatch) -> None:
    """End to end: a real streamed body is capped, and a 302 is followed once.

    The helper is backend-agnostic (it only needs ``get``), so this covers the
    aiohttp implementation of the same contract.
    """
    from llm_proxy.core.exceptions import ValidationError
    from llm_proxy.http.client import download_image_as_base64

    monkeypatch.setattr("llm_proxy.http.client._allowed_hosts", lambda: frozenset({"127.0.0.1"}))
    session = AiohttpSession(timeout=(5.0, 10.0))
    try:
        # A declared length over the cap fails before the body travels.
        with pytest.raises(ValidationError, match="over the 512-byte"):
            await download_image_as_base64(session, f"{server}/blob", max_bytes=512)

        # A chunked body with no length header is caught while streaming.
        with pytest.raises(ValidationError, match="exceeds the"):
            await download_image_as_base64(session, f"{server}/chunked", max_bytes=512)

        data_url, mime = await download_image_as_base64(
            session, f"{server}/redirect-blob", max_bytes=4096
        )
        assert mime == "image/png"
        assert data_url.startswith("data:image/png;base64,")

        # The redirect target is validated like the URL that pointed at it.
        monkeypatch.setattr("llm_proxy.http.client._allowed_hosts", lambda: frozenset())
        with pytest.raises(ValidationError):
            await download_image_as_base64(session, f"{server}/redirect-blob", max_bytes=4096)
    finally:
        await session.close()


async def test_resolver_rejects_a_hostname_that_resolves_privately(monkeypatch) -> None:
    """DNS is checked when the connection is made, closing the rebinding gap."""
    from llm_proxy.http.aiohttp_backend import _SsrfGuardResolver

    resolver = _SsrfGuardResolver()
    try:

        async def private(host, port=0, family=socket.AF_INET):
            return [{"hostname": host, "host": "10.1.2.3", "port": port, "family": family}]

        monkeypatch.setattr(resolver._inner, "resolve", private)

        with pytest.raises(OSError, match="disallowed address 10.1.2.3"):
            await resolver.resolve("images.example.com")
    finally:
        await resolver.close()


async def test_resolver_passes_public_answers_through(monkeypatch) -> None:
    from llm_proxy.http.aiohttp_backend import _SsrfGuardResolver

    resolver = _SsrfGuardResolver()
    try:

        async def public(host, port=0, family=socket.AF_INET):
            return [{"hostname": host, "host": "93.184.216.34", "port": port, "family": family}]

        monkeypatch.setattr(resolver._inner, "resolve", public)

        records = await resolver.resolve("images.example.com")
        assert records[0]["host"] == "93.184.216.34"
    finally:
        await resolver.close()


async def test_resolver_honours_the_allowed_hosts_exemption(monkeypatch) -> None:
    """A LAN provider opted in through HTTP_ALLOWED_HOSTS keeps working."""
    from llm_proxy.http.aiohttp_backend import _SsrfGuardResolver

    resolver = _SsrfGuardResolver()
    try:

        async def private(host, port=0, family=socket.AF_INET):
            return [{"hostname": host, "host": "10.1.2.3", "port": port, "family": family}]

        monkeypatch.setattr(resolver._inner, "resolve", private)
        monkeypatch.setattr(
            "llm_proxy.http.client._allowed_hosts", lambda: frozenset({"images.example.com"})
        )

        records = await resolver.resolve("images.example.com")
        assert records[0]["host"] == "10.1.2.3"
    finally:
        await resolver.close()


async def test_resolver_leaves_literal_loopback_alone(monkeypatch) -> None:
    """Literal addresses are the pre-flight checks' job; local dev keeps working."""
    from llm_proxy.http.aiohttp_backend import _SsrfGuardResolver

    resolver = _SsrfGuardResolver()
    try:

        async def loopback(host, port=0, family=socket.AF_INET):
            return [{"hostname": host, "host": "127.0.0.1", "port": port, "family": family}]

        monkeypatch.setattr(resolver._inner, "resolve", loopback)

        records = await resolver.resolve("127.0.0.1")
        assert records[0]["host"] == "127.0.0.1"
    finally:
        await resolver.close()


async def test_sessions_install_the_guard_resolver() -> None:
    from llm_proxy.http.aiohttp_backend import _SsrfGuardResolver

    session = AiohttpSession(timeout=(5.0, 10.0))
    try:
        client_session = await session._get_session()
        assert isinstance(client_session.connector._resolver, _SsrfGuardResolver)
    finally:
        await session.close()


async def test_rejects_redirect_to_a_hostname_resolving_privately(server: str, monkeypatch) -> None:
    """A redirect target's hostname is resolved, not just pattern-matched."""
    from llm_proxy.core.exceptions import ValidationError

    monkeypatch.setattr(
        "llm_proxy.http.client._is_banned_resolved_ip",
        lambda host, **kwargs: host == "private.example.com",
    )
    session = AiohttpSession(timeout=(5.0, 10.0))
    try:
        with pytest.raises(ValidationError):
            await session.get(f"{server}/redirect-hostname")
    finally:
        await session.close()
