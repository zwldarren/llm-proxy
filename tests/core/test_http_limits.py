"""Read limits on upstream responses, exercised against both HTTP backends.

The proxy buffers a provider's answer to convert it, so an endpoint that sends an
enormous body — or a stream that never sends a newline — must be cut off while it
arrives rather than after it is in memory. Both backends read through the same
``llm_proxy.http.limits`` helpers, so each case runs twice.
"""

import socket

import pytest
from aiohttp import web

from llm_proxy.http.aiohttp_backend import AiohttpSession
from llm_proxy.http.client import _HttpxSession
from llm_proxy.http.errors import UpstreamResponseTooLarge
from llm_proxy.http.limits import split_lines


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
async def server():
    async def big(request: web.Request) -> web.Response:
        return web.Response(body=b"x" * 4096, content_type="application/octet-stream")

    async def payload(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def long_line(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"x" * 1024)
        await response.write(b"x" * 1024)
        await response.write_eof()
        return response

    async def lines(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"a\r\nb\rc\nd\re\r")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/big", big)
    app.router.add_get("/payload", payload)
    app.router.add_get("/long-line", long_line)
    app.router.add_get("/lines", lines)

    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@pytest.fixture(params=["httpx2", "aiohttp"])
async def session(request):
    """One session per backend: both must enforce the same limits."""
    if request.param == "httpx2":
        client = _HttpxSession(timeout=(5.0, 10.0))
    else:
        client = AiohttpSession(timeout=(5.0, 10.0))
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
def small_cap(monkeypatch):
    """Shrink the response and line caps so the tests move kilobytes, not MiBs."""
    import llm_proxy.http.aiohttp_backend as aiohttp_backend
    import llm_proxy.http.client as client
    import llm_proxy.http.limits as limits

    for module in (client, aiohttp_backend):
        monkeypatch.setattr(module, "max_response_bytes", lambda: 1024)
    monkeypatch.setattr(limits, "MAX_STREAM_LINE_BYTES", 512)


async def test_buffered_read_is_capped(server: str, session, small_cap) -> None:
    """A buffered (non-streaming) call stops reading past the cap."""
    with pytest.raises(UpstreamResponseTooLarge, match="HTTP_MAX_RESPONSE_BYTES"):
        await session.get(f"{server}/big")


async def test_buffered_read_under_the_cap_returns_the_body(
    server: str, session, monkeypatch
) -> None:
    """Under the cap the call is an ordinary buffered response."""
    import llm_proxy.http.aiohttp_backend as aiohttp_backend
    import llm_proxy.http.client as client

    module = client if isinstance(session, _HttpxSession) else aiohttp_backend
    monkeypatch.setattr(module, "max_response_bytes", lambda: 8192)

    response = await session.get(f"{server}/payload")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert b'"ok"' in response.content


async def test_streamed_read_is_capped(server: str, session, small_cap) -> None:
    """``aread()`` on a streaming response enforces the same cap."""
    response = await session.get(f"{server}/big", stream=True)
    try:
        with pytest.raises(UpstreamResponseTooLarge, match="HTTP_MAX_RESPONSE_BYTES"):
            await response.aread()
    finally:
        await response.close()


async def test_stream_line_is_capped(server: str, session, small_cap) -> None:
    """A stream that never sends a newline cannot grow the read buffer."""
    response = await session.get(f"{server}/long-line", stream=True)
    try:
        with pytest.raises(UpstreamResponseTooLarge, match="longer than"):
            async for _line in response.iter_lines():
                pass
    finally:
        await response.close()


async def test_stream_lines_split_on_cr_lf_and_crlf(server: str, session, monkeypatch) -> None:
    """SSE treats CR, LF and CRLF as terminators; the byte splitter must too."""
    import llm_proxy.http.limits as limits

    monkeypatch.setattr(limits, "MAX_STREAM_LINE_BYTES", 1024)

    response = await session.get(f"{server}/lines", stream=True)
    try:
        lines = [line async for line in response.iter_lines()]
    finally:
        await response.close()

    assert lines == [b"a", b"b", b"c", b"d", b"e"]


@pytest.mark.parametrize(
    ("buffer", "lines", "tail"),
    [
        (b"", [], b""),
        (b"a\nb\n", [b"a", b"b"], b""),
        (b"a\r\nb\r\n", [b"a", b"b"], b""),
        (b"a\rb\rc", [b"a", b"b"], b"c"),
        (b"a\r", [], b"a\r"),
        (b"a\r\r", [b"a"], b"\r"),
        (b"\n\n", [b"", b""], b""),
        (b"a\nb\rc\nd\re\r", [b"a", b"b", b"c", b"d"], b"e\r"),
    ],
)
def test_split_lines_keeps_a_trailing_cr_pending(
    buffer: bytes, lines: list[bytes], tail: bytes
) -> None:
    """The splitter carries a final CR so a CRLF split across reads rejoins."""
    assert split_lines(buffer) == (lines, tail)
