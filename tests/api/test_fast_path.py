"""Tests for the exact-path protocol fast path.

The dispatcher must call the registered handler for a matching POST and fall
back to the original router (with the body replayed) for every other case, so
FastAPI's canonical error responses still apply.
"""

import pytest
from pydantic import BaseModel
from starlette.responses import JSONResponse, PlainTextResponse

from llm_proxy.api.fast_path import FastPathEntry, ProtocolFastPath
from llm_proxy.core.identity import RequestIdentity


class _Body(BaseModel):
    model_config = {"extra": "allow"}

    model: str
    stream: bool = False


def _scope(
    *,
    path: str = "/v1/chat/completions",
    method: str = "POST",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> dict:
    if headers is None:
        headers = [(b"content-type", b"application/json")]
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers,
        "state": {"identity": RequestIdentity(api_key_name="test-key")},
        "app": None,
    }
    return scope


def _receiver(body: bytes):
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return receive


def _collector():
    messages: list[dict] = []
    responded = False

    async def send(message):
        nonlocal responded
        messages.append(message)
        if message["type"] == "http.response.start":
            responded = True

    return messages, send


class _Fallback:
    def __init__(self) -> None:
        self.called = 0
        self.body_seen: bytes | None = None

    async def __call__(self, scope, receive, send) -> None:
        self.called += 1
        # Drain the receive channel to prove the body was replayed.
        chunks = []
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        self.body_seen = b"".join(chunks)
        response = PlainTextResponse("fallback")
        await response(scope, receive, send)


def _dispatcher(handler, fallback=None, model=_Body):
    entry = FastPathEntry(request_model=model, handler=handler)
    return ProtocolFastPath(fallback or _Fallback(), {"/v1/chat/completions": entry})


async def _run(app, scope, body: bytes):
    messages, send = _collector()
    await app(scope, _receiver(body), send)
    return messages


@pytest.mark.asyncio
async def test_matching_post_calls_handler_without_fastapi(monkeypatch):
    calls: list[_Body] = []

    async def handler(parsed, request):
        calls.append(parsed)
        assert request.state.parsed_request_body is parsed
        return JSONResponse({"ok": True, "model": parsed.model})

    fallback = _Fallback()
    app = _dispatcher(handler, fallback)
    messages = await _run(app, _scope(), b'{"model": "gpt-4o", "stream": true}')

    assert fallback.called == 0
    assert len(calls) == 1
    assert calls[0].model == "gpt-4o"
    assert calls[0].stream is True
    assert messages[0]["type"] == "http.response.start"


@pytest.mark.asyncio
async def test_unknown_path_falls_back():
    async def handler(parsed, request):  # pragma: no cover - must not run
        raise AssertionError("handler must not run")

    fallback = _Fallback()
    app = _dispatcher(handler, fallback)
    await _run(app, _scope(path="/v1/other"), b"{}")
    assert fallback.called == 1


@pytest.mark.asyncio
async def test_non_post_falls_back():
    async def handler(parsed, request):  # pragma: no cover - must not run
        raise AssertionError("handler must not run")

    fallback = _Fallback()
    app = _dispatcher(handler, fallback)
    await _run(app, _scope(method="GET"), b"{}")
    assert fallback.called == 1


@pytest.mark.asyncio
async def test_non_json_content_type_falls_back_and_replays_body():
    async def handler(parsed, request):  # pragma: no cover - must not run
        raise AssertionError("handler must not run")

    fallback = _Fallback()
    app = _dispatcher(handler, fallback)
    scope = _scope(headers=[(b"content-type", b"application/x-www-form-urlencoded")])
    await _run(app, scope, b"model=gpt-4o")
    assert fallback.called == 1
    assert fallback.body_seen == b"model=gpt-4o"


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not json", b"[1, 2, 3]", b""])
async def test_undecodable_or_non_object_body_falls_back(body):
    async def handler(parsed, request):  # pragma: no cover - must not run
        raise AssertionError("handler must not run")

    fallback = _Fallback()
    app = _dispatcher(handler, fallback)
    await _run(app, _scope(), body)
    assert fallback.called == 1
    assert fallback.body_seen == body


@pytest.mark.asyncio
async def test_validation_error_falls_back_with_body():
    async def handler(parsed, request):  # pragma: no cover - must not run
        raise AssertionError("handler must not run")

    fallback = _Fallback()
    app = _dispatcher(handler, fallback)
    await _run(app, _scope(), b'{"stream": true}')
    assert fallback.called == 1
    assert fallback.body_seen == b'{"stream": true}'
