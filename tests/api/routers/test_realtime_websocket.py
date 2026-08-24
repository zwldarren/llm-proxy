"""Integration tests for the Realtime API WebSocket relay endpoint."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.server import serve

from llm_proxy.config.types.model import ModelConfig, ModelProviderConfig, ProviderSelectionStrategy
from llm_proxy.config.types.provider import ProviderConfig


class FakeUpstream:
    """Fake upstream Realtime server connection (duck-typed for the relay).

    Serves the scripted messages, then stays open (blocks on recv) until the
    relay cancels the pump — mirroring a real upstream that outlives the
    client connection.
    """

    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self.sent: list[str | bytes] = []
        self.close_code: int | None = None

    async def recv(self):
        if self._messages:
            return self._messages.pop(0)
        await asyncio.Event().wait()

    async def send(self, data):
        self.sent.append(data)

    async def close(self, code):
        self.close_code = code


def _session_created():
    return orjson.dumps(
        {"type": "session.created", "event_id": "event_1", "session": {"model": "gpt-realtime"}}
    ).decode()


def _response_done():
    return orjson.dumps(
        {
            "type": "response.done",
            "event_id": "event_2",
            "response": {
                "id": "resp_123",
                "status": "completed",
                "usage": {
                    "total_tokens": 1000,
                    "input_tokens": 500,
                    "output_tokens": 500,
                    "input_token_details": {
                        "cached_tokens": 100,
                        "text_tokens": 300,
                        "audio_tokens": 100,
                        "image_tokens": 0,
                    },
                    "output_token_details": {"text_tokens": 200, "audio_tokens": 300},
                },
            },
        }
    ).decode()


def _response_done_without_usage():
    return orjson.dumps(
        {
            "type": "response.done",
            "event_id": "event_3",
            "response": {"id": "resp_124", "status": "failed"},
        }
    ).decode()


def _build_app(monkeypatch, *, provider_configs, model_configs):
    """Build the test app with fake auth, config manager, and recording log service."""
    from llm_proxy.api.routers import realtime as router_module

    app = FastAPI()
    app.include_router(router_module.ws_router)

    async def fake_verify(api_key: str):
        return {
            "principal_id": "test-key",
            "allowed_models": None,
            "allowed_mcp_servers": None,
            "user_id": None,
        }

    monkeypatch.setattr("llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp", fake_verify)

    config = SimpleNamespace(
        provider_configs=provider_configs,
        server_params=SimpleNamespace(max_fallback_attempts=1, max_retries=1),
        provider_selection=SimpleNamespace(strategy=ProviderSelectionStrategy.RANDOM),
    )
    config_manager = MagicMock()
    config_manager.get_config = AsyncMock(return_value=config)
    config_manager.get_model_config = AsyncMock(side_effect=lambda name: model_configs.get(name))
    app.state.config_manager = config_manager
    app.state.redis_client = None
    app.state.circuit_breaker = None
    app.state.provider_stats = None

    # Recording log service (the real one would write to the database).
    recorded_logs = []
    monkeypatch.setattr(
        "llm_proxy.realtime.usage.RequestLogService",
        lambda config: SimpleNamespace(create_log_background=recorded_logs.append),
    )
    app.state._realtime_logs = recorded_logs
    return app


@pytest.fixture
def app(monkeypatch):
    """App with a fake scripted upstream connection (connect_upstream patched)."""
    provider_configs = {
        "openai": ProviderConfig(
            type="openai",
            api_key="sk-upstream",
            base_url="https://api.openai.com/v1",
        ),
        "anthropic": ProviderConfig(type="anthropic", api_key="sk-anthropic"),
    }
    model_configs = {
        "gpt-realtime": ModelConfig(
            providers=[ModelProviderConfig(provider="openai", provider_model_name="gpt-realtime")],
            supports_realtime=True,
        ),
        "gpt-4o-realtime-preview-2024-12-17": ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    provider_model_name="gpt-4o-realtime-preview-2024-12-17",
                )
            ],
            supports_realtime=True,
        ),
        "claude-realtime": ModelConfig(
            providers=[ModelProviderConfig(provider="anthropic")], supports_realtime=True
        ),
        # Deliberately not marked realtime-capable (tests the gate).
        "gpt-nonrealtime": ModelConfig(
            providers=[ModelProviderConfig(provider="openai", provider_model_name="gpt-4o")]
        ),
    }
    app = _build_app(monkeypatch, provider_configs=provider_configs, model_configs=model_configs)

    # Capture the upstream connection attempt and return a scripted connection.
    captured = {}

    async def fake_connect(url, headers, timeout=30.0, subprotocols=None):
        captured["url"] = url
        captured["headers"] = headers
        return captured["upstream"]

    monkeypatch.setattr("llm_proxy.api.routers.realtime.connect_upstream", fake_connect)

    app.state._realtime_captured = captured
    return app


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


def _connect(client, model="gpt-realtime", headers=None):
    return client.websocket_connect(
        f"/v1/realtime?model={model}",
        headers=headers if headers is not None else {"Authorization": "Bearer test-key"},
    )


def _expect_close(ws, code: int) -> None:
    """The next receive must raise WebSocketDisconnect with the given close code."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        ws.receive_json()
    assert exc_info.value.code == code


class TestRealtimeWebSocket:
    def test_auth_required(self, client):
        """No API key → error event and close 4401."""
        with _connect(client, headers={}) as ws:
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["event_id"].startswith("event_")
            assert error["error"]["code"] == "authentication_failed"
            _expect_close(ws, 4401)

    def test_invalid_key_rejected(self, client, monkeypatch):
        """An unknown key is rejected like a missing one."""

        monkeypatch.setattr(
            "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
            AsyncMock(return_value=None),
        )
        with _connect(client) as ws:
            error = ws.receive_json()
            assert error["error"]["code"] == "authentication_failed"
            _expect_close(ws, 4401)

    def test_missing_model_param(self, client):
        """No model query param → error event and close 4004 (invalid model)."""
        with client.websocket_connect(
            "/v1/realtime", headers={"Authorization": "Bearer test-key"}
        ) as ws:
            error = ws.receive_json()
            assert error["event_id"].startswith("event_")
            assert error["error"]["code"] == "invalid_request"
            _expect_close(ws, 4004)

    def test_unknown_model(self, client):
        """Unknown model → error event and close 4004 (official invalid-model code)."""
        with _connect(client, model="nope") as ws:
            error = ws.receive_json()
            assert error["event_id"].startswith("event_")
            assert error["error"]["code"] == "model_not_found"
            _expect_close(ws, 4004)

    def test_model_not_realtime_capable_rejected(self, client):
        """A configured model not marked Realtime-capable is rejected."""
        with _connect(client, model="gpt-nonrealtime") as ws:
            error = ws.receive_json()
            assert error["error"]["code"] == "model_not_supported"
            _expect_close(ws, 4004)

    def test_unsupported_provider_type(self, client):
        """A model routed to a non-realtime provider type is rejected."""
        with _connect(client, model="claude-realtime") as ws:
            error = ws.receive_json()
            assert error["error"]["code"] == "provider_not_supported"
            assert "anthropic" in error["error"]["message"]
            _expect_close(ws, 1011)

    def test_model_restriction_enforced(self, client, monkeypatch):
        """A key whose allowlist excludes the model is rejected."""

        async def restricted_verify(api_key: str):
            return {
                "principal_id": "test-key",
                "allowed_models": ["other-model"],
                "allowed_mcp_servers": None,
                "user_id": None,
            }

        monkeypatch.setattr(
            "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp", restricted_verify
        )
        with _connect(client) as ws:
            error = ws.receive_json()
            assert error["error"]["code"] == "forbidden"
            _expect_close(ws, 4403)

    def test_budget_exceeded_rejected(self, client, monkeypatch):
        """A key/account at its spending cap → error event and close 4007."""
        from llm_proxy.api.middleware.mcp_proxy import BudgetCheckStatus

        monkeypatch.setattr(
            "llm_proxy.api.middleware.mcp_proxy.check_key_budget",
            AsyncMock(return_value=BudgetCheckStatus.EXCEEDED),
        )
        with _connect(client) as ws:
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "budget_exceeded"
            _expect_close(ws, 4007)

    def test_budget_unavailable_fails_closed(self, client, monkeypatch):
        """Unconfirmable spend fails closed → server_error event and close 1011."""
        from llm_proxy.api.middleware.mcp_proxy import BudgetCheckStatus

        monkeypatch.setattr(
            "llm_proxy.api.middleware.mcp_proxy.check_key_budget",
            AsyncMock(return_value=BudgetCheckStatus.UNAVAILABLE),
        )
        with _connect(client) as ws:
            error = ws.receive_json()
            assert error["error"]["type"] == "server_error"
            assert error["error"]["code"] == "budget_unavailable"
            _expect_close(ws, 1011)

    def test_relay_roundtrip_and_usage_log(self, app, client):
        """Messages relay both ways; response.done writes a usage log."""
        upstream = FakeUpstream(messages=[_session_created(), _response_done()])
        app.state._realtime_captured["upstream"] = upstream

        with _connect(client) as ws:
            ws.send_json({"type": "session.update", "session": {"instructions": "hi"}})
            first = ws.receive_json()
            second = ws.receive_json()
            assert first["type"] == "session.created"
            assert second["type"] == "response.done"

        # The client message reached the upstream verbatim.
        assert upstream.sent == [
            orjson.dumps({"type": "session.update", "session": {"instructions": "hi"}}).decode()
        ]
        # The upstream connection used the derived URL and injected auth.
        # GA models get no legacy beta header (official GA migration guidance).
        captured = app.state._realtime_captured
        assert captured["url"] == "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        assert captured["headers"]["Authorization"] == "Bearer sk-upstream"
        assert "OpenAI-Beta" not in captured["headers"]

        # response.done produced one background request log with usage.
        logs = app.state._realtime_logs
        assert len(logs) == 1
        log = logs[0]
        assert log.request_id == "rt_resp_123"
        assert log.model == "gpt-realtime"
        assert log.provider == "openai"
        assert log.prompt_tokens == 500
        assert log.completion_tokens == 500
        assert log.audio_input_tokens == 100
        assert log.audio_output_tokens == 300

    def test_preview_model_gets_beta_header_and_safety_id_forwarded(self, app, client):
        """Legacy preview models get the beta header; the client's safety
        identifier is forwarded upstream."""
        upstream = FakeUpstream(messages=[_session_created()])
        app.state._realtime_captured["upstream"] = upstream

        with _connect(
            client,
            model="gpt-4o-realtime-preview-2024-12-17",
            headers={
                "Authorization": "Bearer test-key",
                "OpenAI-Safety-Identifier": "hashed-user-id",
            },
        ) as ws:
            ws.receive_json()

        captured = app.state._realtime_captured
        assert captured["url"] == (
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        )
        assert captured["headers"]["OpenAI-Beta"] == "realtime=v1"
        assert captured["headers"]["OpenAI-Safety-Identifier"] == "hashed-user-id"

    def test_no_safety_identifier_header_sent_when_client_omits_it(self, app, client):
        """The safety identifier header is absent when the client sends none."""
        upstream = FakeUpstream(messages=[_session_created()])
        app.state._realtime_captured["upstream"] = upstream

        with _connect(client) as ws:
            ws.receive_json()

        captured = app.state._realtime_captured
        assert "OpenAI-Safety-Identifier" not in captured["headers"]

    def test_session_id_captured_from_upstream(self, app, client):
        """session.created announces the upstream session id for log correlation."""
        upstream = FakeUpstream(
            messages=[
                orjson.dumps(
                    {
                        "type": "session.created",
                        "event_id": "event_1",
                        "session": {"id": "sess_xyz", "model": "gpt-realtime"},
                    }
                ).decode(),
                _response_done(),
            ]
        )
        app.state._realtime_captured["upstream"] = upstream

        with _connect(client) as ws:
            ws.send_json({"type": "response.create"})
            ws.receive_json()  # session.created
            ws.receive_json()  # response.done

        logs = app.state._realtime_logs
        assert len(logs) == 1
        assert logs[0].session_id == "sess_xyz"

    def test_response_done_without_usage_logs_zero_entry(self, app, client):
        """A turn without usage still logs (zero tokens, usage_missing flag)."""
        upstream = FakeUpstream(messages=[_session_created(), _response_done_without_usage()])
        app.state._realtime_captured["upstream"] = upstream

        with _connect(client) as ws:
            ws.send_json({"type": "response.create"})
            ws.receive_json()  # session.created
            ws.receive_json()  # response.done

        logs = app.state._realtime_logs
        assert len(logs) == 1
        log = logs[0]
        assert log.request_id == "rt_resp_124"
        # No usage → no token data (None, matching the REST logging paths).
        assert log.prompt_tokens is None
        assert log.completion_tokens is None
        assert log.log_metadata["response_status"] == "failed"
        assert log.log_metadata["usage_missing"] is True

    def test_upstream_connect_failure(self, app, client, monkeypatch):
        """Upstream connection failure → server_error event and close 1011."""

        async def failing_connect(url, headers, timeout=30.0, subprotocols=None):
            raise ConnectionError("upstream refused")

        monkeypatch.setattr("llm_proxy.api.routers.realtime.connect_upstream", failing_connect)

        with _connect(client) as ws:
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["error"]["type"] == "server_error"
            assert error["error"]["code"] == "upstream_connection_failed"
            _expect_close(ws, 1011)

    def test_browser_subprotocol_auth(self, app, client):
        """The openai-insecure-api-key subprotocol authenticates browser clients."""
        upstream = FakeUpstream(messages=[_session_created()])
        app.state._realtime_captured["upstream"] = upstream

        with client.websocket_connect(
            "/v1/realtime?model=gpt-realtime",
            subprotocols=["realtime", "openai-insecure-api-key.test-key"],
        ) as ws:
            event = ws.receive_json()
            assert event["type"] == "session.created"


# =============================================================================
# End-to-end relay test against a real in-process upstream WebSocket server.
# Exercises the actual websockets client: handshake, subprotocol negotiation,
# header injection, and bidirectional frame relay.
# =============================================================================


class _UpstreamServer:
    """Real websockets server speaking a minimal Realtime protocol."""

    def __init__(self):
        self.received: list[str] = []
        self.negotiated_subprotocol: str | None = None
        self.request_headers: dict[str, str] = {}
        self._stop = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.port: int | None = None

    async def _handler(self, websocket):
        self.negotiated_subprotocol = websocket.subprotocol
        self.request_headers = dict(websocket.request.headers)
        await websocket.send(_session_created())
        async for message in websocket:
            self.received.append(message)
            if orjson.loads(message).get("type") == "session.update":
                await websocket.send(_response_done())

    async def _main(self):
        async with serve(self._handler, "127.0.0.1", 0, subprotocols=["realtime"]) as server:
            self.port = server.sockets[0].getsockname()[1]
            await self._stop.wait()

    def start(self):
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._main())

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        for _ in range(200):
            if self.port is not None:
                return
            time.sleep(0.01)
        raise RuntimeError("Upstream server did not start")

    def stop(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)


@pytest.fixture
def upstream_server():
    server = _UpstreamServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def e2e_app(monkeypatch, upstream_server):
    """App wired to the real upstream server (no connect_upstream patch)."""
    provider_configs = {
        "openai": ProviderConfig(
            type="openai",
            api_key="sk-upstream",
            base_url=f"http://127.0.0.1:{upstream_server.port}/v1",
        ),
    }
    model_configs = {
        "gpt-realtime": ModelConfig(
            providers=[ModelProviderConfig(provider="openai", provider_model_name="gpt-realtime")],
            supports_realtime=True,
        ),
    }
    return _build_app(monkeypatch, provider_configs=provider_configs, model_configs=model_configs)


class TestRealtimeWebSocketE2E:
    def test_full_relay_through_real_websockets_client(self, e2e_app, upstream_server):
        """A real client message reaches the real upstream and back."""
        with (
            TestClient(e2e_app) as client,
            client.websocket_connect(
                "/v1/realtime?model=gpt-realtime",
                headers={"Authorization": "Bearer test-key"},
                subprotocols=["realtime"],
            ) as ws,
        ):
            ws.send_json({"type": "session.update", "session": {"instructions": "hi"}})
            first = ws.receive_json()
            second = ws.receive_json()
            assert first["type"] == "session.created"
            assert second["type"] == "response.done"

        # The real websockets client negotiated the realtime subprotocol and
        # injected the provider auth header on the upstream handshake (the
        # server-side header dict lowercases keys). GA models get no legacy
        # beta header (official GA migration guidance).
        assert upstream_server.negotiated_subprotocol == "realtime"
        assert upstream_server.request_headers.get("authorization") == "Bearer sk-upstream"
        assert upstream_server.request_headers.get("openai-beta") is None

        # The client message arrived at the upstream verbatim.
        assert upstream_server.received == [
            orjson.dumps({"type": "session.update", "session": {"instructions": "hi"}}).decode()
        ]

        # response.done produced the per-turn usage log.
        logs = e2e_app.state._realtime_logs
        assert len(logs) == 1
        assert logs[0].request_id == "rt_resp_123"
        assert logs[0].audio_input_tokens == 100
        assert logs[0].audio_output_tokens == 300
