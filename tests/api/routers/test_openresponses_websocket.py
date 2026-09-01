"""Integration tests for the OpenResponses WebSocket transport."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from llm_proxy.api.dependencies import require_api_key_auth
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.openresponses import _parse_sse_blocks
from llm_proxy.protocols.openresponses.store import ResponseStore


@pytest.mark.parametrize(
    ("buffer", "expected"),
    [
        (
            # No-space spelling: relays may omit the space after the SSE
            # field colon (spec-equivalent); the WS transport must not drop
            # those events.
            'event:response.completed\ndata:{"type":"response.completed","response":{"id":"r1"}}\n\n',
            [{"type": "response.completed", "response": {"id": "r1"}}],
        ),
        (
            'event: response.completed\ndata: {"type":"response.completed"}\n\n',
            [{"type": "response.completed"}],
        ),
    ],
    ids=["no-space", "spaced"],
)
def test_parse_sse_blocks_accepts_both_sse_spellings(buffer, expected):
    """Both spaced and no-space SSE field spellings parse identically."""
    events, remainder = _parse_sse_blocks(buffer)
    assert events == expected
    assert remainder == ""


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    redis.delete = AsyncMock(return_value=0)
    return redis


@pytest.fixture
def app(mock_redis, monkeypatch):
    """Create test FastAPI app with a mocked processor."""
    from llm_proxy.api.routers import openresponses as router_module
    from llm_proxy.api.routers.openresponses import router, ws_router

    app = FastAPI()
    app.dependency_overrides[require_api_key_auth] = lambda: None

    async def mock_get_response_store_required():
        return ResponseStore(redis_client=mock_redis, ttl=86400)

    app.dependency_overrides[router_module.get_response_store_required] = (
        mock_get_response_store_required
    )
    register_exception_handlers(app)
    app.include_router(router)
    app.include_router(ws_router)

    # Mock API key verification so any key authenticates.
    async def fake_verify(api_key: str):
        return {
            "principal_id": "test-key",
            "allowed_models": None,
            "allowed_mcp_servers": None,
            "user_id": None,
        }

    monkeypatch.setattr("llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp", fake_verify)

    # Mock the context builder (the real one needs DB config).
    async def fake_build_context(body, req, protocol_name=None):
        return MagicMock()

    monkeypatch.setattr(router_module, "build_request_context", fake_build_context)

    # Default processor: echoes a canned stream.
    async def fake_process(protocol_request=None, req=None, context=None):
        chunks = [
            "event: response.created\n"
            'data: {"type":"response.created","response":{"id":"resp_ws1",'
            '"status":"in_progress"}}\n\n',
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","output_index":0,'
            '"content_index":0,"delta":"Hello"}\n\n',
            "event: response.completed\n"
            'data: {"type":"response.completed","response":{"id":"resp_ws1",'
            '"status":"completed",'
            '"output":[{"type":"message","role":"assistant","content":'
            '[{"type":"output_text","text":"Hello"}]}]}}\n\n',
            "data: [DONE]\n\n",
        ]
        return StreamingResponse(iter(chunks), media_type="text/event-stream")

    processor = MagicMock()
    processor.process = fake_process
    app.state.openresponses_processor = processor
    app.state.redis_client = MagicMock()
    app.state.redis_client.client = mock_redis
    return app


@pytest.fixture
def client(app):
    """Test client."""
    with TestClient(app) as client:
        yield client


def _client_for(app: FastAPI) -> TestClient:
    """Open a fresh TestClient for an app (used when tests swap app state)."""
    return TestClient(app)


def _ws_connect(client):
    """Open a WebSocket connection with an API key header."""
    return client.websocket_connect(
        "/v1/responses", headers={"Authorization": "Bearer test-key-123"}
    )


def _receive_events(ws, count):
    events = []
    for _ in range(count):
        events.append(ws.receive_json())
    return events


class TestOpenResponsesWebSocket:
    """WebSocket transport tests."""

    def test_response_create_streams_events(self, client):
        """A response.create message streams the same events as SSE."""
        with _ws_connect(client) as ws:
            ws.send_json({"type": "response.create", "model": "gpt-5.2", "input": "hi"})
            events = _receive_events(ws, 3)
            types = [e["type"] for e in events]
            assert types == [
                "response.created",
                "response.output_text.delta",
                "response.completed",
            ]
            assert events[0]["response"]["id"] == "resp_ws1"

    def test_sequential_response_create_messages(self, client):
        """Multiple response.create messages are processed sequentially."""
        with _ws_connect(client) as ws:
            ws.send_json({"type": "response.create", "model": "gpt-5.2", "input": "one"})
            _receive_events(ws, 3)
            ws.send_json({"type": "response.create", "model": "gpt-5.2", "input": "two"})
            events = _receive_events(ws, 3)
            assert [e["type"] for e in events] == [
                "response.created",
                "response.output_text.delta",
                "response.completed",
            ]

    def test_sse_blocks_split_across_chunks_are_buffered(self, app):
        """Events split across body_iterator chunks are not lost."""

        async def split_process(protocol_request=None, req=None, context=None):
            from fastapi.responses import StreamingResponse

            # Deliberately split every SSE block across chunk boundaries.
            block = (
                "event: response.output_text.delta\n"
                'data: {"type":"response.output_text.delta","output_index":0,'
                '"content_index":0,"delta":"Hi"}\n\n'
            )
            return StreamingResponse(
                (block[:7], block[7:14], block[14:]), media_type="text/event-stream"
            )

        processor = MagicMock()
        processor.process = split_process
        app.state.openresponses_processor = processor

        with _ws_connect(_client_for(app)) as ws:
            ws.send_json({"type": "response.create", "model": "gpt-5.2", "input": "hi"})
            event = ws.receive_json()
            assert event["type"] == "response.output_text.delta"
            assert event["delta"] == "Hi"

    def test_connection_limit_closes_idle_connections(self, app, monkeypatch):
        """The 60-minute limit is enforced even when no message arrives."""
        from llm_proxy.api.routers import openresponses as router_module

        monkeypatch.setattr(router_module, "WS_MAX_CONNECTION_SECONDS", 0.1)

        with _ws_connect(_client_for(app)) as ws:
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "websocket_connection_limit_reached"
            # The server closes the connection after the error envelope.
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()

    def test_continuation_from_connection_local_state(self, client):
        """A follow-up with previous_response_id uses connection-local state."""
        with _ws_connect(client) as ws:
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.2",
                    "input": "hi",
                    "store": False,
                }
            )
            _receive_events(ws, 3)
            # Continuation: only new input + previous_response_id.
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.2",
                    "previous_response_id": "resp_ws1",
                    "input": [{"type": "message", "role": "user", "content": "more"}],
                }
            )
            events = _receive_events(ws, 3)
            assert events[-1]["type"] == "response.completed"

    def test_missing_previous_response_returns_error(self, client):
        """An uncached previous_response_id fails with previous_response_not_found."""
        with _ws_connect(client) as ws:
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.2",
                    "previous_response_id": "resp_unknown",
                    "input": "hi",
                }
            )
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["status"] == 400
            assert error["error"]["code"] == "previous_response_not_found"
            assert error["error"]["param"] == "previous_response_id"

    def test_dangling_function_call_output_rejects_and_evicts(self, client):
        """A continuation whose function_call_output references an unknown
        call_id must fail with invalid_request and evict the referenced
        previous_response_id from connection-local state (spec 2026-04-24:
        failed continuations evict)."""
        with _ws_connect(client) as ws:
            # Seed connection-local state with a completed response.
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.2",
                    "input": "hi",
                    "store": False,
                }
            )
            _receive_events(ws, 3)

            # Continuation with a dangling function_call_output -> rejected.
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.2",
                    "previous_response_id": "resp_ws1",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_openresponses_missing",
                            "output": "No matching tool call exists.",
                        }
                    ],
                }
            )
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["status"] == 400
            assert error["error"]["code"] == "invalid_request"
            assert error["error"]["param"] == "call_id"

            # The failed continuation must have evicted the cached id.
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.2",
                    "previous_response_id": "resp_ws1",
                    "input": "retry",
                }
            )
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "previous_response_not_found"

    def test_valid_continuation_with_tool_output_passes(self, client):
        """A continuation whose function_call_output matches a function_call in
        the materialized context proceeds to the model."""
        with _ws_connect(client) as ws:
            # Seed connection-local state with a response carrying a tool call.
            async def fake_process_with_tool(protocol_request=None, req=None, context=None):
                chunks = [
                    "event: response.created\n"
                    'data: {"type":"response.created","response":{"id":"resp_ws1",'
                    '"status":"in_progress"}}\n\n',
                    "event: response.output_item.added\n"
                    'data: {"type":"response.output_item.added","item":{"id":"fc_1",'
                    '"type":"function_call","call_id":"call_123","name":"get_weather",'
                    '"arguments":"{}","status":"completed"}}\n\n',
                    "event: response.completed\n"
                    'data: {"type":"response.completed","response":{"id":"resp_ws1",'
                    '"status":"completed","output":[{"id":"fc_1","type":"function_call",'
                    '"call_id":"call_123","name":"get_weather","arguments":"{}",'
                    '"status":"completed"}]}}\n\n',
                    "data: [DONE]\n\n",
                ]
                return StreamingResponse(iter(chunks), media_type="text/event-stream")

            processor = client.app.state.openresponses_processor
            processor.process = fake_process_with_tool
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.2",
                    "input": "what is the weather",
                    "store": False,
                }
            )
            _receive_events(ws, 3)

            # Continuation with a matching function_call_output -> model runs.
            ws.send_json(
                {
                    "type": "response.create",
                    "model": "gpt-5.2",
                    "previous_response_id": "resp_ws1",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_123",
                            "output": "sunny",
                        }
                    ],
                }
            )
            events = _receive_events(ws, 3)
            assert events[-1]["type"] == "response.completed"

    def test_invalid_message_type_returns_error(self, client):
        """Non-response.create messages are rejected with an error envelope."""
        with _ws_connect(client) as ws:
            ws.send_json({"type": "response.cancel"})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "invalid_request"

    def test_unauthenticated_connection_is_rejected(self, client, monkeypatch):
        """Connections without a valid API key are rejected."""

        async def fake_verify_none(api_key: str):
            return None

        monkeypatch.setattr(
            "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
            fake_verify_none,
        )
        with client.websocket_connect(
            "/v1/responses", headers={"Authorization": "Bearer bad-key"}
        ) as ws:
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "authentication_failed"
            # The server closes the connection after the error envelope.
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
