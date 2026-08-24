"""Integration tests for OpenResponses API endpoints."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import require_api_key_auth
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.protocols.openresponses.store import ResponseStore


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    redis.delete = AsyncMock(return_value=0)
    return redis


@pytest.fixture
def response_store(mock_redis):
    """ResponseStore fixture."""
    return ResponseStore(redis_client=mock_redis, ttl=86400)


@pytest.fixture
def app(mock_redis):
    """Create test FastAPI app."""
    from llm_proxy.api.routers.openresponses import (
        get_response_store_required,
        router,
    )

    app = FastAPI()
    app.dependency_overrides[require_api_key_auth] = lambda: None

    @app.middleware("http")
    async def set_test_identity_middleware(request, call_next):
        """Set a test API-key identity so storage is tenant-scoped."""
        from llm_proxy.core.identity import RequestIdentity, set_request_identity

        set_request_identity(
            request,
            RequestIdentity(api_key_name="test-key", auth_method="api_key"),
        )
        request.state.api_key_name = "test-key"
        return await call_next(request)

    from llm_proxy.api.middleware.form_encoded import form_encoded_middleware

    app.middleware("http")(form_encoded_middleware)

    async def mock_get_response_store_required():
        return ResponseStore(redis_client=mock_redis, ttl=86400)

    app.dependency_overrides[get_response_store_required] = mock_get_response_store_required
    register_exception_handlers(app)
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Test client."""
    with TestClient(app) as client:
        yield client


class TestOpenResponsesRouter:
    """Integration tests for OpenResponses router."""

    def test_get_response_endpoint_exists(self, client):
        """Test that GET endpoint exists."""
        # This will return 404 for nonexistent response, but endpoint exists
        response = client.get("/v1/responses/resp_test")
        # Should not get 405 Method Not Allowed
        assert response.status_code != 405

    def test_delete_response_endpoint_exists(self, client):
        """Test that DELETE endpoint exists."""
        response = client.delete("/v1/responses/resp_test")
        # Should not get 405 Method Not Allowed
        assert response.status_code != 405


class TestGetResponseOutputItems:
    """GET /v1/responses/{id} must round-trip proxy-emitted output items.

    Regression: ItemField previously lacked a catch-all, so stored responses
    containing custom_tool_call / web_search_call / tool_search_call /
    local_shell_call output items (all emitted by format_response) failed
    ResponsesResponse validation and the endpoint returned 500.
    """

    @pytest.mark.parametrize(
        "item",
        [
            {
                "type": "custom_tool_call",
                "id": "i1",
                "call_id": "c1",
                "name": "apply_patch",
                "input": "***patch***",
                "status": "completed",
            },
            {
                "type": "web_search_call",
                "id": "ws1",
                "status": "completed",
                "action": {"type": "search", "query": "q", "queries": ["q"]},
            },
            {
                "type": "tool_search_call",
                "id": "ts1",
                "call_id": "c1",
                "status": "completed",
                "execution": "client",
                "arguments": {},
            },
            {
                "type": "local_shell_call",
                "id": "ls1",
                "call_id": "c1",
                "status": "completed",
                "action": {"type": "exec"},
            },
        ],
        ids=lambda i: i["type"],
    )
    def test_get_response_with_proxy_emitted_items(self, client, mock_redis, item):
        import orjson

        stored = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5",
            "output": [item],
        }
        mock_redis.get = AsyncMock(return_value=orjson.dumps(stored))
        resp = client.get("/v1/responses/resp_test")
        assert resp.status_code == 200, resp.text
        assert resp.json()["output"][0]["type"] == item["type"]

    def test_get_response_strips_stored_input(self, client, mock_redis):
        """GET must not leak the internal materialized ``input`` to clients.

        The stored body carries ``input`` so previous_response_id
        continuations can replay the conversation, but the spec's
        ResponseResource has no ``input`` field.
        """
        import orjson

        stored = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5",
            "output": [],
            "input": [{"type": "message", "role": "user", "content": "secret context"}],
        }
        mock_redis.get = AsyncMock(return_value=orjson.dumps(stored))
        resp = client.get("/v1/responses/resp_test")
        assert resp.status_code == 200, resp.text
        assert "input" not in resp.json()

    def test_delete_response_returns_deleted_true(self, client, mock_redis):
        mock_redis.delete = AsyncMock(return_value=1)
        resp = client.delete("/v1/responses/resp_test")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "id": "resp_test",
            "deleted": True,
            "object": "response.deleted",
        }

    def test_delete_response_not_found_returns_404(self, client, mock_redis):
        """OpenAI parity: deleting an unknown response id is a 404."""
        mock_redis.delete = AsyncMock(return_value=0)
        resp = client.delete("/v1/responses/resp_missing")
        assert resp.status_code == 404, resp.text


class TestOpenResponsesPostValidation:
    """HTTP-layer validation of POST /v1/responses request bodies.

    Reproduces the Codex 422 ``Input should be a valid string`` regression at the
    FastAPI body-validation layer (the real app declares ``ResponsesRequest``
    as the body model for POST /v1/responses).
    """

    @pytest.fixture
    def post_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from llm_proxy.api.middleware.exceptions import register_exception_handlers
        from llm_proxy.protocols.openresponses.schemas import ResponsesRequest

        app = FastAPI()

        @app.post("/v1/responses")
        async def _responses(body: ResponsesRequest) -> dict:
            # If we get here, body validation passed (no 422).
            return {
                "ok": True,
                "input_len": len(body.input) if hasattr(body.input, "__len__") else 1,
            }

        register_exception_handlers(app)
        with TestClient(app) as c:
            yield c

    def test_codex_turn2_payload_accepted(self, post_client):
        """A Codex turn-2 payload with function_call_output.output as a content
        items array must pass body validation (no 422 string_type)."""
        payload = {
            "model": "deepseek-v4-pro",
            "instructions": "You are a coding assistant.",
            "stream": True,
            "store": False,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "text": {"type": "text"},
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "<permissions instructions>"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "check the latest CI run"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Let me run gh to check."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_00_x",
                    "name": "exec_command",
                    "arguments": '{"cmd": "gh run list --limit 5"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_00_x",
                    "output": [{"type": "input_text", "text": "1  ci  failure"}],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Runs a command.",
                    "parameters": {"type": "object"},
                }
            ],
        }
        resp = post_client.post("/v1/responses", json=payload)
        assert resp.status_code == 200, resp.text
        assert resp.json()["ok"] is True

    def test_function_call_output_string_output_still_accepted(self, post_client):
        """The original plain-string output form must keep working."""
        payload = {
            "model": "gpt-4",
            "input": [{"type": "function_call_output", "call_id": "c1", "output": "done"}],
        }
        resp = post_client.post("/v1/responses", json=payload)
        assert resp.status_code == 200, resp.text

    def test_function_call_output_array_output_no_string_type_422(self, post_client):
        """The array output form must not produce the string_type 422 the user saw."""
        payload = {
            "model": "gpt-4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": [{"type": "input_text", "text": "ok"}],
                }
            ],
        }
        resp = post_client.post("/v1/responses", json=payload)
        # The user's regression was a 422 with detail
        # {"type": "string_type", "loc": ["body", "input", "str"], ...}.
        assert resp.status_code == 200, resp.text

    def test_codex_local_shell_call_payload_accepted(self, post_client):
        """A Codex payload using the native local_shell_call item (and other
        Codex item types) must pass body validation."""
        payload = {
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": "list files"},
                {
                    "type": "local_shell_call",
                    "call_id": "call_1",
                    "status": "completed",
                    "action": {"type": "exec", "command": ["ls"]},
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
                {
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "q"},
                },
                {"type": "some_future_call", "x": 1},
            ],
        }
        resp = post_client.post("/v1/responses", json=payload)
        assert resp.status_code == 200, resp.text


class _FakeRedis:
    """Minimal in-memory stand-in for the async redis client."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    async def setex(self, key: str, ttl: int, value: bytes) -> None:
        self.data[key] = value

    async def get(self, key: str) -> bytes | None:
        return self.data.get(key)

    async def delete(self, key: str) -> int:
        return 1 if self.data.pop(key, None) is not None else 0


class TestBackgroundMode:
    """OpenResponses background mode returns immediately and persists."""

    def test_background_returns_in_progress_immediately(self, client, monkeypatch):
        """A background request returns an in_progress response right away."""
        from fastapi.responses import JSONResponse

        from llm_proxy.api.routers import protocol as protocol_module
        from llm_proxy.api.routers.openresponses import get_response_store_required
        from llm_proxy.protocols.openresponses.store import ResponseStore
        from llm_proxy.protocols.registry import get_protocol

        # The default fixture app only has GET/DELETE; add the POST router.
        client.app.include_router(
            protocol_module.create_protocol_router(get_protocol("openresponses"))
        )

        async def fake_process(protocol_request=None, req=None, context=None):
            return JSONResponse(
                content={
                    "id": "resp_inner",
                    "object": "response",
                    "status": "completed",
                    "model": "gpt-5.2",
                    "output": [],
                }
            )

        processor = MagicMock()
        processor.process = fake_process
        client.app.state.openresponses_processor = processor

        async def fake_build_context(request, req, protocol_name=None):
            return MagicMock()

        monkeypatch.setattr(protocol_module, "build_request_context", fake_build_context)

        # Background mode requires response storage to be pollable.
        fake_redis = _FakeRedis()
        client.app.state.redis_client = MagicMock()
        client.app.state.redis_client.client = fake_redis
        client.app.dependency_overrides[get_response_store_required] = lambda: ResponseStore(
            redis_client=fake_redis
        )

        response = client.post(
            "/v1/responses",
            json={"model": "gpt-5.2", "input": "hi", "background": True},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "in_progress"
        assert body["background"] is True
        assert body["store"] is True
        assert body["id"].startswith("resp_")

    def test_background_without_redis_fails_fast(self, client, monkeypatch):
        """Without response storage, background mode returns 503 instead of a
        dangling in_progress id that could never resolve."""
        from fastapi.responses import JSONResponse

        from llm_proxy.api.routers import protocol as protocol_module
        from llm_proxy.protocols.registry import get_protocol

        client.app.include_router(
            protocol_module.create_protocol_router(get_protocol("openresponses"))
        )

        async def fake_process(protocol_request=None, req=None, context=None):
            return JSONResponse(content={"id": "resp_inner", "status": "completed"})

        processor = MagicMock()
        processor.process = fake_process
        client.app.state.openresponses_processor = processor

        async def fake_build_context(request, req, protocol_name=None):
            return MagicMock()

        monkeypatch.setattr(protocol_module, "build_request_context", fake_build_context)

        # No app.state.redis_client: storage unavailable.
        response = client.post(
            "/v1/responses",
            json={"model": "gpt-5.2", "input": "hi", "background": True},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "redis_not_available"

    def test_background_persists_in_progress_then_completed(self, client, monkeypatch):
        """The id is pollable immediately and the final body carries the input."""
        from fastapi.responses import JSONResponse

        from llm_proxy.api.routers import protocol as protocol_module
        from llm_proxy.api.routers.openresponses import get_response_store_required
        from llm_proxy.protocols.openresponses.store import ResponseStore
        from llm_proxy.protocols.registry import get_protocol

        client.app.include_router(
            protocol_module.create_protocol_router(get_protocol("openresponses"))
        )

        async def fake_process(protocol_request=None, req=None, context=None):
            # Keep the response in_progress long enough for the test to observe
            # the pollable placeholder state.
            import asyncio

            await asyncio.sleep(1)
            return JSONResponse(
                content={
                    "id": "resp_inner",
                    "object": "response",
                    "status": "completed",
                    "model": "gpt-5.2",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "id": "msg_1",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "hi there"}],
                        }
                    ],
                }
            )

        processor = MagicMock()
        processor.process = fake_process
        client.app.state.openresponses_processor = processor

        async def fake_build_context(request, req, protocol_name=None):
            return MagicMock()

        monkeypatch.setattr(protocol_module, "build_request_context", fake_build_context)

        fake_redis = _FakeRedis()
        client.app.state.redis_client = MagicMock()
        client.app.state.redis_client.client = fake_redis
        client.app.dependency_overrides[get_response_store_required] = lambda: ResponseStore(
            redis_client=fake_redis
        )

        response = client.post(
            "/v1/responses",
            json={"model": "gpt-5.2", "input": "hi", "background": True},
        )
        response_id = response.json()["id"]

        # The in_progress placeholder is persisted synchronously, so the id is
        # pollable (and returns in_progress) before the task has completed.
        poll = client.get(f"/v1/responses/{response_id}")
        assert poll.status_code == 200
        assert poll.json()["status"] == "in_progress"

        # Wait for the background task to finish and overwrite the stored body.
        import time

        deadline = time.monotonic() + 5
        final = poll.json()
        while time.monotonic() < deadline:
            final = client.get(f"/v1/responses/{response_id}").json()
            if final.get("status") != "in_progress":
                break
            time.sleep(0.01)
        assert final["status"] == "completed"
        assert final["id"] == response_id
        # The spec's ResponseResource has no ``input`` field: GET must not
        # leak the stored materialized input to clients.
        assert "input" not in final
        assert final["output"][0]["content"][0]["text"] == "hi there"

        # The stored body (used by previous_response_id continuations) still
        # carries the materialized input.
        import orjson

        stored_bodies = [orjson.loads(v) for v in fake_redis.data.values()]
        completed = next(b for b in stored_bodies if b.get("status") == "completed")
        assert completed["input"] == [{"type": "message", "role": "user", "content": "hi"}]


class TestCompactionEndpointHTTP:
    """HTTP-level tests for POST /v1/responses/compact."""

    def test_compact_materializes_previous_response_input_and_output(self, client):
        """A previous response with string input is wrapped, not shredded."""
        from llm_proxy.api.routers.openresponses import get_response_store_required
        from llm_proxy.protocols.openresponses.store import ResponseStore

        fake_redis = _FakeRedis()
        client.app.state.redis_client = MagicMock()
        client.app.state.redis_client.client = fake_redis
        client.app.dependency_overrides[get_response_store_required] = lambda: ResponseStore(
            redis_client=fake_redis
        )

        store = ResponseStore(redis_client=fake_redis)
        asyncio.run(
            store.store(
                "test-key",
                "resp_prev",
                {
                    "id": "resp_prev",
                    "input": "What is 2+2?",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "4"}],
                        }
                    ],
                },
            )
        )

        response = client.post(
            "/v1/responses/compact",
            json={
                "model": "gpt-5.2",
                "previous_response_id": "resp_prev",
                "input": "What is 2+2?",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "response.compaction"
        assert body["output"][0]["type"] == "compaction"

    def test_compact_unknown_previous_response_returns_not_found(self, client):
        """Missing previous responses fail with previous_response_not_found."""
        fake_redis = _FakeRedis()
        client.app.state.redis_client = MagicMock()
        client.app.state.redis_client.client = fake_redis

        response = client.post(
            "/v1/responses/compact",
            json={
                "model": "gpt-5.2",
                "previous_response_id": "resp_missing",
                "input": "hi",
            },
        )
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "previous_response_not_found"

    def test_compact_accepts_codex_extra_fields(self, client):
        """Codex-style compact requests (tools/reasoning/text/...) are not rejected."""
        response = client.post(
            "/v1/responses/compact",
            json={
                "model": "gpt-5.3-codex",
                "input": "hi",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "d",
                        "parameters": {"type": "object"},
                    }
                ],
                "parallel_tool_calls": True,
                "reasoning": {"effort": "high"},
                "service_tier": "auto",
                "text": {"format": {"type": "text"}, "verbosity": "low"},
                "prompt_cache_options": {"control": {"type": "ephemeral"}},
                "prompt_cache_retention": "24h",
                "prompt_cache_key": "cache-1",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "response.compaction"
        assert body["output"][0]["type"] == "compaction"


class TestCompactPassthrough:
    """POST /v1/responses/compact forwards to a native Responses upstream."""

    @staticmethod
    def _patch_context(monkeypatch, adapter):
        from llm_proxy.api.routers import openresponses as or_module

        class FakeOrchestrator:
            def select_next_provider(self):
                return object()

        class FakeContext:
            orchestrator = FakeOrchestrator()

            async def adapter_factory(self, req, selection):
                return adapter

        async def fake_build_request_context(request, req, protocol_name=None):
            return FakeContext()

        monkeypatch.setattr(or_module, "build_request_context", fake_build_request_context)

    def test_compact_passthrough_to_native_upstream(self, client, monkeypatch):
        """A native Responses upstream performs the compaction; the raw body is
        forwarded verbatim and the upstream response returned as-is."""

        class FakeAdapter:
            def _target_endpoint(self):
                return "responses"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def compact_response(self, raw):
                assert raw["model"] == "gpt-5.2"
                assert raw["previous_response_id"] == "resp_x"
                return 200, {
                    "id": "compaction_upstream",
                    "object": "response.compaction",
                    "output": [{"type": "compaction", "encrypted_content": "UPSTREAM_BLOB"}],
                }

        self._patch_context(monkeypatch, FakeAdapter())
        response = client.post(
            "/v1/responses/compact",
            json={"model": "gpt-5.2", "previous_response_id": "resp_x", "input": "hi"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == "compaction_upstream"
        assert body["output"][0]["encrypted_content"] == "UPSTREAM_BLOB"

    def test_compact_passthrough_upstream_error_falls_back_to_local(self, client, monkeypatch):
        """Upstream error statuses fall back to the local lossless packing
        instead of surfacing the upstream error."""

        class FakeAdapter:
            def _target_endpoint(self):
                return "responses"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def compact_response(self, raw):
                return 429, {"error": {"message": "rate limited", "type": "rate_limit_error"}}

        self._patch_context(monkeypatch, FakeAdapter())
        response = client.post("/v1/responses/compact", json={"model": "gpt-5.2", "input": "hi"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "response.compaction"
        assert body["output"][0]["type"] == "compaction"

    def test_compact_falls_back_to_local_for_non_native(self, client, monkeypatch):
        """Non-native providers keep the local lossless packing behavior."""

        class FakeAdapter:
            def _target_endpoint(self):
                return "chat_completions"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        self._patch_context(monkeypatch, FakeAdapter())
        response = client.post(
            "/v1/responses/compact", json={"model": "claude-sonnet", "input": "hi"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "response.compaction"
        assert body["output"][0]["type"] == "compaction"
        # Locally produced blob, not an upstream one.
        assert body["output"][0]["created_by"] == "llm-proxy"

    def test_compact_falls_back_to_local_on_passthrough_error(self, client, monkeypatch):
        """Provider resolution/transport failures degrade to local compaction."""
        from llm_proxy.api.routers import openresponses as or_module

        async def failing_build_request_context(request, req, protocol_name=None):
            raise RuntimeError("no providers configured")

        monkeypatch.setattr(or_module, "build_request_context", failing_build_request_context)
        response = client.post("/v1/responses/compact", json={"model": "gpt-5.2", "input": "hi"})
        assert response.status_code == 200, response.text
        assert response.json()["output"][0]["type"] == "compaction"


class TestPathAliases:
    """/responses and /v1/v1/responses aliases exist for base_url-tolerant routing."""

    def test_post_responses_path_aliases_registered(self):
        from llm_proxy.api.routers.protocol import create_protocol_router
        from llm_proxy.protocols.registry import get_protocol

        endpoint = get_protocol("openresponses")
        assert endpoint is not None
        router = create_protocol_router(endpoint)
        post_paths = {route.path for route in router.routes if "POST" in (route.methods or set())}
        assert "/v1/responses" in post_paths
        assert "/responses" in post_paths
        assert "/v1/v1/responses" in post_paths

    def test_compact_path_aliases(self, client):
        """The compact endpoint answers on the alias paths too."""
        for path in ("/responses/compact", "/v1/v1/responses/compact"):
            resp = client.post(path, json={"model": "gpt-5.2", "input": "hi"})
            assert resp.status_code == 200, (path, resp.text)
            assert resp.json()["object"] == "response.compaction"


class TestFormEncodedBody:
    """/v1/responses accepts application/x-www-form-urlencoded bodies."""

    def test_form_encoded_body_is_converted(self, client, monkeypatch):
        """Form-encoded params reach the handler as a JSON body."""
        from llm_proxy.api.routers import protocol as protocol_module
        from llm_proxy.protocols.registry import get_protocol

        client.app.include_router(
            protocol_module.create_protocol_router(get_protocol("openresponses"))
        )

        captured = {}

        async def fake_process(protocol_request=None, req=None, context=None):
            captured["raw"] = protocol_request
            from fastapi.responses import JSONResponse

            return JSONResponse(
                content={
                    "id": "resp_1",
                    "object": "response",
                    "status": "completed",
                    "model": protocol_request.model,
                    "output": [],
                }
            )

        processor = MagicMock()
        processor.process = fake_process
        client.app.state.openresponses_processor = processor

        async def fake_build_context(request, req, protocol_name=None):
            return MagicMock()

        monkeypatch.setattr(protocol_module, "build_request_context", fake_build_context)

        response = client.post(
            "/v1/responses",
            content="model=gpt-5.2&input=hello&temperature=0.5",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        assert captured["raw"].model == "gpt-5.2"
        assert captured["raw"].input == "hello"
        assert captured["raw"].temperature == 0.5

    def test_form_encoded_json_field_is_decoded(self, client, monkeypatch):
        """Structured form fields (input array) are JSON-decoded."""
        from llm_proxy.api.routers import protocol as protocol_module
        from llm_proxy.protocols.registry import get_protocol

        client.app.include_router(
            protocol_module.create_protocol_router(get_protocol("openresponses"))
        )

        captured = {}

        async def fake_process(protocol_request=None, req=None, context=None):
            captured["raw"] = protocol_request
            from fastapi.responses import JSONResponse

            return JSONResponse(
                content={
                    "id": "resp_1",
                    "object": "response",
                    "status": "completed",
                    "model": protocol_request.model,
                    "output": [],
                }
            )

        processor = MagicMock()
        processor.process = fake_process
        client.app.state.openresponses_processor = processor

        async def fake_build_context(request, req, protocol_name=None):
            return MagicMock()

        monkeypatch.setattr(protocol_module, "build_request_context", fake_build_context)

        import urllib.parse

        form = urllib.parse.urlencode(
            {
                "model": "gpt-5.2",
                "input": '[{"type": "message", "role": "user", "content": "hi"}]',
            }
        )
        response = client.post(
            "/v1/responses",
            content=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        assert isinstance(captured["raw"].input, list)
        assert captured["raw"].input[0].role == "user"
