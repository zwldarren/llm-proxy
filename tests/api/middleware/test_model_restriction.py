"""Tests for the model restriction middleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request

from llm_proxy.api.middleware import model_restriction as mr
from llm_proxy.api.middleware.model_restriction import (
    check_model_restriction,
    get_model_from_request_body,
    model_restriction_middleware,
)
from llm_proxy.core.identity import RequestIdentity


def _make_request(path: str, body: bytes | None = None) -> Request:
    """Build a Starlette Request with the given path and raw body."""

    async def receive():
        return {"type": "http.request", "body": body or b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": [],
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return Request(scope, receive)


class TestGetModelFromRequestBody:
    """Extracting the requested model from the request body."""

    def test_none_body_returns_none(self):
        assert get_model_from_request_body(None) is None

    def test_empty_body_returns_none(self):
        assert get_model_from_request_body(b"") is None

    def test_extracts_model_field(self):
        assert get_model_from_request_body(b'{"model": "gpt-4"}') == "gpt-4"

    def test_extracts_model_id_field(self):
        assert get_model_from_request_body(b'{"model_id": "claude-3"}') == "claude-3"

    def test_model_takes_precedence_over_model_id(self):
        body = b'{"model": "gpt-4", "model_id": "claude-3"}'
        assert get_model_from_request_body(body) == "gpt-4"

    def test_invalid_json_returns_none(self):
        assert get_model_from_request_body(b"not json") is None

    def test_non_dict_json_returns_none(self):
        assert get_model_from_request_body(b"[1, 2, 3]") is None


class TestCheckModelRestriction:
    """Evaluating whether a requested model is allowed for an API key."""

    @pytest.mark.parametrize(
        "allowed_models, requested_model, expected_allowed",
        [
            (None, "gpt-4", True),
            # Empty list means deny-all (e.g. a user-level constraint that
            # intersected to nothing), distinct from None = unrestricted.
            ([], "gpt-4", False),
            (["gpt-4"], None, True),
            (["gpt-4", "claude-3"], "gpt-4", True),
            (["gpt-4"], "claude-3", False),
        ],
    )
    def test_restriction_decision(self, allowed_models, requested_model, expected_allowed):
        allowed, error = check_model_restriction("key", allowed_models, requested_model)
        assert allowed == expected_allowed
        if expected_allowed:
            assert error is None
        else:
            assert error is not None
            assert "key" in error


class TestModelRestrictionMiddleware:
    """End-to-end behavior of the middleware's allow/deny decisions."""

    @pytest.fixture
    def call_next(self):
        return AsyncMock(return_value="NEXT_RESPONSE")

    @pytest.fixture
    def lockout_manager(self):
        manager = MagicMock()
        manager.record_failed_attempt = MagicMock()
        return manager

    async def _run(
        self,
        request: Request,
        call_next,
        *,
        auth_method: str = "api_key",
        lockout_manager: MagicMock | None = None,
    ):
        request.state.identity = RequestIdentity(auth_method=auth_method)
        with (
            patch.object(mr, "get_request_identity", lambda r: request.state.identity),
            patch.object(mr, "get_client_ip", return_value="1.2.3.4"),
            patch.object(mr, "get_api_key_lockout_manager", lambda: lockout_manager or MagicMock()),
            patch.object(mr, "add_auth_failure_delay", AsyncMock()),
        ):
            return await model_restriction_middleware(request, call_next)

    async def test_jwt_auth_bypasses_restriction(self, call_next, lockout_manager):
        """JWT-authenticated (admin) requests are never model-restricted."""
        request = _make_request("/v1/chat/completions", b'{"model": "claude-3"}')
        request.state.allowed_models = ["gpt-4"]

        result = await self._run(
            request, call_next, auth_method="jwt", lockout_manager=lockout_manager
        )

        assert result == "NEXT_RESPONSE"
        call_next.assert_awaited_once()
        lockout_manager.record_failed_attempt.assert_not_called()

    async def test_non_api_path_bypasses(self, call_next, lockout_manager):
        """Paths outside /v1/ and /servers/ are not restricted."""
        request = _make_request("/api/config", b'{"model": "claude-3"}')
        request.state.allowed_models = ["gpt-4"]

        result = await self._run(request, call_next, lockout_manager=lockout_manager)

        assert result == "NEXT_RESPONSE"
        lockout_manager.record_failed_attempt.assert_not_called()

    async def test_servers_path_bypasses_without_reading_body(self, call_next, lockout_manager):
        """MCP /servers/* JSON-RPC is bypassed without consuming the body."""
        request = _make_request("/servers/abc", b'{"model": "claude-3"}')
        request.state.allowed_models = ["gpt-4"]

        result = await self._run(request, call_next, lockout_manager=lockout_manager)

        assert result == "NEXT_RESPONSE"
        lockout_manager.record_failed_attempt.assert_not_called()

    async def test_no_restrictions_bypasses(self, call_next, lockout_manager):
        request = _make_request("/v1/chat/completions", b'{"model": "claude-3"}')
        request.state.allowed_models = None

        result = await self._run(request, call_next, lockout_manager=lockout_manager)

        assert result == "NEXT_RESPONSE"
        lockout_manager.record_failed_attempt.assert_not_called()

    async def test_allowed_model_passes(self, call_next, lockout_manager):
        request = _make_request("/v1/chat/completions", b'{"model": "gpt-4"}')
        request.state.allowed_models = ["gpt-4", "claude-3"]

        result = await self._run(request, call_next, lockout_manager=lockout_manager)

        assert result == "NEXT_RESPONSE"
        call_next.assert_awaited_once()
        lockout_manager.record_failed_attempt.assert_not_called()

    async def test_no_model_field_with_restrictions_allows(self, call_next, lockout_manager):
        """A missing model field is treated as 'no model specified' -> allow."""
        request = _make_request("/v1/chat/completions", b'{"messages": []}')
        request.state.allowed_models = ["gpt-4"]

        result = await self._run(request, call_next, lockout_manager=lockout_manager)

        assert result == "NEXT_RESPONSE"
        lockout_manager.record_failed_attempt.assert_not_called()

    async def test_disallowed_model_rejected(self, call_next, lockout_manager):
        request = _make_request("/v1/chat/completions", b'{"model": "claude-3"}')
        request.state.allowed_models = ["gpt-4"]
        request.state.api_key_name = "my-key"

        result = await self._run(request, call_next, lockout_manager=lockout_manager)

        assert result.status_code == 403
        payload = result.body
        assert b"model_not_allowed" in payload
        assert b"claude-3" in payload
        call_next.assert_not_awaited()
        lockout_manager.record_failed_attempt.assert_called_once_with("1.2.3.4")
