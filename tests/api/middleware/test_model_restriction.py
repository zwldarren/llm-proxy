"""Tests for the model restriction middleware."""

from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from llm_proxy.api.middleware import model_restriction as mr
from llm_proxy.api.middleware.model_restriction import (
    check_model_restriction,
    get_model_from_request_body,
    model_restriction_middleware,
)
from llm_proxy.core.request_facts import RequestIdentity, facts_for, set_request_identity


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

    async def _run(
        self,
        request: Request,
        call_next,
        *,
        identity: RequestIdentity | None = None,
    ):
        set_request_identity(request, identity or RequestIdentity(auth_method="api_key"))
        return await model_restriction_middleware(request, call_next)

    async def test_jwt_auth_bypasses_restriction(self, call_next):
        """JWT-authenticated (admin) requests are never model-restricted."""
        request = _make_request("/v1/chat/completions", b'{"model": "claude-3"}')
        facts_for(request).allowed_models = ["gpt-4"]

        result = await self._run(
            request,
            call_next,
            identity=RequestIdentity(auth_method="jwt"),
        )

        assert result == "NEXT_RESPONSE"
        call_next.assert_awaited_once()

    async def test_non_api_path_bypasses(self, call_next):
        """Paths outside /v1/ and /servers/ are not restricted."""
        request = _make_request("/api/config", b'{"model": "claude-3"}')
        facts_for(request).allowed_models = ["gpt-4"]

        result = await self._run(request, call_next)

        assert result == "NEXT_RESPONSE"

    async def test_servers_path_bypasses_without_reading_body(self, call_next):
        """MCP /servers/* JSON-RPC is bypassed without consuming the body."""
        request = _make_request("/servers/abc", b'{"model": "claude-3"}')
        facts_for(request).allowed_models = ["gpt-4"]

        result = await self._run(request, call_next)

        assert result == "NEXT_RESPONSE"

    async def test_no_restrictions_bypasses(self, call_next):
        request = _make_request("/v1/chat/completions", b'{"model": "claude-3"}')
        facts_for(request).allowed_models = None

        result = await self._run(request, call_next)

        assert result == "NEXT_RESPONSE"

    async def test_allowed_model_passes(self, call_next):
        request = _make_request("/v1/chat/completions", b'{"model": "gpt-4"}')
        facts_for(request).allowed_models = ["gpt-4", "claude-3"]

        result = await self._run(request, call_next)

        assert result == "NEXT_RESPONSE"
        call_next.assert_awaited_once()

    async def test_no_model_field_with_restrictions_allows(self, call_next):
        """A missing model field is treated as 'no model specified' -> allow."""
        request = _make_request("/v1/chat/completions", b'{"messages": []}')
        facts_for(request).allowed_models = ["gpt-4"]

        result = await self._run(request, call_next)

        assert result == "NEXT_RESPONSE"

    async def test_disallowed_model_rejected(self, call_next):
        request = _make_request("/v1/chat/completions", b'{"model": "claude-3"}')
        facts_for(request).allowed_models = ["gpt-4"]
        identity = RequestIdentity(auth_method="api_key", api_key_name="my-key")

        result = await self._run(request, call_next, identity=identity)

        assert result.status_code == 403
        payload = result.body
        assert b"model_not_allowed" in payload
        assert b"claude-3" in payload
        call_next.assert_not_awaited()

    @pytest.mark.parametrize(
        "path",
        [
            "/chat/completions",
            "/messages",
            "/v1/v1/chat/completions",
            "/v1/v1/messages",
        ],
    )
    async def test_alias_path_enforces_restriction(self, path, call_next):
        """Alias paths carry no /v1 prefix but are API endpoints: a model
        outside the key's allowlist must still be rejected."""
        request = _make_request(path, b'{"model": "claude-3"}')
        facts_for(request).allowed_models = ["gpt-4"]
        identity = RequestIdentity(auth_method="api_key", api_key_name="my-key")

        result = await self._run(request, call_next, identity=identity)

        assert result.status_code == 403
        assert b"model_not_allowed" in result.body
        call_next.assert_not_awaited()


def test_model_restriction_holds_no_lockout_coupling():
    """A model-allowlist rejection comes from an already-authenticated key.

    It is an authorization miss, not credential brute force: it must not feed
    the client IP's auth-failure lockout budget, and it must not make the client
    pay the auth failure delay. Otherwise one misconfigured allowlist gets the
    whole address locked out — every other key from the same client, and in
    development, every request from 127.0.0.1.
    """
    assert not hasattr(mr, "get_api_key_lockout_manager")
    assert not hasattr(mr, "add_auth_failure_delay")
