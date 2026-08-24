"""Tests for the per-key rate limit enforced by the /v1/* and /servers/* middlewares."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from llm_proxy.api.middleware.api_key_auth import api_key_auth_middleware
from llm_proxy.api.middleware.rate_limiting import get_rate_limiter


class TestPerKeyRateLimit:
    """Keys with rate_limit_rpm set are capped; others pass through."""

    @pytest.fixture(autouse=True)
    def _reset_rate_limiter(self):
        """Isolate tests from the global limiter's in-memory windows."""
        limiter = get_rate_limiter()
        limiter._use_redis = False
        limiter._memory_windows.clear()
        yield
        limiter._memory_windows.clear()

    @pytest.fixture
    def app(self):
        app = FastAPI()
        # The middleware skips auth entirely when no config manager is present.
        app.state.config_manager = MagicMock()

        @app.middleware("http")
        async def auth(request, call_next):
            return await api_key_auth_middleware(request, call_next)

        @app.get("/v1/models")
        async def models():
            return {"data": []}

        return app

    def _patchers(self, auth_info: dict):
        return (
            patch(
                "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.middleware.mcp_proxy.is_key_budget_exceeded",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "llm_proxy.api.middleware.api_key_auth._update_key_last_used",
                new=AsyncMock(),
            ),
        )

    @pytest.mark.asyncio
    async def test_under_limit_passes(self, app, make_auth_info):
        p1, p2, p3 = self._patchers(make_auth_info(rate_limit_rpm=60))
        with p1, p2, p3:
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                for _ in range(3):
                    response = await client.get(
                        "/v1/models", headers={"Authorization": "Bearer sk-test"}
                    )
                    assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_over_limit_returns_429(self, app, make_auth_info):
        p1, p2, p3 = self._patchers(make_auth_info(rate_limit_rpm=2))
        with p1, p2, p3:
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                headers = {"Authorization": "Bearer sk-test"}
                assert (await client.get("/v1/models", headers=headers)).status_code == 200
                assert (await client.get("/v1/models", headers=headers)).status_code == 200
                response = await client.get("/v1/models", headers=headers)

        assert response.status_code == 429
        body = response.json()
        assert body["error"]["code"] == "rate_limit_exceeded"
        assert body["error"]["type"] == "rate_limit_error"
        assert "Retry-After" in response.headers

    @pytest.mark.asyncio
    async def test_no_limit_means_unlimited(self, app, make_auth_info):
        p1, p2, p3 = self._patchers(make_auth_info(rate_limit_rpm=None))
        with p1, p2, p3:
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                for _ in range(10):
                    response = await client.get(
                        "/v1/models", headers={"Authorization": "Bearer sk-test"}
                    )
                    assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_session_key_carries_no_rate_limit(self, app, make_auth_info):
        """Session keys (sk-ui-) never carry a rate-limit configuration."""
        info = make_auth_info(principal_id="session:abc-123")
        p1, p2, p3 = self._patchers(info)
        with p1, p2, p3:
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                for _ in range(5):
                    response = await client.get(
                        "/v1/models", headers={"Authorization": "Bearer sk-ui-test"}
                    )
                    assert response.status_code == 200


class TestMCPMiddlewareRateLimit:
    """MCP requests (/servers/*) are capped by the same per-key rpm."""

    @pytest.fixture(autouse=True)
    def _reset_rate_limiter(self):
        """Isolate tests from the global limiter's in-memory windows."""
        limiter = get_rate_limiter()
        limiter._use_redis = False
        limiter._memory_windows.clear()
        yield
        limiter._memory_windows.clear()

    @pytest.mark.asyncio
    async def test_over_limit_returns_429(self, run_mcp_middleware, make_auth_info):
        """A capped key is throttled on /servers/* once its window fills."""
        run = run_mcp_middleware
        status, _, _ = await run(make_auth_info(rate_limit_rpm=2))
        assert status == 200
        status, _, _ = await run(make_auth_info(rate_limit_rpm=2))
        assert status == 200
        status, body, headers = await run(make_auth_info(rate_limit_rpm=2))

        assert status == 429
        assert body["error"]["code"] == "rate_limit_exceeded"
        assert any(h[0].lower() == b"retry-after" for h in headers)

    @pytest.mark.asyncio
    async def test_within_limit_passes(self, run_mcp_middleware, make_auth_info):
        status, _, _ = await run_mcp_middleware(make_auth_info(rate_limit_rpm=60))
        assert status == 200

    @pytest.mark.asyncio
    async def test_session_key_not_capped(self, run_mcp_middleware, make_auth_info):
        status, _, _ = await run_mcp_middleware(make_auth_info(principal_id="session:abc-123"))
        assert status == 200
