"""Tests for the /api/config/server/security endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import (
    get_async_session_dep,
    require_admin_role,
    require_authenticated,
)
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.config.server import router


@pytest.fixture
def app():
    """Create a test FastAPI app with the server config router."""
    app = FastAPI()
    app.dependency_overrides[require_authenticated] = lambda: None
    app.dependency_overrides[require_admin_role] = lambda: None
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/config")
    return app


@pytest.fixture
def client(app):
    """Test client for the server config router."""
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestGetSecurityConfig:
    """Test GET /security returns default and stored values."""

    @pytest.mark.asyncio
    async def test_default_values(self):
        from llm_proxy.api.routers.config.server import get_security_config

        request = MagicMock()
        session = AsyncMock()
        repo = MagicMock()
        repo.get_server_config = AsyncMock(return_value=None)

        with patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=repo,
        ):
            result = await get_security_config(request, session)

        from llm_proxy.api.schemas.admin import SecurityConfig

        defaults = SecurityConfig().model_dump()
        for key, expected in defaults.items():
            assert result[key] == expected, f"{key}: expected {expected}, got {result[key]}"

    @pytest.mark.asyncio
    async def test_stored_values_returned(self):
        """Stored values are returned, with defaults filling missing keys."""
        from llm_proxy.api.routers.config.server import get_security_config

        request = MagicMock()
        session = AsyncMock()
        repo = MagicMock()
        stored = MagicMock()
        stored.value = {"max_failed_login_attempts": 3, "hsts_enabled": False}
        repo.get_server_config = AsyncMock(return_value=stored)

        with patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=repo,
        ):
            result = await get_security_config(request, session)

        assert result["max_failed_login_attempts"] == 3
        assert result["hsts_enabled"] is False
        # Untouched keys fall back to defaults
        assert result["lockout_duration_seconds"] == 900


class TestSecurityConfigAPI:
    """Integration tests for GET/PUT /api/config/server/security."""

    def test_get_default(self, client):
        mock_repo = MagicMock()
        mock_repo.get_server_config = AsyncMock(return_value=None)
        mock_session = AsyncMock()

        app = client._transport.app  # type: ignore[attr-defined]

        async def override_session():
            return mock_session

        app.dependency_overrides[get_async_session_dep] = override_session

        try:
            with patch(
                "llm_proxy.api.routers.config.server.get_config_repository",
                return_value=mock_repo,
            ):
                res = client.get("/api/config/server/security")

            assert res.status_code == 200
            data = res.json()
            assert data["max_failed_login_attempts"] == 5
            assert data["hsts_enabled"] is True
        finally:
            app.dependency_overrides.pop(get_async_session_dep, None)

    def test_put_persists_and_reloads(self, client):
        """PUT /security persists values and reloads the config manager."""
        mock_repo = MagicMock()
        stored = MagicMock()
        stored.value = {"max_failed_login_attempts": 3, "hsts_enabled": False}
        mock_repo.get_server_config = AsyncMock(return_value=stored)
        mock_repo.set_server_config = AsyncMock(return_value=stored)
        mock_session = AsyncMock()

        app = client._transport.app  # type: ignore[attr-defined]

        async def override_session():
            return mock_session

        app.dependency_overrides[get_async_session_dep] = override_session

        config_manager = MagicMock()
        config_manager.reload = AsyncMock()
        app.state.config_manager = config_manager

        try:
            with patch(
                "llm_proxy.api.routers.config.server.get_config_repository",
                return_value=mock_repo,
            ):
                res = client.put(
                    "/api/config/server/security",
                    json={"max_failed_login_attempts": 3, "hsts_enabled": False},
                )

            assert res.status_code == 200
            mock_repo.set_server_config.assert_called_once()
            args = mock_repo.set_server_config.call_args
            assert args.args[0] == "security"
            assert args.args[1]["max_failed_login_attempts"] == 3
            assert args.args[1]["hsts_enabled"] is False
            config_manager.reload.assert_awaited_once()
        finally:
            app.dependency_overrides.pop(get_async_session_dep, None)

    def test_put_rejects_invalid_values(self, client):
        """Values below the minimum are rejected with a validation error."""
        mock_session = AsyncMock()
        app = client._transport.app  # type: ignore[attr-defined]

        async def override_session():
            return mock_session

        app.dependency_overrides[get_async_session_dep] = override_session

        try:
            res = client.put(
                "/api/config/server/security",
                json={"max_failed_login_attempts": 0},
            )
            assert res.status_code in (400, 422)
        finally:
            app.dependency_overrides.pop(get_async_session_dep, None)
