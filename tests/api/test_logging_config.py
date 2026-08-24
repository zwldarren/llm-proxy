"""Tests for the /api/config/server/logging endpoint."""

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


# --- Unit tests for the endpoint handler directly ---


class TestGetLoggingConfigDefault:
    """Test GET /logging returns verbose_routing_logs in default response."""

    @pytest.mark.asyncio
    async def test_default_includes_verbose_routing_logs(self):
        from llm_proxy.api.routers.config.server import get_logging_config

        request = MagicMock()
        session = AsyncMock()
        repo = MagicMock()
        repo.get_server_config = AsyncMock(return_value=None)

        with patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=repo,
        ):
            result = await get_logging_config(request, session)

        assert result["verbose_routing_logs"] is False
        assert result["log_input_output"] is True

    @pytest.mark.asyncio
    async def test_existing_config_inherits_verbose_routing_logs(self):
        """When DB config lacks verbose_routing_logs, it defaults to False."""
        from llm_proxy.api.routers.config.server import get_logging_config

        request = MagicMock()
        session = AsyncMock()
        repo = MagicMock()
        stored = MagicMock()
        stored.value = {"log_input_output": True, "log_retention_days": 30}
        repo.get_server_config = AsyncMock(return_value=stored)

        with patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=repo,
        ):
            result = await get_logging_config(request, session)

        assert result["verbose_routing_logs"] is False

    @pytest.mark.asyncio
    async def test_existing_config_preserves_verbose_routing_logs(self):
        """When DB config has verbose_routing_logs=True, it is preserved."""
        from llm_proxy.api.routers.config.server import get_logging_config

        request = MagicMock()
        session = AsyncMock()
        repo = MagicMock()
        stored = MagicMock()
        stored.value = {
            "log_input_output": True,
            "log_retention_days": 30,
            "verbose_routing_logs": True,
        }
        repo.get_server_config = AsyncMock(return_value=stored)

        with patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=repo,
        ):
            result = await get_logging_config(request, session)

        assert result["verbose_routing_logs"] is True


# --- Integration-style tests via TestClient ---


class TestLoggingConfigAPI:
    """Integration tests for GET/PUT /api/config/server/logging."""

    def test_get_default_returns_verbose_routing_logs(self, client):
        """GET /logging with no stored config returns verbose_routing_logs: False."""
        mock_repo = MagicMock()
        mock_repo.get_server_config = AsyncMock(return_value=None)
        mock_session = AsyncMock()

        app = client._transport.app  # type: ignore[attr-defined]

        # Override the session dependency
        async def override_session():
            return mock_session

        app.dependency_overrides[get_async_session_dep] = override_session

        with patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=mock_repo,
        ):
            res = client.get("/api/config/server/logging")

        assert res.status_code == 200
        data = res.json()
        assert data["verbose_routing_logs"] is False
        assert data["log_input_output"] is True
        assert data["log_retention_days"] == 30

        # Clean up
        del app.dependency_overrides[get_async_session_dep]

    def test_put_persists_verbose_routing_logs(self, client):
        """PUT /logging with verbose_routing_logs=True persists it."""
        mock_repo = MagicMock()

        # First call: get_server_config for the existing check
        existing = MagicMock()
        existing.value = {
            "log_input_output": True,
            "log_retention_days": 30,
            "verbose_routing_logs": False,
        }
        existing.description = "Logging configuration"
        mock_repo.get_server_config = AsyncMock(return_value=existing)
        mock_session = AsyncMock()
        mock_config_manager = AsyncMock()
        mock_config_manager.reload = AsyncMock()

        app = client._transport.app  # type: ignore[attr-defined]

        async def override_session():
            return mock_session

        app.dependency_overrides[get_async_session_dep] = override_session

        with (
            patch(
                "llm_proxy.api.routers.config.server.get_config_repository",
                return_value=mock_repo,
            ),
            patch(
                "llm_proxy.api.dependencies.get_config_manager",
                return_value=mock_config_manager,
            ),
        ):
            res = client.put(
                "/api/config/server/logging",
                json={
                    "log_input_output": True,
                    "log_retention_days": 30,
                    "verbose_routing_logs": True,
                },
            )

        assert res.status_code == 200
        data = res.json()
        assert data["verbose_routing_logs"] is True

        # Clean up
        del app.dependency_overrides[get_async_session_dep]

    def test_put_omits_verbose_routing_logs_preserves_existing(self, client):
        """PUT /logging without verbose_routing_logs should not reset existing value."""
        mock_repo = MagicMock()

        existing = MagicMock()
        existing.value = {
            "log_input_output": True,
            "log_retention_days": 30,
            "verbose_routing_logs": True,
        }
        existing.description = "Logging configuration"
        mock_repo.get_server_config = AsyncMock(return_value=existing)
        mock_session = AsyncMock()
        mock_config_manager = AsyncMock()
        mock_config_manager.reload = AsyncMock()

        app = client._transport.app  # type: ignore[attr-defined]

        async def override_session():
            return mock_session

        app.dependency_overrides[get_async_session_dep] = override_session

        with (
            patch(
                "llm_proxy.api.routers.config.server.get_config_repository",
                return_value=mock_repo,
            ),
            patch(
                "llm_proxy.api.dependencies.get_config_manager",
                return_value=mock_config_manager,
            ),
        ):
            res = client.put(
                "/api/config/server/logging",
                json={
                    "log_input_output": False,
                    "log_retention_days": 60,
                },
            )

        assert res.status_code == 200
        data = res.json()
        assert data["log_input_output"] is False
        assert data["log_retention_days"] == 60
        assert data["verbose_routing_logs"] is True

        # Clean up
        del app.dependency_overrides[get_async_session_dep]

    def test_put_can_disable_verbose_routing_logs(self, client):
        """PUT /logging with verbose_routing_logs=False explicitly disables it."""
        mock_repo = MagicMock()

        existing = MagicMock()
        existing.value = {
            "log_input_output": True,
            "log_retention_days": 30,
            "verbose_routing_logs": True,
        }
        existing.description = "Logging configuration"
        mock_repo.get_server_config = AsyncMock(return_value=existing)
        mock_session = AsyncMock()
        mock_config_manager = AsyncMock()
        mock_config_manager.reload = AsyncMock()

        app = client._transport.app  # type: ignore[attr-defined]

        async def override_session():
            return mock_session

        app.dependency_overrides[get_async_session_dep] = override_session

        with (
            patch(
                "llm_proxy.api.routers.config.server.get_config_repository",
                return_value=mock_repo,
            ),
            patch(
                "llm_proxy.api.dependencies.get_config_manager",
                return_value=mock_config_manager,
            ),
        ):
            res = client.put(
                "/api/config/server/logging",
                json={
                    "log_input_output": True,
                    "log_retention_days": 30,
                    "verbose_routing_logs": False,
                },
            )

        assert res.status_code == 200
        data = res.json()
        assert data["verbose_routing_logs"] is False

        # Clean up
        del app.dependency_overrides[get_async_session_dep]
