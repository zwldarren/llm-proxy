"""Tests for the /api/config/server/request-policy endpoint."""

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


class TestGetRequestPolicyConfig:
    """Test GET /request-policy returns default and stored values."""

    @pytest.mark.asyncio
    async def test_default_values(self):
        from llm_proxy.api.routers.config.server import get_request_policy_config

        request = MagicMock()
        session = AsyncMock()
        repo = MagicMock()
        repo.get_server_config = AsyncMock(return_value=None)

        with patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=repo,
        ):
            result = await get_request_policy_config(request, session)

        assert result["unknown_fields_policy"] == "ignore"
        assert result["unsupported_block_policy"] == "drop"

    @pytest.mark.asyncio
    async def test_stored_values_returned(self):
        """When DB config exists, stored values are returned."""
        from llm_proxy.api.routers.config.server import get_request_policy_config

        request = MagicMock()
        session = AsyncMock()
        repo = MagicMock()
        stored = MagicMock()
        stored.value = {
            "unknown_fields_policy": "passthrough",
            "unsupported_block_policy": "error",
        }
        repo.get_server_config = AsyncMock(return_value=stored)

        with patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=repo,
        ):
            result = await get_request_policy_config(request, session)

        assert result["unknown_fields_policy"] == "passthrough"
        assert result["unsupported_block_policy"] == "error"


# --- Integration-style tests via TestClient ---


class TestRequestPolicyConfigAPI:
    """Integration tests for GET/PUT /api/config/server/request-policy."""

    def test_get_default(self, client):
        """GET /request-policy with no stored config returns defaults."""
        mock_repo = MagicMock()
        mock_repo.get_server_config = AsyncMock(return_value=None)
        mock_session = AsyncMock()

        app = client._transport.app  # type: ignore[attr-defined]

        async def override_session():
            return mock_session

        app.dependency_overrides[get_async_session_dep] = override_session

        with patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=mock_repo,
        ):
            res = client.get("/api/config/server/request-policy")

        assert res.status_code == 200
        data = res.json()
        assert data["unknown_fields_policy"] == "ignore"
        assert data["unsupported_block_policy"] == "drop"

        del app.dependency_overrides[get_async_session_dep]

    def test_put_persists_and_reloads(self, client):
        """PUT /request-policy persists values and reloads the config manager."""
        mock_repo = MagicMock()
        stored = MagicMock()
        stored.value = {
            "unknown_fields_policy": "error",
            "unsupported_block_policy": "degrade",
        }
        mock_repo.get_server_config = AsyncMock(return_value=stored)
        mock_repo.set_server_config = AsyncMock(return_value=stored)
        mock_session = AsyncMock()

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
                "llm_proxy.api.routers.config.server.commit_and_reload",
                return_value=None,
            ) as mock_commit_and_reload,
        ):
            res = client.put(
                "/api/config/server/request-policy",
                json={
                    "unknown_fields_policy": "error",
                    "unsupported_block_policy": "degrade",
                },
            )

        assert res.status_code == 200
        data = res.json()
        assert data["unknown_fields_policy"] == "error"
        assert data["unsupported_block_policy"] == "degrade"

        mock_repo.set_server_config.assert_awaited_once()
        call_args = mock_repo.set_server_config.await_args.args
        call_kwargs = mock_repo.set_server_config.await_args.kwargs
        assert call_args[0] == "request_policy"
        assert call_args[1]["unknown_fields_policy"] == "error"
        assert call_args[1]["unsupported_block_policy"] == "degrade"
        assert call_kwargs["description"] == "Global request policy configuration"
        mock_commit_and_reload.assert_awaited_once()

        del app.dependency_overrides[get_async_session_dep]

    def test_put_invalid_policy_returns_422(self, client):
        """PUT /request-policy rejects invalid enum values."""
        mock_repo = MagicMock()
        mock_session = AsyncMock()

        app = client._transport.app  # type: ignore[attr-defined]

        async def override_session():
            return mock_session

        app.dependency_overrides[get_async_session_dep] = override_session

        with patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=mock_repo,
        ):
            res = client.put(
                "/api/config/server/request-policy",
                json={
                    "unknown_fields_policy": "invalid",
                    "unsupported_block_policy": "drop",
                },
            )

        assert res.status_code == 422

        del app.dependency_overrides[get_async_session_dep]
