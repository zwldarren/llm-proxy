"""Tests for provider config — policy fields are now global, not per-provider."""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import require_authenticated
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.config.config import router as config_router


@dataclass
class MockProviderRecord:
    """Minimal mock of ProviderRecord for tests."""

    id: int = 1
    name: str = "p"
    type: str = "openai-compatible"
    api_key: str = "encrypted-k"
    base_url: str | None = None
    api_version: str | None = None
    timeout: float = 300.0
    max_retries: int = 3
    rate_limit: int | None = None
    custom_headers: dict = field(default_factory=dict)
    provider_models: list = field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    provider_metadata: dict = field(default_factory=dict)
    parameter_overrides: dict = field(default_factory=dict)
    endpoint_base_urls: dict = field(default_factory=dict)
    icon_url: str | None = None


def _setup_app_state(app):
    """Configure the minimal app.state and dependency overrides for config router tests."""
    mock_config_manager = MagicMock()
    mock_config_manager.reload = AsyncMock()
    mock_config_manager.get_config = AsyncMock(return_value=MagicMock())
    app.state.config_manager = mock_config_manager

    # Override admin auth
    from llm_proxy.api.dependencies import require_admin_role

    app.dependency_overrides[require_authenticated] = lambda: None
    app.dependency_overrides[require_admin_role] = lambda: None

    # Override the DB session dependency so commit_and_reload works
    from llm_proxy.api.dependencies import get_async_session

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    # Also ensure _API_LOG_COLUMNS includes user_id so the field appears in responses
    app.dependency_overrides[get_async_session] = lambda: mock_session

    return app


@pytest.fixture
def app():
    """Create a test FastAPI app with the config router and mock state."""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(config_router)
    return _setup_app_state(app)


@pytest.fixture
def client(app):
    """Test client for the config router."""
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestProviderConfigPolicies:
    """Tests that policy fields are NOT present in provider API responses."""

    def test_provider_create_ignores_policy_fields(self, app, client):
        """POST /api/config/providers should ignore policy fields."""
        record = MockProviderRecord(id=1, name="p")

        mock_repo = MagicMock()
        mock_repo.create_provider = AsyncMock(return_value=record)
        mock_repo.get_all_providers = AsyncMock(return_value=[])

        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            resp = client.post(
                "/api/config/providers",
                json={
                    "name": "p",
                    "type": "openai-compatible",
                    "base_url": "https://api.example.com",
                    "api_key": "k",
                    "unknown_fields_policy": "passthrough",
                    "unsupported_block_policy": "degrade",
                },
            )
        assert resp.status_code in (200, 201), resp.text
        data = resp.json()
        # Policy fields are global, not per-provider — should not appear in response
        assert "unknown_fields_policy" not in data
        assert "unsupported_block_policy" not in data

    def test_provider_create_rejects_dead_fields(self, app, client):
        """POST /api/config/providers should not expose unsupported_params."""
        record = MockProviderRecord(id=2, name="p2")

        mock_repo = MagicMock()
        mock_repo.create_provider = AsyncMock(return_value=record)
        mock_repo.get_all_providers = AsyncMock(return_value=[])

        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            resp = client.post(
                "/api/config/providers",
                json={
                    "name": "p2",
                    "type": "openai-compatible",
                    "base_url": "https://api.example.com",
                    "api_key": "k",
                    "unsupported_params": ["x"],
                },
            )
        assert resp.status_code in (200, 201), resp.text
        data = resp.json()
        assert "unsupported_params" not in data

    def test_provider_create_does_not_expose_policy_fields(self, app, client):
        """POST /api/config/providers should not return policy fields."""
        record = MockProviderRecord(id=3, name="p3")

        mock_repo = MagicMock()
        mock_repo.create_provider = AsyncMock(return_value=record)
        mock_repo.get_all_providers = AsyncMock(return_value=[])

        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            resp = client.post(
                "/api/config/providers",
                json={
                    "name": "p3",
                    "type": "openai-compatible",
                    "base_url": "https://api.example.com",
                    "api_key": "k",
                },
            )
        assert resp.status_code in (200, 201), resp.text
        data = resp.json()
        assert "unknown_fields_policy" not in data
        assert "unsupported_block_policy" not in data

    def test_provider_update_ignores_policy_fields(self, app, client):
        """PUT /api/config/providers/{name} should ignore policy fields."""
        record = MockProviderRecord(id=1, name="p")

        mock_repo = MagicMock()
        mock_repo.get_all_providers = AsyncMock(return_value=[record])
        mock_repo.get_provider = AsyncMock(return_value=record)
        mock_repo.update_provider = AsyncMock(return_value=record)

        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            resp = client.put(
                "/api/config/providers/p",
                json={"unsupported_block_policy": "degrade"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "unsupported_block_policy" not in data
        assert "unknown_fields_policy" not in data

    def test_provider_update_rejects_dead_fields(self, app, client):
        """PUT /api/config/providers/{name} should not expose unsupported_params."""
        record = MockProviderRecord(id=2, name="p2")

        mock_repo = MagicMock()
        mock_repo.get_all_providers = AsyncMock(return_value=[record])
        mock_repo.get_provider = AsyncMock(return_value=record)
        mock_repo.update_provider = AsyncMock(return_value=record)

        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            resp = client.put(
                "/api/config/providers/p2",
                json={
                    "unsupported_params": ["x"],
                    "param_transformers": {"a": "b"},
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "unsupported_params" not in data
        assert "param_transformers" not in data
