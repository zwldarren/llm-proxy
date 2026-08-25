"""Tests for provider API key masking and the explicit reveal endpoint.

The plaintext upstream key must never appear in list/detail/update responses
(only ``masked_api_key`` is serialized); the only way to read it back is the
explicit, audited ``POST /api/config/providers/{name}/api-key/reveal``
endpoint, which is admin-only.
"""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import (
    get_async_session,
    require_admin_role,
    require_authenticated,
)
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.config.config import router as config_router
from llm_proxy.core.exceptions import AuthenticationFailedError

PLAINTEXT_KEY = "sk-abcdefghijklmnop1234"


@dataclass
class MockProviderRecord:
    """Minimal mock of ProviderRecord for tests."""

    id: int = 1
    name: str = "p"
    type: str = "openai-compatible"
    api_key: str = PLAINTEXT_KEY
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
    model_provider_mappings: list = field(default_factory=list)


def _setup_app_state(app):
    """Configure the minimal app.state and dependency overrides for config router tests."""
    mock_config_manager = MagicMock()
    mock_config_manager.reload = AsyncMock()
    mock_config_manager.get_config = AsyncMock(return_value=MagicMock())
    app.state.config_manager = mock_config_manager

    app.dependency_overrides[require_authenticated] = lambda: None
    app.dependency_overrides[require_admin_role] = lambda: None

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
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


@pytest.fixture
def mock_repo():
    """A repository mock returning a single provider named 'p'."""
    record = MockProviderRecord()
    repo = MagicMock()
    repo.get_all_providers = AsyncMock(return_value=[record])
    repo.get_provider_with_models = AsyncMock(return_value=record)
    repo.get_provider = AsyncMock(side_effect=lambda name: record if name == "p" else None)
    repo.update_provider = AsyncMock(return_value=record)
    repo.create_provider = AsyncMock(return_value=record)
    return repo


class TestProviderApiKeyMasking:
    """List/detail/update responses must never contain the plaintext key."""

    def test_list_providers_masks_api_key(self, client, mock_repo):
        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            resp = client.get("/api/config/providers")
        assert resp.status_code == 200, resp.text
        assert PLAINTEXT_KEY not in resp.text
        data = resp.json()
        assert len(data) == 1
        assert data[0]["masked_api_key"] == "sk-...1234"
        assert "api_key" not in data[0]

    def test_get_provider_masks_api_key(self, client, mock_repo):
        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            resp = client.get("/api/config/providers/p")
        assert resp.status_code == 200, resp.text
        assert PLAINTEXT_KEY not in resp.text
        data = resp.json()
        assert data["masked_api_key"] == "sk-...1234"
        assert "api_key" not in data

    def test_update_provider_masks_api_key(self, client, mock_repo):
        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            resp = client.put("/api/config/providers/p", json={"base_url": "https://x.example"})
        assert resp.status_code == 200, resp.text
        assert PLAINTEXT_KEY not in resp.text
        data = resp.json()
        assert data["masked_api_key"] == "sk-...1234"
        assert "api_key" not in data

    def test_create_provider_masks_api_key(self, client, mock_repo):
        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            resp = client.post(
                "/api/config/providers",
                json={"name": "p", "type": "openai-compatible", "api_key": PLAINTEXT_KEY},
            )
        assert resp.status_code in (200, 201), resp.text
        assert PLAINTEXT_KEY not in resp.text
        data = resp.json()
        assert data["masked_api_key"] == "sk-...1234"
        assert "api_key" not in data


class TestProviderApiKeyReveal:
    """The reveal endpoint is the only place the plaintext key is returned."""

    def test_reveal_returns_plaintext_key(self, client, mock_repo):
        with (
            patch(
                "llm_proxy.api.routers.config.providers.get_config_repository",
                return_value=mock_repo,
            ),
            patch(
                "llm_proxy.api.routers.config.providers.write_provider_key_reveal_audit_log",
                new=AsyncMock(),
            ) as audit,
        ):
            resp = client.post("/api/config/providers/p/api-key/reveal")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"name": "p", "api_key": PLAINTEXT_KEY}
        audit.assert_awaited_once()
        assert audit.await_args.kwargs["provider_name"] == "p"

    def test_reveal_missing_provider_404(self, client, mock_repo):
        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            resp = client.post("/api/config/providers/does-not-exist/api-key/reveal")
        assert resp.status_code == 404

    def test_reveal_non_admin_rejected(self, app, client, mock_repo):
        """The router-level admin gate rejects non-admin reveal requests."""

        async def _non_admin():
            raise AuthenticationFailedError(
                message="Admin role required",
                code="forbidden",
                status_code=403,
            )

        app.dependency_overrides[require_admin_role] = _non_admin
        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            resp = client.post("/api/config/providers/p/api-key/reveal")
        assert resp.status_code == 403
        assert PLAINTEXT_KEY not in resp.text
