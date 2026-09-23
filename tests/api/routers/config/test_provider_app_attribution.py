"""Admin API surface for the typed ``app_attribution`` provider field.

The field is stored inside the generic ``provider_metadata`` JSON column (the
same pattern as ``parameter_overrides``/``endpoint_base_urls``) but is exposed
as a typed, validated object so the admin UI can render it without free-form
JSON editing.
"""

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import require_authenticated
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.config.providers import (
    provider_record_to_read,
)
from llm_proxy.api.routers.config.providers import (
    router as providers_router,
)


@dataclass
class MockProviderRecord:
    """Minimal mock of ProviderRecord for tests."""

    id: int = 1
    name: str = "or"
    type: str = "openrouter"
    api_key: str = "encrypted-k"
    base_url: str | None = None
    api_version: str | None = None
    timeout: float = 300.0
    rate_limit: int | None = None
    custom_headers: dict = field(default_factory=dict)
    provider_models: list = field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    provider_metadata: dict = field(default_factory=dict)
    definition_path: str | None = None
    icon_url: str | None = None


class TestProviderRecordToRead:
    def test_typed_field_is_split_out_of_metadata(self):
        record = MockProviderRecord(
            provider_metadata={
                "app_attribution": {
                    "url": "https://my.app",
                    "title": "My App",
                    "categories": ["cli-agent"],
                    "visibility": "hidden",
                },
                "kept": "yes",
            }
        )
        read = provider_record_to_read(record)

        assert read.app_attribution.url == "https://my.app"
        assert read.app_attribution.title == "My App"
        assert read.app_attribution.categories == ["cli-agent"]
        assert read.app_attribution.visibility == "hidden"
        # Never duplicated into the free-form metadata bag.
        assert read.provider_metadata == {"kept": "yes"}

    def test_absent_field_defaults_to_empty(self):
        read = provider_record_to_read(MockProviderRecord())
        assert read.app_attribution.url is None
        assert read.app_attribution.categories == []


@pytest.fixture
def client():
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(providers_router)

    mock_config_manager = MagicMock()
    mock_config_manager.reload = AsyncMock()
    app.state.config_manager = mock_config_manager

    from llm_proxy.api.dependencies import get_async_session, require_admin_role

    app.dependency_overrides[require_authenticated] = lambda: None
    app.dependency_overrides[require_admin_role] = lambda: None
    session = AsyncMock()
    session.commit = AsyncMock()
    app.dependency_overrides[get_async_session] = lambda: session

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


class TestProviderCreate:
    def test_app_attribution_is_forwarded_to_the_repository(self, client):
        record = MockProviderRecord(
            provider_metadata={"app_attribution": {"url": "https://my.app", "title": "My App"}}
        )
        repo = MagicMock()
        repo.create_provider = AsyncMock(return_value=record)

        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=repo,
        ):
            resp = client.post(
                "/providers",
                json={
                    "name": "or",
                    "type": "openrouter",
                    "api_key": "k",
                    "app_attribution": {
                        "url": "https://my.app",
                        "title": "My App",
                        "categories": ["cli-agent"],
                    },
                },
            )

        assert resp.status_code in (200, 201), resp.text
        _, kwargs = repo.create_provider.call_args
        assert kwargs["app_attribution"]["url"] == "https://my.app"
        assert kwargs["app_attribution"]["title"] == "My App"
        assert kwargs["app_attribution"]["categories"] == ["cli-agent"]
        assert resp.json()["app_attribution"]["title"] == "My App"

    def test_unknown_visibility_is_rejected(self, client):
        repo = MagicMock()
        repo.create_provider = AsyncMock(return_value=MockProviderRecord())
        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=repo,
        ):
            resp = client.post(
                "/providers",
                json={
                    "name": "or",
                    "type": "openrouter",
                    "api_key": "k",
                    "app_attribution": {"visibility": "secret"},
                },
            )
        assert resp.status_code == 422, resp.text


class TestProviderUpdate:
    def test_partial_update_only_sends_the_supplied_keys(self, client):
        record = MockProviderRecord(provider_metadata={"app_attribution": {"title": "Renamed"}})
        repo = MagicMock()
        repo.get_provider = AsyncMock(return_value=record)
        repo.update_provider = AsyncMock(return_value=record)

        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=repo,
        ):
            resp = client.put("/providers/or", json={"app_attribution": {"title": "Renamed"}})

        assert resp.status_code == 200, resp.text
        _, kwargs = repo.update_provider.call_args
        assert kwargs["app_attribution"] == {"title": "Renamed"}


class TestAppAttributionValidation:
    """Attribution values become raw HTTP headers; reject unsafe input up front."""

    @pytest.mark.parametrize(
        "app_attribution",
        [
            {"categories": ["a", "b", "c"]},
            {"title": "中文应用"},
            {"title": "bad\x01title"},
            {"url": "ftp://my.app"},
            {"url": "not a url"},
            {"categories": ["ok", "café"]},
        ],
    )
    def test_invalid_attribution_is_rejected(self, client, app_attribution):
        repo = MagicMock()
        repo.create_provider = AsyncMock(return_value=MockProviderRecord())
        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=repo,
        ):
            resp = client.post(
                "/providers",
                json={
                    "name": "or",
                    "type": "openrouter",
                    "api_key": "k",
                    "app_attribution": app_attribution,
                },
            )
        assert resp.status_code == 422, resp.text
        repo.create_provider.assert_not_awaited()

    def test_categories_string_is_normalized_to_a_list(self, client):
        record = MockProviderRecord(
            provider_metadata={"app_attribution": {"categories": ["cli-agent", "cloud-agent"]}}
        )
        repo = MagicMock()
        repo.create_provider = AsyncMock(return_value=record)
        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=repo,
        ):
            resp = client.post(
                "/providers",
                json={
                    "name": "or",
                    "type": "openrouter",
                    "api_key": "k",
                    "app_attribution": {"categories": "cli-agent, cloud-agent"},
                },
            )
        assert resp.status_code in (200, 201), resp.text
        _, kwargs = repo.create_provider.call_args
        assert kwargs["app_attribution"]["categories"] == ["cli-agent", "cloud-agent"]
