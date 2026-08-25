"""Tests for the provider-type catalog endpoint (api/routers/config/providers.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Populate the adapter registry, mirroring the app's startup import
# (llm_proxy.api.__init__ imports llm_proxy.providers).
import llm_proxy.providers  # noqa: F401
from llm_proxy.api.dependencies import get_async_session_dep, require_admin_role
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.config.config import router as config_router
from llm_proxy.core.adapter import list_provider_types
from llm_proxy.core.exceptions import AuthenticationFailedError


@pytest.fixture
def app():
    """Create a test FastAPI app with the config router and admin auth stubbed."""
    app = FastAPI()
    register_exception_handlers(app)
    app.dependency_overrides[require_admin_role] = lambda: None
    # The catch-all /providers/{name:path} route resolves a DB session; stub it
    # so tests never touch a real database (CI has none).
    app.dependency_overrides[get_async_session_dep] = lambda: AsyncMock()
    app.include_router(config_router)
    return app


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestProviderTypesCatalog:
    """The catalog derives from the adapter registry, not configured providers."""

    def test_returns_all_registered_types(self, client):
        res = client.get("/api/config/providers/provider-types")
        assert res.status_code == 200
        types = res.json()
        assert len(types) == len(list_provider_types()) > 0
        for entry in types:
            assert entry["type"]
            assert entry["name_en"]
            assert entry["name_zh"]
            assert entry["icon_variant"] in {"mono", "color"}

    def test_sorted_by_display_name(self, client):
        types = client.get("/api/config/providers/provider-types").json()
        names = [t["name_en"].casefold() for t in types]
        assert names == sorted(names)

    def test_branding_metadata(self, client):
        types = {t["type"]: t for t in client.get("/api/config/providers/provider-types").json()}
        qwen = types["qwen"]
        assert qwen["name_en"] == "Qwen (Model Studio)"
        assert qwen["name_zh"] == "通义千问 (百炼)"
        assert qwen["icon_id"] == "qwen"
        assert qwen["icon_variant"] == "color"
        assert types["kimi-code"]["name_en"] == "Kimi Code"
        # Providers without a Lobe icon degrade to icon_id=None.
        assert types["nanogpt"]["icon_id"] is None

    def test_catch_all_route_does_not_swallow_catalog(self, client):
        """GET /providers/{name:path} must not shadow the literal provider-types path."""
        res = client.get("/api/config/providers/provider-types")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_missing_provider_still_404s(self, client):
        mock_repo = MagicMock()
        mock_repo.get_provider_with_models = AsyncMock(return_value=None)
        with patch(
            "llm_proxy.api.routers.config.providers.get_config_repository",
            return_value=mock_repo,
        ):
            res = client.get("/api/config/providers/does-not-exist")
        assert res.status_code == 404

    def test_non_admin_gets_403(self, app, client):
        """The router-level admin gate rejects non-admin requests.

        Mirrors the real ``require_admin_role`` non-admin path (raises
        AuthenticationFailedError with status 403); the catalog must never
        be reachable without admin privileges.
        """

        async def _non_admin():
            raise AuthenticationFailedError(
                message="Admin role required",
                code="forbidden",
                status_code=403,
            )

        app.dependency_overrides[require_admin_role] = _non_admin
        res = client.get("/api/config/providers/provider-types")
        assert res.status_code == 403
