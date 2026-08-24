"""E2E tests for smart routing virtual model interception.

Strategy: Focused integration tests that validate the interception in
_build_request_context and /v1/models listing. Uses real DatabaseConfigManager
against a temp SQLite DB rather than full HTTP e2e (create_app() lifecycle
requires too many subsystems to mock cleanly for this integration point).
The _build_request_context path validates the exact interception logic;
/v1/models validates the listing change.
"""

from unittest.mock import MagicMock, patch

import httpx2
import pytest

from llm_proxy.config.manager import DatabaseConfigManager
from llm_proxy.config.types.smart_routing import SmartRoutingConfig
from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.core.request_type import RequestType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_db_state(monkeypatch):
    """Reset global DB state so each test gets a fresh engine/session factory."""
    # Set a minimum JWT_SECRET for tests
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-for-testing-purposes-32chars")
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Create a temp SQLite DB, run migrations, and set DATABASE_URL."""
    db_path = tmp_path / "test_sr.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    return db_path


@pytest.fixture
def app_with_config(tmp_db):
    """Build a minimal FastAPI app with real DB-backed config manager."""
    from fastapi import FastAPI

    from llm_proxy.api.routers.models import router as models_router

    app = FastAPI()
    app.include_router(models_router)
    return app


async def _bootstrap_db_and_config(
    tmp_db, *, smart_routing_enabled: bool = False, auto_eligible_models: list[str] | None = None
) -> DatabaseConfigManager:
    """Init DB, create provider + models, return config_manager."""
    from llm_proxy.database.connection import init_db

    await init_db()

    from llm_proxy.database.connection import get_async_session_context
    from llm_proxy.database.tables import ModelProviderRecord, ModelRecord, ProviderRecord

    config_manager = DatabaseConfigManager()
    await config_manager.load()

    # Insert a provider
    async with get_async_session_context() as session:
        provider = ProviderRecord(
            name="test-provider",
            type="openai",
            api_key="sk-test",
            base_url="https://api.example.com/v1",
        )
        session.add(provider)
        await session.flush()

        # Insert auto-eligible models
        for model_name in auto_eligible_models or []:
            model = ModelRecord(
                name=model_name,
                auto_eligible=True,
            )
            session.add(model)
            await session.flush()
            mapping = ModelProviderRecord(
                model_id=model.id,
                provider_id=provider.id,
                priority=1,
                provider_model_name=model_name,
            )
            session.add(mapping)
        await session.commit()

    # Set smart routing config
    from llm_proxy.database.repositories import ServerConfigRepository

    async with get_async_session_context() as session:
        repo = ServerConfigRepository(session)
        await repo.set_server_config(
            "smart_routing",
            SmartRoutingConfig(enabled=smart_routing_enabled).to_row(),
            description="Smart routing global configuration",
        )
        await session.commit()
    await config_manager.reload()

    return config_manager


# ---------------------------------------------------------------------------
# Test: _build_request_context interception
# ---------------------------------------------------------------------------


class TestVirtualModelInterception:
    """Tests for the virtual model interception in _build_request_context."""

    @pytest.mark.asyncio
    async def test_auto_request_resolves_to_eligible_model(self, tmp_db, monkeypatch):
        """When smart routing is enabled and model='auto', the resolved model_name
        should be a real eligible model, not 'auto'."""
        eligible_models = ["openai/gpt-4o-mini", "deepseek/deepseek-chat"]
        config_manager = await _bootstrap_db_and_config(
            tmp_db, smart_routing_enabled=True, auto_eligible_models=eligible_models
        )

        # Mock the routing resolver to return a deterministic decision
        from llm_proxy.routing.types import RoutingDecision, Tier

        mock_decision = RoutingDecision(
            model="openai/gpt-4o-mini",
            tier=Tier.SIMPLE,
            complexity=0.2,
            confidence=0.9,
        )

        with (
            patch(
                "llm_proxy.routing.orchestrator.resolve_virtual_model",
                return_value=mock_decision,
            ),
            patch(
                "llm_proxy.routing.orchestrator.extract_messages_for_routing",
                return_value=[{"role": "user", "content": "hello"}],
            ),
        ):
            from llm_proxy.api.context import _build_request_context

            # Build a mock request
            request = MagicMock()
            request.model = "auto"
            request.messages = [{"role": "user", "content": "hello"}]

            req = MagicMock()
            req.app.state.config_manager = config_manager
            req.app.state.web_search_interceptor = None
            req.app.state.redis_client = None
            req.headers = {}
            req.client = MagicMock()
            req.client.host = "127.0.0.1"

            ctx = await _build_request_context(request, req, request_type=RequestType.CHAT)

            assert ctx.requested_model == "auto"
            assert ctx.routing_decision is mock_decision
            # The model_name was rewritten to the resolved model
            assert ctx.orchestrator is not None  # confirms it went past model_config lookup

    @pytest.mark.asyncio
    async def test_auto_disabled_raises_configuration_error(self, tmp_db):
        """When smart routing is disabled, requesting 'auto' raises ConfigurationError."""
        config_manager = await _bootstrap_db_and_config(
            tmp_db, smart_routing_enabled=False, auto_eligible_models=[]
        )

        from llm_proxy.api.context import _build_request_context

        request = MagicMock()
        request.model = "auto"
        request.messages = [{"role": "user", "content": "hello"}]

        req = MagicMock()
        req.app.state.config_manager = config_manager
        req.app.state.web_search_interceptor = None
        req.app.state.redis_client = None
        req.headers = {}
        req.client = MagicMock()
        req.client.host = "127.0.0.1"

        with pytest.raises(ConfigurationError, match="smart routing"):
            await _build_request_context(request, req, request_type=RequestType.CHAT)

    @pytest.mark.asyncio
    async def test_auto_embedding_request_raises_configuration_error(self, tmp_db):
        """Virtual model 'auto' on an embeddings request raises ConfigurationError."""
        config_manager = await _bootstrap_db_and_config(
            tmp_db, smart_routing_enabled=True, auto_eligible_models=["openai/gpt-4o-mini"]
        )

        from llm_proxy.api.context import _build_request_context

        request = MagicMock()
        request.model = "auto"
        request.messages = [{"role": "user", "content": "hello"}]

        req = MagicMock()
        req.app.state.config_manager = config_manager
        req.app.state.web_search_interceptor = None
        req.app.state.redis_client = None
        req.headers = {}
        req.client = MagicMock()
        req.client.host = "127.0.0.1"

        with pytest.raises(ConfigurationError, match="only supports chat"):
            await _build_request_context(request, req, request_type=RequestType.EMBEDDING)

    @pytest.mark.asyncio
    async def test_non_virtual_model_passes_through(self, tmp_db):
        """A normal (non-virtual) model should pass through without interception."""
        config_manager = await _bootstrap_db_and_config(
            tmp_db, smart_routing_enabled=True, auto_eligible_models=["openai/gpt-4o-mini"]
        )

        from llm_proxy.api.context import _build_request_context

        request = MagicMock()
        request.model = "openai/gpt-4o-mini"
        request.messages = [{"role": "user", "content": "hello"}]

        req = MagicMock()
        req.app.state.config_manager = config_manager
        req.app.state.web_search_interceptor = None
        req.app.state.redis_client = None
        req.headers = {}
        req.client = MagicMock()
        req.client.host = "127.0.0.1"

        ctx = await _build_request_context(request, req, request_type=RequestType.CHAT)
        assert ctx.requested_model is None
        assert ctx.routing_decision is None


# ---------------------------------------------------------------------------
# Test: /v1/models listing
# ---------------------------------------------------------------------------


class TestVirtualModelsListing:
    """Tests for auto/fast/best appearing in /v1/models when enabled."""

    @pytest.mark.asyncio
    async def test_virtual_models_listed_when_enabled(self, tmp_db, app_with_config):
        """When smart routing is enabled, /v1/models includes auto/fast/best."""
        config_manager = await _bootstrap_db_and_config(
            tmp_db, smart_routing_enabled=True, auto_eligible_models=["openai/gpt-4o-mini"]
        )
        app_with_config.state.config_manager = config_manager

        # Mock the identity to bypass auth
        mock_identity = MagicMock()
        mock_identity.is_authenticated = True
        mock_identity.api_key_name = "test-key"

        with patch("llm_proxy.api.routers.models.get_request_identity", return_value=mock_identity):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app_with_config), base_url="http://test"
            ) as client:
                resp = await client.get("/v1/models")
                assert resp.status_code == 200
                ids = {m["id"] for m in resp.json()["data"]}
                assert {"auto", "fast", "best"} <= ids

    @pytest.mark.asyncio
    async def test_virtual_models_not_listed_when_disabled(self, tmp_db, app_with_config):
        """When smart routing is disabled, /v1/models excludes auto/fast/best."""
        config_manager = await _bootstrap_db_and_config(
            tmp_db, smart_routing_enabled=False, auto_eligible_models=["openai/gpt-4o-mini"]
        )
        app_with_config.state.config_manager = config_manager

        # Mock the identity to bypass auth
        mock_identity = MagicMock()
        mock_identity.is_authenticated = True
        mock_identity.api_key_name = "test-key"

        with patch("llm_proxy.api.routers.models.get_request_identity", return_value=mock_identity):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app_with_config), base_url="http://test"
            ) as client:
                resp = await client.get("/v1/models")
                assert resp.status_code == 200
                ids = {m["id"] for m in resp.json()["data"]}
                assert "auto" not in ids
                assert "fast" not in ids
                assert "best" not in ids


# ---------------------------------------------------------------------------
# Test: per-API-key model allowlist filtering of /v1/models
# ---------------------------------------------------------------------------


class TestApiKeyModelRestriction:
    """Tests that /v1/models is filtered by the API key's allowed_models.

    The auth middleware stores the allowlist on request.state.allowed_models;
    None means unrestricted, a non-empty list restricts the listing, and an
    empty list is a valid deny-all restriction.
    """

    @staticmethod
    def _app_with_allowlist(app, allowed_models: list[str] | None):
        """Set request.state.allowed_models like the real auth middleware does."""
        from starlette.middleware.base import BaseHTTPMiddleware

        async def set_allowlist(request, call_next):
            if allowed_models is not None:
                request.state.allowed_models = allowed_models
            return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=set_allowlist)
        return app

    @pytest.mark.asyncio
    async def test_unrestricted_lists_all_models(self, tmp_db, app_with_config):
        """Without an allowlist, the full model list is returned."""
        config_manager = await _bootstrap_db_and_config(
            tmp_db, smart_routing_enabled=True, auto_eligible_models=["openai/gpt-4o-mini"]
        )
        app_with_config.state.config_manager = config_manager
        self._app_with_allowlist(app_with_config, None)

        mock_identity = MagicMock()
        mock_identity.is_authenticated = True
        mock_identity.api_key_name = "test-key"

        with patch("llm_proxy.api.routers.models.get_request_identity", return_value=mock_identity):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app_with_config), base_url="http://test"
            ) as client:
                resp = await client.get("/v1/models")
                assert resp.status_code == 200
                ids = {m["id"] for m in resp.json()["data"]}
                assert "openai/gpt-4o-mini" in ids
                assert {"auto", "fast", "best"} <= ids

    @pytest.mark.asyncio
    async def test_allowlist_filters_models(self, tmp_db, app_with_config):
        """Only models in the key's allowlist are listed, virtual models included."""
        config_manager = await _bootstrap_db_and_config(
            tmp_db, smart_routing_enabled=True, auto_eligible_models=["openai/gpt-4o-mini"]
        )
        app_with_config.state.config_manager = config_manager
        self._app_with_allowlist(app_with_config, ["openai/gpt-4o-mini", "auto"])

        mock_identity = MagicMock()
        mock_identity.is_authenticated = True
        mock_identity.api_key_name = "test-key"

        with patch("llm_proxy.api.routers.models.get_request_identity", return_value=mock_identity):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app_with_config), base_url="http://test"
            ) as client:
                resp = await client.get("/v1/models")
                assert resp.status_code == 200
                ids = {m["id"] for m in resp.json()["data"]}
                assert ids == {"openai/gpt-4o-mini", "auto"}

    @pytest.mark.asyncio
    async def test_empty_allowlist_lists_nothing(self, tmp_db, app_with_config):
        """An empty allowlist (deny-all) yields an empty model list."""
        config_manager = await _bootstrap_db_and_config(
            tmp_db, smart_routing_enabled=True, auto_eligible_models=["openai/gpt-4o-mini"]
        )
        app_with_config.state.config_manager = config_manager
        self._app_with_allowlist(app_with_config, [])

        mock_identity = MagicMock()
        mock_identity.is_authenticated = True
        mock_identity.api_key_name = "test-key"

        with patch("llm_proxy.api.routers.models.get_request_identity", return_value=mock_identity):
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app_with_config), base_url="http://test"
            ) as client:
                resp = await client.get("/v1/models")
                assert resp.status_code == 200
                assert resp.json()["data"] == []
