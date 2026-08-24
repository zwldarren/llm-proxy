"""Tests for the self-service per-user tracing router (/api/me/tracing)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import get_current_user, require_authenticated
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.me_tracing import router as me_tracing_router
from llm_proxy.database.tables import UserRecord
from llm_proxy.observability.user_tracing import UserTracingManager


@pytest.fixture
def app():
    """Create a test FastAPI app with the me_tracing router, auth stubbed."""
    app = FastAPI()
    mock_user = MagicMock(spec=UserRecord, id=42, role="viewer", is_active=True)
    app.dependency_overrides[require_authenticated] = lambda: None
    app.dependency_overrides[get_current_user] = lambda: mock_user
    register_exception_handlers(app)
    app.include_router(me_tracing_router)
    return app


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _mock_session_context(repo: MagicMock) -> MagicMock:
    """Build an async context manager yielding a session-like object whose
    UserRepository returns the provided mock repo."""
    mock_session = MagicMock()
    mock_session.commit = AsyncMock(return_value=None)
    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_session)
    mock_context.__aexit__ = AsyncMock(return_value=False)
    return mock_context


class TestMeTracingRouter:
    def test_get_returns_default_when_no_personal_config(self, client):
        """GET /api/me/tracing returns disabled defaults when the user has no config."""
        repo = MagicMock()
        repo.get_tracing_config = AsyncMock(return_value=None)
        ctx = _mock_session_context(repo)
        with (
            patch(
                "llm_proxy.api.routers.me_tracing.get_async_session_context",
                return_value=ctx,
            ),
            patch("llm_proxy.api.routers.me_tracing.UserRepository", return_value=repo),
        ):
            response = client.get("/api/me/tracing/")
        assert response.status_code == 200
        data = response.json()
        assert data["config"]["enabled"] is False
        assert data["config"]["providers"] == []
        assert data["status"]["is_configured"] is False

    def test_get_returns_stored_personal_config(self, client):
        stored = {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "my-langfuse",
                    "enabled": True,
                    "settings": {
                        "public_key": "pk-secret",
                        "secret_key": "sk-secret",
                        "base_url": "https://cloud.langfuse.com",
                    },
                }
            ],
        }
        repo = MagicMock()
        repo.get_tracing_config = AsyncMock(return_value=stored)
        ctx = _mock_session_context(repo)
        with (
            patch(
                "llm_proxy.api.routers.me_tracing.get_async_session_context",
                return_value=ctx,
            ),
            patch("llm_proxy.api.routers.me_tracing.UserRepository", return_value=repo),
        ):
            response = client.get("/api/me/tracing/")
        assert response.status_code == 200
        data = response.json()
        assert data["config"]["enabled"] is True
        assert len(data["config"]["providers"]) == 1
        provider = data["config"]["providers"][0]
        assert provider["name"] == "my-langfuse"
        # Secrets are masked in the response
        assert "****" in provider["masked_settings"]["secret_key"]
        assert provider["settings"]["secret_key"] == "sk-secret"
        assert data["status"]["is_configured"] is True

    def test_put_persists_and_invalidates_user_registry(self, client):
        stored = {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "my-langfuse",
                    "enabled": True,
                    "settings": {
                        "public_key": "pk-existing",
                        "secret_key": "sk-existing",
                        "base_url": "https://cloud.langfuse.com",
                    },
                }
            ],
        }
        repo = MagicMock()
        repo.get_tracing_config = AsyncMock(return_value=stored)
        saved = []
        repo.set_tracing_config = AsyncMock(side_effect=lambda uid, cfg: saved.append((uid, cfg)))
        ctx = _mock_session_context(repo)
        invalidate_mock = AsyncMock()
        with (
            patch(
                "llm_proxy.api.routers.me_tracing.get_async_session_context",
                return_value=ctx,
            ),
            patch("llm_proxy.api.routers.me_tracing.UserRepository", return_value=repo),
            patch(
                "llm_proxy.api.routers.me_tracing.get_user_tracing_manager",
                return_value=MagicMock(invalidate=invalidate_mock),
            ),
        ):
            response = client.put(
                "/api/me/tracing/",
                json={
                    "enabled": True,
                    "providers": [
                        {
                            "provider": "langfuse",
                            "name": "my-langfuse",
                            "enabled": True,
                            "settings": {
                                "public_key": "pk-ex****ting",
                                "secret_key": "****",
                                "base_url": "https://cloud.langfuse.com",
                            },
                        }
                    ],
                },
            )
        assert response.status_code == 200, response.text
        assert response.json()["message"] == "Tracing configuration updated successfully"

        # Persisted against the authenticated user's id, with secrets restored
        assert len(saved) == 1
        user_id, config = saved[0]
        assert user_id == 42
        settings = config["providers"][0]["settings"]
        assert settings["public_key"] == "pk-existing"
        assert settings["secret_key"] == "sk-existing"

        # The per-user registry cache was invalidated so it rebuilds on next request
        invalidate_mock.assert_awaited_once_with(42)

    def test_put_rejects_invalid_base_url(self, client):
        repo = MagicMock()
        repo.get_tracing_config = AsyncMock(return_value=None)
        ctx = _mock_session_context(repo)
        with (
            patch(
                "llm_proxy.api.routers.me_tracing.get_async_session_context",
                return_value=ctx,
            ),
            patch("llm_proxy.api.routers.me_tracing.UserRepository", return_value=repo),
        ):
            response = client.put(
                "/api/me/tracing/",
                json={
                    "enabled": True,
                    "providers": [
                        {
                            "provider": "langfuse",
                            "name": "bad",
                            "enabled": True,
                            "settings": {
                                "public_key": "pk",
                                "secret_key": "sk",
                                "base_url": "ftp://example.com",
                            },
                        }
                    ],
                },
            )
        assert response.status_code == 400

    def test_providers_lists_langfuse(self, client):
        response = client.get("/api/me/tracing/providers")
        assert response.status_code == 200
        names = {p["name"] for p in response.json()["providers"]}
        assert "langfuse" in names

    # ── Masked-secret preservation (ported from the former admin router) ──
    # These cases exercise build_persisted_tracing_dict via /api/me/tracing.

    def _put(self, client, existing_config, payload):
        """PUT a personal config; return (response, saved_configs)."""
        repo = MagicMock()
        repo.get_tracing_config = AsyncMock(return_value=existing_config)
        saved = []
        repo.set_tracing_config = AsyncMock(side_effect=lambda uid, cfg: saved.append((uid, cfg)))
        ctx = _mock_session_context(repo)
        invalidate_mock = AsyncMock()
        with (
            patch(
                "llm_proxy.api.routers.me_tracing.get_async_session_context",
                return_value=ctx,
            ),
            patch("llm_proxy.api.routers.me_tracing.UserRepository", return_value=repo),
            patch(
                "llm_proxy.api.routers.me_tracing.get_user_tracing_manager",
                return_value=MagicMock(invalidate=invalidate_mock),
            ),
        ):
            response = client.put("/api/me/tracing/", json=payload)
        return response, saved

    def test_put_preserves_nested_authorization_masked_value(self, client):
        existing = {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "lf",
                    "enabled": True,
                    "settings": {
                        "base_url": "https://cloud.langfuse.com",
                        "public_key": "pk",
                        "secret_key": "sk",
                        "headers": {"Authorization": "Bearer real-secret-token"},
                    },
                }
            ],
        }
        payload = {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "lf",
                    "enabled": True,
                    "settings": {
                        "base_url": "https://cloud.langfuse.com",
                        "public_key": "pk",
                        "secret_key": "sk",
                        "headers": {"Authorization": "Bear****oken"},
                    },
                }
            ],
        }
        response, saved = self._put(client, existing, payload)
        assert response.status_code == 200
        assert len(saved) == 1
        headers = saved[0][1]["providers"][0]["settings"]["headers"]
        assert headers["Authorization"] == "Bearer real-secret-token"

    def test_put_preserves_secrets_when_providers_reordered(self, client):
        existing = {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "prod",
                    "enabled": True,
                    "settings": {"public_key": "pk-prod", "secret_key": "sk-prod"},
                },
                {
                    "provider": "langfuse",
                    "name": "dev",
                    "enabled": True,
                    "settings": {"public_key": "pk-dev", "secret_key": "sk-dev"},
                },
            ],
        }
        payload = {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "dev",
                    "enabled": True,
                    "settings": {"public_key": "****", "secret_key": "****"},
                },
                {
                    "provider": "langfuse",
                    "name": "prod",
                    "enabled": True,
                    "settings": {"public_key": "****", "secret_key": "****"},
                },
            ],
        }
        response, saved = self._put(client, existing, payload)
        assert response.status_code == 200
        by_name = {p["name"]: p for p in saved[0][1]["providers"]}
        assert by_name["dev"]["settings"]["secret_key"] == "sk-dev"
        assert by_name["prod"]["settings"]["secret_key"] == "sk-prod"

    def test_put_drops_masked_value_without_existing_value(self, client):
        existing = {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "lf",
                    "enabled": True,
                    "settings": {"public_key": "pk", "secret_key": "sk"},
                }
            ],
        }
        payload = {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "lf",
                    "enabled": True,
                    "settings": {
                        "public_key": "pk",
                        "secret_key": "sk",
                        "api_key": "****",  # masked but no existing → dropped
                    },
                }
            ],
        }
        response, saved = self._put(client, existing, payload)
        assert response.status_code == 200
        settings = saved[0][1]["providers"][0]["settings"]
        assert "api_key" not in settings

    def test_put_saves_new_unmasked_secret(self, client):
        payload = {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "lf",
                    "enabled": True,
                    "settings": {
                        "public_key": "pk",
                        "secret_key": "sk-new-secret-value",
                        "base_url": "https://cloud.langfuse.com",
                    },
                }
            ],
        }
        response, saved = self._put(client, None, payload)
        assert response.status_code == 200
        assert saved[0][1]["providers"][0]["settings"]["secret_key"] == "sk-new-secret-value"

    def test_put_rejects_base_url_with_credentials(self, client):
        payload = {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "bad",
                    "enabled": True,
                    "settings": {
                        "public_key": "pk",
                        "secret_key": "sk",
                        "base_url": "http://user:pass@example.com",
                    },
                }
            ],
        }
        response, _ = self._put(client, None, payload)
        assert response.status_code == 400


class TestUserTracingManager:
    """Unit tests for the per-user tracing registry manager."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_personal_config(self, monkeypatch):
        manager = UserTracingManager()
        manager.set_system_handlers([])

        async def fake_ctx():
            class _CM:
                async def __aenter__(self):
                    session = MagicMock()
                    repo = MagicMock()
                    repo.get_tracing_config = AsyncMock(return_value=None)
                    return session

                async def __aexit__(self, *a):
                    return False

            return _CM()

        # Patch UserRepository.get_tracing_config via the session path used by the manager.
        with patch("llm_proxy.observability.user_tracing.get_async_session_context") as get_ctx:
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(
                return_value=MagicMock()  # session
            )
            cm.__aexit__ = AsyncMock(return_value=False)
            get_ctx.return_value = cm
            with patch("llm_proxy.observability.user_tracing.UserRepository") as Repo:
                repo = MagicMock()
                repo.get_tracing_config = AsyncMock(return_value=None)
                Repo.return_value = repo
                result = await manager.get_registry(user_id=7)
        assert result is None

    @pytest.mark.asyncio
    async def test_builds_registry_with_system_handlers_when_configured(self):
        manager = UserTracingManager()
        system_handler = MagicMock()
        manager.set_system_handlers([system_handler])

        config_dict = {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "lf",
                    "enabled": True,
                    "settings": {
                        "public_key": "pk",
                        "secret_key": "sk",
                        "base_url": "https://cloud.langfuse.com",
                    },
                }
            ],
        }
        with patch("llm_proxy.observability.user_tracing.get_async_session_context") as get_ctx:
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=MagicMock())
            cm.__aexit__ = AsyncMock(return_value=False)
            get_ctx.return_value = cm
            with patch("llm_proxy.observability.user_tracing.UserRepository") as Repo:
                repo = MagicMock()
                repo.get_tracing_config = AsyncMock(return_value=config_dict)
                Repo.return_value = repo
                registry = await manager.get_registry(user_id=5)

        assert registry is not None
        # Registry contains the system handler plus the user's langfuse handler
        handler_names = {type(h).__name__ for h in registry.handlers}
        assert "MagicMock" in handler_names
        # Cached: a second lookup must not reload from the DB
        with patch("llm_proxy.observability.user_tracing.UserRepository") as Repo:
            Repo.side_effect = AssertionError("should not reload; registry is cached")
            cached = await manager.get_registry(user_id=5)
        assert cached is registry

    @pytest.mark.asyncio
    async def test_caches_negative_result_to_avoid_per_request_db_hit(self):
        manager = UserTracingManager()
        manager.set_system_handlers([])
        call_count = 0

        with (
            patch("llm_proxy.observability.user_tracing.get_async_session_context") as get_ctx,
            patch("llm_proxy.observability.user_tracing.UserRepository") as Repo,
        ):
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=MagicMock())
            cm.__aexit__ = AsyncMock(return_value=False)
            get_ctx.return_value = cm

            def make_repo(*a, **kw):
                nonlocal call_count
                call_count += 1
                repo = MagicMock()
                repo.get_tracing_config = AsyncMock(return_value=None)
                return repo

            Repo.side_effect = make_repo
            first = await manager.get_registry(user_id=99)
            second = await manager.get_registry(user_id=99)

        assert first is None
        assert second is None
        # The DB-backed repository was constructed only once (cached negative).
        assert call_count == 1
