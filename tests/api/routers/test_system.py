"""Tests for the system info endpoint (api/routers/system.py)."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import require_admin_role
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers import system as system_module
from llm_proxy.api.routers.system import router as system_router
from llm_proxy.config.settings import (
    Settings,
    UpdateCheckSettings,
    reset_settings,
    set_settings,
)
from llm_proxy.core.exceptions import AuthenticationFailedError
from llm_proxy.version import get_version


@pytest.fixture
def mock_github_client(mock_response_cls):
    """Build a mock AsyncSession whose ``get`` stands in for the GitHub tags call."""

    def _build(
        tags: object = None,
        status_code: int = 200,
        side_effect: Exception | None = None,
    ) -> MagicMock:
        client = MagicMock()
        if side_effect is not None:
            client.get = AsyncMock(side_effect=side_effect)
        else:
            client.get = AsyncMock(
                return_value=mock_response_cls(json_data=tags, status_code=status_code)
            )
        return client

    return _build


def _use_update_check(enabled: bool) -> None:
    set_settings(Settings(update_check=UpdateCheckSettings(enabled=enabled)))


def _mount_github_client(app: FastAPI, github_client: MagicMock) -> None:
    # get_http_client returns manager.client for non-pool managers.
    app.state.http_client = SimpleNamespace(client=github_client)


@pytest.fixture(autouse=True)
def _reset_system_state():
    """Reset the module-level update-check cache and settings between tests."""
    system_module._reset_cache()
    yield
    system_module._reset_cache()
    reset_settings()


@pytest.fixture
def app():
    """FastAPI app with the system router and admin auth stubbed."""
    app = FastAPI()
    register_exception_handlers(app)
    app.dependency_overrides[require_admin_role] = lambda: None
    app.include_router(system_router)
    return app


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestUpdateCheckDisabled:
    def test_short_circuits_without_outbound_call(self, app, client, mock_github_client):
        _use_update_check(False)
        github = mock_github_client(tags=[{"name": "v999.0.0"}])
        _mount_github_client(app, github)

        res = client.get("/api/system/info")

        assert res.status_code == 200
        assert res.json() == {
            "version": get_version(),
            "update_check_enabled": False,
            "latest_version": None,
            "update_available": False,
            "checked_at": None,
            "check_failed": False,
        }
        github.get.assert_not_called()


class TestUpdateCheck:
    def test_empty_tags_means_no_update_info(self, app, client, mock_github_client):
        _use_update_check(True)
        github = mock_github_client(tags=[])
        _mount_github_client(app, github)

        res = client.get("/api/system/info")

        assert res.status_code == 200
        body = res.json()
        assert body["update_check_enabled"] is True
        assert body["latest_version"] is None
        assert body["update_available"] is False
        assert body["check_failed"] is False
        assert body["checked_at"] is not None

        github.get.assert_awaited_once()
        args, kwargs = github.get.call_args
        assert args[0] == system_module._GITHUB_TAGS_URL
        assert kwargs["params"] == {"per_page": 100}
        assert kwargs["headers"]["Accept"] == "application/vnd.github+json"
        assert kwargs["headers"]["User-Agent"]
        assert kwargs["timeout"] == 5.0

    def test_newer_tag_marks_update_available(self, app, client, mock_github_client):
        _use_update_check(True)
        github = mock_github_client(
            tags=[
                {"name": "0.1.0"},
                {"name": "v999.0.0"},
                {"name": "not-a-version"},
                {"no_name": 1},
                "junk-entry",
            ]
        )
        _mount_github_client(app, github)

        body = client.get("/api/system/info").json()

        assert body["latest_version"] == "999.0.0"
        assert body["update_available"] is True
        assert body["check_failed"] is False
        assert body["checked_at"] is not None

    def test_current_is_latest(self, app, client, mock_github_client):
        _use_update_check(True)
        github = mock_github_client(tags=[{"name": f"v{get_version()}"}, {"name": "v0.0.1"}])
        _mount_github_client(app, github)

        body = client.get("/api/system/info").json()

        assert body["latest_version"] == get_version()
        assert body["update_available"] is False
        assert body["check_failed"] is False

    def test_outbound_exception_fails_silently(self, app, client, mock_github_client):
        _use_update_check(True)
        github = mock_github_client(side_effect=RuntimeError("boom"))
        _mount_github_client(app, github)

        res = client.get("/api/system/info")

        assert res.status_code == 200
        body = res.json()
        assert body["check_failed"] is True
        assert body["latest_version"] is None
        assert body["update_available"] is False
        assert body["checked_at"] is not None

    def test_http_error_status_fails_silently(self, app, client, mock_github_client):
        _use_update_check(True)
        github = mock_github_client(status_code=500)
        _mount_github_client(app, github)

        body = client.get("/api/system/info").json()

        assert body["check_failed"] is True
        assert body["latest_version"] is None
        assert body["update_available"] is False


class TestUpdateCheckCaching:
    def test_second_request_uses_ttl_cache(self, app, client, mock_github_client):
        _use_update_check(True)
        github = mock_github_client(tags=[{"name": "v999.0.0"}])
        _mount_github_client(app, github)

        first = client.get("/api/system/info").json()
        second = client.get("/api/system/info").json()

        assert github.get.await_count == 1
        assert second["checked_at"] == first["checked_at"]
        assert second["update_available"] is True

    def test_force_within_cooldown_returns_cached_state(self, app, client, mock_github_client):
        _use_update_check(True)
        github = mock_github_client(tags=[{"name": "v999.0.0"}])
        _mount_github_client(app, github)

        client.get("/api/system/info")
        forced = client.get("/api/system/info?force=true").json()

        assert github.get.await_count == 1
        assert forced["update_available"] is True

    def test_force_after_cooldown_refetches(self, app, client, mock_github_client):
        _use_update_check(True)
        github = mock_github_client(tags=[{"name": "v999.0.0"}])
        _mount_github_client(app, github)

        client.get("/api/system/info")
        system_module._state.checked_at = datetime.now(UTC) - timedelta(seconds=61)
        forced = client.get("/api/system/info?force=true").json()

        assert github.get.await_count == 2
        assert forced["update_available"] is True

    def test_stale_cache_triggers_refresh_without_force(self, app, client, mock_github_client):
        _use_update_check(True)
        github = mock_github_client(tags=[{"name": "v999.0.0"}])
        _mount_github_client(app, github)

        client.get("/api/system/info")
        system_module._state.checked_at = datetime.now(UTC) - timedelta(hours=7)
        client.get("/api/system/info")

        assert github.get.await_count == 2


class TestSystemInfoAuth:
    def test_non_admin_gets_403(self, app, client):
        """The router-level admin gate rejects non-admin requests."""

        async def _non_admin():
            raise AuthenticationFailedError(
                message="Admin role required",
                code="forbidden",
                status_code=403,
            )

        app.dependency_overrides[require_admin_role] = _non_admin
        assert client.get("/api/system/info").status_code == 403


class TestVersionSource:
    def test_fastapi_app_uses_get_version(self):
        """create_app must source its version from version.get_version, not a literal."""
        import llm_proxy.api as api_module

        sentinel = "0.0.0-hardcode-sentinel"
        with patch.object(api_module, "get_version", return_value=sentinel):
            assert api_module.create_app().version == sentinel
