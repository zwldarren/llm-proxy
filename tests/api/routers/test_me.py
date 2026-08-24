"""Integration tests for the self-service endpoints (/api/me/*)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import get_current_user
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.logs import _user_role_cache
from llm_proxy.api.routers.me import router
from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.core.identity import RequestIdentity, set_request_identity
from llm_proxy.database import get_async_session
from llm_proxy.security.passwords import hash_password

TEST_SECRET = "test-secret-at-least-32-characters-long-aaaaaa"
CURRENT_PASSWORD = "current-password-123"


def _mock_user(username: str = "admin") -> MagicMock:
    """A mock user record with a real bcrypt hash so password verification works."""
    user = MagicMock()
    user.id = 1
    user.username = username
    user.password_hash = hash_password(CURRENT_PASSWORD)
    user.is_active = True
    user.role = "admin"
    user.token_version = 0
    return user


@pytest.fixture
def me_client():
    """Test app with an authenticated identity and a mocked user repository."""
    user = _mock_user()
    mock_repo = AsyncMock()
    mock_repo.update_username = AsyncMock(return_value=user)

    mock_config = MagicMock()
    mock_config.server_params.auth = ProxyAuthConfig(jwt_secret=TEST_SECRET)
    mock_config_manager = MagicMock()
    mock_config_manager.get_config = AsyncMock(return_value=mock_config)

    with patch("llm_proxy.api.routers.me.UserRepository", return_value=mock_repo):
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router)
        app.state.config_manager = mock_config_manager

        @app.middleware("http")
        async def _set_identity(request: Request, call_next):
            set_request_identity(
                request, RequestIdentity(user=user.username, auth_method="jwt", user_id=user.id)
            )
            return await call_next(request)

        # The repo is mocked, so the session is never actually used.
        app.dependency_overrides[get_async_session] = lambda: AsyncMock()
        app.dependency_overrides[get_current_user] = lambda: user

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, user, mock_repo


class TestChangeUsername:
    def test_success_returns_fresh_token(self, me_client):
        client, user, mock_repo = me_client
        renamed = _mock_user("newname")
        renamed.token_version = 1  # repo.update_username bumped token_version
        mock_repo.update_username = AsyncMock(return_value=renamed)

        response = client.put(
            "/api/me/username",
            json={"current_password": CURRENT_PASSWORD, "new_username": "NewName"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "newname"
        mock_repo.update_username.assert_awaited_once_with(user.id, "NewName")

        # The returned JWT must reference the NEW username and the bumped
        # token_version so the old token (and any recycled-username leak) is
        # invalidated.
        payload = jwt.decode(body["access_token"], TEST_SECRET, algorithms=["HS256"])
        assert payload["sub"] == "newname"
        assert payload["role"] == "admin"
        assert payload["tv"] == 1

    def test_wrong_current_password_rejected(self, me_client):
        client, _, mock_repo = me_client

        response = client.put(
            "/api/me/username",
            json={"current_password": "wrong-password", "new_username": "newname"},
        )

        assert response.status_code == 401
        mock_repo.update_username.assert_not_awaited()

    def test_taken_username_conflicts(self, me_client):
        client, _, mock_repo = me_client
        mock_repo.update_username = AsyncMock(side_effect=ValueError("User 'taken' already exists"))

        response = client.put(
            "/api/me/username",
            json={"current_password": CURRENT_PASSWORD, "new_username": "taken"},
        )

        assert response.status_code == 409

    def test_invalid_username_format_rejected(self, me_client):
        client, _, mock_repo = me_client

        response = client.put(
            "/api/me/username",
            json={"current_password": CURRENT_PASSWORD, "new_username": "bad name!"},
        )

        assert response.status_code == 422
        mock_repo.update_username.assert_not_awaited()

    def test_too_long_username_rejected(self, me_client):
        client, _, mock_repo = me_client

        response = client.put(
            "/api/me/username",
            json={"current_password": CURRENT_PASSWORD, "new_username": "a" * 65},
        )

        assert response.status_code == 422
        mock_repo.update_username.assert_not_awaited()

    def test_rename_invalidates_user_role_cache(self, me_client):
        """The logs router's role cache is keyed by username: after a rename,
        both the old and new names must be dropped so a recycled username
        cannot inherit the previous owner's cached role/user_id."""
        client, user, mock_repo = me_client
        renamed = _mock_user("newname")
        renamed.token_version = 1
        mock_repo.update_username = AsyncMock(return_value=renamed)
        _user_role_cache["admin"] = ("admin", 1, True, time.monotonic())
        _user_role_cache["newname"] = ("viewer", 99, True, time.monotonic())
        try:
            response = client.put(
                "/api/me/username",
                json={"current_password": CURRENT_PASSWORD, "new_username": "newname"},
            )

            assert response.status_code == 200
            assert "admin" not in _user_role_cache
            assert "newname" not in _user_role_cache
        finally:
            _user_role_cache.pop("admin", None)
            _user_role_cache.pop("newname", None)


class TestChangeUsernameUnauthenticated:
    def test_requires_auth(self):
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router)
        app.dependency_overrides[get_async_session] = lambda: AsyncMock()

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/me/username",
                json={"current_password": CURRENT_PASSWORD, "new_username": "newname"},
            )

        assert response.status_code == 401


class TestChangePassword:
    def test_change_password_clears_must_change_flag(self, me_client):
        """A successful self-service change clears the forced-change flag."""
        client, user, mock_repo = me_client

        with patch("llm_proxy.api.routers.me.UserSessionRepository") as session_repo_cls:
            session_repo_cls.return_value.deactivate_user_sessions = AsyncMock()
            response = client.put(
                "/api/me/password",
                json={"current_password": CURRENT_PASSWORD, "new_password": "New-password-456!"},
            )

        assert response.status_code == 200
        mock_repo.update_password.assert_awaited_once()
        mock_repo.set_must_change_password.assert_awaited_once_with(user.id, False)
        # Existing tokens are revoked, so the user logs in again with the new password.
        mock_repo.increment_token_version.assert_awaited_once_with(user.id)

    def test_wrong_current_password_rejected(self, me_client):
        client, user, mock_repo = me_client

        response = client.put(
            "/api/me/password",
            json={"current_password": "wrong-password", "new_password": "New-password-456!"},
        )

        assert response.status_code == 401
        mock_repo.set_must_change_password.assert_not_awaited()


class TestGetBudget:
    """GET /api/me/budget reports the caller's account-level budget."""

    def test_no_budget_returns_nulls(self, me_client):
        client, user, _ = me_client
        user.budget_usd = None

        response = client.get("/api/me/budget")

        assert response.status_code == 200
        body = response.json()
        assert body["budget_usd"] is None
        assert body["budget_period"] is None
        assert body["period_start"] is None
        assert body["period_spend_usd"] is None

    def test_budget_reports_current_window_spend(self, me_client):
        client, user, _ = me_client
        user.budget_usd = 100.0
        user.budget_period = "monthly"
        user.budget_reset_day = None
        user.budget_reset_at = None

        usage_repo = AsyncMock()
        usage_repo.get_user_spend_since = AsyncMock(return_value=42.5)
        with patch("llm_proxy.api.routers.me.UsageRepository", return_value=usage_repo):
            response = client.get("/api/me/budget")

        assert response.status_code == 200
        body = response.json()
        assert body["budget_usd"] == 100.0
        assert body["budget_period"] == "monthly"
        assert body["period_spend_usd"] == 42.5
        assert body["period_start"] is not None
        # The spend query is scoped to the caller's user id.
        call = usage_repo.get_user_spend_since.await_args
        assert call.args[0] == user.id
