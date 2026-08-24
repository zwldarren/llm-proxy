"""Integration tests for the authentication endpoints (/api/auth/*)."""

from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.middleware.rate_limiting import RateLimitManager
from llm_proxy.api.routers.auth import router
from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.database import get_async_session
from llm_proxy.security.passwords import hash_password

TEST_SECRET = "test-secret-at-least-32-characters-long-aaaaaa"


def _mock_user(username: str = "admin", password: str = "test-password-123") -> MagicMock:
    """A mock user record with a real bcrypt hash so verify_admin_password works."""
    user = MagicMock()
    user.username = username
    user.password_hash = hash_password(password)
    user.is_active = True
    user.role = "admin"
    user.token_version = 0
    user.must_change_password = False
    return user


@pytest.fixture
def auth_client_with_lockout():
    """Build a test app with mocked user repo, jwt config, and neutralized rate limiting."""
    mock_repo = AsyncMock()

    mock_config = MagicMock()
    mock_config.server_params.auth = ProxyAuthConfig(jwt_secret=TEST_SECRET)
    mock_config_manager = MagicMock()
    mock_config_manager.get_config = AsyncMock(return_value=mock_config)

    lockout_mock = MagicMock()
    lockout_mock.is_locked_out.return_value = False

    mock_session_repo = AsyncMock()
    mock_session_repo.create_session = AsyncMock(return_value=(MagicMock(), "sk-session-test-key"))
    mock_session_repo.deactivate_user_sessions = AsyncMock()

    with (
        patch("llm_proxy.api.routers.auth.UserRepository", return_value=mock_repo),
        patch("llm_proxy.api.routers.auth.UserSessionRepository", return_value=mock_session_repo),
        patch("llm_proxy.api.routers.auth.get_lockout_manager", return_value=lockout_mock),
        patch.object(
            RateLimitManager,
            "check_rate_limit",
            new=AsyncMock(return_value=(False, {})),
        ),
    ):
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router)
        app.state.config_manager = mock_config_manager
        # The repo is mocked, so the session is never actually used.
        app.dependency_overrides[get_async_session] = lambda: AsyncMock()

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_repo, lockout_mock


@pytest.fixture
def auth_client(auth_client_with_lockout):
    """Convenience fixture that returns (client, mock_repo) without lockout mock."""
    client, mock_repo, _ = auth_client_with_lockout
    return client, mock_repo


@pytest.fixture
def auth_client_locked(auth_client_with_lockout):
    """Fixture with lockout enabled for testing lockout scenarios."""
    client, mock_repo, lockout_mock = auth_client_with_lockout
    lockout_mock.is_locked_out.return_value = True
    lockout_mock.get_lockout_remaining.return_value = 300
    return client, mock_repo, lockout_mock


def _decode(token: str) -> dict:
    payload = jwt.decode(token, TEST_SECRET, algorithms=["HS256"])
    assert "exp" in payload, "JWT token missing 'exp' (expiration) claim"
    assert "iat" in payload, "JWT token missing 'iat' (issued-at) claim"
    return payload


class TestSetupStatus:
    def test_needs_setup_when_no_admin(self, auth_client):
        client, mock_repo = auth_client
        mock_repo.has_admin = AsyncMock(return_value=False)

        response = client.get("/api/auth/setup-status")

        assert response.status_code == 200
        assert response.json() == {"needs_setup": True}

    def test_already_setup_when_admin_exists(self, auth_client):
        client, mock_repo = auth_client
        mock_repo.has_admin = AsyncMock(return_value=True)

        response = client.get("/api/auth/setup-status")

        assert response.status_code == 200
        assert response.json() == {"needs_setup": False}


class TestSetup:
    def test_creates_admin_and_returns_token(self, auth_client):
        client, mock_repo = auth_client
        mock_repo.has_admin = AsyncMock(return_value=False)
        mock_repo.get_by_username = AsyncMock(return_value=None)
        mock_repo.create_initial_admin = AsyncMock(
            return_value=_mock_user("admin", "Strong-Pass1!")
        )

        response = client.post(
            "/api/auth/setup",
            json={"username": "admin", "password": "Strong-Pass1!"},
        )

        assert response.status_code == 200
        body = response.json()
        token = body["access_token"]
        payload = _decode(token)
        assert payload["sub"] == "admin"
        assert payload["type"] == "admin"
        mock_repo.create_initial_admin.assert_called_once()
        # The password must be hashed (bcrypt), not stored in plain text.
        args, _ = mock_repo.create_initial_admin.call_args
        assert args[0] == "admin"
        assert args[1].startswith("$2")  # bcrypt hash

    def test_rejects_when_already_complete(self, auth_client):
        client, mock_repo = auth_client
        mock_repo.has_admin = AsyncMock(return_value=True)

        response = client.post(
            "/api/auth/setup",
            json={"username": "admin", "password": "Strong-Pass1!"},
        )

        assert response.status_code == 400

    def test_rejects_taken_username(self, auth_client):
        client, mock_repo = auth_client
        mock_repo.has_admin = AsyncMock(return_value=False)
        mock_repo.create_initial_admin = AsyncMock(
            side_effect=ValueError("User 'admin' already exists")
        )

        response = client.post(
            "/api/auth/setup",
            json={"username": "admin", "password": "Strong-Pass1!"},
        )

        assert response.status_code == 409
        mock_repo.create_initial_admin.assert_called_once()

    def test_handles_create_user_failure(self, auth_client):
        client, mock_repo = auth_client
        mock_repo.has_admin = AsyncMock(return_value=False)
        mock_repo.create_initial_admin = AsyncMock(
            side_effect=ValueError("Admin user already exists")
        )

        response = client.post(
            "/api/auth/setup",
            json={"username": "admin", "password": "Strong-Pass1!"},
        )

        assert response.status_code == 409

    def test_rejects_short_password(self, auth_client):
        client, mock_repo = auth_client
        mock_repo.has_admin = AsyncMock(return_value=False)

        response = client.post(
            "/api/auth/setup",
            json={"username": "admin", "password": "short"},
        )

        # Pydantic validation error for the min_length=8 constraint.
        assert response.status_code == 422
        mock_repo.create_initial_admin.assert_not_called()


class TestLogin:
    def test_login_success(self, auth_client):
        client, mock_repo = auth_client
        user = _mock_user("admin", "test-password-123")
        mock_repo.get_by_username = AsyncMock(return_value=user)

        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "test-password-123"},
        )

        assert response.status_code == 200
        payload = _decode(response.json()["access_token"])
        assert payload["sub"] == "admin"
        assert payload["type"] == "admin"
        assert response.json()["must_change_password"] is False

    def test_login_returns_must_change_password_flag(self, auth_client):
        """Flagged accounts learn about the forced change right at login."""
        client, mock_repo = auth_client
        user = _mock_user("member", "test-password-123")
        user.must_change_password = True
        mock_repo.get_by_username = AsyncMock(return_value=user)

        response = client.post(
            "/api/auth/login",
            json={"username": "member", "password": "test-password-123"},
        )

        assert response.status_code == 200
        assert response.json()["must_change_password"] is True

    def test_login_unknown_user(self, auth_client):
        client, mock_repo = auth_client
        mock_repo.get_by_username = AsyncMock(return_value=None)

        response = client.post(
            "/api/auth/login",
            json={"username": "ghost", "password": "whatever-password"},
        )

        assert response.status_code == 401

    def test_login_wrong_password(self, auth_client):
        client, mock_repo = auth_client
        mock_repo.get_by_username = AsyncMock(return_value=_mock_user("admin", "test-password-123"))

        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrong-password-here"},
        )

        assert response.status_code == 401

    def test_login_inactive_user(self, auth_client):
        client, mock_repo = auth_client
        user = _mock_user("admin", "test-password-123")
        user.is_active = False
        mock_repo.get_by_username = AsyncMock(return_value=user)

        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "test-password-123"},
        )

        assert response.status_code == 401


class TestLoginLockout:
    """Tests for account lockout during login."""

    def test_login_locked_out(self, auth_client_locked):
        client, mock_repo, lockout_mock = auth_client_locked

        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "test-password-123"},
        )

        assert response.status_code == 400
        body = response.json()
        error_msg = body.get("error", {}).get("message", "").lower()
        assert "locked" in error_msg


class TestLogout:
    """Tests for the logout endpoint."""

    def test_logout_without_token(self, auth_client):
        client, _ = auth_client

        response = client.post("/api/auth/logout")

        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_logout_with_valid_jwt(self, auth_client):
        client, mock_repo = auth_client
        user = _mock_user("admin", "test-password-123")
        mock_repo.get_by_username = AsyncMock(return_value=user)

        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "test-password-123"},
        )

        assert response.status_code == 200
        token = response.json()["access_token"]

        response = client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json() == {"success": True}
