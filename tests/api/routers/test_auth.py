"""Integration tests for the authentication endpoints (/api/auth/*)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.middleware.rate_limiting import RateLimitManager
from llm_proxy.api.middleware.security import BaseLockoutManager
from llm_proxy.api.routers.auth import router
from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.core.request_facts import RequestIdentity, set_request_identity
from llm_proxy.database import get_async_session
from llm_proxy.security.passwords import hash_password, verify_admin_password

TEST_SECRET = "test-secret-at-least-32-characters-long-aaaaaa"


def _mock_user(username: str = "admin", password: str = "test-password-123") -> MagicMock:
    """A mock user record with a real bcrypt hash so verify_admin_password works."""
    user = MagicMock()
    user.id = 1
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
    mock_repo.get_by_email = AsyncMock(return_value=None)

    mock_config = MagicMock()
    mock_config.server_params.auth = ProxyAuthConfig(jwt_secret=TEST_SECRET)
    mock_config_manager = MagicMock()
    mock_config_manager.get_config = AsyncMock(return_value=mock_config)

    lockout_mock = MagicMock()
    lockout_mock.is_locked_out.return_value = False

    mock_session_repo = AsyncMock()
    session_record = MagicMock()
    session_record.id = "session-1"
    mock_session_repo.create_session = AsyncMock(
        return_value=(session_record, "sk-session-test-key")
    )
    mock_session_repo.deactivate_user_sessions = AsyncMock()

    mock_totp_repo = AsyncMock()
    mock_totp_repo.get = AsyncMock(return_value=None)

    with (
        patch("llm_proxy.api.routers.auth.UserRepository", return_value=mock_repo),
        patch("llm_proxy.api.auth_session.UserRepository", return_value=mock_repo),
        patch("llm_proxy.api.auth_session.UserSessionRepository", return_value=mock_session_repo),
        patch("llm_proxy.api.auth_session.TotpRepository", return_value=mock_totp_repo),
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

    @pytest.mark.parametrize("username", ["bad user", "bad/name", "a" * 65])
    def test_rejects_username_outside_console_format(self, auth_client, username):
        client, mock_repo = auth_client
        mock_repo.has_admin = AsyncMock(return_value=False)

        response = client.post(
            "/api/auth/setup",
            json={"username": username, "password": "Strong-Pass1!"},
        )

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
        body = response.json()
        payload = _decode(body["access_token"])
        assert payload["sub"] == "admin"
        assert payload["type"] == "admin"
        assert body["must_change_password"] is False
        assert body["username"] == "admin"
        assert body["role"] == "admin"

        cookies = "; ".join(response.headers.get_list("set-cookie"))
        assert "llm_proxy_console=" in cookies
        assert "llm_proxy_session_key=" in cookies
        assert "llm_proxy_csrf=" in cookies
        console_cookie = next(
            value
            for value in response.headers.get_list("set-cookie")
            if "llm_proxy_console=" in value
        )
        assert "HttpOnly" in console_cookie
        assert "Path=/api" in console_cookie
        csrf_cookie = next(
            value for value in response.headers.get_list("set-cookie") if "llm_proxy_csrf=" in value
        )
        assert "HttpOnly" not in csrf_cookie
        assert "Path=/" in csrf_cookie

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

    def test_login_unknown_user_still_runs_password_verify(self, auth_client):
        """Unknown usernames must still run a bcrypt verify (timing equalization).

        Without this, the short-circuit skips the bcrypt check for unknown
        users and the response time reveals whether a username exists.
        """
        client, mock_repo = auth_client
        mock_repo.get_by_username = AsyncMock(return_value=None)

        with patch(
            "llm_proxy.api.routers.auth.verify_admin_password",
            wraps=verify_admin_password,
        ) as spy:
            response = client.post(
                "/api/auth/login",
                json={"username": "ghost", "password": "whatever-password"},
            )

        assert response.status_code == 401
        spy.assert_called_once()
        # The verify ran against a real bcrypt hash (the precomputed dummy),
        # so the timing cost matches the real-password path.
        _, password_hash = spy.call_args.args
        assert password_hash.startswith("$2") and len(password_hash) == 60

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

    def test_lockout_accumulates_across_ips_for_same_username(self, auth_client):
        """Lockout is keyed by username: failures from different IPs accumulate.

        An attacker rotating source IPs must not be able to bypass the
        per-account lockout.
        """
        client, mock_repo = auth_client
        mock_repo.get_by_username = AsyncMock(return_value=None)

        real_lockout = BaseLockoutManager(max_attempts=3, lockout_duration=60)
        ips = iter(["203.0.113.1", "203.0.113.2", "203.0.113.3", "203.0.113.4"])

        with (
            patch("llm_proxy.api.routers.auth.get_lockout_manager", return_value=real_lockout),
            patch(
                "llm_proxy.api.routers.auth.get_client_ip",
                side_effect=lambda _request: next(ips),
            ),
        ):
            for _ in range(3):
                response = client.post(
                    "/api/auth/login",
                    json={"username": "admin", "password": "wrong-password-1"},
                )
                assert response.status_code == 401

            # Fourth attempt from yet another IP is locked out: the failures
            # accumulated on the username, not on any single IP.
            response = client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "wrong-password-1"},
            )
            assert response.status_code == 400
            error_msg = response.json().get("error", {}).get("message", "").lower()
            assert "locked" in error_msg

    def test_disabled_lockout_never_returns_account_locked(self, auth_client):
        """With login lockout off, repeated failures stay plain 401s."""
        client, mock_repo = auth_client
        mock_repo.get_by_username = AsyncMock(return_value=None)

        disabled_lockout = BaseLockoutManager(
            max_attempts=1, lockout_duration=60, enabled_getter=lambda: False
        )

        with patch("llm_proxy.api.routers.auth.get_lockout_manager", return_value=disabled_lockout):
            for _ in range(5):
                response = client.post(
                    "/api/auth/login",
                    json={"username": "admin", "password": "wrong-password-1"},
                )
                assert response.status_code == 401


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


@pytest.fixture
def logout_client():
    """A client whose request identity is stamped the way the JWT gate does it.

    The identity holder is mutable so a test can present a token without a
    ``sid`` (minted before sessions were bound) without rebuilding the app.
    """
    user = _mock_user()
    mock_repo = AsyncMock()
    mock_repo.get_by_username = AsyncMock(return_value=user)
    mock_repo.increment_token_version = AsyncMock(return_value=True)
    mock_session_repo = AsyncMock()
    mock_session_repo.deactivate_session = AsyncMock(return_value=True)
    mock_session_repo.deactivate_user_sessions = AsyncMock()
    identity: dict[str, Any] = {
        "user": user.username,
        "user_id": user.id,
        "session_id": "session-1",
    }

    with (
        patch("llm_proxy.api.routers.auth.UserRepository", return_value=mock_repo),
        patch("llm_proxy.api.routers.auth.UserSessionRepository", return_value=mock_session_repo),
        patch.object(
            RateLimitManager,
            "check_rate_limit",
            new=AsyncMock(return_value=(False, {})),
        ),
    ):
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router)
        app.dependency_overrides[get_async_session] = lambda: AsyncMock()

        @app.middleware("http")
        async def _set_identity(request: Request, call_next):
            set_request_identity(
                request,
                RequestIdentity(
                    user=identity["user"],
                    auth_method="jwt",
                    user_id=identity["user_id"],
                    console_session_id=identity["session_id"],
                ),
            )
            return await call_next(request)

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, user, mock_repo, mock_session_repo, identity


class TestLogoutScope:
    """Signing out ends this browser's session, not the account's."""

    def test_revokes_only_the_callers_own_session(self, logout_client):
        client, user, mock_repo, mock_session_repo, _ = logout_client

        response = client.post("/api/auth/logout")

        assert response.status_code == 200
        assert response.json() == {"success": True}
        mock_session_repo.deactivate_session.assert_awaited_once_with("session-1", user.id)
        # The bug: an old tab whose session had been rotated away revoking every
        # session of the account, ending the live session on another device.
        mock_session_repo.deactivate_user_sessions.assert_not_awaited()
        mock_repo.increment_token_version.assert_not_awaited()

    def test_legacy_token_without_sid_falls_back_to_account_wide_revoke(self, logout_client):
        """A pre-``sid`` token has no session to end, so it is revoked account-wide."""
        client, user, mock_repo, mock_session_repo, identity = logout_client
        identity["session_id"] = None

        response = client.post("/api/auth/logout")

        assert response.status_code == 200
        mock_session_repo.deactivate_session.assert_not_awaited()
        mock_session_repo.deactivate_user_sessions.assert_awaited_once_with(user.id)
        mock_repo.increment_token_version.assert_awaited_once_with(user.id)

    def test_unknown_caller_is_not_attributed_to_a_user(self, logout_client):
        """An identity whose subject no longer resolves revokes nothing."""
        client, _, mock_repo, mock_session_repo, _ = logout_client
        mock_repo.get_by_username = AsyncMock(return_value=None)

        response = client.post("/api/auth/logout")

        assert response.status_code == 200
        mock_session_repo.deactivate_session.assert_not_awaited()
        mock_session_repo.deactivate_user_sessions.assert_not_awaited()
