"""Self-service password reset (the ADR-0049 amendment).

The three invariants worth pinning here, because each one is a way to hand an
account to the wrong person or to lock its owner out:

- the request endpoint answers identically for every input, so it is not an
  account oracle,
- a reset email only goes to a verified address on an account that actually has
  a password to reset,
- a link is spent once it is used (the password it was minted against moved), so
  a replay cannot undo a newer password.
"""

import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.auth import router as auth_router
from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.config.types.console_auth import ConsoleAuthSettings, SmtpSettings
from llm_proxy.database import get_async_session
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.passwords import hash_password, verify_admin_password

TEST_SECRET = "test-secret-at-least-32-characters-long-aaaaaa"
OLD_PASSWORD = "Original-Password-123!"
NEW_PASSWORD = "Brand-New-Password-456!"


@pytest.fixture(autouse=True)
def _no_rate_limit():
    """Keep the shared in-memory limiter from throttling these routes."""
    from llm_proxy.api.middleware.rate_limiting import RateLimitManager

    with patch.object(
        RateLimitManager, "check_rate_limit", new=AsyncMock(return_value=(False, {}))
    ):
        yield


def _user(
    *,
    email: str | None = "alice@example.com",
    verified: bool = True,
    active: bool = True,
    passwordless: bool = False,
) -> MagicMock:
    user = MagicMock()
    user.id = 1
    user.username = "alice"
    user.email = email
    user.email_verified_at = datetime.now(tz=UTC) if verified else None
    user.is_active = active
    user.password_hash = "" if passwordless else hash_password(OLD_PASSWORD)
    user.role = "viewer"
    user.token_version = 0
    return user


def _console_auth(*, smtp: bool = True, public_url: str = "https://proxy.example.com"):
    return ConsoleAuthSettings(
        public_base_url=public_url,
        smtp=SmtpSettings(
            enabled=smtp,
            host="smtp.example.com" if smtp else "",
            from_address="proxy@example.com" if smtp else "",
        ),
    )


def _config(*, smtp: bool = True, public_url: str = "https://proxy.example.com") -> MagicMock:
    config = MagicMock()
    config.server_params.auth = ProxyAuthConfig(jwt_secret=TEST_SECRET)
    config.server_params.console_auth = _console_auth(smtp=smtp, public_url=public_url)
    manager = MagicMock()
    manager.get_config = AsyncMock(return_value=config)
    manager.get_cached_config = MagicMock(return_value=config)
    manager.publish_generation = AsyncMock()
    return manager


def _client(*, smtp: bool = True, public_url: str = "https://proxy.example.com") -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router)
    app.state.config_manager = _config(smtp=smtp, public_url=public_url)
    app.dependency_overrides[get_async_session] = lambda: AsyncMock()
    return TestClient(app, raise_server_exceptions=False)


def _error_code(response) -> str | None:
    """The error code from whichever envelope this handler produced."""
    body = response.json()
    nested = body.get("error")
    if isinstance(nested, dict):
        return nested.get("code")
    return body.get("code")


def _fingerprint(password_hash: str) -> str:
    return hashlib.sha256(password_hash.encode("utf-8")).hexdigest()


def _reset_token(user: MagicMock, *, purpose: str = "password_reset") -> str:
    manager = JWTManager(ProxyAuthConfig(jwt_secret=TEST_SECRET))
    return manager.create_scoped_token(
        purpose,
        {"uid": user.id, "pwf": _fingerprint(user.password_hash)},
        ttl_seconds=600,
    )


class TestCapabilities:
    """The sign-in page needs to know whether emailing a link is even possible."""

    def test_unavailable_without_smtp(self) -> None:
        response = _client(smtp=False).get("/api/auth/capabilities")
        assert response.status_code == 200, response.text
        assert response.json() == {"password_reset": False}

    def test_unavailable_without_a_public_base_url(self) -> None:
        """No canonical origin means the link would point nowhere usable."""
        response = _client(public_url="").get("/api/auth/capabilities")
        assert response.status_code == 200, response.text
        assert response.json() == {"password_reset": False}

    def test_available_with_smtp_and_a_public_base_url(self) -> None:
        response = _client().get("/api/auth/capabilities")
        assert response.status_code == 200, response.text
        assert response.json() == {"password_reset": True}


class TestRequestReset:
    """Every answer is the same 202: the endpoint must not be an oracle."""

    @pytest.mark.parametrize(
        ("user", "smtp", "public_url"),
        [
            (None, True, "https://proxy.example.com"),  # unknown account
            (_user(), False, "https://proxy.example.com"),  # no SMTP configured
            (_user(), True, ""),  # no public base URL
            (_user(verified=False), True, "https://proxy.example.com"),  # unverified
            (_user(active=False), True, "https://proxy.example.com"),  # deactivated
            (_user(passwordless=True), True, "https://proxy.example.com"),  # nothing to reset
            (_user(email=None), True, "https://proxy.example.com"),  # no address at all
        ],
    )
    def test_never_delivers_and_never_says_why(self, user, smtp, public_url) -> None:
        repo = AsyncMock()
        repo.get_by_login = AsyncMock(return_value=user)

        with (
            patch("llm_proxy.api.routers.auth.UserRepository", return_value=repo),
            patch("llm_proxy.api.routers.auth.send_email", new=AsyncMock()) as send,
        ):
            response = _client(smtp=smtp, public_url=public_url).post(
                "/api/auth/password/forgot", json={"identifier": "alice@example.com"}
            )

        assert response.status_code == 202, response.text
        assert response.json() == {"requested": True}
        send.assert_not_awaited()

    def test_emails_a_link_for_an_eligible_account(self) -> None:
        user = _user()
        repo = AsyncMock()
        repo.get_by_login = AsyncMock(return_value=user)

        with (
            patch("llm_proxy.api.routers.auth.UserRepository", return_value=repo),
            patch("llm_proxy.api.routers.auth.send_email", new=AsyncMock()) as send,
        ):
            response = _client().post(
                "/api/auth/password/forgot", json={"identifier": "alice@example.com"}
            )

        assert response.status_code == 202, response.text
        send.assert_awaited_once()
        address = send.await_args.args[1]
        body = send.await_args.args[3]
        assert address == "alice@example.com"
        assert "https://proxy.example.com/reset-password?token=" in body
        # The link must be the reset credential, not a token usable elsewhere.
        token = body.split("token=")[1].split()[0]
        claims = JWTManager(ProxyAuthConfig(jwt_secret=TEST_SECRET)).verify_scoped_token(
            token, "password_reset"
        )
        assert claims["uid"] == user.id
        assert claims["pwf"] == _fingerprint(user.password_hash)


class TestConfirmReset:
    """Applying a link replaces the password and ends every older credential."""

    def test_sets_the_password_and_revokes_everything(self) -> None:
        user = _user()
        repo = AsyncMock()
        repo.get_by_id = AsyncMock(return_value=user)
        sessions = AsyncMock()

        with (
            patch("llm_proxy.api.routers.auth.UserRepository", return_value=repo),
            patch("llm_proxy.api.routers.auth.UserSessionRepository", return_value=sessions),
        ):
            response = _client().post(
                "/api/auth/password/reset",
                json={"token": _reset_token(user), "password": NEW_PASSWORD},
            )

        assert response.status_code == 200, response.text
        assert response.json() == {"success": True}
        repo.update_password.assert_awaited_once()
        new_hash = repo.update_password.await_args.args[1]
        assert verify_admin_password(NEW_PASSWORD, new_hash)
        # The user chose this password, so no forced change on top of it.
        repo.set_must_change_password.assert_awaited_once_with(user.id, False)
        # Stolen sessions and older JWTs must stop working immediately.
        repo.increment_token_version.assert_awaited_once_with(user.id)
        sessions.deactivate_user_sessions.assert_awaited_once_with(user.id)

    def test_a_link_cannot_be_replayed(self) -> None:
        """After the reset the password moved, so the pinned fingerprint is stale."""
        changed = _user()
        changed.password_hash = hash_password(NEW_PASSWORD)
        repo = AsyncMock()
        repo.get_by_id = AsyncMock(return_value=changed)
        minted_against_old_password = _reset_token(_user())

        with patch("llm_proxy.api.routers.auth.UserRepository", return_value=repo):
            response = _client().post(
                "/api/auth/password/reset",
                json={"token": minted_against_old_password, "password": NEW_PASSWORD},
            )

        assert response.status_code == 401
        assert _error_code(response) == "password_reset_superseded"
        repo.update_password.assert_not_awaited()

    def test_a_token_for_another_flow_is_refused(self) -> None:
        user = _user()
        repo = AsyncMock()
        repo.get_by_id = AsyncMock(return_value=user)

        with patch("llm_proxy.api.routers.auth.UserRepository", return_value=repo):
            response = _client().post(
                "/api/auth/password/reset",
                json={
                    "token": _reset_token(user, purpose="email_verify"),
                    "password": NEW_PASSWORD,
                },
            )

        assert response.status_code == 401
        assert _error_code(response) == "password_reset_invalid"
        repo.update_password.assert_not_awaited()

    @pytest.mark.parametrize(
        "user",
        [
            _user(active=False),
            _user(verified=False),
            _user(passwordless=True),
            _user(email=None),
        ],
    )
    def test_an_ineligible_account_cannot_be_reset(self, user) -> None:
        repo = AsyncMock()
        repo.get_by_id = AsyncMock(return_value=user)

        with patch("llm_proxy.api.routers.auth.UserRepository", return_value=repo):
            response = _client().post(
                "/api/auth/password/reset",
                json={"token": _reset_token(user), "password": NEW_PASSWORD},
            )

        assert response.status_code == 401
        assert _error_code(response) == "password_reset_invalid"
        repo.update_password.assert_not_awaited()

    def test_a_deleted_account_cannot_be_reset(self) -> None:
        repo = AsyncMock()
        repo.get_by_id = AsyncMock(return_value=None)
        token = _reset_token(_user())

        with patch("llm_proxy.api.routers.auth.UserRepository", return_value=repo):
            response = _client().post(
                "/api/auth/password/reset", json={"token": token, "password": NEW_PASSWORD}
            )

        assert response.status_code == 401
        assert _error_code(response) == "password_reset_invalid"

    def test_garbage_token_is_refused(self) -> None:
        repo = AsyncMock()
        repo.get_by_id = AsyncMock(return_value=_user())

        with patch("llm_proxy.api.routers.auth.UserRepository", return_value=repo):
            response = _client().post(
                "/api/auth/password/reset", json={"token": "not-a-token", "password": NEW_PASSWORD}
            )

        assert response.status_code == 401
        assert _error_code(response) == "password_reset_invalid"

    def test_weak_password_is_rejected_by_the_schema(self) -> None:
        user = _user()
        repo = AsyncMock()
        repo.get_by_id = AsyncMock(return_value=user)

        with patch("llm_proxy.api.routers.auth.UserRepository", return_value=repo):
            response = _client().post(
                "/api/auth/password/reset",
                json={"token": _reset_token(user), "password": "short"},
            )

        assert response.status_code == 422, response.text
        repo.update_password.assert_not_awaited()
