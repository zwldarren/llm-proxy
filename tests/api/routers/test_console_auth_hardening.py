"""Regression tests for the console-auth hardening in ADR-0049.

Covers three invariants that are easy to lose again:

- an email-verification link is single-use (it is pinned to the address the
  account had when it was minted),
- an account link completes only in the console session that started it and
  only for the account that started it,
- recovery codes are drawn from a transcription-safe alphabet and hashed with
  a work factor rather than a bare digest.
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import get_current_user, require_recent_auth
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.auth import router as auth_router
from llm_proxy.api.routers.oauth import router as oauth_router
from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.core.request_facts import RequestIdentity, set_request_identity
from llm_proxy.database import get_async_session
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.tokens import (
    _RECOVERY_ALPHABET,
    generate_recovery_codes,
    hash_recovery_code,
)

TEST_SECRET = "test-secret-at-least-32-characters-long-aaaaaa"


@pytest.fixture(autouse=True)
def _no_rate_limit():
    """Keep the shared in-memory limiter from throttling these routes."""
    from llm_proxy.api.middleware.rate_limiting import RateLimitManager

    with patch.object(
        RateLimitManager, "check_rate_limit", new=AsyncMock(return_value=(False, {}))
    ):
        yield


def _user(email: str | None = "old@example.com") -> MagicMock:
    user = MagicMock()
    user.id = 1
    user.username = "alice"
    user.email = email
    user.is_active = True
    user.role = "admin"
    user.token_version = 0
    return user


def _config_manager() -> MagicMock:
    config = MagicMock()
    config.server_params.auth = ProxyAuthConfig(jwt_secret=TEST_SECRET)
    manager = MagicMock()
    manager.get_config = AsyncMock(return_value=config)
    manager.get_cached_config = MagicMock(return_value=config)
    manager.publish_generation = AsyncMock()
    return manager


def _error_code(response) -> str | None:
    """The error code from whichever envelope this handler produced."""
    body = response.json()
    nested = body.get("error")
    if isinstance(nested, dict):
        return nested.get("code")
    return body.get("code")


def _app(*routers, user: MagicMock, session_id: str = "sess-1") -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    for router in routers:
        app.include_router(router)
    app.state.config_manager = _config_manager()
    app.dependency_overrides[get_async_session] = lambda: AsyncMock()
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[require_recent_auth] = lambda: user

    @app.middleware("http")
    async def _identity(request: Request, call_next):
        set_request_identity(
            request,
            RequestIdentity(
                user=user.username,
                auth_method="jwt",
                user_id=user.id,
                console_session_id=session_id,
            ),
        )
        return await call_next(request)

    return app


def _email_token(*, to: str, pinned_from: str | None) -> str:
    manager = JWTManager(ProxyAuthConfig(jwt_secret=TEST_SECRET))
    return manager.create_scoped_token(
        "email_verify",
        {"uid": 1, "email": to, "from": pinned_from},
        ttl_seconds=600,
    )


class TestEmailVerificationIsSingleUse:
    def test_applies_when_the_address_is_unchanged(self) -> None:
        user = _user("old@example.com")
        updated = _user("new@example.com")
        repo = AsyncMock()
        repo.get_by_id = AsyncMock(return_value=user)
        repo.is_email_taken = AsyncMock(return_value=False)
        repo.set_email = AsyncMock(return_value=updated)

        with patch("llm_proxy.api.routers.auth.UserRepository", return_value=repo):
            client = TestClient(_app(auth_router, user=user), raise_server_exceptions=False)
            response = client.post(
                "/api/auth/email/verify",
                json={"token": _email_token(to="new@example.com", pinned_from="old@example.com")},
            )

        assert response.status_code == 200, response.text
        repo.set_email.assert_awaited_once_with(1, "new@example.com", verified_by="smtp")

    def test_replay_after_use_is_refused(self) -> None:
        """After the first use the address no longer matches the pinned one."""
        user = _user("new@example.com")  # already applied once

        with patch("llm_proxy.api.routers.auth.UserRepository", return_value=AsyncMock()):
            client = TestClient(_app(auth_router, user=user), raise_server_exceptions=False)
            response = client.post(
                "/api/auth/email/verify",
                json={"token": _email_token(to="new@example.com", pinned_from="old@example.com")},
            )

        assert response.status_code == 401
        assert _error_code(response) == "email_verification_superseded"

    def test_stale_link_cannot_overwrite_a_newer_address(self) -> None:
        user = _user("admin-attested@example.com")

        with patch("llm_proxy.api.routers.auth.UserRepository", return_value=AsyncMock()):
            client = TestClient(_app(auth_router, user=user), raise_server_exceptions=False)
            response = client.post(
                "/api/auth/email/verify",
                json={"token": _email_token(to="new@example.com", pinned_from="old@example.com")},
            )

        assert response.status_code == 401


def _link_token(*, uid: int = 1, sid: str | None = "sess-1") -> str:
    manager = JWTManager(ProxyAuthConfig(jwt_secret=TEST_SECRET))
    return manager.create_scoped_token(
        "oauth_link",
        {
            "uid": uid,
            "sid": sid,
            "provider": "corp",
            "subject": "subject-1",
            "email": "alice@example.com",
        },
        ttl_seconds=600,
    )


class TestOAuthLinkCompletion:
    """The provider callback is unauthenticated; binding needs the session."""

    def _client(self, *, user: MagicMock) -> TestClient:
        return TestClient(
            _app(oauth_router, user=user, session_id="sess-1"),
            raise_server_exceptions=False,
        )

    def test_rejects_a_token_minted_for_another_account(self) -> None:
        user = _user("alice@example.com")
        identities = AsyncMock()
        identities.get = AsyncMock(return_value=None)

        with patch("llm_proxy.api.routers.oauth.OAuthIdentityRepository", return_value=identities):
            response = self._client(user=user).post(
                "/api/me/oauth/link/complete", json={"link_token": _link_token(uid=2)}
            )

        assert response.status_code == 401
        assert _error_code(response) == "oauth_link_mismatch"
        identities.create.assert_not_awaited()

    def test_rejects_a_token_from_a_different_console_session(self) -> None:
        user = _user("alice@example.com")
        identities = AsyncMock()
        identities.get = AsyncMock(return_value=None)

        with patch("llm_proxy.api.routers.oauth.OAuthIdentityRepository", return_value=identities):
            response = self._client(user=user).post(
                "/api/me/oauth/link/complete",
                json={"link_token": _link_token(sid="another-session")},
            )

        assert response.status_code == 401
        assert _error_code(response) == "oauth_link_session_changed"
        identities.create.assert_not_awaited()

    def test_links_the_identity_for_the_signed_in_account(self) -> None:
        user = _user("alice@example.com")
        identities = AsyncMock()
        identities.get = AsyncMock(return_value=None)
        identities.create = AsyncMock(return_value=MagicMock())
        users = AsyncMock()
        users.get_for_update = AsyncMock(return_value=user)

        with (
            patch("llm_proxy.api.routers.oauth.OAuthIdentityRepository", return_value=identities),
            patch("llm_proxy.api.routers.oauth.UserRepository", return_value=users),
        ):
            response = self._client(user=user).post(
                "/api/me/oauth/link/complete", json={"link_token": _link_token()}
            )

        assert response.status_code == 200, response.text
        assert response.json() == {"success": True}
        identities.create.assert_awaited_once()
        assert identities.create.await_args.kwargs["user_id"] == user.id
        assert identities.create.await_args.kwargs["email"] == "alice@example.com"


class TestRecoveryCodeMaterial:
    def test_codes_only_use_the_alphabet(self) -> None:
        for code in generate_recovery_codes(20):
            assert set(code.replace("-", "")) <= set(_RECOVERY_ALPHABET)

    def test_hashing_uses_a_work_factor(self) -> None:
        """A bare SHA-256 of a ~50-bit code is enumerable offline; the stored
        digest must not be that."""
        code = "ABCDE-FGHJK"
        expected_sha256 = hashlib.sha256(code.replace("-", "").encode()).hexdigest()
        assert hash_recovery_code(code) != expected_sha256
        # Deterministic: the code is looked up by hash in SQL.
        assert hash_recovery_code(code) == hash_recovery_code("abcde fghjk")


@pytest.mark.parametrize("ambiguous", ["0", "O", "1", "I", "L"])
def test_recovery_code_generation_never_emits_ambiguous_glyphs(ambiguous: str) -> None:
    assert all(ambiguous not in code for code in generate_recovery_codes(50)), (
        f"{ambiguous!r} must never appear in a hand-transcribed recovery code"
    )
