"""Tests for authentication middleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from llm_proxy.api.gate import VerifiedKeyAuth
from llm_proxy.api.middleware.api_key_auth import api_key_auth_middleware
from llm_proxy.api.middleware.jwt_auth import jwt_auth_middleware
from llm_proxy.api.middleware.rate_limiting import RateLimitManager
from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.core.request_facts import facts_for, get_request_identity

_TEST_JWT_SECRET = "test-secret-at-least-32-characters-long"


def _auth_context_response(request: Request) -> dict:
    """What the test handlers report back: the facts the middleware stamped."""
    identity = get_request_identity(request)
    facts = facts_for(request)
    return {
        "scope_auth": {
            "principal_type": identity.auth_method,
            "principal_id": identity.api_key_name or identity.user,
            "allowed_models": facts.allowed_models,
            "allowed_mcp_servers": facts.allowed_mcp_servers,
        }
    }


def _make_auth_config() -> ProxyAuthConfig:
    """Build a minimal auth config (jwt_secret only; auth is always enabled)."""
    return ProxyAuthConfig(jwt_secret=_TEST_JWT_SECRET)


class TestAuthBypassPaths:
    """Test that auth bypass uses exact path matching, not prefix matching."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager with auth enabled."""
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_config.server_params.auth = _make_auth_config()
        mock_manager.get_config = AsyncMock(return_value=mock_config)
        return mock_manager

    @pytest.fixture
    def app_with_auth(self, mock_config_manager):
        """Create a test app with auth middleware."""
        app = FastAPI()

        @app.get("/api/auth/login")
        async def auth_login():
            return {"endpoint": "login"}

        @app.get("/api/auth/setup")
        async def auth_setup():
            return {"endpoint": "setup"}

        @app.get("/api/auth/setup-status")
        async def auth_setup_status():
            return {"endpoint": "setup-status"}

        @app.get("/api/health")
        async def health():
            return {"status": "ok"}

        @app.get("/api/health/live")
        async def health_live():
            return {"status": "alive"}

        @app.get("/api/health-check")
        async def health_check():
            return {"status": "check"}

        @app.get("/api/authenticated")
        async def authenticated():
            return {"endpoint": "authenticated"}

        app.state.config_manager = mock_config_manager

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await jwt_auth_middleware(request, call_next)

        return app

    def test_auth_login_bypasses_auth(self, app_with_auth):
        """Legitimate /api/auth/login should bypass authentication."""
        client = TestClient(app_with_auth, raise_server_exceptions=False)
        response = client.get("/api/auth/login")
        assert response.status_code == 200
        assert response.json() == {"endpoint": "login"}

    def test_auth_setup_bypasses_auth(self, app_with_auth):
        """The first-run setup endpoints should bypass authentication."""
        client = TestClient(app_with_auth, raise_server_exceptions=False)
        assert client.get("/api/auth/setup").status_code == 200
        assert client.get("/api/auth/setup-status").status_code == 200

    def test_health_bypasses_auth(self, app_with_auth):
        """Legitimate /api/health should bypass authentication."""
        client = TestClient(app_with_auth, raise_server_exceptions=False)
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_live_bypasses_auth(self, app_with_auth):
        """Legitimate /api/health/live should bypass authentication."""
        client = TestClient(app_with_auth, raise_server_exceptions=False)
        response = client.get("/api/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    def test_authenticated_path_does_not_bypass_auth(self, app_with_auth):
        """Path /api/authenticated should NOT bypass auth (security fix).

        Using startswith("/api/auth") would incorrectly allow paths like
        /api/authenticated to bypass authentication.
        """
        client = TestClient(app_with_auth, raise_server_exceptions=False)
        response = client.get("/api/authenticated")
        assert response.status_code == 401, (
            f"Path /api/authenticated should not bypass auth. Got {response.status_code}"
        )

    def test_health_check_path_does_not_bypass_auth(self, app_with_auth):
        """Path /api/health-check should NOT bypass auth (security fix).

        Using startswith("/api/health") would incorrectly allow paths like
        /api/health-check to bypass authentication.
        """
        client = TestClient(app_with_auth, raise_server_exceptions=False)
        response = client.get("/api/health-check")
        assert response.status_code == 401, (
            f"Path /api/health-check should not bypass auth. Got {response.status_code}"
        )


class TestPublicApiPaths:
    """Test that PUBLIC_API_PATHS is explicit and not broad."""

    def test_only_auth_public_paths_are_listed(self):
        """The exact set of public auth endpoints is explicit.

        Logout uses optional JWT auth: if a valid token is provided, the user is
        identified; otherwise the request is allowed but will be attributed to
        client IP in audit logs.
        """
        from llm_proxy.api.middleware.jwt_auth import PUBLIC_API_PATHS

        assert (
            frozenset(
                {
                    "/api/auth/login",
                    "/api/auth/setup",
                    "/api/auth/setup-status",
                    "/api/auth/capabilities",
                    "/api/auth/2fa/verify",
                    "/api/auth/2fa/recover",
                    "/api/auth/email/verify",
                    "/api/auth/password/forgot",
                    "/api/auth/password/reset",
                }
            )
            == PUBLIC_API_PATHS
        )

    def test_first_factor_and_invitation_prefixes_are_public(self):
        """Passkey login and OAuth/invite flows must not need a JWT."""
        from llm_proxy.api.middleware.jwt_auth import is_public_api_path

        for path in (
            "/api/auth/passkeys/login/begin",
            "/api/auth/passkeys/login/finish",
            "/api/auth/oauth/providers",
            "/api/auth/oauth/google/callback",
            "/api/auth/invitations/token/accept",
        ):
            assert is_public_api_path(path), path

        assert not is_public_api_path("/api/auth/passkeysX/login")
        assert not is_public_api_path("/api/me/passkeys/register/begin")

    def test_logout_optional_auth(self):
        """Logout endpoint supports optional JWT auth."""
        from llm_proxy.api.middleware.jwt_auth import _OPTIONAL_AUTH_PATHS

        assert "/api/auth/logout" in _OPTIONAL_AUTH_PATHS
        assert "/api/auth/session" in _OPTIONAL_AUTH_PATHS

    def test_auth_other_paths_not_public(self):
        """Paths like /api/auth/other should not be public."""
        from llm_proxy.api.middleware.jwt_auth import _OPTIONAL_AUTH_PATHS, PUBLIC_API_PATHS

        assert "/api/auth/other" not in PUBLIC_API_PATHS
        assert "/api/auth/other" not in _OPTIONAL_AUTH_PATHS
        assert "/api/auth/refresh" not in PUBLIC_API_PATHS
        assert "/api/auth/refresh" not in _OPTIONAL_AUTH_PATHS


class TestApiKeyAuthMCPPermissions:
    """Test that API key auth middleware stores MCP permissions on scope."""

    @pytest.mark.asyncio
    async def test_api_key_auth_stores_mcp_permissions_on_scope(self) -> None:
        """API key auth middleware should put auth context on request.state
        and ASGI scope with allowed_mcp_servers and mcp_tool_permissions."""
        from llm_proxy.api.middleware.api_key_auth import api_key_auth_middleware

        app = FastAPI()

        @app.get("/servers/test/mcp")
        async def handler(request: Request):
            return _auth_context_response(request)

        app.middleware("http")(api_key_auth_middleware)

        # Middleware bails out if config_manager is None
        app.state.config_manager = MagicMock()

        mock_lockout = MagicMock()
        mock_lockout.is_locked_out.return_value = False

        mock_sleep = AsyncMock()

        auth_info = VerifiedKeyAuth(
            principal_id="agent",
            allowed_models=None,
            allowed_mcp_servers=["github_mcp"],
            user_id=1,
        )

        with (
            patch(
                "llm_proxy.api.gate.verify_api_key",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.gate.get_api_key_lockout_manager",
                return_value=mock_lockout,
            ),
            patch("llm_proxy.api.gate.apply_auth_failure_delay", new=mock_sleep),
            patch("llm_proxy.api.gate._update_key_last_used", new=AsyncMock()),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/servers/test/mcp", headers={"Authorization": "Bearer sk_test"})

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["scope_auth"] is not None, "scope_auth is None"
        assert data["scope_auth"]["allowed_mcp_servers"] == ["github_mcp"]
        assert data["scope_auth"]["principal_type"] == "api_key"
        assert data["scope_auth"]["principal_id"] == "agent"

    @pytest.mark.asyncio
    async def test_api_key_auth_no_mcp_permissions_when_none(self) -> None:
        """When an API key has no MCP restrictions, the scope auth should
        have None for mcp fields."""
        from llm_proxy.api.middleware.api_key_auth import api_key_auth_middleware

        app = FastAPI()

        @app.get("/servers/test/mcp")
        async def handler(request: Request):
            return _auth_context_response(request)

        app.middleware("http")(api_key_auth_middleware)

        # Middleware bails out if config_manager is None
        app.state.config_manager = MagicMock()

        mock_lockout = MagicMock()
        mock_lockout.is_locked_out.return_value = False

        mock_sleep = AsyncMock()

        auth_info = VerifiedKeyAuth(
            principal_id="basic",
            allowed_models=["gpt-4"],
            allowed_mcp_servers=None,
            user_id=1,
        )

        with (
            patch(
                "llm_proxy.api.gate.verify_api_key",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.gate.get_api_key_lockout_manager",
                return_value=mock_lockout,
            ),
            patch("llm_proxy.api.gate.apply_auth_failure_delay", new=mock_sleep),
            patch("llm_proxy.api.gate._update_key_last_used", new=AsyncMock()),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/servers/test/mcp", headers={"Authorization": "Bearer sk_test"})

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["scope_auth"] is not None
        assert data["scope_auth"]["allowed_mcp_servers"] is None
        assert data["scope_auth"]["principal_type"] == "api_key"


class TestJwtAuthMCPPermissions:
    """Test that JWT auth middleware stores basic auth context on scope."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager with auth enabled."""
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_config.server_params.auth = _make_auth_config()
        mock_manager.get_config = AsyncMock(return_value=mock_config)
        return mock_manager

    @pytest.mark.asyncio
    async def test_jwt_admin_stamps_identity(self, mock_config_manager) -> None:
        """Admin JWT on /api/* paths should stamp the request facts' identity."""
        app = FastAPI()

        @app.get("/api/admin/test")
        async def handler(request: Request):
            return _auth_context_response(request)

        app.state.config_manager = mock_config_manager

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await jwt_auth_middleware(request, call_next)

        valid_admin_token = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiJhZG1pbiIsInR5cGUiOiJhZG1pbiIsImV4cCI6OTk5OTk5OTk5OX0"
            ".INVALID_SIGNATURE"
        )

        mock_jwt = MagicMock()
        mock_jwt.verify_token.return_value = {"sub": "admin", "type": "admin"}

        # The middleware now validates the token subject against the database:
        # mock the repository so the subject resolves to an active admin.
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.is_active = True
        mock_user.token_version = 0
        mock_user.must_change_password = False
        mock_repo = MagicMock()
        mock_repo.get_by_username = AsyncMock(return_value=mock_user)

        with (
            patch("llm_proxy.api.middleware.jwt_auth.JWTManager", return_value=mock_jwt),
            patch("llm_proxy.database.UserRepository", return_value=mock_repo),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(
                "/api/admin/test",
                headers={"Authorization": f"Bearer {valid_admin_token}"},
            )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["scope_auth"] is not None
        assert data["scope_auth"]["principal_type"] == "jwt"
        assert data["scope_auth"]["principal_id"] == "admin"
        # JWT on /api/* should not have allowed_models/allowed_mcp_servers (those are for API keys)
        assert data["scope_auth"]["allowed_models"] is None
        assert data["scope_auth"]["allowed_mcp_servers"] is None

    @pytest.mark.asyncio
    async def test_jwt_rejected_on_v1_paths(self, mock_config_manager) -> None:
        """JWT is NOT accepted on /v1/* paths — they require API key authentication."""
        app = FastAPI()

        @app.get("/v1/chat")
        async def handler(request: Request):
            return {"ok": True}

        app.state.config_manager = mock_config_manager

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await jwt_auth_middleware(request, call_next)

        valid_admin_token = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiJhZG1pbiIsInR5cGUiOiJhZG1pbiIsImV4cCI6OTk5OTk5OTk5OX0"
            ".INVALID_SIGNATURE"
        )

        mock_jwt = MagicMock()
        mock_jwt.verify_token.return_value = {"sub": "admin", "type": "admin"}

        with patch("llm_proxy.api.middleware.jwt_auth.JWTManager", return_value=mock_jwt):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(
                "/v1/chat",
                headers={"Authorization": f"Bearer {valid_admin_token}"},
            )

        # JWT middleware should NOT process /v1/* paths — let API key middleware handle it
        # The request should pass through without JWT identity being set
        # (API key auth will then require an API key, not JWT)
        assert response.status_code == 200, (
            f"Expected 200 (JWT passes through, API key required), got {response.status_code}"
        )


class TestApiKeyAuthSessionKeys:
    """Test that api_key_auth_middleware handles sk-ui- session API keys.

    Session API keys go through the same permission checks as regular API keys:
    - allowed_models=None means all models allowed (default)
    - allowed_mcp_servers=None means all MCP servers allowed (permissive default)
    """

    @pytest.mark.asyncio
    async def test_session_api_key_valid(self) -> None:
        """A valid sk-ui- token should authenticate and set identity."""
        from llm_proxy.api.middleware.api_key_auth import api_key_auth_middleware

        app = FastAPI()

        @app.get("/v1/chat")
        async def handler(request: Request):
            from llm_proxy.core.request_facts import facts_for, get_request_identity

            identity = get_request_identity(request)
            return {
                "identity_user": identity.user,
                "identity_api_key_name": identity.api_key_name,
                "identity_auth_method": identity.auth_method,
                "scope_auth": {
                    "principal_type": identity.auth_method,
                    "principal_id": identity.api_key_name or identity.user,
                    "allowed_models": facts_for(request).allowed_models,
                    "allowed_mcp_servers": facts_for(request).allowed_mcp_servers,
                },
            }

        app.middleware("http")(api_key_auth_middleware)
        app.state.config_manager = MagicMock()

        mock_session_record = MagicMock()
        mock_session_record.id = "session-abc"
        mock_session_record.user_id = 1

        mock_user = MagicMock()
        mock_user.username = "admin"

        mock_lockout = MagicMock()
        mock_lockout.is_locked_out.return_value = False

        mock_sleep = AsyncMock()

        auth_info = VerifiedKeyAuth(
            principal_id="session:session-abc",
            allowed_models=None,
            allowed_mcp_servers=None,
            user_id=1,
            is_session=True,
            username="admin",
        )

        with (
            patch(
                "llm_proxy.api.gate.verify_api_key",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.gate.get_api_key_lockout_manager",
                return_value=mock_lockout,
            ),
            patch(
                "llm_proxy.api.gate.apply_auth_failure_delay",
                new=mock_sleep,
            ),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/v1/chat", headers={"Authorization": "Bearer sk-ui-validtoken"})

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["identity_user"] == "admin"
        assert data["identity_auth_method"] == "session_api_key"
        assert data["identity_api_key_name"] == "session:session-abc"
        # Session API key: allowed_models=None (all models), allowed_mcp_servers=None (all MCP)
        assert data["scope_auth"]["allowed_models"] is None
        assert data["scope_auth"]["allowed_mcp_servers"] is None
        # The session identity's auth method surfaces as the principal type.
        assert data["scope_auth"]["principal_type"] == "session_api_key"
        assert data["scope_auth"]["principal_id"] == "session:session-abc"

    @pytest.mark.asyncio
    async def test_session_api_key_expired(self) -> None:
        """An expired/deactivated sk-ui- token should return 401."""
        from llm_proxy.api.middleware.api_key_auth import api_key_auth_middleware

        app = FastAPI()

        @app.get("/v1/chat")
        async def handler(request: Request):
            return {"ok": True}

        app.middleware("http")(api_key_auth_middleware)
        app.state.config_manager = MagicMock()

        mock_lockout = MagicMock()
        mock_lockout.is_locked_out.return_value = False

        mock_sleep = AsyncMock()

        with (
            patch(
                "llm_proxy.api.gate.verify_api_key",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "llm_proxy.api.gate.get_api_key_lockout_manager",
                return_value=mock_lockout,
            ),
            patch(
                "llm_proxy.api.gate.apply_auth_failure_delay",
                new=mock_sleep,
            ),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(
                "/v1/chat", headers={"Authorization": "Bearer sk-ui-expiredtoken"}
            )

        assert response.status_code == 401, (
            f"Expected 401, got {response.status_code}: {response.text}"
        )


def _build_jwt_test_app():
    """App with the JWT middleware and endpoints for each path category."""
    app = FastAPI()

    @app.get("/api/admin/test")
    async def admin_test():
        return {"ok": True}

    @app.put("/api/me/password")
    async def me_password():
        return {"ok": True}

    @app.get("/api/me/profile")
    async def me_profile():
        return {"ok": True}

    @app.post("/api/auth/logout")
    async def logout():
        return {"ok": True}

    @app.middleware("http")
    async def middleware(request: Request, call_next):
        return await jwt_auth_middleware(request, call_next)

    return app


def _mock_config_manager():
    mock_manager = MagicMock()
    mock_config = MagicMock()
    mock_config.server_params.auth = _make_auth_config()
    mock_manager.get_config = AsyncMock(return_value=mock_config)
    mock_manager.publish_generation = AsyncMock()
    return mock_manager


def _jwt_patchers(user_or_error):
    """Patch JWT verification and the user-repository lookup.

    ``user_or_error`` is either a mock user returned by get_by_username, or an
    Exception instance raised when the database session is opened (simulating
    a database outage).
    """
    mock_jwt = MagicMock()
    mock_jwt.verify_token.return_value = {"sub": "admin", "type": "admin"}

    mock_repo = MagicMock()
    if isinstance(user_or_error, Exception):
        mock_repo.get_by_username = AsyncMock(side_effect=user_or_error)
    else:
        mock_repo.get_by_username = AsyncMock(return_value=user_or_error)

    return (
        patch("llm_proxy.api.middleware.jwt_auth.JWTManager", return_value=mock_jwt),
        patch("llm_proxy.database.UserRepository", return_value=mock_repo),
    )


def _active_user(must_change_password: bool = False) -> MagicMock:
    user = MagicMock()
    user.id = 1
    user.is_active = True
    user.token_version = 0
    user.must_change_password = must_change_password
    return user


class TestForcedPasswordChangeGate:
    """Users flagged must_change_password may only reach the change allowlist."""

    def test_flagged_user_blocked_from_other_endpoints(self):
        app = _build_jwt_test_app()
        app.state.config_manager = _mock_config_manager()
        jwt_patch, repo_patch = _jwt_patchers(_active_user(must_change_password=True))
        with jwt_patch, repo_patch:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/api/admin/test", headers={"Authorization": "Bearer x"})

        assert response.status_code == 403
        assert response.json()["code"] == "password_change_required"

    def test_flagged_user_can_change_password_and_read_profile(self):
        app = _build_jwt_test_app()
        app.state.config_manager = _mock_config_manager()
        jwt_patch, repo_patch = _jwt_patchers(_active_user(must_change_password=True))
        with jwt_patch, repo_patch:
            client = TestClient(app, raise_server_exceptions=False)
            password_response = client.put(
                "/api/me/password", headers={"Authorization": "Bearer x"}
            )
            profile_response = client.get("/api/me/profile", headers={"Authorization": "Bearer x"})
            logout_response = client.post("/api/auth/logout", headers={"Authorization": "Bearer x"})

        assert password_response.status_code == 200
        assert profile_response.status_code == 200
        assert logout_response.status_code == 200

    def test_unflagged_user_passes(self):
        app = _build_jwt_test_app()
        app.state.config_manager = _mock_config_manager()
        jwt_patch, repo_patch = _jwt_patchers(_active_user(must_change_password=False))
        with jwt_patch, repo_patch:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/api/admin/test", headers={"Authorization": "Bearer x"})

        assert response.status_code == 200

    def test_flagged_user_unlocked_after_password_change(self):
        """End-to-end forced-change flow: blocked until the password is set.

        The real /api/me router runs behind the JWT middleware: a flagged
        user is refused everywhere outside the change allowlist, and the
        moment the password endpoint succeeds (clearing the flag through
        the repository) the same token is served normally again.
        """
        from llm_proxy.api.middleware.exceptions import register_exception_handlers
        from llm_proxy.api.routers.me import router as me_router
        from llm_proxy.security.passwords import hash_password

        user = MagicMock()
        user.id = 1
        user.username = "admin"
        user.password_hash = hash_password("current-password-123")
        user.is_active = True
        user.role = "admin"
        user.token_version = 0
        user.must_change_password = True

        # Stateful repository: the password endpoint clears the DB flag and
        # the middleware re-reads it on every request.
        mock_repo = MagicMock()
        mock_repo.get_by_username = AsyncMock(return_value=user)
        mock_repo.update_password = AsyncMock(return_value=True)
        mock_repo.set_must_change_password = AsyncMock(
            side_effect=lambda _user_id, value: (
                user.__setattr__("must_change_password", value),
                True,
            )[1]
        )
        mock_repo.increment_token_version = AsyncMock(return_value=True)
        mock_session_repo = MagicMock()
        mock_session_repo.deactivate_user_sessions = AsyncMock()

        mock_jwt = MagicMock()
        mock_jwt.verify_token.return_value = {"sub": "admin", "type": "admin", "tv": 0}

        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(me_router)
        # The repository is mocked, so the DI session/current-user lookups
        # must be stubbed too (same wiring as the /api/me test fixture).
        from llm_proxy.api.dependencies import get_current_user
        from llm_proxy.database import get_async_session

        app.dependency_overrides[get_async_session] = lambda: AsyncMock()
        app.dependency_overrides[get_current_user] = lambda: user

        @app.get("/api/admin/test")
        async def admin_test():
            return {"ok": True}

        @app.get("/api/auth/session")
        async def session_probe():
            return {"authenticated": True}

        @app.middleware("http")
        async def middleware(request: Request, call_next):
            return await jwt_auth_middleware(request, call_next)

        app.state.config_manager = _mock_config_manager()
        headers = {"Authorization": "Bearer x"}

        with (
            patch("llm_proxy.api.middleware.jwt_auth.JWTManager", return_value=mock_jwt),
            patch("llm_proxy.database.UserRepository", return_value=mock_repo),
            patch("llm_proxy.api.routers.me.UserRepository", return_value=mock_repo),
            patch(
                "llm_proxy.api.routers.me.UserSessionRepository",
                return_value=mock_session_repo,
            ),
            patch.object(
                RateLimitManager,
                "check_rate_limit",
                new=AsyncMock(return_value=(False, {})),
            ),
        ):
            client = TestClient(app, raise_server_exceptions=False)

            # Flagged: every endpoint outside the change allowlist is refused.
            blocked = client.get("/api/admin/test", headers=headers)
            assert blocked.status_code == 403
            assert blocked.json()["code"] == "password_change_required"

            # The console's session probe stays reachable, otherwise a reload
            # cannot learn that this account is in the forced-change flow.
            probe = client.get("/api/auth/session", headers=headers)
            assert probe.status_code == 200
            assert probe.json() == {"authenticated": True}

            # The change-password endpoint is reachable and clears the flag.
            changed = client.put(
                "/api/me/password",
                headers=headers,
                json={
                    "current_password": "current-password-123",
                    "new_password": "New-password-456!",
                },
            )
            assert changed.status_code == 200
            mock_repo.set_must_change_password.assert_awaited_once_with(1, False)
            assert user.must_change_password is False

            # Unlocked: the same token is now served normally.
            unlocked = client.get("/api/admin/test", headers=headers)
            assert unlocked.status_code == 200


class TestJwtValidationFailClosed:
    """A database outage during subject validation rejects the request (503)."""

    def test_db_error_returns_503_on_protected_path(self):
        app = _build_jwt_test_app()
        app.state.config_manager = _mock_config_manager()
        jwt_patch, repo_patch = _jwt_patchers(RuntimeError("db down"))
        with jwt_patch, repo_patch:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/api/admin/test", headers={"Authorization": "Bearer x"})

        assert response.status_code == 503
        assert response.json()["code"] == "authentication_unavailable"

    def test_db_error_stays_lenient_on_optional_auth_path(self):
        """Logout must keep working during a database outage."""
        app = _build_jwt_test_app()
        app.state.config_manager = _mock_config_manager()
        jwt_patch, repo_patch = _jwt_patchers(RuntimeError("db down"))
        with jwt_patch, repo_patch:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/auth/logout", headers={"Authorization": "Bearer x"})

        assert response.status_code == 200


class TestApiKeyAuthAliasPaths:
    """Alias paths are API endpoints and must require an API key."""

    @pytest.mark.parametrize(
        "path",
        ["/chat/completions", "/messages", "/v1/v1/chat/completions", "/v1/v1/messages"],
    )
    def test_alias_path_requires_api_key(self, path: str) -> None:
        from llm_proxy.api.middleware.api_key_auth import api_key_auth_middleware

        app = FastAPI()

        @app.post(path)
        async def handler():
            return {"ok": True}

        app.middleware("http")(api_key_auth_middleware)
        app.state.config_manager = MagicMock()

        mock_lockout = MagicMock()
        mock_lockout.is_locked_out.return_value = False

        with (
            patch(
                "llm_proxy.api.gate.get_api_key_lockout_manager",
                return_value=mock_lockout,
            ),
            patch(
                "llm_proxy.api.gate.apply_auth_failure_delay",
                new=AsyncMock(),
            ),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post(path, json={"model": "m"})

        assert response.status_code == 401, response.text
        assert response.json()["error"]["type"] == "authentication_error"


class TestMissingConfigManagerRefusesToServe:
    """A lifespan that never installed the config manager must not silently no-op.

    Both gates used to return early when the config manager was missing, leaving
    authentication to whatever dependency the route declared (the protocol routes'
    ``require_any_auth``, the protocol fast path's re-check, the admin routers'
    ``require_admin_role``). That holds today — nothing is served unauthenticated —
    but it makes every route the load-bearing part of a decision no reviewer can see
    at the gate. The gates now fail fast instead (ADR-0021).
    """

    def _app_without_a_config_manager(self, path: str, method: str) -> tuple[FastAPI, list]:
        from llm_proxy.api.middleware.exceptions import register_exception_handlers

        app = FastAPI()
        reached: list = []

        async def handler():
            reached.append(True)
            return {"ok": True}

        getattr(app, method)(path)(handler)
        register_exception_handlers(app)
        # No app.state.config_manager: the lifespan did not run.
        return app, reached

    def test_api_key_gate_refuses_the_request(self) -> None:
        from llm_proxy.api.middleware.api_key_auth import api_key_auth_middleware

        app, reached = self._app_without_a_config_manager("/v1/chat/completions", "post")
        app.middleware("http")(api_key_auth_middleware)

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/v1/chat/completions", json={"model": "m"})

        assert response.status_code == 500, response.text
        assert not reached, "a request with no readable auth policy reached the handler"

    def test_jwt_gate_refuses_the_request(self) -> None:
        app, reached = self._app_without_a_config_manager("/api/me", "get")
        app.middleware("http")(jwt_auth_middleware)

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/me", headers={"Authorization": "Bearer nope"})

        assert response.status_code == 500, response.text
        assert not reached, "a request with no readable auth policy reached the handler"


class TestConsoleSessionBinding:
    """A JWT bound to a revoked console session must stop working at once."""

    @pytest.fixture
    def revoked_session_client(self):
        """A client whose token's ``sid`` no longer resolves to a live session."""
        from llm_proxy.security.jwt import JWTManager

        auth_config = _make_auth_config()
        config = MagicMock()
        config.server_params.auth = auth_config
        config_manager = MagicMock()
        config_manager.get_config = AsyncMock(return_value=config)

        app = FastAPI()

        @app.get("/api/authenticated")
        async def authenticated():
            return {"ok": True}

        @app.get("/api/auth/session")
        async def session_probe(request: Request):
            identity = get_request_identity(request)
            return {"authenticated": identity.user is not None, "username": identity.user}

        @app.post("/api/auth/logout")
        async def logout_probe(request: Request):
            return {"user": get_request_identity(request).user}

        app.state.config_manager = config_manager
        app.middleware("http")(jwt_auth_middleware)

        user = MagicMock()
        user.id = 1
        user.username = "admin"
        user.is_active = True
        user.role = "admin"
        user.token_version = 0
        user.must_change_password = False
        user_repo = AsyncMock()
        user_repo.get_by_username = AsyncMock(return_value=user)
        session_repo = AsyncMock()
        session_repo.get_valid_session = AsyncMock(return_value=None)
        totp_repo = AsyncMock()
        totp_repo.get = AsyncMock(return_value=None)
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=AsyncMock())
        context.__aexit__ = AsyncMock(return_value=False)

        token = JWTManager(auth_config).create_token(
            "admin", role="admin", token_version=0, session_id="sess-1"
        )
        with (
            patch("llm_proxy.database.UserRepository", return_value=user_repo),
            patch("llm_proxy.database.UserSessionRepository", return_value=session_repo),
            patch("llm_proxy.database.TotpRepository", return_value=totp_repo),
            patch("llm_proxy.database.get_async_session_context", return_value=context),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            yield client, token, user_repo

    def test_revoked_session_returns_stable_code(self, revoked_session_client):
        client, token, _ = revoked_session_client

        response = client.get(
            "/api/authenticated",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 401
        assert response.json()["code"] == "session_revoked"

    def test_revoked_session_is_not_reported_as_signed_in(self, revoked_session_client):
        """The console probe must not name a user whose session is gone.

        Optional-auth paths stay reachable without a live session; the bug was
        that they were handed the *rejected* token's identity, so a tab whose
        session another sign-in had rotated away kept reporting a signed-in
        console and could act as that account (including signing it out).
        """
        client, token, _ = revoked_session_client

        response = client.get("/api/auth/session", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        assert response.json() == {"authenticated": False, "username": None}

    def test_revoked_session_logout_touches_nothing(self, revoked_session_client):
        """A stale tab's sign-out must resolve to an anonymous request."""
        client, token, user_repo = revoked_session_client

        response = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        # Anonymous: the handler sees no identity, so it cannot revoke the
        # account's sessions on behalf of a session that is already gone. The
        # only subject lookup is the gate's own, which rejected the token.
        assert response.json() == {"user": None}
        user_repo.get_by_username.assert_awaited_once()


class TestMfaEnrollmentGate:
    """An admin without TOTP must still reach the page that enrolls it."""

    #: Everything SecurityView reads on mount, plus the flows that let a
    #: passwordless admin step up to begin/confirm enrollment.
    _SECURITY_PAGE_PATHS = (
        "/api/me/2fa",
        "/api/me/2fa/begin",
        "/api/me/2fa/confirm",
        "/api/me/passkeys",
        "/api/me/passkeys/step-up/begin",
        "/api/me/passkeys/step-up/finish",
        "/api/me/oauth/identities",
        "/api/me/profile",
        "/api/auth/logout",
        "/api/auth/session",
        "/api/auth/step-up",
    )

    def test_allowlist_covers_the_security_page(self):
        """The gate's allowlist is exactly what the enrollment page needs."""
        from llm_proxy.api.middleware.jwt_auth import _MFA_ENROLLMENT_ALLOWED_PATHS

        assert frozenset(self._SECURITY_PAGE_PATHS) == _MFA_ENROLLMENT_ALLOWED_PATHS

    @pytest.fixture
    def enrollment_gate_client(self):
        """A client whose JWT belongs to an admin without a confirmed TOTP."""
        from llm_proxy.security.jwt import JWTManager

        auth_config = _make_auth_config()
        config = MagicMock()
        config.server_params.auth = auth_config
        config.server_params.console_auth.require_2fa_for_admins = True
        config_manager = MagicMock()
        config_manager.get_config = AsyncMock(return_value=config)

        app = FastAPI()

        @app.api_route("/{path:path}", methods=["GET", "POST"])
        async def catch_all(path: str):
            return {"path": path}

        app.state.config_manager = config_manager
        app.middleware("http")(jwt_auth_middleware)

        user = MagicMock()
        user.id = 1
        user.username = "admin"
        user.is_active = True
        user.role = "admin"
        user.token_version = 0
        user.must_change_password = False
        user_repo = AsyncMock()
        user_repo.get_by_username = AsyncMock(return_value=user)
        session_repo = AsyncMock()
        session_repo.get_valid_session = AsyncMock(return_value=MagicMock())
        totp_repo = AsyncMock()
        totp_repo.get = AsyncMock(return_value=None)
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=AsyncMock())
        context.__aexit__ = AsyncMock(return_value=False)

        token = JWTManager(auth_config).create_token(
            "admin", role="admin", token_version=0, session_id="sess-1"
        )
        with (
            patch("llm_proxy.database.UserRepository", return_value=user_repo),
            patch("llm_proxy.database.UserSessionRepository", return_value=session_repo),
            patch("llm_proxy.database.TotpRepository", return_value=totp_repo),
            patch("llm_proxy.database.get_async_session_context", return_value=context),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            yield client, token

    @pytest.mark.parametrize(
        "method,path",
        [
            ("get", "/api/me/2fa"),
            ("post", "/api/me/2fa/begin"),
            ("post", "/api/me/2fa/confirm"),
            ("get", "/api/me/passkeys"),
            ("post", "/api/me/passkeys/step-up/begin"),
            ("post", "/api/me/passkeys/step-up/finish"),
            ("get", "/api/me/oauth/identities"),
            ("get", "/api/me/profile"),
            ("post", "/api/auth/step-up"),
        ],
    )
    def test_security_page_requests_are_reachable(
        self, enrollment_gate_client, method: str, path: str
    ) -> None:
        client, token = enrollment_gate_client

        response = client.request(method, path, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200, response.text

    @pytest.mark.parametrize(
        "method,path",
        [
            ("get", "/api/team/members"),
            ("get", "/api/me/passkeys/register/begin"),
            ("post", "/api/me/passkeys/register/begin"),
        ],
    )
    def test_other_console_requests_stay_blocked(
        self, enrollment_gate_client, method: str, path: str
    ) -> None:
        client, token = enrollment_gate_client

        response = client.request(method, path, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 403, response.text
        assert response.json()["code"] == "mfa_enrollment_required"


class TestConsoleCookieAuth:
    """The browser cookie is ambient, so unsafe calls need the CSRF header."""

    def _app(self):

        auth_config = _make_auth_config()
        config = MagicMock()
        config.server_params.auth = auth_config
        config_manager = MagicMock()
        config_manager.get_config = AsyncMock(return_value=config)

        app = FastAPI()

        @app.get("/api/authenticated")
        async def authenticated_get():
            return {"ok": True}

        @app.post("/api/authenticated")
        async def authenticated_post():
            return {"ok": True}

        app.state.config_manager = config_manager
        app.middleware("http")(jwt_auth_middleware)
        return app, auth_config

    def _repos(self):
        user = MagicMock()
        user.username = "admin"
        user.is_active = True
        user.role = "admin"
        user.token_version = 0
        user.must_change_password = False
        user_repo = AsyncMock()
        user_repo.get_by_username = AsyncMock(return_value=user)
        session_repo = AsyncMock()
        session_repo.get_valid_session = AsyncMock(return_value=MagicMock())
        totp_repo = AsyncMock()
        totp_repo.get = AsyncMock(return_value=None)
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=AsyncMock())
        context.__aexit__ = AsyncMock(return_value=False)
        return user_repo, session_repo, totp_repo, context

    def test_cookie_authenticates_get(self):
        from llm_proxy.api.session_cookies import CONSOLE_SESSION_COOKIE
        from llm_proxy.security.jwt import JWTManager

        app, auth_config = self._app()
        user_repo, session_repo, totp_repo, context = self._repos()
        token = JWTManager(auth_config).create_token(
            "admin", role="admin", token_version=0, session_id="sess-1"
        )
        with (
            patch("llm_proxy.database.UserRepository", return_value=user_repo),
            patch("llm_proxy.database.UserSessionRepository", return_value=session_repo),
            patch("llm_proxy.database.TotpRepository", return_value=totp_repo),
            patch("llm_proxy.database.get_async_session_context", return_value=context),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.get(
                "/api/authenticated",
                headers={"Cookie": f"{CONSOLE_SESSION_COOKIE}={token}"},
            )

        assert response.status_code == 200, response.text

    def test_cookie_post_requires_csrf_header(self):
        from llm_proxy.api.session_cookies import (
            CONSOLE_SESSION_COOKIE,
            CSRF_COOKIE,
            CSRF_HEADER,
        )
        from llm_proxy.security.jwt import JWTManager

        app, auth_config = self._app()
        user_repo, session_repo, totp_repo, context = self._repos()
        token = JWTManager(auth_config).create_token(
            "admin", role="admin", token_version=0, session_id="sess-1"
        )
        cookie_header = f"{CONSOLE_SESSION_COOKIE}={token}; {CSRF_COOKIE}=csrf-value"
        with (
            patch("llm_proxy.database.UserRepository", return_value=user_repo),
            patch("llm_proxy.database.UserSessionRepository", return_value=session_repo),
            patch("llm_proxy.database.TotpRepository", return_value=totp_repo),
            patch("llm_proxy.database.get_async_session_context", return_value=context),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            rejected = client.post("/api/authenticated", headers={"Cookie": cookie_header})
            accepted = client.post(
                "/api/authenticated",
                headers={"Cookie": cookie_header, CSRF_HEADER: "csrf-value"},
            )

        assert rejected.status_code == 403, rejected.text
        assert rejected.json()["code"] == "csrf_failed"
        assert accepted.status_code == 200, accepted.text


class TestV1SessionCookieAuth:
    """The /v1 session key cookie is HttpOnly and CSRF-gated."""

    def test_cookie_key_requires_csrf_for_post(self) -> None:
        from llm_proxy.api.session_cookies import (
            CSRF_COOKIE,
            CSRF_HEADER,
            SESSION_API_KEY_COOKIE,
        )

        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat():
            return {"ok": True}

        app.middleware("http")(api_key_auth_middleware)
        app.state.config_manager = MagicMock()
        mock_lockout = MagicMock()
        mock_lockout.is_locked_out.return_value = False
        auth_info = VerifiedKeyAuth(
            principal_id="ui-session",
            allowed_models=None,
            allowed_mcp_servers=None,
            user_id=1,
        )
        cookie = f"{SESSION_API_KEY_COOKIE}=sk-ui-test; {CSRF_COOKIE}=csrf-value"

        with (
            patch(
                "llm_proxy.api.gate.verify_api_key",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.gate.get_api_key_lockout_manager",
                return_value=mock_lockout,
            ),
            patch("llm_proxy.api.gate.apply_auth_failure_delay", new=AsyncMock()),
            patch("llm_proxy.api.gate._update_key_last_used", new=AsyncMock()),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            rejected = client.post(
                "/v1/chat/completions",
                headers={"Cookie": cookie},
                json={"model": "m"},
            )
            accepted = client.post(
                "/v1/chat/completions",
                headers={"Cookie": cookie, CSRF_HEADER: "csrf-value"},
                json={"model": "m"},
            )

        assert rejected.status_code == 403, rejected.text
        assert rejected.json()["code"] == "csrf_failed"
        assert accepted.status_code == 200, accepted.text
