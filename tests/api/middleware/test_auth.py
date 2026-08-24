"""Tests for authentication middleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from llm_proxy.api.middleware.jwt_auth import jwt_auth_middleware
from llm_proxy.config.types.auth import ProxyAuthConfig

_TEST_JWT_SECRET = "test-secret-at-least-32-characters-long"


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
        """Only the public auth endpoints (login + first-run setup) are truly public.

        Logout uses optional JWT auth: if a valid token is provided, the user is
        identified; otherwise the request is allowed but will be attributed to
        client IP in audit logs.
        """
        from llm_proxy.api.middleware.jwt_auth import PUBLIC_API_PATHS

        assert (
            frozenset({"/api/auth/login", "/api/auth/setup", "/api/auth/setup-status"})
            == PUBLIC_API_PATHS
        )

    def test_logout_optional_auth(self):
        """Logout endpoint supports optional JWT auth."""
        from llm_proxy.api.middleware.jwt_auth import _OPTIONAL_AUTH_PATHS

        assert "/api/auth/logout" in _OPTIONAL_AUTH_PATHS

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
            return {"scope_auth": request.scope.get("llm_proxy_auth")}

        app.middleware("http")(api_key_auth_middleware)

        # Middleware bails out if config_manager is None
        app.state.config_manager = MagicMock()

        mock_lockout = MagicMock()
        mock_lockout.is_locked_out.return_value = False

        mock_sleep = AsyncMock()

        auth_info = {
            "principal_type": "api_key",
            "principal_id": "agent",
            "allowed_models": None,
            "allowed_mcp_servers": ["github_mcp"],
        }

        with (
            patch(
                "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.middleware.api_key_auth.get_api_key_lockout_manager",
                return_value=mock_lockout,
            ),
            patch("llm_proxy.api.middleware.api_key_auth.add_auth_failure_delay", new=mock_sleep),
            patch("llm_proxy.api.middleware.api_key_auth._update_key_last_used", new=AsyncMock()),
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
            return {"scope_auth": request.scope.get("llm_proxy_auth")}

        app.middleware("http")(api_key_auth_middleware)

        # Middleware bails out if config_manager is None
        app.state.config_manager = MagicMock()

        mock_lockout = MagicMock()
        mock_lockout.is_locked_out.return_value = False

        mock_sleep = AsyncMock()

        auth_info = {
            "principal_type": "api_key",
            "principal_id": "basic",
            "allowed_models": ["gpt-4"],
            "allowed_mcp_servers": None,
        }

        with (
            patch(
                "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.middleware.api_key_auth.get_api_key_lockout_manager",
                return_value=mock_lockout,
            ),
            patch("llm_proxy.api.middleware.api_key_auth.add_auth_failure_delay", new=mock_sleep),
            patch("llm_proxy.api.middleware.api_key_auth._update_key_last_used", new=AsyncMock()),
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
    async def test_jwt_admin_sets_scope_auth(self, mock_config_manager) -> None:
        """Admin JWT on /api/* paths should set scope['llm_proxy_auth']."""
        app = FastAPI()

        @app.get("/api/admin/test")
        async def handler(request: Request):
            return {"scope_auth": request.scope.get("llm_proxy_auth")}

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
        assert "allowed_models" not in data["scope_auth"]
        assert "allowed_mcp_servers" not in data["scope_auth"]

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
            from llm_proxy.core.identity import get_request_identity

            identity = get_request_identity(request)
            return {
                "identity_user": identity.user,
                "identity_api_key_name": identity.api_key_name,
                "identity_auth_method": identity.auth_method,
                "scope_auth": request.scope.get("llm_proxy_auth"),
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

        auth_info = {
            "principal_type": "api_key",
            "principal_id": "session:session-abc",
            "allowed_models": None,
            "allowed_mcp_servers": None,
            "user_id": 1,
        }

        async def set_identity(api_key: str, request: Request) -> str | None:
            from llm_proxy.core.identity import RequestIdentity, set_request_identity

            set_request_identity(
                request,
                RequestIdentity(
                    user="admin",
                    api_key_name="session:session-abc",
                    auth_method="session_api_key",
                    user_id=1,
                ),
            )
            return "session:session-abc"

        with (
            patch(
                "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.middleware.api_key_auth._set_session_identity",
                new=set_identity,
            ),
            patch(
                "llm_proxy.api.middleware.api_key_auth.get_api_key_lockout_manager",
                return_value=mock_lockout,
            ),
            patch(
                "llm_proxy.api.middleware.api_key_auth.add_auth_failure_delay",
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
        assert data["scope_auth"]["principal_type"] == "api_key"
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
                "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "llm_proxy.api.middleware.api_key_auth.get_api_key_lockout_manager",
                return_value=mock_lockout,
            ),
            patch(
                "llm_proxy.api.middleware.api_key_auth.add_auth_failure_delay",
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
        ):
            client = TestClient(app, raise_server_exceptions=False)

            # Flagged: every endpoint outside the change allowlist is refused.
            blocked = client.get("/api/admin/test", headers=headers)
            assert blocked.status_code == 403
            assert blocked.json()["code"] == "password_change_required"

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
