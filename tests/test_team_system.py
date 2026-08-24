"""Integration tests for the team/multi-user system.

Verifies the full auth + authorization flow: admin creates members, members
log in with JWT, viewer role enforcement, API key ownership, log filtering,
password management, and cascade deactivation on member deletion.

Uses an in-memory SQLite database with FastAPI TestClient for real
middleware-chain testing.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import bcrypt
import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql import select

from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.middleware.jwt_auth import jwt_auth_middleware
from llm_proxy.api.routers.api_keys import router as api_keys_router
from llm_proxy.api.routers.auth import router as auth_router
from llm_proxy.api.routers.config import router as config_router
from llm_proxy.api.routers.logs import router as logs_router
from llm_proxy.api.routers.mcp import router as mcp_router
from llm_proxy.api.routers.me import router as me_router
from llm_proxy.api.routers.team import router as team_router
from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.config.types.main import ProxyConfig
from llm_proxy.config.types.server import ServerParams
from llm_proxy.database import (
    ApiKeyRepository,
    UserRepository,
    get_async_session,
)
from llm_proxy.database.tables import Base, UserRecord, UserSessionRecord
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.passwords import hash_password

TEST_JWT_SECRET = "test-jwt-secret-for-team-system-32chars"

# ---------------------------------------------------------------------------
# Fixtures — global database state for the session factory
# ---------------------------------------------------------------------------


def _reset_db_globals(monkeypatch):
    """Reset the global connection module state so every test gets a fresh
    in-memory SQLite database."""
    monkeypatch.setenv("JWT_SECRET", TEST_JWT_SECRET)
    # Rate limiting is UI-managed (server_config "security" key); disable it
    # for tests by stubbing the params resolver.
    from llm_proxy.config.types.server import SecurityParams

    monkeypatch.setattr(
        "llm_proxy.api.middleware.security.get_security_params",
        lambda: SecurityParams(rate_limit_disabled=True),
    )
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", True)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)


def _make_test_config() -> ProxyConfig:
    """Return a minimal ProxyConfig that lets middleware + dependencies run."""
    return ProxyConfig(
        server_params=ServerParams(
            auth=ProxyAuthConfig(jwt_secret=TEST_JWT_SECRET),
        ),
    )


# ---------------------------------------------------------------------------
# Shared fixture — app, client, db helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _per_test_db(monkeypatch):
    """Before every test: reset globals so each test starts with a fresh DB."""
    _reset_db_globals(monkeypatch)


@pytest.fixture(autouse=True)
def _fast_test_overrides(monkeypatch):
    """Speed up the test suite by replacing expensive production defaults.

    * Lower bcrypt rounds: password hashing/verification dominates per-test
      runtime when every test seeds multiple users.
    * Disable background log/usage writers: these are not exercised by the
      assertions in this file, and their startup/cleanup loops add teardown
      latency to every request-driven test.
    """
    _orig_gensalt = bcrypt.gensalt
    monkeypatch.setattr(
        "llm_proxy.security.passwords._hash_with_bcrypt",
        lambda value: bcrypt.hashpw(value.encode("utf-8"), _orig_gensalt(rounds=4)).decode("utf-8"),
    )
    monkeypatch.setattr(
        "llm_proxy.observability.service.start_background_log_writer",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "llm_proxy.observability.service.start_background_usage_writer",
        lambda _retention_days=365: None,
    )


async def _create_test_engine():
    """Create a fresh in-memory SQLite engine for a single test.

    A new engine per test (rather than a class-scoped shared one) keeps every
    async resource — including the global background log/usage writers that
    error paths may spawn — bound to the test's own event loop. Sharing an
    engine across per-test event loops previously caused hangs when a writer
    task performed DB I/O on a connection created by a closed loop.
    """
    import os

    os.environ["JWT_SECRET"] = TEST_JWT_SECRET

    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    return engine, factory


@pytest.fixture
async def app_and_db(monkeypatch, request):
    """Build a test FastAPI app backed by an in-memory SQLite database.

    Each test gets its own engine (see _create_test_engine) plus clean global
    state via the ``_per_test_db`` autouse fixture.

    Returns (app, engine, session_factory) so tests can create their own
    sessions for seeding data.
    """
    engine, session_factory = await _create_test_engine()

    # Point the connection module globals at this engine: the JWT middleware
    # enforces token subjects via get_async_session_context(), and background
    # log writers use it as well.
    import llm_proxy.config.settings as settings_mod
    import llm_proxy.database.connection as conn_mod

    conn_mod._db_initialized = False
    conn_mod._migrations_run = True
    conn_mod._engine = engine
    conn_mod._async_session_factory = session_factory
    settings_mod._settings = None

    # Build the app
    app = FastAPI()

    # Override the session dependency so router-level code uses our engine
    async def _override_get_session():
        async with session_factory() as s:
            yield s

    app.dependency_overrides[get_async_session] = _override_get_session

    # Mock config manager — must be on app.state for the JWT middleware
    mock_config_manager = MagicMock()
    mock_config_manager.get_config = AsyncMock(return_value=_make_test_config())
    app.state.config_manager = mock_config_manager

    # Mock MCP manager
    mock_mcp_manager = MagicMock()
    mock_mcp_manager.list_active_servers = AsyncMock(return_value=[])
    app.state.mcp_manager = mock_mcp_manager

    # Add JWT middleware
    app.middleware("http")(jwt_auth_middleware)

    # Register routers
    register_exception_handlers(app)
    app.include_router(auth_router)
    app.include_router(team_router)
    app.include_router(api_keys_router)
    app.include_router(logs_router)
    app.include_router(me_router)
    app.include_router(config_router)
    app.include_router(mcp_router)

    yield app, engine, session_factory

    # Stop the global background log/usage writers: they are process-wide
    # singletons bound to whichever test's loop first triggered them. Only
    # writers owned by THIS test's loop can be stopped gracefully; foreign
    # writers (created on since-closed loops) can never run again, so the
    # singletons are reset and will be re-created on demand.
    from llm_proxy.observability import service as observability_service

    current_loop = asyncio.get_running_loop()
    for attr in (
        "_background_writer",
        "_background_audit_writer",
        "_background_usage_writer",
    ):
        writer = getattr(observability_service, attr, None)
        if writer is None:
            continue
        writer_task = getattr(writer, "_writer_task", None)
        if writer_task is not None and writer_task.get_loop() is current_loop:
            with contextlib.suppress(Exception):
                await writer.stop()
        setattr(observability_service, attr, None)

    await engine.dispose()


@pytest.fixture
async def client(app_and_db):
    """Return an httpx.AsyncClient bound to the test app.

    Uses ASGITransport so the app runs in the same event loop as the test
    (and seeding), avoiding the cross-loop deadlock that Starlette's
    TestClient portal causes with a shared async SQLite engine.
    """
    app, _, _ = app_and_db
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_token(username: str, role: str = "admin") -> str:
    """Create a signed JWT for *username* with the given role (payload field)."""
    mgr = JWTManager(ProxyAuthConfig(jwt_secret=TEST_JWT_SECRET))
    return mgr.create_token(username, role=role)


async def _seed_user(
    app_and_db, username: str = "viewer1", password: str = "Viewer@pass123", role: str = "viewer"
) -> int:
    """Create a user in the DB and return their id."""
    _, _, session_factory = app_and_db
    async with session_factory() as s:
        user = await UserRepository(s).create_user(username, hash_password(password), role=role)
        await s.commit()
        return user.id


async def _seed_logs(app_and_db, member1_id: int, member2_id: int) -> None:
    """Seed two log entries -- one for each member."""
    _, _, session_factory = app_and_db
    import time

    async with session_factory() as s:
        from llm_proxy.database.tables import RequestLog

        log1 = RequestLog(
            timestamp=time.time(),
            request_id="req-001",
            endpoint="/v1/chat/completions",
            method="POST",
            status_code=200,
            user_id=member1_id,
            model="test-model",
            log_type="endpoint",
        )
        log2 = RequestLog(
            timestamp=time.time(),
            request_id="req-002",
            endpoint="/v1/chat/completions",
            method="POST",
            status_code=200,
            user_id=member2_id,
            model="test-model",
            log_type="endpoint",
        )
        s.add_all([log1, log2])
        await s.commit()


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# Tests
# ============================================================================


class TestAdminCreatesMember:
    """Admin can create a viewer member via POST /api/team/members."""

    @pytest.mark.asyncio
    async def test_create_member(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        token = _make_token("admin")

        resp = await client.post(
            "/api/team/members",
            json={"username": "new-viewer", "password": "Strong@pass12"},
            headers=_auth_header(token),
        )

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["username"] == "new-viewer"
        assert body["role"] == "viewer"
        assert body["is_active"] is True

    @pytest.mark.asyncio
    async def test_create_member_duplicate(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        token = _make_token("admin")

        resp1 = await client.post(
            "/api/team/members",
            json={"username": "dup-viewer", "password": "Strong@pass12"},
            headers=_auth_header(token),
        )
        assert resp1.status_code == 201

        resp2 = await client.post(
            "/api/team/members",
            json={"username": "dup-viewer", "password": "Another@pass12"},
            headers=_auth_header(token),
        )
        assert resp2.status_code == 409


class TestMemberLogin:
    """Member can log in and receive a JWT token."""

    @pytest.mark.asyncio
    async def test_login_success(self, app_and_db, client):
        await _seed_user(app_and_db, "viewer1", "Viewer@pass123")

        resp = await client.post(
            "/api/auth/login",
            json={"username": "viewer1", "password": "Viewer@pass123"},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

        # Verify JWT payload role
        from llm_proxy.config.types.auth import ProxyAuthConfig
        from llm_proxy.security.jwt import JWTManager

        jwt_mgr = JWTManager(ProxyAuthConfig(jwt_secret=TEST_JWT_SECRET))
        payload = jwt_mgr.verify_token(body["access_token"])
        assert payload["role"] == "viewer"

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, app_and_db, client):
        await _seed_user(app_and_db, "viewer1", "Viewer@pass123")

        resp = await client.post(
            "/api/auth/login",
            json={"username": "viewer1", "password": "wrong"},
        )

        assert resp.status_code == 401


class TestViewerRoleAccess:
    """Viewer role members are denied access to admin-only endpoints (403)."""

    @pytest.mark.asyncio
    async def test_cannot_access_config_providers(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.get("/api/config/providers", headers=_auth_header(token))
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_cannot_access_config_models(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.get("/api/config/models", headers=_auth_header(token))
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_cannot_access_config_server(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.get("/api/config/server/logging", headers=_auth_header(token))
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_cannot_access_config_server_jwt_secret(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.get("/api/config/server/mcp-security", headers=_auth_header(token))
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_cannot_access_mcp(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.get("/api/mcp/servers", headers=_auth_header(token))
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_cannot_access_audit_verify(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.get("/api/logs/audit/verify-integrity", headers=_auth_header(token))
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_cannot_access_team_members(self, app_and_db, client):
        """Viewer is denied access to team management endpoints."""
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.get("/api/team/members", headers=_auth_header(token))
        assert resp.status_code == 403, resp.text


class TestMemberManagesOwnApiKeys:
    """Member can create and list their own API keys, but not others'."""

    @pytest.mark.asyncio
    async def test_create_api_key(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        member_id = await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.post(
            "/api/api-keys",
            json={"name": "my-key", "allowed_models": None},
            headers=_auth_header(token),
        )

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["name"] == "my-key"
        assert "key" in body

        # Verify the key is attributed to the member
        _, _, session_factory = app_and_db
        async with session_factory() as s:
            repo = ApiKeyRepository(s)
            key = await repo.get_api_key_by_name("my-key")
            assert key is not None
            assert key.user_id == member_id

    @pytest.mark.asyncio
    async def test_sees_only_own_keys(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")

        # Create keys for both members
        token1 = _make_token("viewer1")
        await client.post(
            "/api/api-keys",
            json={"name": "key-v1", "allowed_models": None},
            headers=_auth_header(token1),
        )

        await _seed_user(app_and_db, "viewer2", "viewer2pass")
        token2 = _make_token("viewer2")
        await client.post(
            "/api/api-keys",
            json={"name": "key-v2", "allowed_models": None},
            headers=_auth_header(token2),
        )

        # viewer1 lists keys — should only see "key-v1"
        resp = await client.get("/api/api-keys", headers=_auth_header(token1))
        assert resp.status_code == 200, resp.text
        keys = resp.json()
        names = {k["name"] for k in keys}
        assert "key-v1" in names
        assert "key-v2" not in names

    @pytest.mark.asyncio
    async def test_cannot_access_other_members_key(self, app_and_db, client):
        """Member cannot update another member's key."""
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")
        await _seed_user(app_and_db, "viewer2", "viewer2pass")

        # viewer1 creates a key
        token1 = _make_token("viewer1")
        await client.post(
            "/api/api-keys",
            json={"name": "v1-key", "allowed_models": None},
            headers=_auth_header(token1),
        )

        # viewer2 tries to rename viewer1's key
        token2 = _make_token("viewer2")
        resp = await client.put(
            "/api/api-keys/v1-key",
            json={"name": "stolen-key"},
            headers=_auth_header(token2),
        )
        assert resp.status_code == 403, resp.text

        # viewer2 tries to delete viewer1's key
        resp = await client.delete(
            "/api/api-keys/v1-key",
            headers=_auth_header(token2),
        )
        assert resp.status_code == 403, resp.text


class TestMemberLogVisibility:
    """Member sees only own logs; admin sees all."""

    @pytest.mark.asyncio
    async def test_member_sees_only_own_logs(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        member1_id = await _seed_user(app_and_db, "viewer1")
        member2_id = await _seed_user(app_and_db, "viewer2", "viewer2pass")

        await _seed_logs(app_and_db, member1_id, member2_id)

        # viewer1 should only see req-001 (not req-002)
        token = _make_token("viewer1")
        resp = await client.get("/api/logs", headers=_auth_header(token))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["items"]) >= 1
        request_ids = {item.get("request_id") for item in body["items"]}
        assert "req-001" in request_ids, "viewer1 should see their own log"
        assert "req-002" not in request_ids, "viewer1 should not see viewer2's log"

    @pytest.mark.asyncio
    async def test_admin_sees_all_logs(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        member1_id = await _seed_user(app_and_db, "viewer1")
        member2_id = await _seed_user(app_and_db, "viewer2", "viewer2pass")

        await _seed_logs(app_and_db, member1_id, member2_id)

        token = _make_token("admin")
        resp = await client.get("/api/logs", headers=_auth_header(token))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        request_ids = {item.get("request_id") for item in body["items"]}
        assert "req-001" in request_ids
        assert "req-002" in request_ids


class TestKeyOwnershipScoping:
    """API keys are strictly owner-scoped — admins see only their own keys."""

    @pytest.mark.asyncio
    async def test_admin_sees_only_own_keys(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")
        await _seed_user(app_and_db, "viewer2", "viewer2pass")

        # Create keys as each member
        token1 = _make_token("viewer1")
        await client.post(
            "/api/api-keys",
            json={"name": "member1-key", "allowed_models": None},
            headers=_auth_header(token1),
        )

        token2 = _make_token("viewer2")
        await client.post(
            "/api/api-keys",
            json={"name": "member2-key", "allowed_models": None},
            headers=_auth_header(token2),
        )

        # Admin lists keys: only their own (none) are visible, not members'.
        admin_token = _make_token("admin")
        resp = await client.get("/api/api-keys", headers=_auth_header(admin_token))
        assert resp.status_code == 200, resp.text
        keys = resp.json()
        names = {k["name"] for k in keys}
        assert "member1-key" not in names
        assert "member2-key" not in names

        # Each member still sees exactly their own key.
        resp = await client.get("/api/api-keys", headers=_auth_header(token1))
        assert [k["name"] for k in resp.json()] == ["member1-key"]

        # And an admin cannot manage a member's key directly.
        resp = await client.put(
            "/api/api-keys/member1-key",
            json={"budget_usd": 10.0},
            headers=_auth_header(admin_token),
        )
        assert resp.status_code == 403, resp.text


class TestPasswordManagement:
    """Member changes own password; admin resets member password."""

    @pytest.mark.asyncio
    async def test_member_changes_own_password(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1", "Old@pass123")
        token = _make_token("viewer1")

        resp = await client.put(
            "/api/me/password",
            json={
                "current_password": "Old@pass123",
                "new_password": "New@pass123!",
            },
            headers=_auth_header(token),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["message"] == "Password changed successfully. Please log in again."

        # The pre-change JWT is revoked (token version bumped)
        resp = await client.get("/api/me/profile", headers=_auth_header(token))
        assert resp.status_code == 401, resp.text

        # Verify old password no longer works
        resp = await client.post(
            "/api/auth/login",
            json={"username": "viewer1", "password": "Old@pass123"},
        )
        assert resp.status_code == 401, resp.text

        # Verify new password works
        resp = await client.post(
            "/api/auth/login",
            json={"username": "viewer1", "password": "New@pass123!"},
        )
        assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_member_cannot_change_with_wrong_current(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1", "Correct@pass1")
        token = _make_token("viewer1")

        resp = await client.put(
            "/api/me/password",
            json={
                "current_password": "wrongcurrent",
                "new_password": "Ignored@pass123",
            },
            headers=_auth_header(token),
        )
        assert resp.status_code == 401, resp.text

    @pytest.mark.asyncio
    async def test_admin_resets_member_password(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        member_id = await _seed_user(app_and_db, "viewer1", "Old@Secret12")
        admin_token = _make_token("admin")

        resp = await client.put(
            f"/api/team/members/{member_id}/password",
            json={"password": "Reset@NewPass1"},
            headers=_auth_header(admin_token),
        )
        assert resp.status_code == 200, resp.text

        # The member's pre-reset JWT is revoked (token version bumped)
        member_token = _make_token("viewer1")
        resp = await client.get("/api/me/profile", headers=_auth_header(member_token))
        assert resp.status_code == 401, resp.text

        # Verify new password works
        resp = await client.post(
            "/api/auth/login",
            json={"username": "viewer1", "password": "Reset@NewPass1"},
        )
        assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_admin_reset_nonexistent_member(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        admin_token = _make_token("admin")

        resp = await client.put(
            "/api/team/members/99999/password",
            json={"password": "Whatever@123456"},
            headers=_auth_header(admin_token),
        )
        assert resp.status_code == 404, resp.text


class TestDeleteMemberCascade:
    """Deleting a member deactivates their API keys and sessions."""

    @pytest.mark.asyncio
    async def test_delete_cascades(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        member_id = await _seed_user(app_and_db, "victim")
        token = _make_token("victim")

        # Create an API key for the member
        resp = await client.post(
            "/api/api-keys",
            json={"name": "victim-key", "allowed_models": None},
            headers=_auth_header(token),
        )
        assert resp.status_code == 201, resp.text

        # Create a session for the member (by logging in, which creates a session)
        resp = await client.post(
            "/api/auth/login",
            json={"username": "victim", "password": "Viewer@pass123"},
        )
        assert resp.status_code == 200, resp.text

        # Verify key exists and is active
        _, _, session_factory = app_and_db
        async with session_factory() as s:
            repo = ApiKeyRepository(s)
            key = await repo.get_api_key_by_name("victim-key")
            assert key is not None
            assert key.is_active is True

            # Verify session exists
            stmt = select(UserSessionRecord).where(
                UserSessionRecord.user_id == member_id,
                UserSessionRecord.is_active.is_(True),
            )
            result = await s.execute(stmt)
            sessions_before = result.scalars().all()
            assert len(sessions_before) > 0

        # Deactivate sessions first (required before user deletion)
        from llm_proxy.database.repositories.user_sessions import UserSessionRepository

        async with session_factory() as s:
            session_repo = UserSessionRepository(s)
            await session_repo.deactivate_user_sessions(member_id)
            await s.commit()

        # Admin deletes the member (now that sessions are deactivated)
        admin_token = _make_token("admin")
        resp = await client.delete(
            f"/api/team/members/{member_id}",
            headers=_auth_header(admin_token),
        )
        assert resp.status_code == 200, resp.text

        # Verify key is now deleted
        async with session_factory() as s:
            repo = ApiKeyRepository(s)
            key = await repo.get_api_key_by_name("victim-key")
            assert key is None, "API key should be deleted after member deletion"

            # Verify sessions are deleted (cascade)
            stmt = select(UserSessionRecord).where(
                UserSessionRecord.user_id == member_id,
            )
            result = await s.execute(stmt)
            sessions = result.scalars().all()
            assert len(sessions) == 0, "Sessions should be deleted after member deletion"

            # Verify user record is deleted
            stmt = select(UserRecord).where(UserRecord.id == member_id)
            result = await s.execute(stmt)
            user = result.scalar_one_or_none()
            assert user is None, "User should be deleted from the database"

    @pytest.mark.asyncio
    async def test_cannot_delete_admin(self, app_and_db, client):
        admin_id = await _seed_user(app_and_db, "admin", role="admin")
        admin_token = _make_token("admin")

        resp = await client.delete(
            f"/api/team/members/{admin_id}",
            headers=_auth_header(admin_token),
        )
        # Should be rejected because admin cannot be deleted
        assert resp.status_code in (400, 422), resp.text


class TestExistingAdminUnaffected:
    """After multi-user migration, existing admin still works normally."""

    @pytest.mark.asyncio
    async def test_admin_login_still_works(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", "adminpass123", role="admin")

        resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "adminpass123"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "access_token" in body

    @pytest.mark.asyncio
    async def test_admin_can_access_config(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        token = _make_token("admin")

        resp = await client.get("/api/config/providers", headers=_auth_header(token))
        assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_admin_can_access_team(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        token = _make_token("admin")

        resp = await client.get("/api/team/members", headers=_auth_header(token))
        assert resp.status_code == 200, resp.text
        members = resp.json()
        # Should include the admin
        usernames = {m["username"] for m in members}
        assert "admin" in usernames

    @pytest.mark.asyncio
    async def test_setup_status(self, app_and_db, client):
        """Setup status endpoint works correctly with multi-user DB."""
        # No admin yet
        resp = await client.get("/api/auth/setup-status")
        assert resp.status_code == 200
        assert resp.json()["needs_setup"] is True

        # Create admin
        await _seed_user(app_and_db, "admin", role="admin")

        resp = await client.get("/api/auth/setup-status")
        assert resp.status_code == 200
        assert resp.json()["needs_setup"] is False


class TestTeamMemberList:
    """Admin can list all team members."""

    @pytest.mark.asyncio
    async def test_list_members(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")
        await _seed_user(app_and_db, "viewer2", "viewer2pass")

        admin_token = _make_token("admin")
        resp = await client.get("/api/team/members", headers=_auth_header(admin_token))
        assert resp.status_code == 200, resp.text
        members = resp.json()
        usernames = {m["username"] for m in members}
        assert "admin" in usernames
        assert "viewer1" in usernames
        assert "viewer2" in usernames

    @pytest.mark.asyncio
    async def test_list_members_denied_for_viewer(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.get("/api/team/members", headers=_auth_header(token))
        assert resp.status_code == 403, resp.text


class TestTokenRevocation:
    """JWT revocation: token version + subject validation at the middleware."""

    @pytest.mark.asyncio
    async def test_deleted_user_token_rejected(self, app_and_db, client):
        """A JWT whose subject no longer exists is rejected with 401."""
        _, _, session_factory = app_and_db
        user_id = await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        # Sanity: token works while the user exists
        resp = await client.get("/api/me/profile", headers=_auth_header(token))
        assert resp.status_code == 200, resp.text

        async with session_factory() as s:
            await UserRepository(s).delete_user(user_id)
            await s.commit()

        resp = await client.get("/api/me/profile", headers=_auth_header(token))
        assert resp.status_code == 401, resp.text
        assert "no longer exists" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_disabled_user_token_rejected(self, app_and_db, client):
        """A JWT whose subject is disabled is rejected with 403."""
        _, _, session_factory = app_and_db
        user_id = await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        async with session_factory() as s:
            await UserRepository(s).deactivate_user(user_id)
            await s.commit()

        resp = await client.get("/api/me/profile", headers=_auth_header(token))
        assert resp.status_code == 403, resp.text
        assert "disabled" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_logout_does_not_crash_with_revoked_token(self, app_and_db, client):
        """Optional-auth paths tolerate revoked/invalid tokens (no hard failure)."""
        _, _, session_factory = app_and_db
        user_id = await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        async with session_factory() as s:
            repo = UserRepository(s)
            await repo.increment_token_version(user_id)
            await s.commit()

        # Logout is an optional-auth path: revoked token is treated as anonymous
        resp = await client.post("/api/auth/logout", headers=_auth_header(token))
        assert resp.status_code == 200, resp.text


class TestMemberModelAllowlist:
    """User-level model allowlist constrains member API keys."""

    @pytest.mark.asyncio
    async def test_admin_sets_member_allowlist(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        member_id = await _seed_user(app_and_db, "viewer1")
        admin_token = _make_token("admin")

        resp = await client.put(
            f"/api/team/members/{member_id}/models",
            json={"allowed_models": ["gpt-4o", "claude-sonnet"]},
            headers=_auth_header(admin_token),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["allowed_models"] == ["gpt-4o", "claude-sonnet"]

    @pytest.mark.asyncio
    async def test_member_cannot_create_key_outside_allowlist(self, app_and_db, client):
        _, _, session_factory = app_and_db
        member_id = await _seed_user(app_and_db, "viewer1")
        async with session_factory() as s:
            await UserRepository(s).set_allowed_models(member_id, ["gpt-4o"])
            await s.commit()
        token = _make_token("viewer1")

        resp = await client.post(
            "/api/api-keys",
            json={"name": "bad-key", "allowed_models": ["gpt-4o", "claude-opus"]},
            headers=_auth_header(token),
        )
        assert resp.status_code == 403, resp.text
        assert "claude-opus" in resp.text

    @pytest.mark.asyncio
    async def test_member_can_create_key_within_allowlist(self, app_and_db, client):
        _, _, session_factory = app_and_db
        member_id = await _seed_user(app_and_db, "viewer1")
        async with session_factory() as s:
            await UserRepository(s).set_allowed_models(member_id, ["gpt-4o", "gpt-4o-mini"])
            await s.commit()
        token = _make_token("viewer1")

        resp = await client.post(
            "/api/api-keys",
            json={"name": "ok-key", "allowed_models": ["gpt-4o"]},
            headers=_auth_header(token),
        )
        assert resp.status_code == 201, resp.text

        # A key with no explicit list is also allowed: the request-time
        # intersection with the user's allowlist keeps it constrained.
        resp = await client.post(
            "/api/api-keys",
            json={"name": "inherit-key"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 201, resp.text

    @pytest.mark.asyncio
    async def test_member_cannot_widen_own_key_beyond_allowlist(self, app_and_db, client):
        _, _, session_factory = app_and_db
        member_id = await _seed_user(app_and_db, "viewer1")
        async with session_factory() as s:
            await UserRepository(s).set_allowed_models(member_id, ["gpt-4o"])
            await s.commit()
        token = _make_token("viewer1")

        resp = await client.post(
            "/api/api-keys",
            json={"name": "my-key", "allowed_models": ["gpt-4o"]},
            headers=_auth_header(token),
        )
        assert resp.status_code == 201, resp.text

        resp = await client.put(
            "/api/api-keys/my-key/models",
            json={"allowed_models": ["gpt-4o", "claude-opus"]},
            headers=_auth_header(token),
        )
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_unconstrained_member_unchanged(self, app_and_db, client):
        """Members without a user allowlist keep the previous behavior."""
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.post(
            "/api/api-keys",
            json={"name": "free-key", "allowed_models": ["anything"]},
            headers=_auth_header(token),
        )
        assert resp.status_code == 201, resp.text


async def _seed_models(app_and_db, names: list[str]) -> None:
    """Insert ModelRecord rows (without provider mappings) for tests."""
    from llm_proxy.database.tables import ModelRecord

    _, _, session_factory = app_and_db
    async with session_factory() as s:
        for name in names:
            s.add(ModelRecord(name=name))
        await s.commit()


class TestModelNamesEndpoint:
    """GET /api/config/model-names returns the effective model list per user.

    This public (require_authenticated) endpoint powers the API-key model dropdown for
    non-admin users, so it must not 403 for viewers the way the admin-only
    ``/api/config/models`` route does. A non-admin with a per-user model
    allowlist sees only the names within that allowlist.
    """

    @pytest.mark.asyncio
    async def test_admin_sees_all_models(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        await _seed_models(app_and_db, ["gpt-4o", "claude-sonnet", "llama-3"])
        token = _make_token("admin")

        resp = await client.get("/api/config/model-names", headers=_auth_header(token))
        assert resp.status_code == 200, resp.text
        assert set(resp.json()) == {"gpt-4o", "claude-sonnet", "llama-3"}

    @pytest.mark.asyncio
    async def test_unrestricted_viewer_sees_all_models(self, app_and_db, client):
        await _seed_user(app_and_db, "viewer1")
        await _seed_models(app_and_db, ["gpt-4o", "claude-sonnet"])
        token = _make_token("viewer1")

        resp = await client.get("/api/config/model-names", headers=_auth_header(token))
        assert resp.status_code == 200, resp.text
        # A viewer with no user allowlist is unrestricted.
        assert set(resp.json()) == {"gpt-4o", "claude-sonnet"}

    @pytest.mark.asyncio
    async def test_restricted_viewer_sees_only_allowed_models(self, app_and_db, client):
        _, _, session_factory = app_and_db
        member_id = await _seed_user(app_and_db, "viewer1")
        await _seed_models(app_and_db, ["gpt-4o", "claude-sonnet", "llama-3"])
        async with session_factory() as s:
            await UserRepository(s).set_allowed_models(member_id, ["gpt-4o", "claude-sonnet"])
            await s.commit()
        token = _make_token("viewer1")

        resp = await client.get("/api/config/model-names", headers=_auth_header(token))
        assert resp.status_code == 200, resp.text
        assert set(resp.json()) == {"gpt-4o", "claude-sonnet"}

    @pytest.mark.asyncio
    async def test_unconfigured_allowlist_entries_are_hidden(self, app_and_db, client):
        # An allowlist entry that is not a configured model is filtered out.
        _, _, session_factory = app_and_db
        member_id = await _seed_user(app_and_db, "viewer1")
        await _seed_models(app_and_db, ["gpt-4o"])
        async with session_factory() as s:
            await UserRepository(s).set_allowed_models(member_id, ["gpt-4o", "ghost-model"])
            await s.commit()
        token = _make_token("viewer1")

        resp = await client.get("/api/config/model-names", headers=_auth_header(token))
        assert resp.status_code == 200, resp.text
        assert resp.json() == ["gpt-4o"]

    @pytest.mark.asyncio
    async def test_unauthenticated_request_rejected(self, app_and_db, client):
        await _seed_models(app_and_db, ["gpt-4o"])

        resp = await client.get("/api/config/model-names")
        assert resp.status_code == 401, resp.text


class TestApiKeyMcpAllowAllDefault:
    """Unconfigured keys default to allow-all for models and MCP servers.

    A member can configure ``allowed_mcp_servers`` on their own keys, and an
    explicit ``null`` on update resets a restriction back to allow-all.
    """

    @pytest.mark.asyncio
    async def test_non_admin_cannot_set_mcp_servers(self, app_and_db, client):
        """Non-admin members cannot configure MCP server permissions on their keys."""
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.post(
            "/api/api-keys",
            json={"name": "k1", "allowed_mcp_servers": ["github_mcp"]},
            headers=_auth_header(token),
        )
        # Request succeeds but MCP servers are stripped for non-admins
        assert resp.status_code == 201, resp.text
        assert resp.json()["allowed_mcp_servers"] is None

    @pytest.mark.asyncio
    async def test_unconfigured_key_defaults_to_allow_all(self, app_and_db, client):
        """A key created without MCP/models config stores null (allow all)."""
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.post(
            "/api/api-keys",
            json={"name": "default-key"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["allowed_models"] is None
        assert data["allowed_mcp_servers"] is None

    @pytest.mark.asyncio
    async def test_update_with_null_resets_to_allow_all(self, app_and_db, client):
        """Updating allowed_mcp_servers with null resets it to allow all."""
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        # Non-admin creates a key - MCP servers are stripped (admin-only)
        resp = await client.post(
            "/api/api-keys",
            json={"name": "restricted", "allowed_mcp_servers": ["github_mcp"]},
            headers=_auth_header(token),
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["allowed_mcp_servers"] is None

        # Non-admin cannot update MCP servers - the update is ignored.
        # An omitted field preserves the stored value (rename keeps None).
        resp = await client.put(
            "/api/api-keys/restricted",
            json={"name": "renamed"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["allowed_mcp_servers"] is None
        assert resp.json()["name"] == "renamed"

    @pytest.mark.asyncio
    async def test_update_models_only_does_not_rename(self, app_and_db, client):
        """PUT /api/api-keys/{name} omitting the name field must not crash or rename."""
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.post(
            "/api/api-keys",
            json={"name": "scope-only-key"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 201, resp.text

        resp = await client.put(
            "/api/api-keys/scope-only-key",
            json={"allowed_models": ["gpt-4"]},
            headers=_auth_header(token),
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["name"] == "scope-only-key"
        assert data["allowed_models"] == ["gpt-4"]

    @pytest.mark.asyncio
    async def test_update_models_endpoint_strips_mcp_for_non_admin(self, app_and_db, client):
        """Non-admins cannot change MCP servers via PUT /{name}/models."""
        await _seed_user(app_and_db, "admin", role="admin")
        admin_token = _make_token("admin")
        await _seed_user(app_and_db, "viewer1")
        viewer_token = _make_token("viewer1")

        # Admin creates a key for themselves, then restricts MCP.
        resp = await client.post(
            "/api/api-keys",
            json={"name": "admin-key", "allowed_mcp_servers": ["github_mcp"]},
            headers=_auth_header(admin_token),
        )
        assert resp.status_code == 201, resp.text

        # Non-admin trying to widen MCP access gets ignored.
        resp = await client.put(
            "/api/api-keys/admin-key/models",
            json={"allowed_mcp_servers": ["filesystem_mcp"]},
            headers=_auth_header(viewer_token),
        )
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_effective_mcp_allows_all_when_unconfigured(self, app_and_db):
        """verify_api_key_for_mcp returns None (allow all) for an unconfigured key."""
        _, _, session_factory = app_and_db
        from llm_proxy.api.middleware.api_key_cache import invalidate_api_key_cache
        from llm_proxy.api.middleware.mcp_proxy import verify_api_key_for_mcp
        from llm_proxy.security.passwords import generate_api_key, hash_api_key

        user_id = await _seed_user(app_and_db, "viewer1")
        plain_key = generate_api_key()
        async with session_factory() as s:
            await ApiKeyRepository(s).create_api_key(
                name="k1", key_hash=hash_api_key(plain_key), user_id=user_id
            )
            await s.commit()

        invalidate_api_key_cache()
        info = await verify_api_key_for_mcp(plain_key)
        assert info is not None
        assert info["allowed_mcp_servers"] is None
        # An unconfigured key must be able to access any configured MCP server.
        from llm_proxy.api.routers.mcp import _server_access_allowed
        from llm_proxy.mcp.security.policy import McpSecurityPolicy

        assert (
            _server_access_allowed(
                "any_server", McpSecurityPolicy(require_key_mcp_permissions=True), info
            )
            is True
        )


class TestLogCleanupAdminOnly:
    """DELETE /api/logs/cleanup is restricted to admins."""

    @pytest.mark.asyncio
    async def test_viewer_cannot_cleanup(self, app_and_db, client):
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1")

        resp = await client.delete("/api/logs/cleanup", headers=_auth_header(token))
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_admin_can_cleanup(self, app_and_db, client):
        await _seed_user(app_and_db, "admin", role="admin")
        token = _make_token("admin")

        resp = await client.delete(
            "/api/logs/cleanup?older_than_days=30", headers=_auth_header(token)
        )
        assert resp.status_code == 200, resp.text
        assert "deleted" in resp.json()


class TestDisabledUserKeyAccess:
    """Keys owned by a disabled user stop working at verification time."""

    @pytest.mark.asyncio
    async def test_api_key_of_disabled_user_rejected(self, app_and_db):
        _, _, session_factory = app_and_db
        from llm_proxy.api.middleware.api_key_cache import invalidate_api_key_cache
        from llm_proxy.api.middleware.mcp_proxy import verify_api_key_for_mcp
        from llm_proxy.security.passwords import generate_api_key, hash_api_key

        user_id = await _seed_user(app_and_db, "viewer1")
        plain_key = generate_api_key()
        async with session_factory() as s:
            await ApiKeyRepository(s).create_api_key(
                name="k1", key_hash=hash_api_key(plain_key), user_id=user_id
            )
            await s.commit()

        invalidate_api_key_cache()
        assert await verify_api_key_for_mcp(plain_key) is not None

        async with session_factory() as s:
            await UserRepository(s).deactivate_user(user_id)
            await s.commit()

        invalidate_api_key_cache()
        assert await verify_api_key_for_mcp(plain_key) is None

    @pytest.mark.asyncio
    async def test_session_key_of_disabled_user_rejected(self, app_and_db):
        _, _, session_factory = app_and_db
        from llm_proxy.api.middleware.mcp_proxy import _verify_session_api_key
        from llm_proxy.database import UserSessionRepository

        user_id = await _seed_user(app_and_db, "viewer1")
        async with session_factory() as s:
            _, session_key = await UserSessionRepository(s).create_session(user_id)
            await s.commit()

        assert await _verify_session_api_key(session_key) is not None

        async with session_factory() as s:
            await UserRepository(s).deactivate_user(user_id)
            await s.commit()

        assert await _verify_session_api_key(session_key) is None

    @pytest.mark.asyncio
    async def test_effective_models_intersected_with_user_allowlist(self, app_and_db):
        """Key verification returns key.allowed_models ∩ user.allowed_models."""
        _, _, session_factory = app_and_db
        from llm_proxy.api.middleware.api_key_cache import invalidate_api_key_cache
        from llm_proxy.api.middleware.mcp_proxy import verify_api_key_for_mcp
        from llm_proxy.security.passwords import generate_api_key, hash_api_key

        user_id = await _seed_user(app_and_db, "viewer1")
        plain_key = generate_api_key()
        async with session_factory() as s:
            repo = UserRepository(s)
            await repo.set_allowed_models(user_id, ["gpt-4o", "claude-sonnet"])
            await ApiKeyRepository(s).create_api_key(
                name="k1",
                key_hash=hash_api_key(plain_key),
                user_id=user_id,
                allowed_models=["gpt-4o", "gpt-4o-mini"],
            )
            await s.commit()

        invalidate_api_key_cache()
        info = await verify_api_key_for_mcp(plain_key)
        assert info is not None
        assert info["allowed_models"] == ["gpt-4o"]

    @pytest.mark.asyncio
    async def test_session_key_inherits_user_allowlist(self, app_and_db):
        _, _, session_factory = app_and_db
        from llm_proxy.api.middleware.mcp_proxy import _verify_session_api_key
        from llm_proxy.database import UserSessionRepository

        user_id = await _seed_user(app_and_db, "viewer1")
        async with session_factory() as s:
            repo = UserRepository(s)
            await repo.set_allowed_models(user_id, ["gpt-4o"])
            _, session_key = await UserSessionRepository(s).create_session(user_id)
            await s.commit()

        info = await _verify_session_api_key(session_key)
        assert info is not None
        assert info["allowed_models"] == ["gpt-4o"]


class TestUserLevelBudget:
    """Admin-set account budgets: set/reset via team endpoints, enforced across
    all of the member's keys (and their session keys) at request time."""

    @pytest.mark.asyncio
    async def test_user_budget_full_flow(self, app_and_db, client):
        import time as _time

        from llm_proxy.api.middleware.api_key_cache import (
            get_budget_spend_cache,
            invalidate_api_key_cache,
        )
        from llm_proxy.api.middleware.mcp_proxy import (
            BudgetCheckStatus,
            check_key_budget,
            verify_api_key_for_mcp,
        )
        from llm_proxy.database.tables import UsageRecord

        # Process-global spend cache: start clean so earlier tests' entries
        # (keyed user:<id>) cannot leak into this flow.
        get_budget_spend_cache().invalidate()

        await _seed_user(app_and_db, "admin", role="admin")
        member_id = await _seed_user(app_and_db, "viewer1")
        admin_token = _make_token("admin")
        member_token = _make_token("viewer1")

        # The member creates a key (self-service).
        resp = await client.post(
            "/api/api-keys",
            json={"name": "member-key"},
            headers=_auth_header(member_token),
        )
        assert resp.status_code == 201, resp.text
        plain_key = resp.json()["key"]

        # Admin caps the member's account at $10/month.
        resp = await client.put(
            f"/api/team/members/{member_id}/budget",
            json={"budget_usd": 10.0, "budget_period": "monthly"},
            headers=_auth_header(admin_token),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["budget_usd"] == 10.0
        assert body["budget_period"] == "monthly"
        assert body["budget_spend_usd"] == 0.0

        # The member can see their own envelope.
        resp = await client.get("/api/me/budget", headers=_auth_header(member_token))
        assert resp.status_code == 200, resp.text
        assert resp.json()["budget_usd"] == 10.0
        assert resp.json()["period_spend_usd"] == 0.0

        # The budget rides along in the verified key's auth info.
        invalidate_api_key_cache()
        info = await verify_api_key_for_mcp(plain_key)
        assert info is not None
        assert info["user_budget"].budget_usd == 10.0
        assert await check_key_budget(info) is BudgetCheckStatus.OK

        # Seed usage over the cap: the member's key is now blocked.
        _, _, session_factory = app_and_db
        async with session_factory() as s:
            s.add(
                UsageRecord(
                    timestamp=_time.time(),
                    model="gpt-4o",
                    cost_usd=12.0,
                    log_type="endpoint",
                    user_id=member_id,
                    api_key_name="member-key",
                )
            )
            await s.commit()

        get_budget_spend_cache().invalidate()
        assert await check_key_budget(info) is BudgetCheckStatus.USER_EXCEEDED

        # /api/me/budget reflects the over-cap spend.
        resp = await client.get("/api/me/budget", headers=_auth_header(member_token))
        assert resp.json()["period_spend_usd"] == 12.0

        # Raising the account budget unblocks the key (the endpoint invalidates
        # the caches, so the new envelope takes effect immediately).
        resp = await client.put(
            f"/api/team/members/{member_id}/budget",
            json={"budget_usd": 50.0},
            headers=_auth_header(admin_token),
        )
        assert resp.status_code == 200, resp.text
        invalidate_api_key_cache()
        info = await verify_api_key_for_mcp(plain_key)
        assert info["user_budget"].budget_usd == 50.0
        assert await check_key_budget(info) is BudgetCheckStatus.OK

        # Lower the cap again and reset the window: spend counts from the
        # reset point, so the key is unblocked despite being "over" the cap.
        resp = await client.put(
            f"/api/team/members/{member_id}/budget",
            json={"budget_usd": 5.0},
            headers=_auth_header(admin_token),
        )
        assert resp.status_code == 200, resp.text
        resp = await client.post(
            f"/api/team/members/{member_id}/budget/reset",
            headers=_auth_header(admin_token),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["budget_spend_usd"] == 0.0

        invalidate_api_key_cache()
        info = await verify_api_key_for_mcp(plain_key)
        assert await check_key_budget(info) is BudgetCheckStatus.OK

    @pytest.mark.asyncio
    async def test_viewer_cannot_set_member_budget(self, app_and_db, client):
        """The account budget endpoints are admin-only."""
        await _seed_user(app_and_db, "viewer1")
        member_id = await _seed_user(app_and_db, "viewer2", "Viewer2@pass123")
        token = _make_token("viewer1", role="viewer")

        resp = await client.put(
            f"/api/team/members/{member_id}/budget",
            json={"budget_usd": 10.0},
            headers=_auth_header(token),
        )
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_viewer_cannot_reset_member_budget(self, app_and_db, client):
        """The budget-reset endpoint is admin-only as well."""
        await _seed_user(app_and_db, "viewer1")
        member_id = await _seed_user(app_and_db, "viewer2", "Viewer2@pass123")
        token = _make_token("viewer1", role="viewer")

        resp = await client.post(
            f"/api/team/members/{member_id}/budget/reset",
            headers=_auth_header(token),
        )
        assert resp.status_code == 403, resp.text

    @pytest.mark.asyncio
    async def test_session_key_subject_to_account_budget(self, app_and_db):
        """Session keys carry the owner's budget envelope — the UI key cannot
        bypass the admin-set account cap.

        This pins the wiring inside ``_verify_session_api_key``: it must
        surface ``user_budget_*`` in the auth info (derived from the real
        user record), and the shared ``check_key_budget`` must then block the
        session principal once the account is over its cap.
        """
        import time as _time

        from llm_proxy.api.middleware.api_key_cache import get_budget_spend_cache
        from llm_proxy.api.middleware.mcp_proxy import (
            BudgetCheckStatus,
            _verify_session_api_key,
            check_key_budget,
        )
        from llm_proxy.database import UserSessionRepository
        from llm_proxy.database.tables import UsageRecord

        # Process-global spend cache: start clean so earlier tests' entries
        # cannot leak into this flow.
        get_budget_spend_cache().invalidate()

        _, _, session_factory = app_and_db
        user_id = await _seed_user(app_and_db, "viewer1")
        async with session_factory() as s:
            repo = UserRepository(s)
            await repo.set_budget(
                user_id, budget_usd=10.0, budget_period="monthly", budget_reset_day=None
            )
            _, session_key = await UserSessionRepository(s).create_session(user_id)
            await s.commit()

        # The wiring: the session token resolves to auth info carrying the
        # owner's account-level budget envelope.
        info = await _verify_session_api_key(session_key)
        assert info is not None
        assert info["user_id"] == user_id
        assert info["user_budget"].budget_usd == 10.0
        assert info["user_budget"].budget_period == "monthly"

        # Under the cap the session key is allowed...
        assert await check_key_budget(info) is BudgetCheckStatus.OK

        # ...and once the account's spend crosses the cap, it is blocked
        # through the same shared check used for regular API keys.
        async with session_factory() as s:
            s.add(
                UsageRecord(
                    timestamp=_time.time(),
                    model="gpt-4o",
                    cost_usd=12.0,
                    log_type="endpoint",
                    user_id=user_id,
                )
            )
            await s.commit()
        get_budget_spend_cache().invalidate()
        assert await check_key_budget(info) is BudgetCheckStatus.USER_EXCEEDED

    @pytest.mark.asyncio
    async def test_clearing_budget_drops_manual_reset_stamp(self, app_and_db):
        """Clearing the cap clears ``budget_reset_at`` so re-setting the cap
        never counts spend from the unlimited period.
        """
        _, _, session_factory = app_and_db
        user_id = await _seed_user(app_and_db, "viewer1")
        async with session_factory() as s:
            repo = UserRepository(s)
            await repo.set_budget(
                user_id, budget_usd=10.0, budget_period="monthly", budget_reset_day=None
            )
            await repo.reset_budget(user_id)
            await s.commit()
            user = await repo.get_by_id(user_id)
            assert user is not None
            assert user.budget_reset_at is not None

            await repo.set_budget(
                user_id, budget_usd=None, budget_period=None, budget_reset_day=None
            )
            await s.commit()
            user = await repo.get_by_id(user_id)
            assert user is not None
            assert user.budget_reset_at is None

    @pytest.mark.asyncio
    async def test_member_budgets_are_fully_self_service(self, app_and_db, client):
        """Members can raise, clear, and reset their own keys' budgets."""
        await _seed_user(app_and_db, "viewer1")
        token = _make_token("viewer1", role="viewer")

        resp = await client.post(
            "/api/api-keys",
            json={"name": "k", "budget_usd": 5.0, "budget_period": "daily"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 201, resp.text

        # Raise the cap.
        resp = await client.put(
            "/api/api-keys/k", json={"budget_usd": 500.0}, headers=_auth_header(token)
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["budget_usd"] == 500.0

        # Reset the window.
        resp = await client.post("/api/api-keys/k/budget/reset", headers=_auth_header(token))
        assert resp.status_code == 200, resp.text
        assert resp.json()["budget_reset_at"] is not None

        # Clear the budget.
        resp = await client.put(
            "/api/api-keys/k", json={"budget_usd": None}, headers=_auth_header(token)
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["budget_usd"] is None
        # The stale manual reset stamp is dropped with the cap, so re-setting
        # a budget later never counts spend from the unlimited period.
        assert resp.json()["budget_reset_at"] is None
