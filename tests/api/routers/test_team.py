"""Integration tests for the team management endpoints (/api/team/*)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.logs import _user_role_cache
from llm_proxy.api.routers.team import router
from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.core.identity import RequestIdentity, set_request_identity
from llm_proxy.database import get_async_session
from llm_proxy.observability.types import ActionCategory, Outcome

TEST_SECRET = "test-secret-at-least-32-characters-long-aaaaaa"


def _mock_user(user_id: int, username: str, role: str = "viewer") -> MagicMock:
    user = MagicMock()
    user.id = user_id
    user.username = username
    user.role = role
    user.is_active = True
    user.must_change_password = False
    user.allowed_models = None
    user.token_version = 0
    user.budget_usd = None
    user.budget_period = None
    user.budget_reset_day = None
    user.budget_reset_at = None
    # Response-computed enrichment fields (not record fields); without explicit
    # values a bare MagicMock would coerce to 1.0 under pydantic validation.
    user.budget_spend_usd = None
    user.budget_period_start = None
    return user


@pytest.fixture
def team_client():
    """Test app with an authenticated admin identity and mocked repositories."""
    admin = _mock_user(1, "admin", role="admin")

    # Repo used by require_admin_role (lives in api.dependencies).
    auth_repo = AsyncMock()
    auth_repo.get_by_username = AsyncMock(return_value=admin)

    # Repo used by the endpoint itself (lives in api.routers.team).
    mock_repo = AsyncMock()

    mock_config = MagicMock()
    mock_config.server_params.auth = ProxyAuthConfig(jwt_secret=TEST_SECRET)
    mock_config_manager = MagicMock()
    mock_config_manager.get_config = AsyncMock(return_value=mock_config)

    with (
        patch("llm_proxy.api.dependencies.UserRepository", return_value=auth_repo),
        patch("llm_proxy.api.routers.team.UserRepository", return_value=mock_repo),
    ):
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router)
        app.state.config_manager = mock_config_manager

        @app.middleware("http")
        async def _set_identity(request: Request, call_next):
            set_request_identity(
                request,
                RequestIdentity(user=admin.username, auth_method="jwt", user_id=admin.id),
            )
            return await call_next(request)

        # The repos are mocked, so the session is never actually used.
        app.dependency_overrides[get_async_session] = lambda: AsyncMock()

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, admin, mock_repo


class TestUpdateMemberUsername:
    def test_rename_other_member(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "oldname"))
        renamed = _mock_user(2, "newname")
        mock_repo.update_username = AsyncMock(return_value=renamed)

        response = client.put("/api/team/members/2/username", json={"username": "NewName"})

        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "newname"
        # Renaming someone else must not issue a token.
        assert body["access_token"] is None
        mock_repo.update_username.assert_awaited_once_with(2, "NewName")

    def test_self_rename_returns_fresh_token(self, team_client):
        client, admin, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=admin)
        renamed = _mock_user(admin.id, "boss", role="admin")
        renamed.token_version = 1  # repo.update_username bumped token_version
        mock_repo.update_username = AsyncMock(return_value=renamed)

        response = client.put("/api/team/members/1/username", json={"username": "boss"})

        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "boss"
        # The admin's old JWT references the old username in `sub`, so a
        # replacement token must be returned for a self-rename.
        payload = jwt.decode(body["access_token"], TEST_SECRET, algorithms=["HS256"])
        assert payload["sub"] == "boss"
        assert payload["role"] == "admin"
        assert payload["tv"] == 1

    def test_user_not_found(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=None)

        response = client.put("/api/team/members/99/username", json={"username": "ghost"})

        assert response.status_code == 404
        mock_repo.update_username.assert_not_awaited()

    def test_taken_username_conflicts(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "oldname"))
        mock_repo.update_username = AsyncMock(side_effect=ValueError("User 'taken' already exists"))

        response = client.put("/api/team/members/2/username", json={"username": "taken"})

        assert response.status_code == 409

    def test_invalid_username_format_rejected(self, team_client):
        client, _, mock_repo = team_client

        response = client.put("/api/team/members/2/username", json={"username": "bad name!"})

        assert response.status_code == 422
        mock_repo.update_username.assert_not_awaited()

    def test_rename_invalidates_user_role_cache(self, team_client):
        """The logs router's role cache is keyed by username: after a rename,
        both the old and new names must be dropped so a recycled username
        cannot inherit the previous owner's cached role/user_id."""
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "oldname"))
        mock_repo.update_username = AsyncMock(return_value=_mock_user(2, "newname"))
        _user_role_cache["oldname"] = ("admin", 2, True, time.monotonic())
        _user_role_cache["newname"] = ("viewer", 99, True, time.monotonic())
        try:
            response = client.put("/api/team/members/2/username", json={"username": "newname"})

            assert response.status_code == 200
            assert "oldname" not in _user_role_cache
            assert "newname" not in _user_role_cache
        finally:
            _user_role_cache.pop("oldname", None)
            _user_role_cache.pop("newname", None)

    def test_requires_admin(self):
        """Non-admin (viewer) users are rejected before reaching the endpoint."""
        viewer = _mock_user(2, "viewer", role="viewer")
        auth_repo = AsyncMock()
        auth_repo.get_by_username = AsyncMock(return_value=viewer)

        with patch("llm_proxy.api.dependencies.UserRepository", return_value=auth_repo):
            app = FastAPI()
            register_exception_handlers(app)
            app.include_router(router)

            @app.middleware("http")
            async def _set_identity(request: Request, call_next):
                set_request_identity(
                    request,
                    RequestIdentity(user=viewer.username, auth_method="jwt", user_id=viewer.id),
                )
                return await call_next(request)

            app.dependency_overrides[get_async_session] = lambda: AsyncMock()

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.put("/api/team/members/1/username", json={"username": "x"})

        assert response.status_code == 403

    def test_requires_auth(self):
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router)
        app.dependency_overrides[get_async_session] = lambda: AsyncMock()

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.put("/api/team/members/1/username", json={"username": "x"})

        assert response.status_code == 401


class TestViewerRejectedMatrix:
    """The full team-management surface rejects viewers, not just rename."""

    # (method, path, body) — one representative call per endpoint.
    ENDPOINTS = [
        ("get", "/api/team/members", None),
        (
            "post",
            "/api/team/members",
            {"username": "newbie", "password": "Temporary-pass-123!", "role": "viewer"},
        ),
        ("put", "/api/team/members/1/role", {"role": "admin"}),
        ("delete", "/api/team/members/1", None),
        ("post", "/api/team/members/1/deactivate", None),
        ("post", "/api/team/members/1/reactivate", None),
        ("put", "/api/team/members/1/models", {"allowed_models": []}),
        ("put", "/api/team/members/1/password", {"password": "Brand-new-pass-456!"}),
        ("put", "/api/team/members/1/username", {"username": "newname"}),
    ]

    def test_viewer_rejected_from_all_endpoints(self):
        """Every member-management endpoint answers 403 to a viewer identity."""
        viewer = _mock_user(2, "viewer", role="viewer")
        auth_repo = AsyncMock()
        auth_repo.get_by_username = AsyncMock(return_value=viewer)

        with patch("llm_proxy.api.dependencies.UserRepository", return_value=auth_repo):
            app = FastAPI()
            register_exception_handlers(app)
            app.include_router(router)

            @app.middleware("http")
            async def _set_identity(request: Request, call_next):
                set_request_identity(
                    request,
                    RequestIdentity(user=viewer.username, auth_method="jwt", user_id=viewer.id),
                )
                return await call_next(request)

            app.dependency_overrides[get_async_session] = lambda: AsyncMock()

            with TestClient(app, raise_server_exceptions=False) as client:
                for method, path, body in self.ENDPOINTS:
                    kwargs = {} if body is None else {"json": body}
                    response = getattr(client, method)(path, **kwargs)
                    assert response.status_code == 403, (
                        f"{method.upper()} {path} -> {response.status_code}"
                    )


def _patch_sessions_and_keys():
    """Patch the session/api-key repositories referenced by team endpoints."""
    return (
        patch("llm_proxy.api.routers.team.UserSessionRepository"),
        patch("llm_proxy.api.routers.team.ApiKeyRepository"),
    )


class TestUpdateMemberRole:
    """PUT /members/{id}/role — promote/demote with self and last-admin guards."""

    def test_promote_viewer_to_admin(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "member"))
        mock_repo.set_role = AsyncMock(return_value=_mock_user(2, "member", role="admin"))

        sessions_patch, _ = _patch_sessions_and_keys()
        with sessions_patch as session_repo_cls:
            session_repo_cls.return_value.deactivate_user_sessions = AsyncMock()
            response = client.put("/api/team/members/2/role", json={"role": "admin"})

        assert response.status_code == 200
        assert response.json()["role"] == "admin"
        mock_repo.set_role.assert_awaited_once_with(2, "admin")
        # The role rides in the JWT, so existing tokens and sessions must be revoked.
        mock_repo.increment_token_version.assert_awaited_once_with(2)
        session_repo_cls.return_value.deactivate_user_sessions.assert_awaited_once_with(2)

    def test_demote_admin_to_viewer_allowed_when_not_last(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "second-admin", role="admin"))
        mock_repo.count_active_admins = AsyncMock(return_value=2)
        mock_repo.set_role = AsyncMock(return_value=_mock_user(2, "second-admin"))

        sessions_patch, _ = _patch_sessions_and_keys()
        with sessions_patch as session_repo_cls:
            session_repo_cls.return_value.deactivate_user_sessions = AsyncMock()
            response = client.put("/api/team/members/2/role", json={"role": "viewer"})

        assert response.status_code == 200
        mock_repo.set_role.assert_awaited_once_with(2, "viewer")

    def test_demote_last_active_admin_rejected(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "only-admin", role="admin"))
        mock_repo.count_active_admins = AsyncMock(return_value=1)

        response = client.put("/api/team/members/2/role", json={"role": "viewer"})

        assert response.status_code == 400
        mock_repo.set_role.assert_not_awaited()
        mock_repo.increment_token_version.assert_not_awaited()

    def test_change_own_role_rejected(self, team_client):
        client, admin, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=admin)

        response = client.put(f"/api/team/members/{admin.id}/role", json={"role": "viewer"})

        assert response.status_code == 400
        mock_repo.set_role.assert_not_awaited()

    def test_noop_same_role_does_not_revoke_sessions(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "member"))

        response = client.put("/api/team/members/2/role", json={"role": "viewer"})

        assert response.status_code == 200
        mock_repo.set_role.assert_not_awaited()
        mock_repo.increment_token_version.assert_not_awaited()

    def test_role_change_user_not_found(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=None)

        response = client.put("/api/team/members/99/role", json={"role": "admin"})

        assert response.status_code == 404
        mock_repo.set_role.assert_not_awaited()

    def test_invalid_role_rejected(self, team_client):
        client, _, mock_repo = team_client

        response = client.put("/api/team/members/2/role", json={"role": "superuser"})

        assert response.status_code == 422
        mock_repo.set_role.assert_not_awaited()


class TestDeleteMember:
    """DELETE /members/{id} — admins deletable, self and last-admin guarded."""

    def test_delete_viewer(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "member"))

        sessions_patch, keys_patch = _patch_sessions_and_keys()
        with keys_patch as key_repo_cls:
            key_repo_cls.return_value.delete_api_keys_by_user = AsyncMock()
            response = client.delete("/api/team/members/2")

        assert response.status_code == 200
        key_repo_cls.return_value.delete_api_keys_by_user.assert_awaited_once_with(2)
        mock_repo.delete_user.assert_awaited_once_with(2)

    def test_delete_admin_allowed_when_not_last(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "second-admin", role="admin"))
        mock_repo.count_active_admins = AsyncMock(return_value=2)

        sessions_patch, keys_patch = _patch_sessions_and_keys()
        with keys_patch as key_repo_cls:
            key_repo_cls.return_value.delete_api_keys_by_user = AsyncMock()
            response = client.delete("/api/team/members/2")

        assert response.status_code == 200
        mock_repo.delete_user.assert_awaited_once_with(2)

    def test_delete_last_active_admin_rejected(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "only-admin", role="admin"))
        mock_repo.count_active_admins = AsyncMock(return_value=1)

        response = client.delete("/api/team/members/2")

        assert response.status_code == 400
        mock_repo.delete_user.assert_not_awaited()

    def test_delete_self_rejected(self, team_client):
        client, admin, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=admin)

        response = client.delete(f"/api/team/members/{admin.id}")

        assert response.status_code == 400
        mock_repo.delete_user.assert_not_awaited()

    def test_delete_not_found(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=None)

        response = client.delete("/api/team/members/99")

        assert response.status_code == 404
        mock_repo.delete_user.assert_not_awaited()

    def test_delete_invalidates_role_cache(self, team_client):
        """Deleting a member drops their username-keyed role cache entry so a
        recycled username cannot inherit the previous owner's cached role."""
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "deleted-user"))

        _user_role_cache["deleted-user"] = ("viewer", 2, True, time.monotonic())
        try:
            sessions_patch, keys_patch = _patch_sessions_and_keys()
            with keys_patch as key_repo_cls:
                key_repo_cls.return_value.delete_api_keys_by_user = AsyncMock()
                response = client.delete("/api/team/members/2")

            assert response.status_code == 200
            assert "deleted-user" not in _user_role_cache
        finally:
            _user_role_cache.pop("deleted-user", None)


class TestDeactivateReactivateMember:
    """POST /members/{id}/deactivate|reactivate — suspend without deletion."""

    def test_deactivate_viewer_revokes_access(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "member"))

        sessions_patch, _ = _patch_sessions_and_keys()
        with sessions_patch as session_repo_cls:
            session_repo_cls.return_value.deactivate_user_sessions = AsyncMock()
            response = client.post("/api/team/members/2/deactivate")

        assert response.status_code == 200
        mock_repo.deactivate_user.assert_awaited_once_with(2)
        mock_repo.increment_token_version.assert_awaited_once_with(2)
        session_repo_cls.return_value.deactivate_user_sessions.assert_awaited_once_with(2)

    def test_deactivate_is_idempotent(self, team_client):
        client, _, mock_repo = team_client
        inactive = _mock_user(2, "member")
        inactive.is_active = False
        mock_repo.get_by_id = AsyncMock(return_value=inactive)

        response = client.post("/api/team/members/2/deactivate")

        assert response.status_code == 200
        mock_repo.set_active.assert_not_awaited()
        mock_repo.increment_token_version.assert_not_awaited()

    def test_deactivate_self_rejected(self, team_client):
        client, admin, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=admin)

        response = client.post(f"/api/team/members/{admin.id}/deactivate")

        assert response.status_code == 400
        mock_repo.set_active.assert_not_awaited()

    def test_deactivate_last_active_admin_rejected(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "only-admin", role="admin"))
        mock_repo.count_active_admins = AsyncMock(return_value=1)

        response = client.post("/api/team/members/2/deactivate")

        assert response.status_code == 400
        mock_repo.set_active.assert_not_awaited()

    def test_deactivate_admin_allowed_when_not_last(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "second-admin", role="admin"))
        mock_repo.count_active_admins = AsyncMock(return_value=2)

        sessions_patch, _ = _patch_sessions_and_keys()
        with sessions_patch as session_repo_cls:
            session_repo_cls.return_value.deactivate_user_sessions = AsyncMock()
            response = client.post("/api/team/members/2/deactivate")

        assert response.status_code == 200
        mock_repo.deactivate_user.assert_awaited_once_with(2)

    def test_deactivate_not_found(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=None)

        response = client.post("/api/team/members/99/deactivate")

        assert response.status_code == 404

    def test_reactivate_member(self, team_client):
        client, _, mock_repo = team_client
        inactive = _mock_user(2, "member")
        inactive.is_active = False
        mock_repo.get_by_id = AsyncMock(return_value=inactive)

        response = client.post("/api/team/members/2/reactivate")

        assert response.status_code == 200
        mock_repo.set_active.assert_awaited_once_with(2, True)

    def test_reactivate_is_idempotent(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "member"))

        response = client.post("/api/team/members/2/reactivate")

        assert response.status_code == 200
        mock_repo.set_active.assert_not_awaited()

    def test_reactivate_not_found(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=None)

        response = client.post("/api/team/members/99/reactivate")

        assert response.status_code == 404


class TestCreateMemberFlags:
    """Admin-created accounts must change the temporary password on first login."""

    def test_create_member_sets_must_change_password(self, team_client):
        client, _, mock_repo = team_client
        created = _mock_user(2, "newbie")
        created.must_change_password = True
        mock_repo.create_user = AsyncMock(return_value=created)

        response = client.post(
            "/api/team/members",
            json={"username": "newbie", "password": "Temporary-pass-123!", "role": "viewer"},
        )

        assert response.status_code == 201
        call = mock_repo.create_user.await_args
        assert call.args[0] == "newbie"
        assert call.kwargs["role"] == "viewer"
        assert call.kwargs["must_change_password"] is True
        assert response.json()["must_change_password"] is True


class TestResetMemberPasswordFlags:
    """Admin password resets also force the member to choose their own password."""

    def test_reset_password_sets_must_change_password(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "member"))

        sessions_patch, _ = _patch_sessions_and_keys()
        with sessions_patch as session_repo_cls:
            session_repo_cls.return_value.deactivate_user_sessions = AsyncMock()
            response = client.put(
                "/api/team/members/2/password", json={"password": "Brand-new-pass-456!"}
            )

        assert response.status_code == 200
        mock_repo.set_must_change_password.assert_awaited_once_with(2, True)
        mock_repo.increment_token_version.assert_awaited_once_with(2)


class TestMemberAuditLog:
    """Every member-management mutation writes an explicit audit entry that
    records the operating admin as actor and the target username as resource."""

    @pytest.fixture
    def audit_patch(self):
        with patch("llm_proxy.api.routers.team.write_member_audit_log", new=AsyncMock()) as audit:
            yield audit

    def test_create_member_writes_audit_log(self, team_client, audit_patch):
        client, _, mock_repo = team_client
        mock_repo.create_user = AsyncMock(return_value=_mock_user(2, "newbie"))

        response = client.post(
            "/api/team/members",
            json={"username": "newbie", "password": "Temporary-pass-123!", "role": "viewer"},
        )

        assert response.status_code == 201, response.text
        audit_patch.assert_awaited_once()
        kwargs = audit_patch.await_args.kwargs
        assert kwargs["actor"] == "admin"
        assert kwargs["action"] == ActionCategory.CREATE
        assert kwargs["target_user"] == "newbie"
        assert kwargs["outcome"] == Outcome.SUCCESS

    def test_delete_member_writes_audit_log(self, team_client, audit_patch):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "member"))
        mock_repo.count_active_admins = AsyncMock(return_value=2)

        sessions_patch, keys_patch = _patch_sessions_and_keys()
        with sessions_patch, keys_patch as key_repo_cls:
            key_repo_cls.return_value.delete_api_keys_by_user = AsyncMock()
            response = client.delete("/api/team/members/2")

        assert response.status_code == 200, response.text
        audit_patch.assert_awaited_once()
        kwargs = audit_patch.await_args.kwargs
        assert kwargs["actor"] == "admin"
        assert kwargs["action"] == ActionCategory.DELETE
        assert kwargs["target_user"] == "member"

    def test_role_change_writes_audit_log(self, team_client, audit_patch):
        client, _, mock_repo = team_client
        target = _mock_user(2, "member")
        mock_repo.get_by_id = AsyncMock(return_value=target)
        promoted = _mock_user(2, "member", role="admin")
        mock_repo.set_role = AsyncMock(return_value=promoted)

        sessions_patch, _ = _patch_sessions_and_keys()
        with sessions_patch as session_repo_cls:
            session_repo_cls.return_value.deactivate_user_sessions = AsyncMock()
            response = client.put("/api/team/members/2/role", json={"role": "admin"})

        assert response.status_code == 200, response.text
        audit_patch.assert_awaited_once()
        kwargs = audit_patch.await_args.kwargs
        assert kwargs["actor"] == "admin"
        assert kwargs["action"] == ActionCategory.UPDATE
        assert kwargs["target_user"] == "member"
        assert kwargs["extra"] == {"operation": "role", "old_role": "viewer", "new_role": "admin"}

    def test_password_reset_writes_audit_log(self, team_client, audit_patch):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "member"))

        sessions_patch, _ = _patch_sessions_and_keys()
        with sessions_patch as session_repo_cls:
            session_repo_cls.return_value.deactivate_user_sessions = AsyncMock()
            response = client.put(
                "/api/team/members/2/password", json={"password": "Brand-new-pass-456!"}
            )

        assert response.status_code == 200, response.text
        audit_patch.assert_awaited_once()
        kwargs = audit_patch.await_args.kwargs
        assert kwargs["actor"] == "admin"
        assert kwargs["action"] == ActionCategory.UPDATE
        assert kwargs["target_user"] == "member"
        assert kwargs["extra"] == {"operation": "password_reset"}

    def test_failed_guard_does_not_write_member_audit_log(self, team_client, audit_patch):
        """Rejected operations (e.g. self-delete) are left to the generic
        exception-handler audit path; the explicit helper only logs successes."""
        client, admin, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=admin)

        response = client.delete(f"/api/team/members/{admin.id}")

        assert response.status_code == 400
        audit_patch.assert_not_awaited()


class TestUpdateMemberBudget:
    """Admin-managed account-level budget endpoints (/members/{id}/budget)."""

    def test_set_budget(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "member"))
        updated = _mock_user(2, "member")
        updated.budget_usd = 50.0
        updated.budget_period = "monthly"
        updated.budget_reset_day = 15
        mock_repo.set_budget = AsyncMock(return_value=updated)

        usage_repo = AsyncMock()
        usage_repo.get_spend_since_by_user = AsyncMock(return_value={2: 12.5})
        with patch("llm_proxy.api.routers.team.UsageRepository", return_value=usage_repo):
            response = client.put(
                "/api/team/members/2/budget",
                json={"budget_usd": 50.0, "budget_period": "monthly", "budget_reset_day": 15},
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["budget_usd"] == 50.0
        assert body["budget_period"] == "monthly"
        assert body["budget_reset_day"] == 15
        assert body["budget_spend_usd"] == 12.5
        assert body["budget_period_start"] is not None
        mock_repo.set_budget.assert_awaited_once_with(
            2, budget_usd=50.0, budget_period="monthly", budget_reset_day=15
        )

    def test_partial_update_keeps_stored_window(self, team_client):
        """Setting only the reset day keeps the stored cap and period."""
        client, _, mock_repo = team_client
        stored = _mock_user(2, "member")
        stored.budget_usd = 50.0
        stored.budget_period = "monthly"
        mock_repo.get_by_id = AsyncMock(return_value=stored)
        updated = _mock_user(2, "member")
        updated.budget_usd = 50.0
        updated.budget_period = "monthly"
        updated.budget_reset_day = 10
        mock_repo.set_budget = AsyncMock(return_value=updated)

        usage_repo = AsyncMock()
        usage_repo.get_spend_since_by_user = AsyncMock(return_value={})
        with patch("llm_proxy.api.routers.team.UsageRepository", return_value=usage_repo):
            response = client.put("/api/team/members/2/budget", json={"budget_reset_day": 10})

        assert response.status_code == 200, response.text
        mock_repo.set_budget.assert_awaited_once_with(
            2, budget_usd=50.0, budget_period="monthly", budget_reset_day=10
        )
        # No spend rows for the window: the response reports 0.0, not null.
        assert response.json()["budget_spend_usd"] == 0.0

    def test_clear_budget_clears_window(self, team_client):
        """Explicit null on budget_usd clears the cap and its window config."""
        client, _, mock_repo = team_client
        stored = _mock_user(2, "member")
        stored.budget_usd = 50.0
        stored.budget_period = "monthly"
        stored.budget_reset_day = 15
        mock_repo.get_by_id = AsyncMock(return_value=stored)
        mock_repo.set_budget = AsyncMock(return_value=_mock_user(2, "member"))

        response = client.put("/api/team/members/2/budget", json={"budget_usd": None})

        assert response.status_code == 200, response.text
        mock_repo.set_budget.assert_awaited_once_with(
            2, budget_usd=None, budget_period=None, budget_reset_day=None
        )
        body = response.json()
        assert body["budget_usd"] is None
        assert body["budget_spend_usd"] is None

    def test_period_without_cap_rejected(self, team_client):
        """A window with no cap in effect (stored or requested) is meaningless."""
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "member"))  # no budget

        response = client.put("/api/team/members/2/budget", json={"budget_period": "monthly"})

        assert response.status_code == 400
        mock_repo.set_budget.assert_not_awaited()

    def test_reset_day_requires_monthly_period(self, team_client):
        client, _, mock_repo = team_client
        stored = _mock_user(2, "member")
        stored.budget_usd = 50.0
        stored.budget_period = "weekly"
        mock_repo.get_by_id = AsyncMock(return_value=stored)

        response = client.put("/api/team/members/2/budget", json={"budget_reset_day": 10})

        assert response.status_code == 400
        mock_repo.set_budget.assert_not_awaited()

    def test_clear_cap_with_window_fields_rejected(self, team_client):
        """Clearing the cap while setting a window is contradictory (400)."""
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=_mock_user(2, "member"))

        response = client.put(
            "/api/team/members/2/budget",
            json={"budget_usd": None, "budget_period": "monthly"},
        )

        assert response.status_code == 400
        mock_repo.set_budget.assert_not_awaited()

    def test_budget_member_not_found(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=None)

        response = client.put("/api/team/members/99/budget", json={"budget_usd": 50.0})

        assert response.status_code == 404
        mock_repo.set_budget.assert_not_awaited()


class TestResetMemberBudget:
    """POST /members/{id}/budget/reset restarts the account budget window."""

    def test_reset_budget_window(self, team_client):
        client, _, mock_repo = team_client
        stored = _mock_user(2, "member")
        stored.budget_usd = 50.0
        stored.budget_period = "monthly"
        mock_repo.get_by_id = AsyncMock(return_value=stored)
        updated = _mock_user(2, "member")
        updated.budget_usd = 50.0
        updated.budget_period = "monthly"
        mock_repo.reset_budget = AsyncMock(return_value=updated)

        usage_repo = AsyncMock()
        usage_repo.get_spend_since_by_user = AsyncMock(return_value={})
        with patch("llm_proxy.api.routers.team.UsageRepository", return_value=usage_repo):
            response = client.post("/api/team/members/2/budget/reset")

        assert response.status_code == 200, response.text
        mock_repo.reset_budget.assert_awaited_once_with(2)

    def test_reset_budget_member_not_found(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.get_by_id = AsyncMock(return_value=None)

        response = client.post("/api/team/members/99/budget/reset")

        assert response.status_code == 404
        mock_repo.reset_budget.assert_not_awaited()


class TestMutationResponsesKeepBudgetSpend:
    """Single-member mutation responses keep the budget spend enrichment.

    The frontend patches the member row in place from these responses; a
    response with a null ``budget_spend_usd`` for a budgeted member would
    regress the budget column to "-" until the next full reload.
    """

    def test_role_change_response_keeps_budget_spend(self, team_client):
        client, _, mock_repo = team_client
        stored = _mock_user(2, "member")
        stored.budget_usd = 50.0
        stored.budget_period = "monthly"
        mock_repo.get_by_id = AsyncMock(return_value=stored)
        updated = _mock_user(2, "member")
        updated.role = "admin"
        updated.budget_usd = 50.0
        updated.budget_period = "monthly"
        mock_repo.set_role = AsyncMock(return_value=updated)

        usage_repo = AsyncMock()
        usage_repo.get_spend_since_by_user = AsyncMock(return_value={2: 7.25})
        with patch("llm_proxy.api.routers.team.UsageRepository", return_value=usage_repo):
            response = client.put("/api/team/members/2/role", json={"role": "admin"})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["role"] == "admin"
        assert body["budget_spend_usd"] == 7.25
        assert body["budget_period_start"] is not None

    def test_deactivate_response_keeps_budget_spend(self, team_client):
        client, _, mock_repo = team_client
        stored = _mock_user(2, "member")
        stored.budget_usd = 50.0
        stored.budget_period = "monthly"
        mock_repo.get_by_id = AsyncMock(return_value=stored)
        mock_repo.deactivate_user = AsyncMock(return_value=stored)

        usage_repo = AsyncMock()
        usage_repo.get_spend_since_by_user = AsyncMock(return_value={2: 4.5})
        with patch("llm_proxy.api.routers.team.UsageRepository", return_value=usage_repo):
            response = client.post("/api/team/members/2/deactivate")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["budget_spend_usd"] == 4.5
        assert body["budget_period_start"] is not None


class TestListMembersBudgetSpend:
    """The members list enriches budgeted members with current-window spend."""

    def test_budgeted_members_include_window_spend(self, team_client):
        client, _, mock_repo = team_client
        budgeted = _mock_user(2, "budgeted")
        budgeted.budget_usd = 100.0
        budgeted.budget_period = "monthly"
        plain = _mock_user(3, "plain")
        mock_repo.list_users = AsyncMock(return_value=[budgeted, plain])

        usage_repo = AsyncMock()
        usage_repo.get_spend_since_by_user = AsyncMock(return_value={2: 42.0})
        with patch("llm_proxy.api.routers.team.UsageRepository", return_value=usage_repo):
            response = client.get("/api/team/members")

        assert response.status_code == 200, response.text
        rows = {m["username"]: m for m in response.json()}
        assert rows["budgeted"]["budget_spend_usd"] == 42.0
        assert rows["budgeted"]["budget_period_start"] is not None
        assert rows["plain"]["budget_spend_usd"] is None
        assert rows["plain"]["budget_period_start"] is None
        # One grouped query covering exactly the budgeted member's window.
        usage_repo.get_spend_since_by_user.assert_awaited_once()
        windows = usage_repo.get_spend_since_by_user.await_args.args[0]
        assert list(windows) == [2]

    def test_no_budgeted_members_skips_spend_query(self, team_client):
        client, _, mock_repo = team_client
        mock_repo.list_users = AsyncMock(return_value=[_mock_user(2, "plain")])

        with patch("llm_proxy.api.routers.team.UsageRepository") as usage_cls:
            response = client.get("/api/team/members")

        assert response.status_code == 200, response.text
        usage_cls.assert_not_called()
