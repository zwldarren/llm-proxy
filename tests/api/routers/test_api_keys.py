"""Integration tests for API key management endpoints."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from llm_proxy.api.dependencies import (
    get_async_session_dep,
    require_admin_role,
    require_authenticated,
)
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers import api_keys as api_keys_module
from llm_proxy.api.routers.api_keys import router
from llm_proxy.core.exceptions import ConflictError, NotFoundError
from llm_proxy.core.identity import RequestIdentity, set_request_identity
from llm_proxy.database.repositories.api_keys import _UNSET, ApiKeyRepository
from llm_proxy.database.tables import ApiKeyRecord


def make_api_key(
    name: str = "test-key",
    allowed_models: list[str] | None = None,
    is_active: bool = True,
    expires_at: datetime | None = None,
    budget_usd: float | None = None,
    budget_period: str | None = None,
    budget_reset_day: int | None = None,
    budget_reset_at: datetime | None = None,
    rate_limit_rpm: int | None = None,
) -> MagicMock:
    """Create a mock API key record."""
    key = MagicMock(spec=ApiKeyRecord)
    key.name = name
    key.key_hash = "hash123"
    key.allowed_models = allowed_models
    key.allowed_mcp_servers = None
    key.mcp_tool_permissions = None
    key.created_at = datetime.now(UTC)
    key.last_used_at = None
    key.is_active = is_active
    key.user_id = 1
    key.expires_at = expires_at
    key.budget_usd = budget_usd
    key.budget_period = budget_period
    key.budget_reset_day = budget_reset_day
    key.budget_reset_at = budget_reset_at
    key.rate_limit_rpm = rate_limit_rpm
    return key


class TestApiKeyRouter:
    """Integration tests for API key router endpoints."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock ApiKeyRepository."""
        repo = AsyncMock(spec=ApiKeyRepository)
        return repo

    @pytest.fixture(autouse=True)
    def _mock_helpers(self):
        """Mock the identity helpers to simulate an admin user."""
        mock_user = MagicMock()
        mock_user.role = "admin"
        with (
            patch.object(api_keys_module, "_get_current_user_id", new=AsyncMock(return_value=1)),
            patch.object(
                api_keys_module, "_get_current_user", new=AsyncMock(return_value=(mock_user, 1))
            ),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            yield

    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        app.dependency_overrides[require_authenticated] = lambda: None
        app.dependency_overrides[get_async_session_dep] = lambda: AsyncMock()

        @app.middleware("http")
        async def set_test_identity(request, call_next):
            set_request_identity(request, RequestIdentity(user="test-admin", user_id=1))
            response = await call_next(request)
            return response

        register_exception_handlers(app)
        app.include_router(router)
        return app

    @pytest.fixture
    async def client(self, app):
        """Async test client using ASGITransport.

        Uses httpx.AsyncClient with ASGITransport instead of Starlette's
        synchronous TestClient to ensure async middleware runs correctly
        in all environments (fixes CI 500 errors).
        """
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            yield c

    @pytest.mark.asyncio
    async def test_create_api_key(self, client, mock_repo):
        """Test creating an API key."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)
        mock_key = make_api_key("new-key")
        mock_repo.create_api_key = AsyncMock(return_value=("sk_abc", mock_key))

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post(
                "/api/api-keys",
                json={"name": "new-key", "allowed_models": None},
            )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "new-key"

    @pytest.mark.asyncio
    async def test_create_api_key_with_mcp_fields(self, client, mock_repo):
        """Test creating an API key with MCP permission fields."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)
        mock_key = make_api_key("agent")
        mock_key.allowed_mcp_servers = ["github_mcp"]
        mock_repo.create_api_key = AsyncMock(return_value=("sk_agent", mock_key))

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post(
                "/api/api-keys",
                json={
                    "name": "agent",
                    "allowed_mcp_servers": ["github_mcp"],
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["allowed_mcp_servers"] == ["github_mcp"]

    @pytest.mark.asyncio
    async def test_non_admin_cannot_set_mcp_servers(self, client, mock_repo):
        """Non-admin members cannot grant MCP server permissions on their own keys."""
        mock_user = MagicMock()
        mock_user.role = "member"
        with (
            patch.object(
                api_keys_module, "_get_current_user", new=AsyncMock(return_value=(mock_user, 2))
            ),
        ):
            mock_repo.get_api_key_by_name = AsyncMock(return_value=None)
            mock_key = make_api_key("member-key")
            mock_key.allowed_mcp_servers = None  # Non-admins cannot set MCP servers
            mock_repo.create_api_key = AsyncMock(return_value=("sk_member", mock_key))

            with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
                response = await client.post(
                    "/api/api-keys",
                    json={
                        "name": "member-key",
                        "allowed_mcp_servers": ["github_mcp"],
                    },
                )

        assert response.status_code == 201
        data = response.json()
        # Non-admin cannot set MCP servers - field is stripped to None
        assert data["allowed_mcp_servers"] is None
        # Ensure repository was called with None (not the requested value).
        mock_repo.create_api_key.assert_called_once()
        call_kwargs = mock_repo.create_api_key.call_args.kwargs
        assert call_kwargs["allowed_mcp_servers"] is None

    @pytest.mark.asyncio
    async def test_create_api_key_conflict(self, client, mock_repo):
        """Test creating an API key with duplicate name."""
        existing_key = make_api_key("existing-key")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=existing_key)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post(
                "/api/api-keys",
                json={"name": "existing-key", "allowed_models": None},
            )

        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_list_api_keys(self, client, mock_repo):
        """Listing keys is owner-scoped: even admins see only their own keys."""
        key1 = make_api_key("key-1")
        key2 = make_api_key("key-2")
        mock_repo.list_api_keys_by_user = AsyncMock(return_value=[key1, key2])

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.get("/api/api-keys")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        mock_repo.list_api_keys_by_user.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_list_api_keys_with_mcp_fields(self, client, mock_repo):
        """Test listing API keys includes MCP permission fields."""
        key1 = make_api_key("key-1")
        key1.allowed_mcp_servers = ["github_mcp"]
        key2 = make_api_key("key-2")
        key2.allowed_mcp_servers = None
        mock_repo.list_api_keys_by_user = AsyncMock(return_value=[key1, key2])

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.get("/api/api-keys")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["allowed_mcp_servers"] == ["github_mcp"]
        assert data[1]["allowed_mcp_servers"] is None

    @pytest.mark.asyncio
    async def test_update_api_key_models(self, client, mock_repo):
        """Test updating allowed models for an API key."""
        mock_key = make_api_key("test-key", allowed_models=["gpt-4"])
        mock_repo.get_api_key_by_name = AsyncMock(return_value=mock_key)
        mock_repo.update_api_key_models = AsyncMock(return_value=True)
        # spec=ApiKeyRepository blocks access to .session, so configure it explicitly
        mock_repo.session = AsyncMock()

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key/models",
                json={"allowed_models": ["gpt-4", "claude-3"]},
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_update_api_key_models_with_mcp_fields(self, client, mock_repo):
        """Test updating models for an API key returns MCP permission fields."""
        mock_key = make_api_key("test-key", allowed_models=["gpt-4"])
        mock_key.allowed_mcp_servers = ["github_mcp"]
        mock_repo.get_api_key_by_name = AsyncMock(return_value=mock_key)
        mock_repo.update_api_key_models = AsyncMock(return_value=True)
        mock_repo.session = AsyncMock()

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key/models",
                json={"allowed_models": ["gpt-4", "claude-3"]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["allowed_mcp_servers"] == ["github_mcp"]

    @pytest.mark.asyncio
    async def test_update_api_key_models_not_found(self, client, mock_repo):
        """Test updating models for a nonexistent key."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/nonexistent/models",
                json={"allowed_models": ["gpt-4"]},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_api_key_rename(self, client, mock_repo):
        """Test renaming an API key."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=make_api_key("old-name"))
        mock_key = make_api_key("renamed-key", allowed_models=["gpt-4"])
        mock_repo.update_api_key = AsyncMock(return_value=mock_key)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/old-name",
                json={"name": "renamed-key"},
            )

        assert response.status_code == 200
        mock_repo.update_api_key.assert_called_once_with(
            current_name="old-name",
            new_name="renamed-key",
            allowed_models=_UNSET,
            allowed_mcp_servers=_UNSET,
            is_active=_UNSET,
            expires_at=_UNSET,
            budget_usd=_UNSET,
            budget_period=_UNSET,
            budget_reset_day=_UNSET,
            rate_limit_rpm=_UNSET,
        )

    @pytest.mark.asyncio
    async def test_update_api_key_models_only(self, client, mock_repo):
        """Test updating only allowed models via the general update endpoint."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=make_api_key("test-key"))
        mock_key = make_api_key("test-key", allowed_models=["gpt-4"])
        mock_repo.update_api_key = AsyncMock(return_value=mock_key)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"allowed_models": ["gpt-4"]},
            )

        assert response.status_code == 200, f"Status: {response.status_code}, body: {response.text}"
        mock_repo.update_api_key.assert_called_once_with(
            current_name="test-key",
            new_name=_UNSET,
            allowed_models=["gpt-4"],
            allowed_mcp_servers=_UNSET,
            is_active=_UNSET,
            expires_at=_UNSET,
            budget_usd=_UNSET,
            budget_period=_UNSET,
            budget_reset_day=_UNSET,
            rate_limit_rpm=_UNSET,
        )

    @pytest.mark.asyncio
    async def test_update_api_key_with_mcp_fields(self, client, mock_repo):
        """Test updating an API key with MCP permission fields."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=make_api_key("test-key"))
        mock_key = make_api_key("test-key")
        mock_key.allowed_mcp_servers = ["github_mcp"]
        mock_repo.update_api_key = AsyncMock(return_value=mock_key)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={
                    "allowed_mcp_servers": ["github_mcp"],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["allowed_mcp_servers"] == ["github_mcp"]
        mock_repo.update_api_key.assert_called_once_with(
            current_name="test-key",
            new_name=_UNSET,
            allowed_models=_UNSET,
            allowed_mcp_servers=["github_mcp"],
            is_active=_UNSET,
            expires_at=_UNSET,
            budget_usd=_UNSET,
            budget_period=_UNSET,
            budget_reset_day=_UNSET,
            rate_limit_rpm=_UNSET,
        )

    @pytest.mark.asyncio
    async def test_update_api_key_both_name_and_models(self, client, mock_repo):
        """Test updating both name and allowed models."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=make_api_key("test-key"))
        mock_key = make_api_key("renamed-key", allowed_models=["gpt-4", "claude-3"])
        mock_repo.update_api_key = AsyncMock(return_value=mock_key)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"name": "renamed-key", "allowed_models": ["gpt-4", "claude-3"]},
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_update_api_key_not_found(self, client, mock_repo):
        """Test updating a nonexistent API key."""
        # The ownership check resolves the key first and 404s before update.
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)
        mock_repo.update_api_key = AsyncMock(
            side_effect=NotFoundError(message="API key 'nonexistent' not found")
        )

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/nonexistent",
                json={"name": "new-name"},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_api_key_name_conflict(self, client, mock_repo):
        """Test renaming to a name that already exists."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=make_api_key("test-key"))
        mock_repo.update_api_key = AsyncMock(
            side_effect=ConflictError(message="API key with name 'existing' already exists")
        )

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"name": "existing"},
            )

        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_delete_api_key(self, client, mock_repo):
        """Test deleting an API key."""
        mock_key = make_api_key("test-key")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=mock_key)
        mock_repo.delete_api_key = AsyncMock(return_value=True)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.delete("/api/api-keys/test-key")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_admin_cannot_manage_another_users_key(self, client, mock_repo):
        """Keys are strictly owner-scoped: even admins get 403 on another user's key.

        Admins control other accounts through the team endpoints (account
        budget, model allowlist, activation), not by touching individual keys.
        """
        other_key = make_api_key("test-key")
        other_key.user_id = 99  # owned by someone else (the identity is user 1)
        mock_repo.get_api_key_by_name = AsyncMock(return_value=other_key)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"budget_usd": 10.0},
            )

        assert response.status_code == 403
        mock_repo.update_api_key.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_api_key_not_found(self, client, mock_repo):
        """Test deleting a nonexistent API key."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.delete("/api/api-keys/nonexistent")

        assert response.status_code == 404


class TestApiKeyBudgetExpiryAndSpend:
    """Integration tests for expiry, budget, and per-key spend endpoints."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock ApiKeyRepository."""
        return AsyncMock(spec=ApiKeyRepository)

    @pytest.fixture(autouse=True)
    def _mock_helpers(self):
        """Mock the identity helpers to simulate an admin user."""
        mock_user = MagicMock()
        mock_user.role = "admin"
        with (
            patch.object(api_keys_module, "_get_current_user_id", new=AsyncMock(return_value=1)),
            patch.object(
                api_keys_module, "_get_current_user", new=AsyncMock(return_value=(mock_user, 1))
            ),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            yield

    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        app.dependency_overrides[require_authenticated] = lambda: None
        app.dependency_overrides[require_admin_role] = lambda: None
        app.dependency_overrides[get_async_session_dep] = lambda: AsyncMock()

        @app.middleware("http")
        async def set_test_identity(request, call_next):
            set_request_identity(request, RequestIdentity(user="test-admin", user_id=1))
            response = await call_next(request)
            return response

        register_exception_handlers(app)
        app.include_router(router)
        return app

    @pytest.fixture
    async def client(self, app):
        """Create test client."""
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            yield c

    # --- Create with expiry / budget -----------------------------------------

    @pytest.mark.asyncio
    async def test_create_with_expiry_and_budget(self, client, mock_repo):
        """Create forwards expires_at/budget fields to the repository."""
        expires = "2030-01-01T00:00:00Z"
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)
        mock_key = make_api_key(
            "budgeted-key",
            expires_at=datetime(2030, 1, 1, tzinfo=UTC),
            budget_usd=25.0,
            budget_period="monthly",
        )
        mock_repo.create_api_key = AsyncMock(return_value=("sk_x", mock_key))

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post(
                "/api/api-keys",
                json={
                    "name": "budgeted-key",
                    "expires_at": expires,
                    "budget_usd": 25.0,
                    "budget_period": "monthly",
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["budget_usd"] == 25.0
        assert data["budget_period"] == "monthly"
        # Exact serialized value: catches timezone/format regressions.
        assert data["expires_at"] == "2030-01-01T00:00:00Z"
        _, kwargs = mock_repo.create_api_key.call_args
        assert kwargs["budget_usd"] == 25.0
        assert kwargs["budget_period"] == "monthly"
        assert kwargs["expires_at"] is not None

    @pytest.mark.asyncio
    async def test_create_budget_without_period_creates_lifetime_budget(self, client, mock_repo):
        """budget_usd without budget_period is a lifetime cap (cumulative spend)."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)
        mock_key = make_api_key("lifetime-key", budget_usd=10.0, budget_period=None)
        mock_repo.create_api_key = AsyncMock(return_value=("sk_x", mock_key))

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post(
                "/api/api-keys",
                json={"name": "lifetime-key", "budget_usd": 10.0},
            )

        assert response.status_code == 201
        assert response.json()["budget_period"] is None
        _, kwargs = mock_repo.create_api_key.call_args
        assert kwargs["budget_usd"] == 10.0
        assert kwargs["budget_period"] is None

    @pytest.mark.asyncio
    async def test_create_period_without_budget_rejected(self, client, mock_repo):
        """A budget window without a cap is meaningless and fails validation."""
        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post(
                "/api/api-keys",
                json={"name": "bad-budget", "budget_period": "daily"},
            )

        # FastAPI turns request-body schema validation failures into 422.
        assert response.status_code == 422
        mock_repo.create_api_key.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_reset_day_requires_monthly_period(self, client, mock_repo):
        """budget_reset_day is only valid with a monthly budget window."""
        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post(
                "/api/api-keys",
                json={
                    "name": "bad-budget",
                    "budget_usd": 10.0,
                    "budget_period": "daily",
                    "budget_reset_day": 15,
                },
            )

        assert response.status_code == 422
        mock_repo.create_api_key.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_monthly_budget_with_reset_day(self, client, mock_repo):
        """A monthly budget forwards its reset day to the repository."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)
        mock_key = make_api_key(
            "monthly-key", budget_usd=25.0, budget_period="monthly", budget_reset_day=15
        )
        mock_repo.create_api_key = AsyncMock(return_value=("sk_x", mock_key))

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post(
                "/api/api-keys",
                json={
                    "name": "monthly-key",
                    "budget_usd": 25.0,
                    "budget_period": "monthly",
                    "budget_reset_day": 15,
                },
            )

        assert response.status_code == 201
        assert response.json()["budget_reset_day"] == 15
        _, kwargs = mock_repo.create_api_key.call_args
        assert kwargs["budget_reset_day"] == 15

    @pytest.mark.asyncio
    async def test_create_negative_budget_rejected(self, client, mock_repo):
        """A non-positive budget fails schema validation."""
        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post(
                "/api/api-keys",
                json={"name": "bad-budget", "budget_usd": -5, "budget_period": "daily"},
            )

        assert response.status_code == 422
        mock_repo.create_api_key.assert_not_called()

    # --- Update is_active / expiry / budget ----------------------------------

    @pytest.mark.asyncio
    async def test_update_is_active(self, client, mock_repo):
        """Disabling a key forwards is_active=False to the repository."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=make_api_key("test-key"))
        mock_key = make_api_key("test-key", is_active=False)
        mock_repo.update_api_key = AsyncMock(return_value=mock_key)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"is_active": False},
            )

        assert response.status_code == 200
        assert response.json()["is_active"] is False
        _, kwargs = mock_repo.update_api_key.call_args
        assert kwargs["is_active"] is False

    @pytest.mark.asyncio
    async def test_update_expiry_set_and_clear(self, client, mock_repo):
        """expires_at can be set and explicitly cleared with null."""
        mock_key = make_api_key("test-key")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=mock_key)
        mock_repo.update_api_key = AsyncMock(return_value=mock_key)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"expires_at": "2030-06-01T12:00:00Z"},
            )
        assert response.status_code == 200
        _, kwargs = mock_repo.update_api_key.call_args
        assert kwargs["expires_at"] is not None

        mock_repo.update_api_key.reset_mock()
        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"expires_at": None},
            )
        assert response.status_code == 200
        _, kwargs = mock_repo.update_api_key.call_args
        assert kwargs["expires_at"] is None

    @pytest.mark.asyncio
    async def test_update_budget_without_period_creates_lifetime_budget(self, client, mock_repo):
        """Setting a cap with no period in the request and none stored is a lifetime cap."""
        mock_key = make_api_key("test-key")  # budget_period is None
        mock_repo.get_api_key_by_name = AsyncMock(return_value=mock_key)
        updated = make_api_key("test-key", budget_usd=10.0, budget_period=None)
        mock_repo.update_api_key = AsyncMock(return_value=updated)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"budget_usd": 10.0},
            )

        assert response.status_code == 200
        _, kwargs = mock_repo.update_api_key.call_args
        assert kwargs["budget_usd"] == 10.0
        assert kwargs["budget_period"] is _UNSET

    @pytest.mark.asyncio
    async def test_update_reset_day_without_monthly_period_rejected(self, client, mock_repo):
        """A reset day against a non-monthly stored period is rejected."""
        mock_key = make_api_key("test-key", budget_usd=5.0, budget_period="weekly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=mock_key)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"budget_reset_day": 15},
            )

        assert response.status_code == 400
        mock_repo.update_api_key.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_reset_day_with_monthly_period_accepted(self, client, mock_repo):
        """A reset day is valid when the effective period is monthly."""
        mock_key = make_api_key("test-key", budget_usd=5.0, budget_period="monthly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=mock_key)
        updated = make_api_key(
            "test-key", budget_usd=5.0, budget_period="monthly", budget_reset_day=15
        )
        mock_repo.update_api_key = AsyncMock(return_value=updated)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"budget_reset_day": 15},
            )

        assert response.status_code == 200
        _, kwargs = mock_repo.update_api_key.call_args
        assert kwargs["budget_reset_day"] == 15

    @pytest.mark.asyncio
    async def test_update_budget_uses_stored_period(self, client, mock_repo):
        """Setting a cap alone is valid when the stored key already has a period."""
        mock_key = make_api_key("test-key", budget_usd=5.0, budget_period="weekly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=mock_key)
        updated = make_api_key("test-key", budget_usd=10.0, budget_period="weekly")
        mock_repo.update_api_key = AsyncMock(return_value=updated)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"budget_usd": 10.0},
            )

        assert response.status_code == 200
        _, kwargs = mock_repo.update_api_key.call_args
        assert kwargs["budget_usd"] == 10.0
        assert kwargs["budget_period"] is _UNSET

    @pytest.mark.asyncio
    async def test_update_budget_clear(self, client, mock_repo):
        """Explicitly nulling budget_usd clears the budget."""
        mock_key = make_api_key("test-key", budget_usd=5.0, budget_period="daily")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=mock_key)
        mock_repo.update_api_key = AsyncMock(return_value=make_api_key("test-key"))

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put(
                "/api/api-keys/test-key",
                json={"budget_usd": None},
            )

        assert response.status_code == 200
        _, kwargs = mock_repo.update_api_key.call_args
        assert kwargs["budget_usd"] is None

    # --- Budget reset ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_reset_budget(self, client, mock_repo):
        """Reset endpoint stamps budget_reset_at via the repository."""
        reset_key = make_api_key(
            "test-key",
            budget_usd=10.0,
            budget_period="daily",
            budget_reset_at=datetime(2026, 7, 15, 12, tzinfo=UTC),
        )
        mock_repo.get_api_key_by_name = AsyncMock(return_value=make_api_key("test-key"))
        mock_repo.reset_budget = AsyncMock(return_value=reset_key)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post("/api/api-keys/test-key/budget/reset")

        assert response.status_code == 200
        data = response.json()
        assert data["budget_reset_at"] is not None
        mock_repo.reset_budget.assert_called_once_with("test-key")

    @pytest.mark.asyncio
    async def test_reset_budget_not_found(self, client, mock_repo):
        """Reset on a missing key returns 404 from the ownership check."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)
        mock_repo.reset_budget = AsyncMock(return_value=None)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post("/api/api-keys/ghost/budget/reset")

        assert response.status_code == 404

    # --- Spend summary --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_spend_summary(self, client, mock_repo):
        """Spend summary merges all-time totals with current-window spend."""
        keys = [
            make_api_key("plain-key"),
            make_api_key("budgeted-key", budget_usd=50.0, budget_period="monthly"),
        ]
        mock_repo.list_api_keys_by_user = AsyncMock(return_value=keys)

        usage_repo = AsyncMock()
        usage_repo.get_spend_by_api_key = AsyncMock(
            return_value=[
                {"api_key_name": "plain-key", "requests": 10, "cost": 1.5},
                {"api_key_name": "budgeted-key", "requests": 5, "cost": 30.0},
            ]
        )
        usage_repo.get_spend_since_by_api_key = AsyncMock(return_value={"budgeted-key": 12.5})

        with (
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "UsageRepository", return_value=usage_repo),
        ):
            response = await client.get("/api/api-keys/spend/summary")

        assert response.status_code == 200
        data = {row["name"]: row for row in response.json()}
        assert data["plain-key"]["total_spend_usd"] == 1.5
        assert data["plain-key"]["period_spend_usd"] is None
        assert data["budgeted-key"]["total_spend_usd"] == 30.0
        assert data["budgeted-key"]["period_spend_usd"] == 12.5
        assert data["budgeted-key"]["period_start"] is not None
        assert data["budgeted-key"]["budget_period"] == "monthly"
        # Window spend is fetched for all budgeted keys in a single bulk query.
        usage_repo.get_spend_since_by_api_key.assert_called_once()

    @pytest.mark.asyncio
    async def test_spend_summary_non_admin_scoped_to_own_keys(self, client, mock_repo):
        """Non-admin users only get spend rows for their own keys."""
        mock_user = MagicMock()
        mock_user.role = "member"
        mock_repo.list_api_keys_by_user = AsyncMock(return_value=[make_api_key("mine")])
        usage_repo = AsyncMock()
        usage_repo.get_spend_by_api_key = AsyncMock(return_value=[])

        with (
            patch.object(
                api_keys_module, "_get_current_user", new=AsyncMock(return_value=(mock_user, 2))
            ),
            patch.object(api_keys_module, "_get_current_user_id", new=AsyncMock(return_value=2)),
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "UsageRepository", return_value=usage_repo),
        ):
            response = await client.get("/api/api-keys/spend/summary")

        assert response.status_code == 200
        assert [row["name"] for row in response.json()] == ["mine"]
        mock_repo.list_api_keys_by_user.assert_called_once_with(2)
        usage_repo.get_spend_by_api_key.assert_called_once_with(user_id=2)

    # --- Per-key usage --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_key_usage(self, client, mock_repo):
        """Per-key usage returns summary, by-model, and daily breakdowns."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=make_api_key("test-key"))

        usage_repo = AsyncMock()
        usage_repo.get_usage_stats = AsyncMock(
            return_value={
                "total_cost": 2.5,
                "total_requests": 7,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "avg_response_time_ms": 120.0,
                "success_rate": 100.0,
                "total_cache_creation_tokens": 0,
                "total_cache_read_tokens": 0,
                "total_cached_prompt_tokens": 0,
                "cache_savings_usd": 0.0,
                "avg_tokens_per_second": 0.0,
                "avg_ttft_ms": 0.0,
            }
        )
        usage_repo.get_usage_by_model = AsyncMock(
            return_value=[
                {
                    "model": "gpt-4",
                    "provider": "openai",
                    "requests": 7,
                    "cost": 2.5,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 0,
                    "cached_prompt_tokens": 0,
                }
            ]
        )
        usage_repo.get_daily_usage = AsyncMock(
            return_value=[
                {
                    "date": "2026-07-15",
                    "requests": 7,
                    "cost": 2.5,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 0,
                    "cached_prompt_tokens": 0,
                    "by_model": [],
                }
            ]
        )

        with (
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "UsageRepository", return_value=usage_repo),
        ):
            response = await client.get("/api/api-keys/test-key/usage")

        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["total_cost"] == 2.5
        assert data["by_model"][0]["model"] == "gpt-4"
        assert data["daily_usage"][0]["date"] == "2026-07-15"
        # All three queries are scoped to the key.
        assert usage_repo.get_usage_stats.call_args.kwargs["api_key_name"] == "test-key"
        assert usage_repo.get_usage_by_model.call_args.kwargs["api_key_name"] == "test-key"
        assert usage_repo.get_daily_usage.call_args.kwargs["api_key_name"] == "test-key"

    @pytest.mark.asyncio
    async def test_key_usage_not_found(self, client, mock_repo):
        """Usage for a missing key returns 404."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.get("/api/api-keys/ghost/usage")

        assert response.status_code == 404


def _viewer_patches():
    """Patch the identity helpers so requests resolve to a non-admin viewer."""
    viewer = MagicMock()
    viewer.role = "viewer"
    viewer.allowed_models = None
    return (
        patch.object(api_keys_module, "_get_current_user", new=AsyncMock(return_value=(viewer, 1))),
        patch.object(api_keys_module, "_get_current_user_id", new=AsyncMock(return_value=1)),
    )


class TestViewerQuotaPermissions:
    """Non-admin quota rules: key budgets are self-service; rate limits are admin-only."""

    @pytest.fixture
    def mock_repo(self):
        return AsyncMock(spec=ApiKeyRepository)

    @pytest.fixture
    def app(self):
        app = FastAPI()
        app.dependency_overrides[require_authenticated] = lambda: None
        app.dependency_overrides[get_async_session_dep] = lambda: AsyncMock()

        @app.middleware("http")
        async def set_test_identity(request, call_next):
            set_request_identity(request, RequestIdentity(user="viewer", user_id=1))
            return await call_next(request)

        register_exception_handlers(app)
        app.include_router(router)
        return app

    @pytest.fixture
    async def client(self, app):
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            yield c

    # --- Budget: fully self-service ------------------------------------------

    @pytest.mark.asyncio
    async def test_viewer_create_with_budget_allowed(self, client, mock_repo):
        """A member may set a cap on a brand-new key."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)
        mock_repo.create_api_key = AsyncMock(
            return_value=("", make_api_key("k", budget_usd=10.0, budget_period="monthly"))
        )

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.post(
                "/api/api-keys",
                json={"name": "k", "budget_usd": 10.0, "budget_period": "monthly"},
            )

        assert response.status_code == 201, response.text
        call = mock_repo.create_api_key.await_args
        assert call.kwargs["budget_usd"] == 10.0
        assert call.kwargs["budget_period"] == "monthly"

    @pytest.mark.asyncio
    async def test_viewer_lower_budget_allowed(self, client, mock_repo):
        """Key budgets are self-service: lowering the cap is allowed."""
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(
            return_value=make_api_key("k", budget_usd=50.0, budget_period="monthly")
        )

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_usd": 50.0})

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["budget_usd"] == 50.0

    @pytest.mark.asyncio
    async def test_viewer_lower_budget_echoing_period_allowed(self, client, mock_repo):
        """Lowering the cap while echoing the window back is allowed."""
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(
            return_value=make_api_key("k", budget_usd=50.0, budget_period="monthly")
        )

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put(
                "/api/api-keys/k", json={"budget_usd": 50.0, "budget_period": "monthly"}
            )

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["budget_usd"] == 50.0
        assert call.kwargs["budget_period"] == "monthly"

    @pytest.mark.asyncio
    async def test_viewer_echo_identical_period_allowed(self, client, mock_repo):
        """Echoing the stored window back unchanged is allowed."""
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(return_value=stored)

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_period": "monthly"})

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["budget_period"] == "monthly"

    @pytest.mark.asyncio
    async def test_viewer_echo_identical_reset_day_allowed(self, client, mock_repo):
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly", budget_reset_day=15)
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(return_value=stored)

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_reset_day": 15})

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["budget_reset_day"] == 15

    @pytest.mark.asyncio
    async def test_viewer_echo_reset_day_null_equals_first_allowed(self, client, mock_repo):
        """NULL and 1 denote the same day (the 1st): echoing either spelling is a no-op."""
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly", budget_reset_day=None)
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(return_value=stored)

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_reset_day": 1})

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["budget_reset_day"] == 1

    @pytest.mark.asyncio
    async def test_viewer_echo_reset_day_first_equals_null_allowed(self, client, mock_repo):
        """The reverse spelling (stored 1, echoed as null) is also a no-op."""
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly", budget_reset_day=1)
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(return_value=stored)

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_reset_day": None})

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["budget_reset_day"] is None

    @pytest.mark.asyncio
    async def test_viewer_raise_budget_allowed(self, client, mock_repo):
        """Key budgets are self-service: the owner may raise their own cap.

        Spend is ultimately bounded by the admin-set account-level budget, so
        relaxing a key-level cap cannot exceed the account envelope.
        """
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(
            return_value=make_api_key("k", budget_usd=200.0, budget_period="monthly")
        )

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_usd": 200.0})

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["budget_usd"] == 200.0

    @pytest.mark.asyncio
    async def test_viewer_clear_budget_allowed(self, client, mock_repo):
        """The owner may clear their own key's budget entirely."""
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(return_value=make_api_key("k"))

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_usd": None})

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["budget_usd"] is None

    @pytest.mark.asyncio
    async def test_viewer_change_period_allowed(self, client, mock_repo):
        """The owner may re-window their own key's budget."""
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(
            return_value=make_api_key("k", budget_usd=100.0, budget_period="daily")
        )

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_period": "daily"})

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["budget_period"] == "daily"

    @pytest.mark.asyncio
    async def test_viewer_change_reset_day_allowed(self, client, mock_repo):
        """The owner may change the monthly anchor day of their own budget."""
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(
            return_value=make_api_key(
                "k", budget_usd=100.0, budget_period="monthly", budget_reset_day=15
            )
        )

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_reset_day": 15})

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["budget_reset_day"] == 15

    @pytest.mark.asyncio
    async def test_viewer_period_without_budget_rejected_as_meaningless(self, client, mock_repo):
        """A window without a cap is meaningless — rejected by schema rules.

        A period with no budget in the same request fails the cap-required
        check against the effective (stored) budget.
        """
        stored = make_api_key("k")  # no budget
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_period": "monthly"})

        assert response.status_code == 400
        mock_repo.update_api_key.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_viewer_reset_day_without_budget_rejected_as_meaningless(self, client, mock_repo):
        stored = make_api_key("k")  # no budget
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
        ):
            response = await client.put(
                "/api/api-keys/k", json={"budget_period": "monthly", "budget_reset_day": 15}
            )

        assert response.status_code == 400
        mock_repo.update_api_key.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_viewer_set_budget_on_unbudgeted_key_allowed(self, client, mock_repo):
        """A cap on an unbudgeted key is allowed (a lifetime cap when no window)."""
        stored = make_api_key("k")  # no budget
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(return_value=make_api_key("k", budget_usd=25.0))

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_usd": 25.0})

        assert response.status_code == 200, response.text

    @pytest.mark.asyncio
    async def test_viewer_set_budget_with_period_on_unbudgeted_key_allowed(self, client, mock_repo):
        """A member may establish a new budget together with its window."""
        stored = make_api_key("k")  # no budget
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        updated = make_api_key("k", budget_usd=25.0, budget_period="daily")
        mock_repo.update_api_key = AsyncMock(return_value=updated)

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.put(
                "/api/api-keys/k", json={"budget_usd": 25.0, "budget_period": "daily"}
            )

        assert response.status_code == 200, response.text
        _, kwargs = mock_repo.update_api_key.call_args
        assert kwargs["budget_usd"] == 25.0
        assert kwargs["budget_period"] == "daily"

    @pytest.mark.asyncio
    async def test_viewer_budget_reset_allowed_for_own_key(self, client, mock_repo):
        """Budget reset restarts accumulation — self-service for the key owner."""
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.reset_budget = AsyncMock(return_value=stored)

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            response = await client.post("/api/api-keys/k/budget/reset")

        assert response.status_code == 200, response.text
        mock_repo.reset_budget.assert_awaited_once_with("k")

    @pytest.mark.asyncio
    async def test_viewer_cannot_manage_others_keys(self, client, mock_repo):
        """Keys are strictly owner-scoped: a viewer gets 403 on someone else's key."""
        other_key = make_api_key("k")
        other_key.user_id = 99  # owned by someone else
        mock_repo.get_api_key_by_name = AsyncMock(return_value=other_key)

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
        ):
            response = await client.put("/api/api-keys/k", json={"budget_usd": 50.0})

        assert response.status_code == 403
        mock_repo.update_api_key.assert_not_awaited()

    # --- Rate limit: admin-only ----------------------------------------------

    @pytest.mark.asyncio
    async def test_viewer_create_with_rate_limit_forbidden(self, client, mock_repo):
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
        ):
            response = await client.post("/api/api-keys", json={"name": "k", "rate_limit_rpm": 60})

        assert response.status_code == 403
        mock_repo.create_api_key.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_viewer_update_rate_limit_forbidden(self, client, mock_repo):
        stored = make_api_key("k")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)

        p1, p2 = _viewer_patches()
        with (
            p1,
            p2,
            patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo),
        ):
            response = await client.put("/api/api-keys/k", json={"rate_limit_rpm": 120})

        assert response.status_code == 403
        mock_repo.update_api_key.assert_not_awaited()


class TestAdminRateLimitField:
    """Admins can set, change, and clear the per-key rate limit."""

    @pytest.fixture
    def mock_repo(self):
        return AsyncMock(spec=ApiKeyRepository)

    @pytest.fixture(autouse=True)
    def _mock_helpers(self):
        mock_user = MagicMock()
        mock_user.role = "admin"
        with (
            patch.object(api_keys_module, "_get_current_user_id", new=AsyncMock(return_value=1)),
            patch.object(
                api_keys_module, "_get_current_user", new=AsyncMock(return_value=(mock_user, 1))
            ),
            patch.object(api_keys_module, "invalidate_api_key_cache"),
        ):
            yield

    @pytest.fixture
    def app(self):
        app = FastAPI()
        app.dependency_overrides[require_authenticated] = lambda: None
        app.dependency_overrides[require_admin_role] = lambda: None
        app.dependency_overrides[get_async_session_dep] = lambda: AsyncMock()

        @app.middleware("http")
        async def set_test_identity(request, call_next):
            set_request_identity(request, RequestIdentity(user="test-admin", user_id=1))
            return await call_next(request)

        register_exception_handlers(app)
        app.include_router(router)
        return app

    @pytest.fixture
    async def client(self, app):
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            yield c

    @pytest.mark.asyncio
    async def test_admin_create_with_rate_limit(self, client, mock_repo):
        mock_repo.get_api_key_by_name = AsyncMock(return_value=None)
        mock_repo.create_api_key = AsyncMock(
            return_value=("", make_api_key("k", rate_limit_rpm=60))
        )

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post("/api/api-keys", json={"name": "k", "rate_limit_rpm": 60})

        assert response.status_code == 201, response.text
        call = mock_repo.create_api_key.await_args
        assert call.kwargs["rate_limit_rpm"] == 60
        assert response.json()["rate_limit_rpm"] == 60

    @pytest.mark.asyncio
    async def test_admin_update_rate_limit(self, client, mock_repo):
        stored = make_api_key("k")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(return_value=make_api_key("k", rate_limit_rpm=120))

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put("/api/api-keys/k", json={"rate_limit_rpm": 120})

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["rate_limit_rpm"] == 120

    @pytest.mark.asyncio
    async def test_admin_clear_rate_limit(self, client, mock_repo):
        stored = make_api_key("k", rate_limit_rpm=60)
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(return_value=make_api_key("k"))

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put("/api/api-keys/k", json={"rate_limit_rpm": None})

        assert response.status_code == 200, response.text
        call = mock_repo.update_api_key.await_args
        assert call.kwargs["rate_limit_rpm"] is None

    @pytest.mark.asyncio
    async def test_admin_reset_budget(self, client, mock_repo):
        """Budget reset is available to the key's owner (admin or not)."""
        mock_repo.get_api_key_by_name = AsyncMock(return_value=make_api_key("k"))
        reset_key = make_api_key(
            "k",
            budget_usd=10.0,
            budget_period="daily",
            budget_reset_at=datetime(2026, 8, 1, tzinfo=UTC),
        )
        mock_repo.reset_budget = AsyncMock(return_value=reset_key)

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.post("/api/api-keys/k/budget/reset")

        assert response.status_code == 200
        mock_repo.reset_budget.assert_awaited_once_with("k")

    @pytest.mark.asyncio
    async def test_admin_raise_budget_allowed(self, client, mock_repo):
        stored = make_api_key("k", budget_usd=100.0, budget_period="monthly")
        mock_repo.get_api_key_by_name = AsyncMock(return_value=stored)
        mock_repo.update_api_key = AsyncMock(
            return_value=make_api_key("k", budget_usd=200.0, budget_period="monthly")
        )

        with patch.object(api_keys_module, "get_api_key_repository", return_value=mock_repo):
            response = await client.put("/api/api-keys/k", json={"budget_usd": 200.0})

        assert response.status_code == 200, response.text
