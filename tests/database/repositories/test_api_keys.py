"""Tests for api_keys repository."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.core.exceptions import ConflictError, NotFoundError
from llm_proxy.database.repositories.api_keys import ApiKeyRepository
from llm_proxy.database.tables import ApiKeyRecord


class TestApiKeyRepository:
    """Tests for ApiKeyRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        session = AsyncMock(spec=AsyncSession)
        return session

    @pytest.fixture
    def repo(self, mock_session):
        """Create an ApiKeyRepository instance."""
        return ApiKeyRepository(mock_session)

    @pytest.fixture
    def mock_api_key(self):
        """Create a mock API key record."""
        key = MagicMock(spec=ApiKeyRecord)
        key.name = "test-key"
        key.key_hash = "hash123"
        key.allowed_models = ["gpt-4", "claude-3"]
        key.created_at = datetime.now(UTC)
        key.last_used_at = None
        key.is_active = True
        key.expires_at = None
        key.budget_usd = None
        key.budget_period = None
        key.budget_reset_day = None
        key.budget_reset_at = None
        return key

    @pytest.mark.asyncio
    async def test_create_api_key(self, repo, mock_session, mock_api_key):
        """Test creating an API key."""
        mock_session.refresh = AsyncMock(return_value=mock_api_key)

        _, result = await repo.create_api_key(
            name="test-key",
            key_hash="hash123",
            allowed_models=["gpt-4"],
            user_id=1,
        )

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        assert result.name == "test-key"
        assert result.key_hash == "hash123"
        assert result.allowed_models == ["gpt-4"]
        assert result.user_id == 1

    @pytest.mark.asyncio
    async def test_list_api_keys(self, repo, mock_session):
        """Test listing API keys."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.list_api_keys()

        mock_session.execute.assert_called_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_api_key_by_name_found(self, repo, mock_session, mock_api_key):
        """Test getting an API key by name when it exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_api_key
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_api_key_by_name("test-key")

        assert result == mock_api_key

    @pytest.mark.asyncio
    async def test_get_api_key_by_name_not_found(self, repo, mock_session):
        """Test getting an API key by name when it doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_api_key_by_name("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_api_key_models(self, repo, mock_session, mock_api_key):
        """Test updating allowed models for an API key."""
        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            result = await repo.update_api_key_models("test-key", ["gpt-4", "claude-3"])

        assert result is True
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_api_key_models_not_found(self, repo, mock_session):
        """Test updating models for a nonexistent key."""
        with patch.object(repo, "get_api_key_by_name", return_value=None):
            result = await repo.update_api_key_models("nonexistent", ["gpt-4"])

        assert result is False

    @pytest.mark.asyncio
    async def test_update_api_key_rename(self, repo, mock_session, mock_api_key):
        """Test renaming an API key."""
        with patch.object(repo, "get_api_key_by_name") as mock_get:
            mock_get.side_effect = [mock_api_key, None]  # found by current name, no conflict

            result = await repo.update_api_key(
                current_name="test-key",
                new_name="renamed-key",
                allowed_models=None,
            )

        assert result == mock_api_key
        assert mock_api_key.name == "renamed-key"
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_api_key_rename_conflict(self, repo, mock_session, mock_api_key):
        """Test renaming an API key to a name that already exists."""
        conflicting_key = MagicMock(spec=ApiKeyRecord)
        conflicting_key.name = "existing-key"

        with patch.object(repo, "get_api_key_by_name") as mock_get:
            mock_get.side_effect = [mock_api_key, conflicting_key]

            with pytest.raises(ConflictError):
                await repo.update_api_key(
                    current_name="test-key",
                    new_name="existing-key",
                    allowed_models=None,
                )

    @pytest.mark.asyncio
    async def test_update_api_key_not_found(self, repo, mock_session):
        """Test updating a nonexistent API key."""
        with (
            patch.object(repo, "get_api_key_by_name", return_value=None),
            pytest.raises(NotFoundError),
        ):
            await repo.update_api_key(
                current_name="nonexistent",
                new_name=None,
                allowed_models=["gpt-4"],
            )

    @pytest.mark.asyncio
    async def test_update_api_key_models_only(self, repo, mock_session, mock_api_key):
        """Test updating only allowed models without renaming."""
        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            result = await repo.update_api_key(
                current_name="test-key",
                new_name=None,
                allowed_models=["gpt-4", "claude-3", "gemini-pro"],
            )

        assert result == mock_api_key
        assert mock_api_key.allowed_models == ["gpt-4", "claude-3", "gemini-pro"]
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_api_key_same_name_updates_models(self, repo, mock_session, mock_api_key):
        """Test that setting new_name same as current_name only updates models."""
        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            result = await repo.update_api_key(
                current_name="test-key",
                new_name="test-key",
                allowed_models=["gpt-4"],
            )

        assert result == mock_api_key
        assert mock_api_key.allowed_models == ["gpt-4"]

    @pytest.mark.asyncio
    async def test_delete_api_key(self, repo, mock_session, mock_api_key):
        """Test deleting an API key."""
        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            result = await repo.delete_api_key("test-key")

        assert result is True
        mock_session.delete.assert_called_once_with(mock_api_key)
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_api_key_not_found(self, repo, mock_session):
        """Test deleting a nonexistent API key."""
        with patch.object(repo, "get_api_key_by_name", return_value=None):
            result = await repo.delete_api_key("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_update_last_used(self, repo, mock_session, mock_api_key):
        """Test updating last_used_at timestamp."""
        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            await repo.update_last_used("test-key")

        assert mock_api_key.last_used_at is not None
        mock_session.flush.assert_called_once()

    # --- MCP permission fields ---

    @pytest.mark.asyncio
    async def test_create_api_key_with_mcp_fields(self, repo, mock_session, mock_api_key):
        """Creating an API key persists the MCP permission fields onto the record."""
        mock_session.refresh = AsyncMock(return_value=mock_api_key)

        _, result = await repo.create_api_key(
            name="agent-key",
            key_hash="hash123",
            allowed_models=["gpt-4"],
            allowed_mcp_servers=["github_mcp"],
            user_id=1,
        )

        mock_session.add.assert_called_once()
        added = mock_session.add.call_args[0][0]
        assert isinstance(added, ApiKeyRecord)
        assert added.name == "agent-key"
        assert added.allowed_mcp_servers == ["github_mcp"]
        # The record passed to session.add is the same one returned to the caller.
        assert result is added

    @pytest.mark.asyncio
    async def test_update_api_key_with_mcp_fields(self, repo, mock_session, mock_api_key):
        """Updating an API key persists the MCP permission fields."""
        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            result = await repo.update_api_key(
                current_name="test-key",
                new_name=None,
                allowed_models=["gpt-4"],
                allowed_mcp_servers=["github_mcp"],
            )

        assert result == mock_api_key
        assert mock_api_key.allowed_mcp_servers == ["github_mcp"]
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_api_key_models_with_mcp_fields(self, repo, mock_session, mock_api_key):
        """update_api_key_models can also persist MCP permission fields."""
        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            result = await repo.update_api_key_models(
                "test-key",
                ["gpt-4"],
                allowed_mcp_servers=["github_mcp"],
            )

        assert result is True
        assert mock_api_key.allowed_mcp_servers == ["github_mcp"]
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_api_key_models_preserves_mcp_fields(
        self, repo, mock_session, mock_api_key
    ):
        """Updating only models must not wipe existing MCP permission fields."""
        mock_api_key.allowed_mcp_servers = ["github_mcp"]

        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            result = await repo.update_api_key_models("test-key", ["gpt-4"])

        assert result is True
        assert mock_api_key.allowed_mcp_servers == ["github_mcp"]

    @pytest.mark.asyncio
    async def test_create_api_key_with_expiry_and_budget(self, repo, mock_session):
        """Create persists expiry and budget fields."""
        expires = datetime(2030, 1, 1, tzinfo=UTC)

        _, result = await repo.create_api_key(
            name="budgeted-key",
            key_hash="hash456",
            user_id=1,
            expires_at=expires,
            budget_usd=25.0,
            budget_period="monthly",
        )

        mock_session.add.assert_called_once()
        added = mock_session.add.call_args[0][0]
        assert added.expires_at == expires
        assert added.budget_usd == 25.0
        assert added.budget_period == "monthly"
        assert result.name == "budgeted-key"

    @pytest.mark.asyncio
    async def test_update_api_key_new_fields(self, repo, mock_session, mock_api_key):
        """Update applies is_active, expires_at, and budget fields."""
        expires = datetime(2030, 6, 1, tzinfo=UTC)

        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            await repo.update_api_key(
                "test-key",
                is_active=False,
                expires_at=expires,
                budget_usd=10.0,
                budget_period="weekly",
            )

        assert mock_api_key.is_active is False
        assert mock_api_key.expires_at == expires
        assert mock_api_key.budget_usd == 10.0
        assert mock_api_key.budget_period == "weekly"

    @pytest.mark.asyncio
    async def test_update_api_key_clear_budget_clears_period(
        self, repo, mock_session, mock_api_key
    ):
        """Clearing budget_usd also clears the stored window configuration."""
        mock_api_key.budget_usd = 10.0
        mock_api_key.budget_period = "monthly"
        mock_api_key.budget_reset_day = 15

        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            await repo.update_api_key("test-key", budget_usd=None)

        assert mock_api_key.budget_usd is None
        assert mock_api_key.budget_period is None
        assert mock_api_key.budget_reset_day is None

    @pytest.mark.asyncio
    async def test_update_api_key_non_monthly_period_clears_reset_day(
        self, repo, mock_session, mock_api_key
    ):
        """Switching to a non-monthly period drops the stored reset day."""
        mock_api_key.budget_usd = 10.0
        mock_api_key.budget_period = "monthly"
        mock_api_key.budget_reset_day = 15

        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            await repo.update_api_key("test-key", budget_period="weekly")

        assert mock_api_key.budget_period == "weekly"
        assert mock_api_key.budget_reset_day is None

    @pytest.mark.asyncio
    async def test_update_api_key_unset_preserves_new_fields(
        self, repo, mock_session, mock_api_key
    ):
        """Omitted fields keep their stored values."""
        expires = datetime(2030, 6, 1, tzinfo=UTC)
        mock_api_key.is_active = False
        mock_api_key.expires_at = expires
        mock_api_key.budget_usd = 10.0
        mock_api_key.budget_period = "daily"

        # The rename path re-fetches by the new name to check for conflicts.
        async def get_by_name(name):
            return mock_api_key if name == "test-key" else None

        with patch.object(repo, "get_api_key_by_name", side_effect=get_by_name):
            await repo.update_api_key("test-key", new_name="renamed")

        assert mock_api_key.name == "renamed"
        assert mock_api_key.is_active is False
        assert mock_api_key.expires_at == expires
        assert mock_api_key.budget_usd == 10.0
        assert mock_api_key.budget_period == "daily"

    @pytest.mark.asyncio
    async def test_reset_budget(self, repo, mock_session, mock_api_key):
        """Reset stamps budget_reset_at with the current time."""
        assert mock_api_key.budget_reset_at is None

        with patch.object(repo, "get_api_key_by_name", return_value=mock_api_key):
            result = await repo.reset_budget("test-key")

        assert result is mock_api_key
        assert mock_api_key.budget_reset_at is not None
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_budget_not_found(self, repo, mock_session):
        """Reset on a missing key returns None."""
        with patch.object(repo, "get_api_key_by_name", return_value=None):
            result = await repo.reset_budget("ghost")

        assert result is None
        mock_session.flush.assert_not_called()
