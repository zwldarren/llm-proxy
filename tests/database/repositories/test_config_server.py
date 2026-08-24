"""Tests for config_server.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.database.repositories.config_server import ServerConfigRepository
from llm_proxy.database.tables import ServerConfigRecord


class TestServerConfigRepository:
    """Tests for ServerConfigRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        session = AsyncMock(spec=AsyncSession)
        return session

    @pytest.fixture
    def repo(self, mock_session):
        """Create a ServerConfigRepository instance."""
        return ServerConfigRepository(mock_session)

    @pytest.fixture
    def mock_config(self):
        """Create a mock server config record."""
        config = MagicMock(spec=ServerConfigRecord)
        config.key = "test-key"
        config.value = {"setting": "value"}
        config.description = "Test config"
        return config

    @pytest.mark.asyncio
    async def test_set_server_config_create(self, repo, mock_session):
        """Test setting a new server config."""
        mock_session.refresh = AsyncMock()
        mock_session.add = MagicMock()

        with patch.object(repo, "get_server_config", return_value=None):
            await repo.set_server_config(
                key="new-key",
                value={"setting": "value"},
                description="New config",
            )

            mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_server_config_update(self, repo, mock_session, mock_config):
        """Test updating an existing server config."""
        mock_session.refresh = AsyncMock()

        with patch.object(repo, "get_server_config", return_value=mock_config):
            await repo.set_server_config(
                key="test-key",
                value={"new": "value"},
                description="Updated",
            )

            assert mock_config.value == {"new": "value"}
            mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_get_server_config_found(self, repo, mock_session, mock_config):
        """Test getting a server config that exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_config
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_server_config("test-key")

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_server_config_not_found(self, repo, mock_session):
        """Test getting a server config that doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_server_config("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_server_config(self, repo, mock_session, mock_config):
        """Test getting all server configs."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_config]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_all_server_config()

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_all_server_config_empty(self, repo, mock_session):
        """Test getting all server configs when none exist."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_all_server_config()

        assert result == []

    @pytest.mark.asyncio
    async def test_delete_server_config(self, repo, mock_session, mock_config):
        """Test deleting a server config."""
        with patch.object(repo, "get_server_config", return_value=mock_config):
            mock_session.delete = AsyncMock()

            result = await repo.delete_server_config("test-key")

            assert result is True
            mock_session.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_server_config_not_found(self, repo, mock_session):
        """Test deleting a server config that doesn't exist."""
        with patch.object(repo, "get_server_config", return_value=None):
            result = await repo.delete_server_config("nonexistent")

            assert result is False

    @pytest.mark.asyncio
    async def test_get_tracing_config(self, repo, mock_config):
        """Test getting tracing config."""
        mock_config.value = {"enabled": True}

        with patch.object(repo, "get_server_config", return_value=mock_config):
            result = await repo.get_tracing_config()

            assert result == {"enabled": True}

    @pytest.mark.asyncio
    async def test_get_tracing_config_not_set(self, repo):
        """Test getting tracing config when not set."""
        with patch.object(repo, "get_server_config", return_value=None):
            result = await repo.get_tracing_config()

            assert result is None

    @pytest.mark.asyncio
    async def test_set_tracing_config(self, repo, mock_session):
        """Test setting tracing config."""
        mock_session.refresh = AsyncMock()

        with patch.object(repo, "set_server_config") as mock_set:
            mock_set.return_value = MagicMock(spec=ServerConfigRecord)
            await repo.set_tracing_config({"enabled": True})

            mock_set.assert_called_once_with(
                key="tracing_config",
                value={"enabled": True},
                description="Tracing configuration",
            )

    @pytest.mark.asyncio
    async def test_set_tracing_config_with_description(self, repo, mock_session):
        """Test setting tracing config with custom description."""
        mock_session.refresh = AsyncMock()

        with patch.object(repo, "set_server_config") as mock_set:
            mock_set.return_value = MagicMock(spec=ServerConfigRecord)
            await repo.set_tracing_config({"enabled": True}, description="Custom desc")

            mock_set.assert_called_once_with(
                key="tracing_config",
                value={"enabled": True},
                description="Custom desc",
            )

    @pytest.mark.asyncio
    async def test_get_web_search_config(self, repo, mock_config):
        """Test getting web search config."""
        mock_config.value = {"provider": "searxng"}

        with patch.object(repo, "get_server_config", return_value=mock_config):
            result = await repo.get_web_search_config()

            assert result == {"provider": "searxng"}

    @pytest.mark.asyncio
    async def test_get_web_search_config_not_set(self, repo):
        """Test getting web search config when not set."""
        with patch.object(repo, "get_server_config", return_value=None):
            result = await repo.get_web_search_config()

            assert result is None

    @pytest.mark.asyncio
    async def test_set_web_search_config(self, repo, mock_session):
        """Test setting web search config."""
        mock_session.refresh = AsyncMock()

        with patch.object(repo, "set_server_config") as mock_set:
            mock_set.return_value = MagicMock(spec=ServerConfigRecord)
            await repo.set_web_search_config({"provider": "searxng"})

            mock_set.assert_called_once()
