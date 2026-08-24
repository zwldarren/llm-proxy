"""Tests for config_providers.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.database.repositories.config_providers import ProviderRepository
from llm_proxy.database.tables import ProviderRecord


class TestProviderRepository:
    """Tests for ProviderRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        session = AsyncMock(spec=AsyncSession)
        return session

    @pytest.fixture
    def repo(self, mock_session):
        """Create a ProviderRepository instance."""
        return ProviderRepository(mock_session)

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider record."""
        provider = MagicMock(spec=ProviderRecord)
        provider.id = 1
        provider.name = "test-provider"
        provider.type = "openai"
        provider.api_key = "test-api-key"
        provider.provider_metadata = {}
        return provider

    def test_prepare_provider_data(self, repo):
        """Test _prepare_provider_data method."""
        data = repo._prepare_provider_data(
            name="test",
            type="openai",
            endpoint_base_urls={"v1": "https://api.example.com"},
        )
        assert "endpoint_base_urls" not in data
        assert data["provider_metadata"]["endpoint_base_urls"] == {"v1": "https://api.example.com"}

    def test_prepare_provider_data_empty_endpoints(self, repo):
        """Test _prepare_provider_data with empty endpoints."""
        data = repo._prepare_provider_data(name="test", type="openai")
        assert "provider_metadata" in data

    @pytest.mark.asyncio
    async def test_create_provider(self, repo, mock_session, mock_provider):
        """Test creating a provider."""
        with (
            patch(
                "llm_proxy.database.repositories.config_providers.encrypt_api_key"
            ) as mock_encrypt,
            patch.object(repo, "_prepare_provider_data", return_value={}),
        ):
            mock_encrypt.return_value = "encrypted-key"
            mock_session.refresh = AsyncMock(return_value=mock_provider)

            await repo.create_provider(
                name="test-provider",
                type="openai",
                api_key="test-api-key",
            )

            mock_session.add.assert_called_once()
            mock_session.flush.assert_called_once()
            mock_encrypt.assert_called_once_with("test-api-key")

    @pytest.mark.asyncio
    async def test_get_provider_found(self, repo, mock_session, mock_provider):
        """Test getting a provider that exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_provider
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "llm_proxy.database.repositories.config_providers.decrypt_api_key"
        ) as mock_decrypt:
            mock_decrypt.return_value = "decrypted-key"
            result = await repo.get_provider("test-provider")

            assert result is not None
            mock_decrypt.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_provider_not_found(self, repo, mock_session):
        """Test getting a provider that doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_provider("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_provider_no_decrypt(self, repo, mock_session, mock_provider):
        """Test getting a provider without decryption."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_provider
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_provider("test-provider", decrypt=False)

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_provider_with_models(self, repo, mock_session, mock_provider):
        """Test getting a provider with models."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_provider
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "llm_proxy.database.repositories.config_providers.decrypt_api_key"
        ) as mock_decrypt:
            mock_decrypt.return_value = "decrypted-key"
            result = await repo.get_provider_with_models("test-provider")

            assert result is not None

    @pytest.mark.asyncio
    async def test_get_all_providers(self, repo, mock_session, mock_provider):
        """Test getting all providers."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_provider]
        mock_session.execute = AsyncMock(return_value=mock_result)

        with (
            patch(
                "llm_proxy.database.repositories.config_providers.decrypt_api_keys"
            ) as mock_decrypt,
        ):
            mock_decrypt.return_value = ["decrypted-key"]
            result = await repo.get_all_providers()

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_all_providers_empty(self, repo, mock_session):
        """Test getting all providers when none exist."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_all_providers()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_providers_no_decrypt(self, repo, mock_session, mock_provider):
        """Test getting all providers without decryption."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_provider]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_all_providers(decrypt=False)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_providers_by_names(self, repo, mock_session, mock_provider):
        """Test getting providers by names."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_provider]
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "llm_proxy.database.repositories.config_providers.decrypt_api_keys"
        ) as mock_decrypt:
            mock_decrypt.return_value = ["decrypted-key"]
            result = await repo.get_providers_by_names(["test-provider"])

            assert "test-provider" in result

    @pytest.mark.asyncio
    async def test_get_providers_by_names_empty(self, repo, mock_session):
        """Test getting providers with empty names list."""
        result = await repo.get_providers_by_names([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_update_provider(self, repo, mock_session, mock_provider):
        """Test updating a provider."""
        mock_provider.provider_metadata = {}

        with (
            patch.object(repo, "get_provider", return_value=mock_provider),
            patch(
                "llm_proxy.database.repositories.config_providers.encrypt_api_key"
            ) as mock_encrypt,
            patch(
                "llm_proxy.database.repositories.config_providers.decrypt_api_key"
            ) as mock_decrypt,
        ):
            mock_encrypt.return_value = "encrypted-key"
            mock_decrypt.return_value = "decrypted-key"
            mock_session.refresh = AsyncMock()

            await repo.update_provider("test-provider", api_key="new-key")

            mock_encrypt.assert_called_once_with("new-key")
            mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_provider_with_metadata(self, repo, mock_session, mock_provider):
        """Test updating a provider with metadata."""
        mock_provider.provider_metadata = {}

        with (
            patch.object(repo, "get_provider", return_value=mock_provider),
            patch(
                "llm_proxy.database.repositories.config_providers.decrypt_api_key"
            ) as mock_decrypt,
        ):
            mock_decrypt.return_value = "decrypted-key"
            mock_session.refresh = AsyncMock()

            result = await repo.update_provider(
                "test-provider",
                parameter_overrides={"temperature": 0.7},
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_update_provider_not_found(self, repo, mock_session):
        """Test updating a provider that doesn't exist."""
        with patch.object(repo, "get_provider", return_value=None):
            result = await repo.update_provider("nonexistent", type="openai")

            assert result is None

    @pytest.mark.asyncio
    async def test_delete_provider(self, repo, mock_session, mock_provider):
        """Test deleting a provider."""
        with patch.object(repo, "get_provider", return_value=mock_provider):
            mock_session.delete = AsyncMock()

            result = await repo.delete_provider("test-provider")

            assert result is True
            mock_session.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_provider_not_found(self, repo, mock_session):
        """Test deleting a provider that doesn't exist."""
        with patch.object(repo, "get_provider", return_value=None):
            result = await repo.delete_provider("nonexistent")

            assert result is False
