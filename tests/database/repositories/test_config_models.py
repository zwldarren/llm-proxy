"""Tests for config_models.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.database.repositories.config_models import ModelRepository
from llm_proxy.database.tables import ModelProviderRecord, ModelRecord


class TestModelRepository:
    """Tests for ModelRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        session = AsyncMock(spec=AsyncSession)
        return session

    @pytest.fixture
    def repo(self, mock_session):
        """Create a ModelRepository instance."""
        return ModelRepository(mock_session)

    @pytest.fixture
    def mock_model(self):
        """Create a mock model record."""
        model = MagicMock(spec=ModelRecord)
        model.id = 1
        model.name = "test-model"
        model.model_metadata = {}
        return model

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider record."""
        provider = MagicMock()
        provider.id = 1
        provider.name = "test-provider"
        return provider

    @pytest.fixture
    def mock_mapping(self):
        """Create a mock model-provider mapping."""
        mapping = MagicMock(spec=ModelProviderRecord)
        mapping.id = 1
        mapping.model_id = 1
        mapping.provider_id = 1
        mapping.input_cost_per_1m = None
        mapping.output_cost_per_1m = None
        return mapping

    def test_prepare_model_data(self, repo):
        """Test _prepare_model_data method."""
        data = repo._prepare_model_data(
            name="test",
            parameter_overrides={"temperature": 0.7},
        )
        assert "model_metadata" in data
        assert "parameter_overrides" not in data

    @pytest.mark.asyncio
    async def test_create_model(self, repo, mock_session, mock_model, mock_provider):
        """Test creating a model."""
        mock_session.refresh = AsyncMock(return_value=mock_model)

        with patch.object(
            repo._provider_repo,
            "get_providers_by_names",
            return_value={"test-provider": mock_provider},
        ):
            await repo.create_model(
                name="test-model",
                providers=[{"provider_name": "test-provider", "priority": 1}],
            )

            mock_session.add.assert_called()
            mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_create_model_no_providers(self, repo, mock_session):
        """Test creating a model with no providers."""
        result = await repo.create_model(name="test-model", providers=[])

        assert result is None

    @pytest.mark.asyncio
    async def test_create_model_invalid_provider(self, repo, mock_session, mock_model):
        """Test creating a model with invalid provider."""
        mock_session.refresh = AsyncMock(return_value=mock_model)
        mock_session.delete = AsyncMock()

        with patch.object(repo._provider_repo, "get_providers_by_names", return_value={}):
            result = await repo.create_model(
                name="test-model",
                providers=[{"provider_name": "nonexistent"}],
            )

            assert result is None
            mock_session.delete.assert_called()

    @pytest.mark.asyncio
    async def test_get_model_found(self, repo, mock_session, mock_model):
        """Test getting a model that exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_model
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_model("test-model")

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_model_not_found(self, repo, mock_session):
        """Test getting a model that doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_model("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_model_with_provider(self, repo, mock_session, mock_model):
        """Test getting a model with provider."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_model
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_model_with_provider("test-model")

        assert result is not None

    @pytest.mark.asyncio
    async def test_get_all_models(self, repo, mock_session, mock_model):
        """Test getting all models."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_model]
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_all_models()

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_all_models_empty(self, repo, mock_session):
        """Test getting all models when none exist."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_all_models()

        assert result == []

    @pytest.mark.asyncio
    async def test_update_model(self, repo, mock_session, mock_model):
        """Test updating a model."""
        mock_model.model_metadata = {}
        mock_session.refresh = AsyncMock()

        with patch.object(repo, "get_model_with_provider", return_value=mock_model):
            await repo.update_model("test-model", description="updated")

            mock_session.flush.assert_called()

    @pytest.mark.asyncio
    async def test_update_model_with_metadata(self, repo, mock_session, mock_model):
        """Test updating a model with metadata."""
        mock_model.model_metadata = {}
        mock_session.refresh = AsyncMock()

        with patch.object(repo, "get_model_with_provider", return_value=mock_model):
            result = await repo.update_model(
                "test-model",
                parameter_overrides={"temperature": 0.7},
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_update_model_with_providers(self, repo, mock_session, mock_model, mock_provider):
        """Test updating a model with new providers."""
        mock_model.model_metadata = {}
        mock_model.id = 1
        mock_session.refresh = AsyncMock()

        with (
            patch.object(repo, "get_model_with_provider", return_value=mock_model),
            patch.object(repo, "_delete_model_provider_mappings", return_value=None),
            patch.object(
                repo._provider_repo,
                "get_providers_by_names",
                return_value={"test-provider": mock_provider},
            ),
        ):
            await repo.update_model(
                "test-model",
                providers=[{"provider_name": "test-provider"}],
            )

            mock_session.add.assert_called()

    @pytest.mark.asyncio
    async def test_update_model_not_found(self, repo, mock_session):
        """Test updating a model that doesn't exist."""
        with patch.object(repo, "get_model_with_provider", return_value=None):
            result = await repo.update_model("nonexistent", description="updated")

            assert result is None

    @pytest.mark.asyncio
    async def test_delete_model_provider_mappings(self, repo, mock_session):
        """Test deleting model provider mappings."""
        mock_session.execute = AsyncMock()

        await repo._delete_model_provider_mappings(1)

        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_model_provider_pricing(self, repo, mock_session, mock_mapping):
        """Test updating model provider pricing."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_mapping
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.refresh = AsyncMock()

        result = await repo.update_model_provider_pricing(
            mapping_id=1,
            input_cost_per_1m=1.5,
            output_cost_per_1m=2.5,
        )

        assert result is not None
        assert mock_mapping.input_cost_per_1m == 1.5
        assert mock_mapping.output_cost_per_1m == 2.5

    @pytest.mark.asyncio
    async def test_update_model_provider_pricing_not_found(self, repo, mock_session):
        """Test updating pricing for non-existent mapping."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.update_model_provider_pricing(
            mapping_id=999,
            input_cost_per_1m=1.5,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_model(self, repo, mock_session, mock_model):
        """Test deleting a model."""
        with patch.object(repo, "get_model", return_value=mock_model):
            mock_session.delete = AsyncMock()

            result = await repo.delete_model("test-model")

            assert result is True
            mock_session.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_model_not_found(self, repo, mock_session):
        """Test deleting a model that doesn't exist."""
        with patch.object(repo, "get_model", return_value=None):
            result = await repo.delete_model("nonexistent")

            assert result is False


class TestApplyMappingPricing:
    """Tests for ModelRepository.apply_mapping_pricing."""

    @pytest.fixture
    def mock_session(self):
        session = AsyncMock(spec=AsyncSession)
        return session

    @pytest.fixture
    def repo(self, mock_session):
        return ModelRepository(mock_session)

    @pytest.fixture
    def mock_mapping(self):
        mapping = MagicMock(spec=ModelProviderRecord)
        mapping.id = 1
        return mapping

    @pytest.mark.asyncio
    async def test_applies_unit_based_pricing_fields(self, repo, mock_session, mock_mapping):
        """All unit-based pricing fields are written when provided."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_mapping
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.refresh = AsyncMock()

        result = await repo.apply_mapping_pricing(
            mapping_id=1,
            updates={
                "cost_per_image": 0.04,
                "audio_cost_per_minute": 0.006,
                "tts_cost_per_1m_chars": 16.0,
                "web_search_cost_per_1k": 0.03,
                "image_input_cost_per_1m": 30.0,
            },
        )

        assert result is mock_mapping
        assert mock_mapping.cost_per_image == 0.04
        assert mock_mapping.audio_cost_per_minute == 0.006
        assert mock_mapping.tts_cost_per_1m_chars == 16.0
        assert mock_mapping.web_search_cost_per_1k == 0.03
        assert mock_mapping.image_input_cost_per_1m == 30.0

    @pytest.mark.asyncio
    async def test_explicit_null_clears_field(self, repo, mock_session, mock_mapping):
        """A provided None is written (clears the field)."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_mapping
        mock_session.execute = AsyncMock(return_value=mock_result)

        await repo.apply_mapping_pricing(mapping_id=1, updates={"cost_per_image": None})

        assert mock_mapping.cost_per_image is None

    @pytest.mark.asyncio
    async def test_unknown_fields_ignored(self, repo, mock_session, mock_mapping):
        """Fields outside _APPLY_PRICING_FIELDS are silently ignored."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_mapping
        mock_session.execute = AsyncMock(return_value=mock_result)

        await repo.apply_mapping_pricing(
            mapping_id=1,
            updates={"cost_per_image": 0.04, "malicious_field": "evil"},
        )

        assert mock_mapping.cost_per_image == 0.04
        assert not hasattr(mock_mapping, "malicious_field") or "malicious_field" not in str(
            mock_mapping.__dict__
        )

    @pytest.mark.asyncio
    async def test_mapping_not_found(self, repo, mock_session):
        """Returns None when the mapping does not exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.apply_mapping_pricing(mapping_id=999, updates={"cost_per_image": 0.04})

        assert result is None
