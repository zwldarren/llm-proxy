from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.database.repositories.config_models import ModelRepository


class TestModelRoutingTags:
    @pytest.fixture
    def repo(self):
        return ModelRepository(MagicMock(spec=AsyncSession))

    def test_prepare_model_data_keeps_routing_fields(self, repo):
        data = repo._prepare_model_data(
            auto_eligible=True,
            quality_tier="PREMIUM",
        )
        assert data["auto_eligible"] is True
        assert data["quality_tier"] == "PREMIUM"
