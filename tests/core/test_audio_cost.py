"""Regression tests for audio endpoint cost calculation.

These tests previously exercised log_response with audio response types.
Since log_response has been removed (duplicate console logging), the tests
now verify that cost calculation works correctly via calculate_cost directly.
"""

import pytest

from llm_proxy.billing.cost import calculate_cost
from llm_proxy.config.types.model import ModelConfig, ModelProviderConfig


class FakeConfigManager:
    """Returns a model config with input pricing for testing."""

    async def get_model_config(self, model_name):
        return ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="whisper-1",
                )
            ],
            input_cost_per_1m=10.0,
        )


@pytest.mark.asyncio
async def test_cost_calculation_with_no_usage():
    """No usage dict means cost should be None."""
    usage_dict = None
    result = await calculate_cost(
        usage_dict, "whisper-1", FakeConfigManager(), None, None, "openai"
    )
    cost_usd = result.cost_usd
    assert cost_usd is None


@pytest.mark.asyncio
async def test_cost_calculation_for_transcription_response():
    """Transcription usage dict should calculate cost."""
    usage_dict = {"prompt_tokens": 100, "output_tokens": 0, "total_tokens": 100}
    result = await calculate_cost(
        usage_dict, "whisper-1", FakeConfigManager(), None, None, "openai"
    )
    assert result.prompt_tokens == 100
    assert result.cost_usd == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_cost_calculation_for_translation_response():
    """Translation usage dict should calculate cost."""
    usage_dict = {"prompt_tokens": 120, "output_tokens": 0, "total_tokens": 120}
    result = await calculate_cost(
        usage_dict, "whisper-1", FakeConfigManager(), None, None, "openai"
    )
    assert result.prompt_tokens == 120
    assert result.cost_usd == pytest.approx(0.0012)
