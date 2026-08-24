"""Tests for cost calculation utilities."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.billing.cost import (
    PricingRates,
    _calculate_audio_cost,
    _calculate_cache_cost,
    _calculate_image_token_cost,
    _calculate_unit_costs,
    _get_provider_pricing,
    calculate_cost,
)
from llm_proxy.billing.tokens import TokenUsage
from llm_proxy.config.types.model import ModelConfig, ModelProviderConfig


class TestGetProviderPricing:
    """Tests for _get_provider_pricing function."""

    def test_model_level_pricing_only(self):
        """Test pricing when only model-level pricing is configured."""
        model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-4",
                )
            ],
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
            cached_read_cost_per_1m=1.0,
            cached_write_cost_per_1m=2.5,
            audio_input_cost_per_1m=15.0,
            audio_output_cost_per_1m=30.0,
        )

        rates = _get_provider_pricing(model_config, "openai")
        input_cost = rates.input_cost_per_1m
        output_cost = rates.output_cost_per_1m
        cached_read_cost = rates.cached_read_cost_per_1m
        cached_write_cost = rates.cached_write_cost_per_1m
        audio_input_cost = rates.audio_input_cost_per_1m
        audio_output_cost = rates.audio_output_cost_per_1m

        assert input_cost == 10.0
        assert output_cost == 30.0
        assert cached_read_cost == 1.0
        assert cached_write_cost == 2.5
        assert audio_input_cost == 15.0
        assert audio_output_cost == 30.0

    def test_provider_level_pricing_overrides_model_level(self):
        """Test that provider-level pricing takes precedence over model-level."""
        model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-4",
                    input_cost_per_1m=5.0,
                    output_cost_per_1m=15.0,
                    cached_read_cost_per_1m=0.5,
                    cached_write_cost_per_1m=1.25,
                    audio_input_cost_per_1m=7.5,
                    audio_output_cost_per_1m=15.0,
                )
            ],
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
            cached_read_cost_per_1m=1.0,
            cached_write_cost_per_1m=2.5,
            audio_input_cost_per_1m=15.0,
            audio_output_cost_per_1m=30.0,
        )

        rates = _get_provider_pricing(model_config, "openai")
        input_cost = rates.input_cost_per_1m
        output_cost = rates.output_cost_per_1m
        cached_read_cost = rates.cached_read_cost_per_1m
        cached_write_cost = rates.cached_write_cost_per_1m
        audio_input_cost = rates.audio_input_cost_per_1m
        audio_output_cost = rates.audio_output_cost_per_1m

        assert input_cost == 5.0
        assert output_cost == 15.0
        assert cached_read_cost == 0.5
        assert cached_write_cost == 1.25
        assert audio_input_cost == 7.5
        assert audio_output_cost == 15.0

    def test_provider_level_partial_override_fills_from_model(self):
        """Test that unset provider-level fields fall back to model-level."""
        model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-4",
                    input_cost_per_1m=5.0,
                    cached_read_cost_per_1m=0.5,
                )
            ],
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
            cached_read_cost_per_1m=1.0,
            cached_write_cost_per_1m=2.5,
            audio_input_cost_per_1m=15.0,
            audio_output_cost_per_1m=30.0,
        )

        rates = _get_provider_pricing(model_config, "openai")
        input_cost = rates.input_cost_per_1m
        output_cost = rates.output_cost_per_1m
        cached_read_cost = rates.cached_read_cost_per_1m
        cached_write_cost = rates.cached_write_cost_per_1m
        audio_input_cost = rates.audio_input_cost_per_1m
        audio_output_cost = rates.audio_output_cost_per_1m

        assert input_cost == 5.0
        assert output_cost == 30.0
        assert cached_read_cost == 0.5
        assert cached_write_cost == 2.5
        assert audio_input_cost == 15.0
        assert audio_output_cost == 30.0

    def test_no_pricing_configured(self):
        """Test when no pricing is configured at any level."""
        model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-4",
                )
            ],
        )

        rates = _get_provider_pricing(model_config, "openai")
        input_cost = rates.input_cost_per_1m
        output_cost = rates.output_cost_per_1m
        cached_read_cost = rates.cached_read_cost_per_1m
        cached_write_cost = rates.cached_write_cost_per_1m
        audio_input_cost = rates.audio_input_cost_per_1m
        audio_output_cost = rates.audio_output_cost_per_1m

        assert input_cost is None
        assert output_cost is None
        assert cached_read_cost is None
        assert cached_write_cost is None
        assert audio_input_cost is None
        assert audio_output_cost is None

    def test_different_provider_uses_model_level(self):
        """Test that requesting a different provider uses model-level pricing."""
        model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="anthropic",
                    priority=0,
                    provider_model_name="claude-3",
                    input_cost_per_1m=3.0,
                    cached_read_cost_per_1m=0.3,
                )
            ],
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
            cached_read_cost_per_1m=1.0,
        )

        rates = _get_provider_pricing(model_config, "openai")
        input_cost = rates.input_cost_per_1m
        output_cost = rates.output_cost_per_1m
        cached_read_cost = rates.cached_read_cost_per_1m
        cached_write_cost = rates.cached_write_cost_per_1m

        assert input_cost == 10.0
        assert output_cost == 30.0
        assert cached_read_cost == 1.0
        assert cached_write_cost is None


class TestCalculateCacheCost:
    """Tests for _calculate_cache_cost function."""

    def test_cache_read_cost_configured(self):
        """Test cache read cost with configured cached_read_cost_per_1m."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cache_read_input_tokens=500,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=10.0,
                cached_read_cost_per_1m=1.0,
            ),
        )

        assert adjustment < 0
        assert savings > 0
        expected_savings = (500 / 1_000_000) * (10.0 - 1.0)
        assert abs(savings - expected_savings) < 0.0001

    def test_cache_write_cost_configured(self):
        """Cache creation tokens adjust by the premium over base input cost."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cache_creation_input_tokens=200,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=10.0,
                cached_write_cost_per_1m=12.5,
            ),
        )

        # Premium = (12.5 - 10.0) * 200 / 1M
        expected_premium = (200 / 1_000_000) * (12.5 - 10.0)
        assert abs(adjustment - expected_premium) < 0.0001
        assert savings == 0.0

    def test_cached_prompt_tokens_openai(self):
        """Test OpenAI cached_prompt_tokens with configured cost."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cached_prompt_tokens=400,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=10.0,
                cached_read_cost_per_1m=1.0,
            ),
        )

        expected_savings = (400 / 1_000_000) * (10.0 - 1.0)
        assert abs(savings - expected_savings) < 0.0001

    def test_no_silent_discount_when_cost_not_configured(self):
        """Tokens remain at full input cost when cache read cost is not configured."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cache_read_input_tokens=500,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=10.0,
                cached_read_cost_per_1m=None,
            ),
        )

        # No silent fallback — tokens remain at full input cost
        assert adjustment == 0.0
        assert savings == 0.0

    def test_cache_write_no_cost_means_no_adjustment(self):
        """No cache write cost means no premium adjustment — tokens stay at base input price."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cache_creation_input_tokens=200,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=10.0,
                cached_write_cost_per_1m=None,
            ),
        )

        assert adjustment == 0.0
        assert savings == 0.0

    def test_zero_cost_returns_zero(self):
        """Test that zero cached cost is properly used (not treated as None)."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cache_read_input_tokens=500,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=10.0,
                cached_read_cost_per_1m=0.0,
            ),
        )

        assert savings == (500 / 1_000_000) * 10.0


class TestCalculateAudioCost:
    """Tests for _calculate_audio_cost function."""

    def test_audio_input_cost(self):
        """Test audio input cost calculation."""
        token_usage = TokenUsage(
            audio_input_tokens=1000,
        )

        cost = _calculate_audio_cost(
            token_usage,
            PricingRates(
                audio_input_cost_per_1m=15.0,
                audio_output_cost_per_1m=None,
            ),
        )

        expected = (1000 / 1_000_000) * 15.0
        assert abs(cost - expected) < 0.0001

    def test_audio_output_cost(self):
        """Test audio output cost calculation."""
        token_usage = TokenUsage(
            audio_output_tokens=500,
        )

        cost = _calculate_audio_cost(
            token_usage,
            PricingRates(
                audio_input_cost_per_1m=None,
                audio_output_cost_per_1m=30.0,
            ),
        )

        expected = (500 / 1_000_000) * 30.0
        assert abs(cost - expected) < 0.0001

    def test_both_audio_costs(self):
        """Test combined audio input and output cost."""
        token_usage = TokenUsage(
            audio_input_tokens=1000,
            audio_output_tokens=500,
        )

        cost = _calculate_audio_cost(
            token_usage,
            PricingRates(
                audio_input_cost_per_1m=15.0,
                audio_output_cost_per_1m=30.0,
            ),
        )

        expected = (1000 / 1_000_000) * 15.0 + (500 / 1_000_000) * 30.0
        assert abs(cost - expected) < 0.0001

    def test_no_audio_tokens(self):
        """Test with no audio tokens."""
        token_usage = TokenUsage()

        cost = _calculate_audio_cost(
            token_usage,
            PricingRates(
                audio_input_cost_per_1m=15.0,
                audio_output_cost_per_1m=30.0,
            ),
        )

        assert cost == 0.0

    def test_no_audio_cost_configured(self):
        """Test with audio tokens but no cost configured."""
        token_usage = TokenUsage(
            audio_input_tokens=1000,
            audio_output_tokens=500,
        )

        cost = _calculate_audio_cost(
            token_usage,
            PricingRates(
                audio_input_cost_per_1m=None,
                audio_output_cost_per_1m=None,
            ),
        )

        assert cost == 0.0


class TestZeroCostPreservation:
    """Tests to ensure zero cost values are preserved (not converted to None)."""

    def test_zero_cached_read_cost_is_used(self):
        """cached_read_cost_per_1m=0.0 is used, not treated as None."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cache_read_input_tokens=500,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=10.0,
                cached_read_cost_per_1m=0.0,
            ),
        )

        assert adjustment == -(500 / 1_000_000) * 10.0
        assert savings == (500 / 1_000_000) * 10.0

    def test_zero_cached_write_cost_is_used(self):
        """cached_write_cost_per_1m=0.0 adjusts from base input to zero."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cache_creation_input_tokens=200,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=10.0,
                cached_write_cost_per_1m=0.0,
            ),
        )

        assert adjustment == -(200 / 1_000_000) * 10.0
        assert savings == 0.0

    def test_cache_creation_anthropic_typical_pricing(self):
        """Anthropic-typical cache creation pricing (1.25x input): adjustment adds the premium."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cache_creation_input_tokens=200,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=3.0,
                cached_write_cost_per_1m=3.75,  # 1.25x input
            ),
        )

        expected_premium = (200 / 1_000_000) * 0.75
        assert abs(adjustment - expected_premium) < 0.0001
        assert savings == 0.0

    def test_cache_cost_without_input_rate(self):
        """Cache tokens charged at cache rate directly when no base input rate."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cache_read_input_tokens=500,
            cache_creation_input_tokens=200,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=None,
                cached_read_cost_per_1m=1.0,
                cached_write_cost_per_1m=2.5,
            ),
        )

        expected = (500 / 1_000_000) * 1.0 + (200 / 1_000_000) * 2.5
        assert abs(adjustment - expected) < 0.0001
        assert savings == 0.0

    def test_cache_cost_without_any_rates(self):
        """No rates at all -> 0."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cache_read_input_tokens=500,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=None,
                cached_read_cost_per_1m=None,
            ),
        )

        assert adjustment == 0.0
        assert savings == 0.0

    def test_cache_cost_openai_cached_prompt_without_input_rate(self):
        """OpenAI cached_prompt_tokens charged at cache rate when no base input rate."""
        token_usage = TokenUsage(
            prompt_tokens=1000,
            cached_prompt_tokens=400,
        )

        adjustment, savings = _calculate_cache_cost(
            token_usage,
            PricingRates(
                input_cost_per_1m=None,
                cached_read_cost_per_1m=0.5,
            ),
        )

        expected = (400 / 1_000_000) * 0.5
        assert abs(adjustment - expected) < 0.0001
        assert savings == 0.0


class TestCalculateCostPublicAPI:
    """Tests for the public calculate_cost async function."""

    @pytest.mark.asyncio
    async def test_calculate_cost_with_usage(self):
        """Basic cost calculation with usage dict."""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        }

        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-4",
                )
            ],
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="gpt-4",
            config_manager=mock_config_manager,
            provider_name="openai",
        )

        cost_usd = result.cost_usd
        assert cost_usd is not None
        expected = (1000 / 1_000_000) * 10.0 + (500 / 1_000_000) * 30.0
        assert abs(cost_usd - expected) < 0.0001

    @pytest.mark.asyncio
    async def test_calculate_cost_no_tokens_no_usage(self):
        """Returns None cost when no tokens and no messages."""
        result = await calculate_cost(
            usage=None,
            model_name="gpt-4",
            config_manager=None,
        )

        assert result.cost_usd is None
        assert result.prompt_tokens is None

    @pytest.mark.asyncio
    async def test_calculate_cost_no_model_name(self):
        """Returns None cost when model_name is None."""
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}

        result = await calculate_cost(
            usage=usage,
            model_name=None,
            config_manager=MagicMock(),
        )

        assert result.cost_usd is None

    @pytest.mark.asyncio
    async def test_calculate_cost_model_not_found(self):
        """Returns None cost when model config not found."""
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}

        mock_config_manager = MagicMock()
        mock_config_manager.get_model_config = AsyncMock(return_value=None)

        result = await calculate_cost(
            usage=usage,
            model_name="unknown-model",
            config_manager=mock_config_manager,
        )

        assert result.cost_usd is None

    @pytest.mark.asyncio
    async def test_calculate_cost_with_cache_tokens(self):
        """Cost calculation with cache tokens includes savings."""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
            "cache_read_input_tokens": 400,
        }

        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="anthropic",
                    priority=0,
                    provider_model_name="claude-3",
                )
            ],
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
            cached_read_cost_per_1m=1.0,
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="claude-3",
            config_manager=mock_config_manager,
            provider_name="anthropic",
        )

        cost_usd, cache_savings = result.cost_usd, result.cache_savings_usd
        assert cost_usd is not None
        assert cache_savings is not None
        assert cache_savings > 0

    @pytest.mark.asyncio
    async def test_calculate_cost_with_audio_tokens(self):
        """Cost calculation with audio tokens includes audio cost.

        Audio tokens are a subset of prompt_tokens (OpenAI gpt-4o-audio-preview
        style). They are subtracted from effective_prompt_tokens so they are
        charged at the audio rate only, not double-charged.
        """
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "prompt_tokens_details": {"audio_tokens": 600},
            "completion_tokens_details": {"audio_tokens": 300},
        }

        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-4o-audio",
                )
            ],
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
            audio_input_cost_per_1m=15.0,
            audio_output_cost_per_1m=30.0,
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="gpt-4o-audio",
            config_manager=mock_config_manager,
        )

        cost_usd = result.cost_usd
        assert cost_usd is not None
        # Audio tokens (600) are subtracted from prompt_tokens (1000) so
        # effective_prompt_tokens = 400; audio output tokens (300) are
        # subtracted from completion_tokens (500) so effective = 200. Audio
        # tokens are charged at the audio rate only, never double-charged.
        expected = (
            (400 / 1_000_000) * 10.0  # text input tokens
            + (200 / 1_000_000) * 30.0  # text output tokens (500 - 300 audio)
            + (600 / 1_000_000) * 15.0  # audio input tokens at audio rate
            + (300 / 1_000_000) * 30.0  # audio output tokens at audio rate
        )
        assert abs(cost_usd - expected) < 0.0001

    @pytest.mark.asyncio
    async def test_calculate_cost_realtime_usage_dialect(self):
        """Realtime response.done usage is priced per component exactly once.

        ``input_tokens``/``output_tokens`` include text, cached, and audio
        tokens; each component must be charged at its own rate, with no
        double charge at the base rate for the audio and cached portions.
        """
        usage = {
            "total_tokens": 1000,
            "input_tokens": 500,
            "output_tokens": 500,
            "input_token_details": {
                "cached_tokens": 100,
                "text_tokens": 300,
                "audio_tokens": 100,
                "image_tokens": 0,
            },
            "output_token_details": {"text_tokens": 200, "audio_tokens": 300},
        }
        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[ModelProviderConfig(provider="openai", priority=0)],
            input_cost_per_1m=10.0,
            output_cost_per_1m=10.0,
            cached_read_cost_per_1m=0.5,
            audio_input_cost_per_1m=40.0,
            audio_output_cost_per_1m=80.0,
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="gpt-realtime",
            config_manager=mock_config_manager,
            provider_name="openai",
        )

        assert result.cost_usd is not None
        expected = (
            (300 / 1_000_000) * 10.0  # text input (500 - 100 audio - 100 cached)
            + (200 / 1_000_000) * 10.0  # text output (500 - 300 audio)
            + (100 / 1_000_000) * 0.5  # cached input at cache-read rate
            + (100 / 1_000_000) * 40.0  # audio input at audio rate
            + (300 / 1_000_000) * 80.0  # audio output at audio rate
        )
        assert abs(result.cost_usd - expected) < 0.0001

    @pytest.mark.asyncio
    async def test_calculate_cost_exception_returns_none(self):
        """Exception in config lookup returns None cost, not raises."""
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}

        mock_config_manager = MagicMock()
        mock_config_manager.get_model_config = AsyncMock(side_effect=RuntimeError("DB error"))

        result = await calculate_cost(
            usage=usage,
            model_name="gpt-4",
            config_manager=mock_config_manager,
        )

        # Should return values with None cost, not raise
        assert result.cost_usd is None
        assert result.prompt_tokens == 1000
        assert result.completion_tokens == 500

    @pytest.mark.asyncio
    async def test_calculate_cost_with_messages_estimates(self):
        """Cost calculation estimates from messages when usage is None."""
        messages = [{"role": "user", "content": "Hello world " * 50}]  # ~9 tokens/word * 100 words

        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-4",
                )
            ],
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=None,
            model_name="gpt-4",
            config_manager=mock_config_manager,
            messages=messages,
            provider_name="openai",
        )

        cost_usd, prompt_tokens = result.cost_usd, result.prompt_tokens
        assert cost_usd is not None
        assert prompt_tokens is not None
        assert prompt_tokens > 0

    @pytest.mark.asyncio
    async def test_calculate_cost_cached_tokens_separate_from_prompt(self):
        """Some providers report cached_tokens in prompt_tokens_details but NOT in
        prompt_tokens. prompt_tokens=2527, cached_tokens=126553, total_tokens=129144.
        Without the fix, this produces a negative cost because _calculate_cache_cost
        assumes cached tokens are already in prompt_tokens and subtracts the full
        input cost that was never charged.
        """
        usage = {
            "prompt_tokens": 2527,
            "completion_tokens": 64,
            "total_tokens": 129144,
            "prompt_tokens_details": {"cached_tokens": 126553},
        }

        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="some-provider",
                    priority=0,
                    provider_model_name="some-model",
                )
            ],
            input_cost_per_1m=2.0,
            output_cost_per_1m=8.0,
            cached_read_cost_per_1m=0.5,
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="some-model",
            config_manager=mock_config_manager,
            provider_name="some-provider",
        )

        cost_usd = result.cost_usd
        assert cost_usd is not None
        assert cost_usd > 0, f"Cost should be positive, got {cost_usd}"

        # Expected: non-cached prompt (2527) at input rate
        #          + cached (126553) at cache rate + output (64) at output rate
        expected = (2527 / 1_000_000) * 2.0 + (126553 / 1_000_000) * 0.5 + (64 / 1_000_000) * 8.0
        assert abs(cost_usd - expected) < 0.0001, f"Expected {expected}, got {cost_usd}"

        # Verify savings are reported correctly
        cache_savings = result.cache_savings_usd
        assert cache_savings is not None
        expected_savings = (126553 / 1_000_000) * (2.0 - 0.5)
        assert abs(cache_savings - expected_savings) < 0.0001

    @pytest.mark.asyncio
    async def test_calculate_cost_cached_tokens_included_in_prompt(self):
        """OpenAI-style: cached_tokens are included in prompt_tokens — unchanged."""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
            "prompt_tokens_details": {"cached_tokens": 400},
        }

        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-4",
                )
            ],
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
            cached_read_cost_per_1m=1.0,
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="gpt-4",
            config_manager=mock_config_manager,
            provider_name="openai",
        )

        cost_usd = result.cost_usd
        assert cost_usd is not None
        assert cost_usd > 0, f"Cost should be positive, got {cost_usd}"

        # Expected: non-cached prompt (600) at input rate
        #          + cached (400) at cache rate + output (500) at output rate
        expected = (600 / 1_000_000) * 10.0 + (400 / 1_000_000) * 1.0 + (500 / 1_000_000) * 30.0
        assert abs(cost_usd - expected) < 0.0001, f"Expected {expected}, got {cost_usd}"

    @pytest.mark.asyncio
    async def test_calculate_cost_provider_fallback(self):
        """Provider-specific pricing overrides model-level pricing."""
        usage = {"prompt_tokens": 1000, "completion_tokens": 500}

        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-4",
                    input_cost_per_1m=5.0,  # Provider-level override
                    output_cost_per_1m=15.0,
                )
            ],
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="gpt-4",
            config_manager=mock_config_manager,
            provider_name="openai",
        )

        cost_usd = result.cost_usd
        assert cost_usd is not None
        # Should use provider-level 5.0, not model-level 10.0
        expected = (1000 / 1_000_000) * 5.0 + (500 / 1_000_000) * 15.0
        assert abs(cost_usd - expected) < 0.0001


class TestCalculateUnitCosts:
    """Tests for _calculate_unit_costs (non-token billing dimensions)."""

    def test_cost_per_image(self):
        """Per-generated-image billing."""
        token_usage = TokenUsage(images_generated=3)
        cost = _calculate_unit_costs(
            token_usage,
            PricingRates(cost_per_image=0.04),
        )
        assert abs(cost - 3 * 0.04) < 0.0001

    def test_audio_cost_per_minute(self):
        """STT duration-based billing (seconds -> minutes)."""
        token_usage = TokenUsage(audio_duration_seconds=120)
        cost = _calculate_unit_costs(
            token_usage,
            PricingRates(audio_cost_per_minute=0.006),
        )
        assert abs(cost - (120 / 60) * 0.006) < 0.0001

    def test_tts_cost_per_1m_chars(self):
        """TTS per-1M-characters billing."""
        token_usage = TokenUsage(tts_characters=2_000_000)
        cost = _calculate_unit_costs(
            token_usage,
            PricingRates(tts_cost_per_1m_chars=16.0),
        )
        assert abs(cost - (2_000_000 / 1_000_000) * 16.0) < 0.0001

    def test_web_search_cost_per_1k(self):
        """Web search per-1k-requests billing."""
        token_usage = TokenUsage(web_search_requests=5)
        cost = _calculate_unit_costs(
            token_usage,
            PricingRates(web_search_cost_per_1k=0.03),
        )
        assert abs(cost - (5 / 1000) * 0.03) < 0.0001

    def test_all_unit_dimensions_combined(self):
        """All four unit dimensions accumulate."""
        token_usage = TokenUsage(
            images_generated=3,
            audio_duration_seconds=120,
            tts_characters=2_000_000,
            web_search_requests=5,
        )
        cost = _calculate_unit_costs(
            token_usage,
            PricingRates(
                cost_per_image=0.04,
                audio_cost_per_minute=0.006,
                tts_cost_per_1m_chars=16.0,
                web_search_cost_per_1k=0.03,
            ),
        )
        expected = (
            3 * 0.04 + (120 / 60) * 0.006 + (2_000_000 / 1_000_000) * 16.0 + (5 / 1000) * 0.03
        )
        assert abs(cost - expected) < 0.0001

    def test_no_unit_cost_configured(self):
        """Usage present but no rates configured -> 0."""
        token_usage = TokenUsage(
            images_generated=3,
            audio_duration_seconds=120,
            tts_characters=2_000_000,
            web_search_requests=5,
        )
        assert _calculate_unit_costs(token_usage, PricingRates()) == 0.0

    def test_no_unit_usage(self):
        """Rates configured but no usage -> 0."""
        token_usage = TokenUsage()
        cost = _calculate_unit_costs(
            token_usage,
            PricingRates(
                cost_per_image=0.04,
                audio_cost_per_minute=0.006,
                tts_cost_per_1m_chars=16.0,
                web_search_cost_per_1k=0.03,
            ),
        )
        assert cost == 0.0


class TestCalculateImageTokenCost:
    """Tests for _calculate_image_token_cost (image input token adjustment)."""

    def test_image_token_premium_adjustment(self):
        """Image tokens charged at the full image rate (not delta)."""
        token_usage = TokenUsage(prompt_tokens=1000, image_input_tokens=500)
        cost = _calculate_image_token_cost(
            token_usage,
            PricingRates(input_cost_per_1m=10.0, image_input_cost_per_1m=30.0),
        )
        # image tokens charged at full image rate (caller subtracts from prompt_tokens)
        assert abs(cost - (500 / 1_000_000) * 30.0) < 0.0001

    def test_no_image_tokens(self):
        """No image input tokens -> 0 adjustment."""
        token_usage = TokenUsage(prompt_tokens=1000)
        cost = _calculate_image_token_cost(
            token_usage,
            PricingRates(input_cost_per_1m=10.0, image_input_cost_per_1m=30.0),
        )
        assert cost == 0.0

    def test_no_image_rate_configured(self):
        """image_input_cost_per_1m not configured -> 0 (charged at base input rate)."""
        token_usage = TokenUsage(prompt_tokens=1000, image_input_tokens=500)
        cost = _calculate_image_token_cost(
            token_usage,
            PricingRates(input_cost_per_1m=10.0, image_input_cost_per_1m=None),
        )
        assert cost == 0.0

    def test_no_input_rate_configured(self):
        """input_cost_per_1m not configured -> charge at full image rate."""
        token_usage = TokenUsage(prompt_tokens=1000, image_input_tokens=500)
        cost = _calculate_image_token_cost(
            token_usage,
            PricingRates(input_cost_per_1m=None, image_input_cost_per_1m=30.0),
        )
        assert abs(cost - (500 / 1_000_000) * 30.0) < 0.0001


class TestCalculateCostUnitBased:
    """End-to-end calculate_cost tests for unit-based billing dimensions."""

    @pytest.mark.asyncio
    async def test_image_generation_per_image_only(self):
        """gpt-image billed per image with no token pricing configured."""
        usage = {"images_generated": 3}
        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-image-1",
                    cost_per_image=0.04,
                )
            ],
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="gpt-image-1",
            config_manager=mock_config_manager,
            provider_name="openai",
        )

        assert result.cost_usd is not None
        assert abs(result.cost_usd - 3 * 0.04) < 0.0001
        assert result.images_generated == 3
        assert result.prompt_tokens is None

    @pytest.mark.asyncio
    async def test_gpt_image_tokens_plus_per_image(self):
        """gpt-image with token usage AND per-image pricing.

        Image tokens are subtracted from effective_prompt_tokens and charged
        at the full image rate. The total is the same as before (delta approach)
        when both input and image rates are configured.
        """
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "input_tokens_details": {"image_tokens": 500},
            "images_generated": 2,
        }
        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-image-1",
                    input_cost_per_1m=10.0,
                    output_cost_per_1m=30.0,
                    image_input_cost_per_1m=30.0,
                    cost_per_image=0.04,
                )
            ],
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="gpt-image-1",
            config_manager=mock_config_manager,
            provider_name="openai",
        )

        assert result.cost_usd is not None
        # effective_prompt_tokens = 1000 - 500 = 500 (image tokens subtracted)
        expected = (
            (500 / 1_000_000) * 10.0  # text input tokens at base rate
            + (500 / 1_000_000) * 30.0  # output tokens
            + (500 / 1_000_000) * 30.0  # image tokens at full image rate
            + 2 * 0.04  # per-image
        )
        assert abs(result.cost_usd - expected) < 0.0001
        assert result.image_input_tokens == 500
        assert result.images_generated == 2

    @pytest.mark.asyncio
    async def test_tts_per_character_billing(self):
        """TTS billed per 1M characters, no token usage."""
        usage = {"tts_characters": 500_000}
        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="tts-1",
                    tts_cost_per_1m_chars=16.0,
                )
            ],
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="tts-1",
            config_manager=mock_config_manager,
            provider_name="openai",
        )

        assert result.cost_usd is not None
        assert abs(result.cost_usd - (500_000 / 1_000_000) * 16.0) < 0.0001
        assert result.tts_characters == 500_000

    @pytest.mark.asyncio
    async def test_stt_duration_billing(self):
        """Whisper-style STT billed per audio minute (duration-based usage)."""
        usage = {"audio_duration_seconds": 90}
        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="whisper-1",
                    audio_cost_per_minute=0.006,
                )
            ],
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="whisper-1",
            config_manager=mock_config_manager,
            provider_name="openai",
        )

        assert result.cost_usd is not None
        assert abs(result.cost_usd - (90 / 60) * 0.006) < 0.0001
        assert result.audio_duration_seconds == 90

    @pytest.mark.asyncio
    async def test_web_search_per_request_billing(self):
        """Native web search billed per 1k requests, no token usage."""
        usage = {"web_search_requests": 10}
        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="anthropic",
                    priority=0,
                    provider_model_name="claude-3",
                    web_search_cost_per_1k=0.03,
                )
            ],
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="claude-3",
            config_manager=mock_config_manager,
            provider_name="anthropic",
        )

        assert result.cost_usd is not None
        assert abs(result.cost_usd - (10 / 1000) * 0.03) < 0.0001
        assert result.web_search_requests == 10

    @pytest.mark.asyncio
    async def test_tokens_plus_web_search_billing(self):
        """Token cost and web search cost both apply."""
        usage = {"prompt_tokens": 1000, "completion_tokens": 500, "web_search_requests": 4}
        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="anthropic",
                    priority=0,
                    provider_model_name="claude-3",
                    input_cost_per_1m=10.0,
                    output_cost_per_1m=30.0,
                    web_search_cost_per_1k=0.03,
                )
            ],
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="claude-3",
            config_manager=mock_config_manager,
            provider_name="anthropic",
        )

        assert result.cost_usd is not None
        expected = (1000 / 1_000_000) * 10.0 + (500 / 1_000_000) * 30.0 + (4 / 1000) * 0.03
        assert abs(result.cost_usd - expected) < 0.0001


class TestGetProviderPricingUnitBased:
    """_get_provider_pricing covers the new unit-based fields with fallback."""

    def test_unit_based_provider_override(self):
        """Provider-level unit pricing overrides model-level."""
        model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-image-1",
                    cost_per_image=0.04,
                    audio_cost_per_minute=0.006,
                    tts_cost_per_1m_chars=16.0,
                    web_search_cost_per_1k=0.03,
                    image_input_cost_per_1m=30.0,
                )
            ],
            cost_per_image=0.08,
            audio_cost_per_minute=0.012,
            tts_cost_per_1m_chars=32.0,
            web_search_cost_per_1k=0.06,
            image_input_cost_per_1m=60.0,
        )
        rates = _get_provider_pricing(model_config, "openai")
        assert rates.cost_per_image == 0.04
        assert rates.audio_cost_per_minute == 0.006
        assert rates.tts_cost_per_1m_chars == 16.0
        assert rates.web_search_cost_per_1k == 0.03
        assert rates.image_input_cost_per_1m == 30.0

    def test_unit_based_model_fallback(self):
        """Model-level unit pricing used when provider doesn't override."""
        model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-image-1",
                )
            ],
            cost_per_image=0.08,
            audio_cost_per_minute=0.012,
            tts_cost_per_1m_chars=32.0,
            web_search_cost_per_1k=0.06,
            image_input_cost_per_1m=60.0,
        )
        rates = _get_provider_pricing(model_config, "openai")
        assert rates.cost_per_image == 0.08
        assert rates.audio_cost_per_minute == 0.012
        assert rates.tts_cost_per_1m_chars == 32.0
        assert rates.web_search_cost_per_1k == 0.06
        assert rates.image_input_cost_per_1m == 60.0


class TestEdgeCases:
    """Edge cases for the fixed cost calculation."""

    @pytest.mark.asyncio
    async def test_image_tokens_without_input_rate(self):
        """Image tokens charged at full image rate when no base input rate."""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "input_tokens_details": {"image_tokens": 500},
        }
        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="openai",
                    priority=0,
                    provider_model_name="gpt-image-1",
                    image_input_cost_per_1m=30.0,
                )
            ],
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="gpt-image-1",
            config_manager=mock_config_manager,
            provider_name="openai",
        )

        assert result.cost_usd is not None
        # No input_rate, so input_cost = 0. Image tokens charged at full image rate.
        # effective_prompt_tokens = 1000 - 500 = 500, but input_cost = 0 since no rate.
        expected = (500 / 1_000_000) * 30.0  # image tokens at full image rate
        assert abs(result.cost_usd - expected) < 0.0001
        assert result.image_input_tokens == 500

    @pytest.mark.asyncio
    async def test_audio_tokens_not_in_prompt_tokens(self):
        """Audio tokens reported separately from prompt_tokens (third-party provider).

        When prompt_tokens < audio_tokens, audio tokens are NOT subtracted from
        effective_prompt_tokens, and audio_cost adds the full audio rate.
        """
        usage = {
            "prompt_tokens": 400,
            "completion_tokens": 200,
            "prompt_tokens_details": {"audio_tokens": 600},
        }
        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="some-provider",
                    priority=0,
                    provider_model_name="audio-model",
                    input_cost_per_1m=10.0,
                    output_cost_per_1m=30.0,
                    audio_input_cost_per_1m=15.0,
                )
            ],
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="audio-model",
            config_manager=mock_config_manager,
            provider_name="some-provider",
        )

        assert result.cost_usd is not None
        # prompt_tokens (400) < audio_tokens (600), so audio tokens are NOT subtracted.
        # input_cost = 400 / 1M * 10.0 = 0.004
        # audio_cost = 600 / 1M * 15.0 = 0.009
        # output_cost = 200 / 1M * 30.0 = 0.006
        expected = (400 / 1_000_000) * 10.0 + (600 / 1_000_000) * 15.0 + (200 / 1_000_000) * 30.0
        assert abs(result.cost_usd - expected) < 0.0001

    @pytest.mark.asyncio
    async def test_cache_tokens_without_input_rate_end_to_end(self):
        """Cache tokens charged at cache rates when no base input rate (end-to-end)."""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "cache_read_input_tokens": 400,
            "cache_creation_input_tokens": 200,
        }
        mock_config_manager = MagicMock()
        mock_model_config = ModelConfig(
            providers=[
                ModelProviderConfig(
                    provider="some-provider",
                    priority=0,
                    provider_model_name="cache-model",
                    cached_read_cost_per_1m=1.0,
                    cached_write_cost_per_1m=2.5,
                )
            ],
        )
        mock_config_manager.get_model_config = AsyncMock(return_value=mock_model_config)

        result = await calculate_cost(
            usage=usage,
            model_name="cache-model",
            config_manager=mock_config_manager,
            provider_name="some-provider",
        )

        assert result.cost_usd is not None
        # No input_rate or output_rate, so only cache costs apply.
        expected = (400 / 1_000_000) * 1.0 + (200 / 1_000_000) * 2.5
        assert abs(result.cost_usd - expected) < 0.0001
