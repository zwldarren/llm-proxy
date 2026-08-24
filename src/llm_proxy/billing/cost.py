"""Cost calculation utilities."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_proxy.billing.tokens import TokenUsage, extract_tokens_from_usage
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager

logger = get_logger(__name__)


@dataclass
class PricingRates:
    """All pricing rates for a model.

    Token rates are per 1M tokens; the remaining fields are per-unit rates
    for non-token billing dimensions.
    """

    # Token-based rates (per 1M tokens)
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    cached_read_cost_per_1m: float | None = None
    cached_write_cost_per_1m: float | None = None
    audio_input_cost_per_1m: float | None = None
    audio_output_cost_per_1m: float | None = None
    image_input_cost_per_1m: float | None = None
    # Unit-based rates
    cost_per_image: float | None = None  # per generated image
    audio_cost_per_minute: float | None = None  # per audio minute (STT)
    tts_cost_per_1m_chars: float | None = None  # per 1M characters (TTS)
    web_search_cost_per_1k: float | None = None  # per 1k search requests


@dataclass
class CostBreakdown:
    """Result of cost calculation: total cost plus every usage dimension.

    ``cost_usd`` is None when the cost cannot be determined (no billable
    usage data, no model pricing configured, or a lookup failure).
    """

    cost_usd: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    cache_savings_usd: float | None = None
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None
    image_input_tokens: int | None = None
    # Non-token billable dimensions
    images_generated: int | None = None
    audio_duration_seconds: float | None = None
    tts_characters: int | None = None
    web_search_requests: int | None = None


_PRICING_FIELDS = tuple(PricingRates.__dataclass_fields__)


def _get_provider_pricing(
    model_config: Any,
    provider_name: str | None,
) -> PricingRates:
    """Get pricing for a specific provider, falling back to model-level pricing."""

    def _rate(obj: Any, attr: str) -> float | None:
        return getattr(obj, attr, None)

    if provider_name:
        provider_config = next(
            (p for p in model_config.providers if p.provider == provider_name),
            None,
        )
        if provider_config:
            return PricingRates(
                **{
                    field: (
                        v
                        if (v := _rate(provider_config, field)) is not None
                        else _rate(model_config, field)
                    )
                    for field in _PRICING_FIELDS
                }
            )

    return PricingRates(**{field: _rate(model_config, field) for field in _PRICING_FIELDS})


def _calculate_cache_cost(
    token_usage: TokenUsage,
    rates: PricingRates,
) -> tuple[float, float]:
    """Calculate cost adjustment and savings for cache token types.

    Cache tokens (read, creation, cached_prompt) are already counted in prompt_tokens
    and charged at base input_cost_per_1m. This function adjusts the difference between
    the base rate and the cache-specific rate.

    When no base input rate is configured but cache rates are set, cache tokens are
    charged at the cache rate directly (no adjustment needed since there's no base
    charge to adjust from).
    """
    if rates.input_cost_per_1m is None:
        # No base input rate — charge cache tokens at cache rates directly.
        cache_cost = 0.0
        if rates.cached_read_cost_per_1m is not None:
            cache_cost += (
                token_usage.cache_read_input_tokens / 1_000_000
            ) * rates.cached_read_cost_per_1m
            cache_cost += (
                token_usage.cached_prompt_tokens / 1_000_000
            ) * rates.cached_read_cost_per_1m
        if rates.cached_write_cost_per_1m is not None:
            cache_cost += (
                token_usage.cache_creation_input_tokens / 1_000_000
            ) * rates.cached_write_cost_per_1m
        return cache_cost, 0.0

    input_cost_per_1m = rates.input_cost_per_1m
    cost_adjustment = 0.0
    cache_savings_usd = 0.0

    # Anthropic cache read tokens — already in prompt_tokens, adjust to cache rate
    if token_usage.cache_read_input_tokens > 0:
        if rates.cached_read_cost_per_1m is not None:
            discounted_cost = (
                token_usage.cache_read_input_tokens / 1_000_000
            ) * rates.cached_read_cost_per_1m
            full_cost = (token_usage.cache_read_input_tokens / 1_000_000) * input_cost_per_1m
            cost_adjustment += discounted_cost - full_cost
            cache_savings_usd += full_cost - discounted_cost
        else:
            logger.warning(
                "cache_read_input_tokens present but no cached_read_cost_per_1m configured; "
                "tokens will be charged at full input price with no discount"
            )

    # Anthropic cache creation tokens — already in prompt_tokens, adjust to cache write rate
    if token_usage.cache_creation_input_tokens > 0:
        if rates.cached_write_cost_per_1m is not None:
            creation_cost = (
                token_usage.cache_creation_input_tokens / 1_000_000
            ) * rates.cached_write_cost_per_1m
            full_cost = (token_usage.cache_creation_input_tokens / 1_000_000) * input_cost_per_1m
            cost_adjustment += creation_cost - full_cost
        else:
            logger.warning(
                "cache_creation_input_tokens present but no cached_write_cost_per_1m configured; "
                "tokens will be charged at base input price instead of cache write price"
            )

    # OpenAI cached prompt tokens — already in prompt_tokens, adjust to cache rate
    if token_usage.cached_prompt_tokens > 0:
        if rates.cached_read_cost_per_1m is not None:
            discounted_cost = (
                token_usage.cached_prompt_tokens / 1_000_000
            ) * rates.cached_read_cost_per_1m
            full_cost = (token_usage.cached_prompt_tokens / 1_000_000) * input_cost_per_1m
            cost_adjustment += discounted_cost - full_cost
            cache_savings_usd += full_cost - discounted_cost
        else:
            logger.warning(
                "cached_prompt_tokens present but no cached_read_cost_per_1m configured; "
                "tokens will be charged at full input price with no discount"
            )

    return cost_adjustment, cache_savings_usd


def _calculate_audio_cost(
    token_usage: TokenUsage,
    rates: PricingRates,
) -> float:
    """Calculate cost for audio tokens.

    Audio tokens (e.g. gpt-4o-audio-preview's ``prompt_tokens_details.audio_tokens``)
    are already counted in prompt_tokens and charged at the base input rate.
    When a dedicated audio input rate is configured, the caller is responsible for
    subtracting audio tokens from ``effective_prompt_tokens`` before computing
    ``input_cost``, so this function charges the full audio rate.
    """
    audio_cost = 0.0

    if rates.audio_input_cost_per_1m is not None:
        audio_cost += (token_usage.audio_input_tokens / 1_000_000) * rates.audio_input_cost_per_1m

    if rates.audio_output_cost_per_1m is not None:
        audio_cost += (token_usage.audio_output_tokens / 1_000_000) * rates.audio_output_cost_per_1m

    return audio_cost


def _calculate_image_token_cost(
    token_usage: TokenUsage,
    rates: PricingRates,
) -> float:
    """Calculate cost for image input tokens.

    Image input tokens (e.g. gpt-image's ``input_tokens_details.image_tokens``)
    are already counted in prompt_tokens and charged at the base input rate.
    When a dedicated image input rate is configured, the caller is responsible for
    subtracting image tokens from ``effective_prompt_tokens`` before computing
    ``input_cost``, so this function charges the full image rate.
    """
    if token_usage.image_input_tokens <= 0:
        return 0.0
    if rates.image_input_cost_per_1m is None:
        return 0.0
    return (token_usage.image_input_tokens / 1_000_000) * rates.image_input_cost_per_1m


def _calculate_unit_costs(
    token_usage: TokenUsage,
    rates: PricingRates,
) -> float:
    """Calculate cost for non-token billing dimensions.

    Covers per-image (image generation), per-minute (STT), per-character
    (TTS), and per-search (web search) pricing.
    """
    unit_cost = 0.0

    if rates.cost_per_image is not None and (token_usage.images_generated or 0) > 0:
        unit_cost += (token_usage.images_generated or 0) * rates.cost_per_image

    if rates.audio_cost_per_minute is not None and (token_usage.audio_duration_seconds or 0) > 0:
        unit_cost += ((token_usage.audio_duration_seconds or 0) / 60) * rates.audio_cost_per_minute

    if rates.tts_cost_per_1m_chars is not None and (token_usage.tts_characters or 0) > 0:
        unit_cost += ((token_usage.tts_characters or 0) / 1_000_000) * rates.tts_cost_per_1m_chars

    if rates.web_search_cost_per_1k is not None and (token_usage.web_search_requests or 0) > 0:
        unit_cost += ((token_usage.web_search_requests or 0) / 1000) * rates.web_search_cost_per_1k

    return unit_cost


def _has_billable_data(token_usage: TokenUsage) -> bool:
    """Check if any billable usage dimension has data."""
    return any(
        [
            token_usage.prompt_tokens,
            token_usage.completion_tokens,
            token_usage.images_generated,
            token_usage.audio_duration_seconds,
            token_usage.tts_characters,
            token_usage.web_search_requests,
        ]
    )


def _breakdown_from_usage(token_usage: TokenUsage, cost_usd: float | None) -> CostBreakdown:
    """Build a CostBreakdown from extracted usage with the given total cost."""
    return CostBreakdown(
        cost_usd=cost_usd,
        prompt_tokens=token_usage.prompt_tokens or None,
        completion_tokens=token_usage.completion_tokens or None,
        total_tokens=token_usage.total_tokens or None,
        cache_creation_input_tokens=token_usage.cache_creation_input_tokens or None,
        cache_read_input_tokens=token_usage.cache_read_input_tokens or None,
        cached_prompt_tokens=token_usage.cached_prompt_tokens or None,
        cache_savings_usd=None,
        audio_input_tokens=token_usage.audio_input_tokens or None,
        audio_output_tokens=token_usage.audio_output_tokens or None,
        image_input_tokens=token_usage.image_input_tokens or None,
        images_generated=token_usage.images_generated or None,
        audio_duration_seconds=token_usage.audio_duration_seconds or None,
        tts_characters=token_usage.tts_characters or None,
        web_search_requests=token_usage.web_search_requests or None,
    )


async def calculate_cost(
    usage: dict[str, Any] | None,
    model_name: str | None,
    config_manager: DatabaseConfigManager | None = None,
    messages: list[dict[str, Any]] | None = None,
    completion_text: str | None = None,
    provider_name: str | None = None,
) -> CostBreakdown:
    """Calculate cost based on usage and model pricing.

    Supports token-based billing (text/cache/audio/image tokens) and
    unit-based billing (per image, per audio minute, per 1M TTS characters,
    per 1k web searches).

    Returns a CostBreakdown; ``cost_usd`` is None when the cost cannot be
    determined.
    """
    token_usage = extract_tokens_from_usage(usage)

    if (
        token_usage.prompt_tokens == 0
        and token_usage.completion_tokens == 0
        and (messages or completion_text)
    ):
        from llm_proxy.billing.tokens import estimate_usage_from_request

        estimated = estimate_usage_from_request(messages, completion_text)
        token_usage.prompt_tokens = estimated["prompt_tokens"]
        token_usage.completion_tokens = estimated["completion_tokens"]
        token_usage.total_tokens = estimated["total_tokens"]

    if not _has_billable_data(token_usage):
        return _breakdown_from_usage(token_usage, None)

    if not model_name or config_manager is None:
        return _breakdown_from_usage(token_usage, None)

    try:
        model_config = await config_manager.get_model_config(model_name)
        if not model_config:
            return _breakdown_from_usage(token_usage, None)

        # Get all pricing rates with provider-specific fallback
        rates = _get_provider_pricing(model_config, provider_name)

        input_cost = 0.0
        output_cost = 0.0

        # Some providers report cached_tokens in prompt_tokens_details
        # but do NOT include them in prompt_tokens (e.g. certain third-party
        # OpenAI-compatible providers). OpenAI includes them.
        effective_prompt_tokens = token_usage.prompt_tokens
        if (token_usage.cached_prompt_tokens or 0) > 0 and (
            (token_usage.prompt_tokens or 0) < (token_usage.cached_prompt_tokens or 0)
        ):
            effective_prompt_tokens = (token_usage.prompt_tokens or 0) + (
                token_usage.cached_prompt_tokens or 0
            )

        # Audio tokens are already in prompt_tokens (OpenAI gpt-4o-audio-preview).
        # Subtract them so they are charged at the audio rate only, not double-charged
        # at both the base input rate and the audio rate.
        if (
            (token_usage.audio_input_tokens or 0) > 0
            and rates.audio_input_cost_per_1m is not None
            and effective_prompt_tokens >= token_usage.audio_input_tokens
        ):
            effective_prompt_tokens -= token_usage.audio_input_tokens

        # Audio tokens are already in completion_tokens (Realtime/Responses
        # usage: ``output_tokens`` includes text and audio). Subtract them so
        # they are charged at the audio rate only, mirroring the input side.
        effective_completion_tokens = token_usage.completion_tokens
        if (
            (token_usage.audio_output_tokens or 0) > 0
            and rates.audio_output_cost_per_1m is not None
            and effective_completion_tokens >= token_usage.audio_output_tokens
        ):
            effective_completion_tokens -= token_usage.audio_output_tokens

        # Image tokens are already in prompt_tokens (gpt-image).
        # Subtract them so they are charged at the image rate only.
        if (
            (token_usage.image_input_tokens or 0) > 0
            and rates.image_input_cost_per_1m is not None
            and effective_prompt_tokens >= token_usage.image_input_tokens
        ):
            effective_prompt_tokens -= token_usage.image_input_tokens

        if rates.input_cost_per_1m is not None:
            input_cost = (effective_prompt_tokens / 1_000_000) * rates.input_cost_per_1m
        if rates.output_cost_per_1m is not None:
            output_cost = (effective_completion_tokens / 1_000_000) * rates.output_cost_per_1m

        # Cost components: cache adjustment, audio tokens, image tokens, unit pricing
        cache_cost, cache_savings_usd = _calculate_cache_cost(token_usage, rates)
        audio_cost = _calculate_audio_cost(token_usage, rates)
        image_token_cost = _calculate_image_token_cost(token_usage, rates)
        unit_cost = _calculate_unit_costs(token_usage, rates)

        cost_usd = input_cost + output_cost + cache_cost + audio_cost + image_token_cost + unit_cost
        breakdown = _breakdown_from_usage(token_usage, cost_usd)
        breakdown.cache_savings_usd = cache_savings_usd
        return breakdown
    except Exception as e:
        logger.warning(
            f"Cost calculation failed for model={model_name}, provider={provider_name}: {e}"
        )
        return _breakdown_from_usage(token_usage, None)
