"""Cost calculation utilities."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from llm_proxy.billing.tokens import ProxyBillableDimensions, TokenUsage, token_usage_from
from llm_proxy.config.types.model import minutes_of_day
from llm_proxy.core.utils import coerce_float, read_field
from llm_proxy.models.types import Usage
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
    reasoning_cost_per_1m: float | None = None
    audio_input_cost_per_1m: float | None = None
    audio_output_cost_per_1m: float | None = None
    image_input_cost_per_1m: float | None = None
    # Unit-based rates
    cost_per_image: float | None = None  # per generated image
    audio_cost_per_minute: float | None = None  # per audio minute (STT)
    tts_cost_per_1m_chars: float | None = None  # per 1M characters (TTS)
    web_search_cost_per_1k: float | None = None  # per 1k search requests


@dataclass
class CostTier:
    """Context-based pricing tier: rates that apply from ``threshold`` input tokens up.

    ``None`` rates fall back to the base rate of the same dimension, so a tier
    only needs to carry the dimensions it actually changes.
    """

    threshold: int
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    cached_read_cost_per_1m: float | None = None
    cached_write_cost_per_1m: float | None = None
    reasoning_cost_per_1m: float | None = None
    audio_input_cost_per_1m: float | None = None
    audio_output_cost_per_1m: float | None = None
    image_input_cost_per_1m: float | None = None


@dataclass
class PricingWindow:
    """One time-of-day price band, read in ``timezone``'s wall clock.

    ``start_minute >= end_minute`` marks a window that crosses midnight (a
    ``22:00``–``08:00`` off-peak band). ``days`` names the ISO weekday the window
    *starts* on, so a Friday band also covers Saturday's small hours; an empty set
    means every day.
    """

    start_minute: int
    end_minute: int
    days: frozenset[int]
    timezone: str
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    cached_read_cost_per_1m: float | None = None
    cached_write_cost_per_1m: float | None = None
    reasoning_cost_per_1m: float | None = None
    audio_input_cost_per_1m: float | None = None
    audio_output_cost_per_1m: float | None = None
    image_input_cost_per_1m: float | None = None


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

# Token rates an overlay (context tier, time window, service tier) may override.
# Unit-based pricing (per image, per minute, per character, per search) is never
# overlaid: no upstream prices those by request size, clock or service tier.
_OVERLAY_RATE_FIELDS = (
    "input_cost_per_1m",
    "output_cost_per_1m",
    "cached_read_cost_per_1m",
    "cached_write_cost_per_1m",
    "reasoning_cost_per_1m",
    "audio_input_cost_per_1m",
    "audio_output_cost_per_1m",
    "image_input_cost_per_1m",
)


def _provider_config(model_config: Any, provider_name: str | None) -> Any:
    """The provider mapping ``provider_name`` names, or ``None`` when the model
    has no mapping for it (or the caller named no provider)."""
    if not provider_name:
        return None
    return next(
        (p for p in model_config.providers if p.provider == provider_name),
        None,
    )


def _effective_overlay(model_config: Any, provider_name: str | None, attribute: str) -> Any:
    """A model's price overlay, unless its provider mapping carries its own.

    A provider mapping may override the model's tiers, windows or service-tier
    rates (the same model can be priced differently per upstream); when it does,
    it replaces the model-level value wholesale rather than extending it. The
    three states matter: ``None`` means the provider sets nothing and inherits
    the model-level value, while an empty list/map is a provider that explicitly
    has no overlay (an upstream with no off-peak rates, say) and must not fall
    back — hence an identity check, not a truthiness test.
    """
    provider_config = _provider_config(model_config, provider_name)
    if provider_config is not None:
        value = getattr(provider_config, attribute, None)
        if value is not None:
            return value
    return getattr(model_config, attribute, None)


def _get_provider_pricing(
    model_config: Any,
    provider_name: str | None,
) -> PricingRates:
    """Get pricing for a specific provider, falling back to model-level pricing."""
    provider_config = _provider_config(model_config, provider_name)
    return PricingRates(
        **{
            field: (
                provider_value
                if provider_config is not None
                and (provider_value := getattr(provider_config, field, None)) is not None
                else getattr(model_config, field, None)
            )
            for field in _PRICING_FIELDS
        }
    )


def _coerce_pricing_tier(raw: Any) -> CostTier | None:
    """Coerce a config model or stored JSON dict into a :class:`CostTier`.

    Returns ``None`` for entries without a usable threshold so malformed rows
    are ignored instead of crashing the billing path.
    """
    if isinstance(raw, CostTier):
        return raw

    threshold = coerce_float(read_field(raw, "threshold"))
    if threshold is None:
        logger.warning(f"Ignoring pricing tier without threshold: {raw!r}")
        return None
    return CostTier(
        threshold=int(threshold),
        **{field: coerce_float(read_field(raw, field)) for field in _OVERLAY_RATE_FIELDS},
    )


def _get_pricing_tiers(
    model_config: Any,
    provider_name: str | None,
) -> list[CostTier]:
    """Get the effective tier list: provider tiers win over model-level ones."""
    raw_tiers = _effective_overlay(model_config, provider_name, "pricing_tiers")
    if not raw_tiers:
        return []
    return [tier for tier in (_coerce_pricing_tier(raw) for raw in raw_tiers) if tier is not None]


def _overlay(base: PricingRates, source: CostTier | PricingWindow | dict[str, Any]) -> PricingRates:
    """Replace the rates ``source`` sets, keeping every rate it leaves unset."""
    overrides = {
        field: value
        for field in _OVERLAY_RATE_FIELDS
        if (value := _overlay_rate(source, field)) is not None
    }
    return replace(base, **overrides) if overrides else base


def _overlay_rate(source: CostTier | PricingWindow | dict[str, Any], field: str) -> float | None:
    """One overlay rate, read from a config/JSON overlay of either shape."""
    return coerce_float(read_field(source, field))


def _select_tier_rates(
    base: PricingRates,
    tiers: list[CostTier],
    input_tokens: int,
) -> PricingRates:
    """Overlay the highest tier the input token count has reached onto base rates.

    Thresholds are inclusive: a request with exactly ``threshold`` input tokens
    already bills at that band. Unset tier dimensions inherit the base rate.
    """
    selected: CostTier | None = None
    for tier in sorted(tiers, key=lambda t: t.threshold):
        if input_tokens >= tier.threshold:
            selected = tier
    if selected is None:
        return base
    return _overlay(base, selected)


def _coerce_pricing_window(raw: Any) -> PricingWindow | None:
    """Coerce a config model or stored JSON dict into a :class:`PricingWindow`.

    Returns ``None`` for entries whose clock, days or timezone cannot be read so
    one malformed row is ignored instead of failing every request that touches
    the model.
    """
    start = minutes_of_day(read_field(raw, "start"))
    end = minutes_of_day(read_field(raw, "end"))
    if start is None or end is None or start == end:
        logger.warning(f"Ignoring pricing time window without a valid start/end: {raw!r}")
        return None

    timezone = str(read_field(raw, "timezone") or "UTC")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as _:
        logger.warning(f"Ignoring pricing time window with unknown timezone {timezone!r}")
        return None

    raw_days = read_field(raw, "days") or []
    days = frozenset(day for day in raw_days if isinstance(day, int) and 0 <= day <= 6)

    return PricingWindow(
        start_minute=start,
        end_minute=end,
        days=days,
        timezone=timezone,
        **{field: coerce_float(read_field(raw, field)) for field in _OVERLAY_RATE_FIELDS},
    )


def _get_pricing_windows(
    model_config: Any,
    provider_name: str | None,
) -> list[PricingWindow]:
    """Get the effective window list: provider windows win over model-level ones."""
    raw_windows = _effective_overlay(model_config, provider_name, "pricing_time_windows")
    if not raw_windows:
        return []
    return [
        window
        for window in (_coerce_pricing_window(raw) for raw in raw_windows)
        if window is not None
    ]


def _window_applies(window: PricingWindow, at: datetime) -> bool:
    """Whether ``at`` falls inside ``window``, read on the window's own clock.

    The start day is resolved first: after midnight, the instant belongs to the
    weekday the window started on, not the weekday it is locally now.
    """
    moment = at if at.tzinfo is not None else at.replace(tzinfo=UTC)
    local = moment.astimezone(ZoneInfo(window.timezone))
    minutes = local.hour * 60 + local.minute

    crosses_midnight = window.start_minute > window.end_minute
    if crosses_midnight:
        # After midnight the instant belongs to the previous weekday's window.
        start_day = (local.weekday() - 1) % 7 if minutes < window.end_minute else local.weekday()
        inside = minutes >= window.start_minute or minutes < window.end_minute
    else:
        start_day = local.weekday()
        inside = window.start_minute <= minutes < window.end_minute

    if not inside:
        return False
    return not window.days or start_day in window.days


def _apply_time_window(
    base: PricingRates,
    windows: list[PricingWindow],
    at: datetime,
) -> PricingRates:
    """Overlay the first time window that covers ``at`` (peak/off-peak repricing).

    Windows in one list never overlap (validated at every boundary that accepts
    them), so the first match is also the only one.
    """
    for window in windows:
        if _window_applies(window, at):
            return _overlay(base, window)
    return base


def _get_service_tier_pricing(
    model_config: Any,
    provider_name: str | None,
) -> dict[str, Any]:
    """Get the effective service-tier map: provider map wins over model-level."""
    return _effective_overlay(model_config, provider_name, "service_tier_pricing") or {}


def _apply_service_tier(
    base: PricingRates,
    pricing: dict[str, Any],
    service_tier: str | None,
) -> PricingRates:
    """Overlay the rates of the service tier the request was served at.

    models.dev publishes these as ``experimental.modes`` (``flex``/``priority``);
    a mode missing from the map means the request keeps the rates the earlier
    overlays resolved, which is what an upstream serving ``default`` charges. Keys are stored
    lowercased by every writer, so one lowering matches them all.
    """
    if not service_tier or not pricing:
        return base
    rates = pricing.get(service_tier.strip().lower())
    return base if rates is None else _overlay(base, rates)


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


def _breakdown_from_usage(
    token_usage: TokenUsage,
    cost_usd: float | None,
    *,
    usage_reported: bool,
) -> CostBreakdown:
    """Build a CostBreakdown from extracted usage with the given total cost.

    ``usage_reported`` decides what a zero means for the usage-derived counts.
    With a record in hand, ``0`` is a reported zero (a prompt served entirely
    from cache, a turn that had produced no output yet) and must survive as
    ``0``; without one the count is unknown and stays ``None``. This is the same
    rule log intake applies to the request-log token columns
    (``store.token_column``), so a Realtime turn's row and the breakdown it
    is written from cannot disagree about "reported nothing" vs "reported zero".

    The proxy's own dimensions (images generated, characters synthesized,
    searches executed) are not provider facts and are known for every priced
    request, so they keep their previous treatment: a zero there is stored as
    ``None`` because the request-log columns they feed are not usage measurements.
    """

    def reported(value: int) -> int | None:
        return value if usage_reported else None

    def reported_seconds(value: float) -> float | None:
        return value if usage_reported else None

    return CostBreakdown(
        cost_usd=cost_usd,
        prompt_tokens=reported(token_usage.prompt_tokens),
        completion_tokens=reported(token_usage.completion_tokens),
        total_tokens=reported(token_usage.total_tokens),
        cache_creation_input_tokens=reported(token_usage.cache_creation_input_tokens),
        cache_read_input_tokens=reported(token_usage.cache_read_input_tokens),
        cached_prompt_tokens=reported(token_usage.cached_prompt_tokens),
        cache_savings_usd=None,
        audio_input_tokens=reported(token_usage.audio_input_tokens),
        audio_output_tokens=reported(token_usage.audio_output_tokens),
        image_input_tokens=reported(token_usage.image_input_tokens),
        images_generated=token_usage.images_generated or None,
        audio_duration_seconds=reported_seconds(token_usage.audio_duration_seconds),
        tts_characters=token_usage.tts_characters or None,
        web_search_requests=token_usage.web_search_requests or None,
    )


async def calculate_cost(
    usage: Usage | None,
    model_name: str | None,
    config_manager: DatabaseConfigManager | None = None,
    messages: list[dict[str, Any]] | None = None,
    completion_text: str | None = None,
    provider_name: str | None = None,
    dimensions: ProxyBillableDimensions | None = None,
    at: datetime | None = None,
    service_tier: str | None = None,
) -> CostBreakdown:
    """Calculate cost from a turn's canonical usage record and model pricing.

    Supports token-based billing (text/cache/reasoning/audio/image tokens) and
    unit-based billing (per image, per audio minute, per 1M TTS characters,
    per 1k web searches). ``dimensions`` carries the quantities the proxy
    itself produced, which the record does not hold (ADR-0025).

    Three overlays can replace the base token rates, applied in this order:

    1. the context tier the input token count reaches;
    2. the time-of-day window covering ``at`` — the request's **start**
       instant, which is what upstreams price a peak/off-peak request by;
    3. the service tier (``flex``/``priority``/...) the request was served at.

    ``at`` is deliberately not defaulted to "now": billing the instant the turn
    *finished* is exactly the mistake ADR-0053 rejects, and a caller that has no
    start instant must say so. Passing ``None`` leaves the time-window overlay
    off (base/tier rates stay in force), so a forgotten ``at`` under-reports an
    off-peak discount instead of silently pricing the wrong band.

    Each overlay only replaces the dimensions it sets, and a later overlay wins
    per dimension. The whole request is repriced at the resulting rates — none
    of the three is a marginal rate over the base.

    Returns a CostBreakdown; ``cost_usd`` is None when the cost cannot be
    determined. A count the provider reported as zero stays ``0`` — only the
    counts of a turn with no usage record at all are ``None``.
    """
    token_usage = token_usage_from(usage, dimensions)
    # A record means the provider answered with its own counters, so its zeros
    # are measurements. An estimate we compute in its absence is a value we
    # produced, and is stored as one.
    usage_reported = usage is not None

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
        usage_reported = True

    if not _has_billable_data(token_usage):
        return _breakdown_from_usage(token_usage, None, usage_reported=usage_reported)

    if not model_name or config_manager is None:
        return _breakdown_from_usage(token_usage, None, usage_reported=usage_reported)

    try:
        model_config = await config_manager.get_model_config(model_name)
        if not model_config:
            return _breakdown_from_usage(token_usage, None, usage_reported=usage_reported)

        # Get all pricing rates with provider-specific fallback
        rates = _get_provider_pricing(model_config, provider_name)
        tiers = _get_pricing_tiers(model_config, provider_name)
        windows = _get_pricing_windows(model_config, provider_name)
        tier_pricing = _get_service_tier_pricing(model_config, provider_name)

        # Model resolved but no pricing configured at any level: report None
        # ("unknown") rather than a fake $0.00 that silently under-reports
        # spend in usage stats. A model priced only by an overlay is not
        # "unpriced": its rates are resolved below, and the same check runs
        # again on the resolved rates so a window or tier that covers nothing
        # here still reports unknown.
        if (
            not tiers
            and not windows
            and not tier_pricing
            and all(getattr(rates, field) is None for field in _PRICING_FIELDS)
        ):
            return _breakdown_from_usage(token_usage, None, usage_reported=usage_reported)

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

        # Context-tier pricing keys off the request's whole input, before the
        # audio/image token carve-outs below shrink the billable text input.
        rates = _select_tier_rates(rates, tiers, effective_prompt_tokens)
        # Peak/off-peak, then the service tier actually served. Both key off
        # facts about the turn rather than its size, so they compose with the
        # context band by replacing only the dimensions they price. No start
        # instant means no window: never fall back to the completion time.
        if at is not None:
            rates = _apply_time_window(rates, windows, at)
        rates = _apply_service_tier(rates, tier_pricing, service_tier)
        if all(getattr(rates, field) is None for field in _PRICING_FIELDS):
            return _breakdown_from_usage(token_usage, None, usage_reported=usage_reported)

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

        # Reasoning tokens are already in completion_tokens, and a dedicated
        # reasoning rate carves them out so they are charged once at that rate.
        # Without a reasoning rate they keep billing at the output rate, which is
        # what every provider that does not publish one charges. The carve-out and
        # the charge below share one count, so a provider reporting more reasoning
        # tokens than completion tokens can never bill a token twice: only the
        # overlapping part is removed from the output count.
        reasoning_tokens = token_usage.reasoning_tokens or 0
        reasoning_cost = 0.0
        if reasoning_tokens > 0 and rates.reasoning_cost_per_1m is not None:
            effective_completion_tokens -= min(reasoning_tokens, effective_completion_tokens)
            reasoning_cost = (reasoning_tokens / 1_000_000) * rates.reasoning_cost_per_1m

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

        cost_usd = (
            input_cost
            + output_cost
            + reasoning_cost
            + cache_cost
            + audio_cost
            + image_token_cost
            + unit_cost
        )
        breakdown = _breakdown_from_usage(token_usage, cost_usd, usage_reported=usage_reported)
        breakdown.cache_savings_usd = cache_savings_usd
        return breakdown
    except Exception as e:
        logger.warning(
            f"Cost calculation failed for model={model_name}, provider={provider_name}: {e}"
        )
        return _breakdown_from_usage(token_usage, None, usage_reported=usage_reported)
