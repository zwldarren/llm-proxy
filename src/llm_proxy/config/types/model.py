"""Model configuration types."""

import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator, model_validator

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.core.utils import read_field

#: Shared description for the ``supports_systemone`` capability flag. The config
#: model and both admin schemas declare the same field, so the copy lives here
#: once instead of drifting across three ``Field(description=...)`` sites.
SUPPORTS_SYSTEMONE_DESCRIPTION = (
    "Whether this model serves evaluation requests — System One "
    "(/v1/systemone) and Decisions (/v1/decisions), which the provider bridges "
    "onto whichever envelope its upstream speaks. Gates both endpoints, and "
    "gates being selected as the routing judge. Additive like the other "
    "capability flags"
)


class ProviderSelectionStrategy(StrEnum):
    """Strategy for picking among same-priority providers of a model.

    Strategies only order candidates *within* the highest available priority
    group; higher-priority groups are always tried first. Per-request fallback
    walks the same ordering, skipping providers already tried.

    Attributes:
        RANDOM: Pick uniformly at random — keeps the provider of a conversation
            rotating, so prompt caches are re-warmed on every turn.
        SESSION_STICKY: Pin a conversation to one provider (cache affinity):
            same session keeps hitting the same provider so prompt caches stay
            warm. Uses Redis when available, rendezvous hashing otherwise.
        COST_OPTIMIZED: Pick the cheapest provider (per-mapping pricing with
            model-level fallback). Deterministic, so also cache-friendly.
        BALANCED: Score = 0.5 * normalized cost + 0.5 * normalized observed
            latency (in-memory EWMA); picks the best trade-off with a small
            exploration share so no candidate's samples go stale. Degrades to
            COST_OPTIMIZED while no latency samples exist.
    """

    RANDOM = "random"
    SESSION_STICKY = "session_sticky"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"


#: The strategy every deployment starts with and the fallback for a missing or
#: unparseable ``server_config.provider_selection`` row. Session sticky is the
#: default because most traffic is multi-turn agent work, where re-warming a
#: prompt cache on a different provider every turn costs more than the even load
#: spread ``random`` buys; a request without a usable conversation key degrades
#: to the random ordering anyway.
DEFAULT_PROVIDER_SELECTION_STRATEGY = ProviderSelectionStrategy.SESSION_STICKY


class TokenRateOverrides(BaseModel):
    """Per-1M token rates an overlay (tier, time window, service tier) may replace.

    Every dimension is optional: an unset rate inherits the effective base rate
    of the same dimension, so an overlay only states what it actually changes.
    Unit-based prices (per image, per minute, per character, per search) are not
    overlayable — no upstream tiers them.
    """

    input_cost_per_1m: float | None = Field(
        default=None, ge=0, description="Cost per 1M input tokens in USD"
    )
    output_cost_per_1m: float | None = Field(
        default=None, ge=0, description="Cost per 1M output tokens in USD"
    )
    cached_read_cost_per_1m: float | None = Field(
        default=None, ge=0, description="Cost per 1M cached read tokens in USD"
    )
    cached_write_cost_per_1m: float | None = Field(
        default=None, ge=0, description="Cost per 1M cached write tokens in USD"
    )
    reasoning_cost_per_1m: float | None = Field(
        default=None, ge=0, description="Cost per 1M reasoning (thinking) tokens in USD"
    )
    audio_input_cost_per_1m: float | None = Field(
        default=None, ge=0, description="Cost per 1M audio input tokens in USD"
    )
    audio_output_cost_per_1m: float | None = Field(
        default=None, ge=0, description="Cost per 1M audio output tokens in USD"
    )
    image_input_cost_per_1m: float | None = Field(
        default=None, ge=0, description="Cost per 1M image input tokens in USD"
    )


class PricingTierConfig(TokenRateOverrides):
    """Context-based pricing tier.

    Applies when a request's input token count reaches ``threshold``. Any rate
    left unset falls back to the model/provider base rate it overrides, so a
    tier can change only the input/output price and inherit cache pricing.
    """

    threshold: int = Field(
        ..., ge=0, description="Input token count at which this tier starts applying"
    )


#: ``HH:MM`` on a 24-hour clock, the only form a time window accepts.
_TIME_OF_DAY_RE = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")


def minutes_of_day(value: Any) -> int | None:
    """Read a ``HH:MM`` wall-clock time as minutes since midnight.

    Returns ``None`` for anything else — a config validator reports the
    offending value, and the billing path can then ignore a malformed stored
    window instead of raising mid-request.
    """
    if not isinstance(value, str):
        return None
    match = _TIME_OF_DAY_RE.fullmatch(value)
    if match is None:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


class PricingTimeWindowConfig(TokenRateOverrides):
    """A time-of-day price band — peak or off-peak pricing.

    Several upstreams price by the wall clock (Volcengine Ark's peak/off-peak
    rates, DeepSeek's former off-peak discount), which models.dev cannot express:
    its cost schema has a single tier type, ``context``. A window therefore lives
    on the model/provider mapping and overlays the token rates of any request
    that **started** inside it; requests outside every window bill at the base
    rates, which is how a deployment states "standard rate, discounted overnight".

    ``end <= start`` means the window crosses midnight (``22:00``–``08:00``).
    ``days`` names the day the window *starts* on, so a Friday 22:00–08:00 window
    also covers Saturday's small hours.
    """

    start: str = Field(..., description="Window start, HH:MM in `timezone` (inclusive)")
    end: str = Field(
        ...,
        description=(
            "Window end, HH:MM in `timezone` (exclusive); a value at or before "
            "`start` wraps the window past midnight"
        ),
    )
    days: list[int] = Field(
        default_factory=list,
        description=(
            "ISO weekdays the window starts on (0=Monday .. 6=Sunday); empty means every day"
        ),
    )
    timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone the window's clock is read in, e.g. Asia/Shanghai. The "
            "request's start time is converted into it before the window is tested."
        ),
    )

    @field_validator("start", "end")
    @classmethod
    def validate_clock_time(cls, v: str) -> str:
        if minutes_of_day(v) is None:
            raise ValidationError("pricing time window start/end must be HH:MM (00:00-23:59)")
        return v

    @field_validator("days")
    @classmethod
    def validate_days(cls, v: list[int]) -> list[int]:
        if any(day < 0 or day > 6 for day in v):
            raise ValidationError("pricing time window days must be 0 (Monday) to 6 (Sunday)")
        return sorted(set(v))

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as _:
            raise ValidationError(f"unknown timezone for pricing time window: {v!r}") from None
        return v

    @model_validator(mode="after")
    def validate_not_empty(self) -> PricingTimeWindowConfig:
        if minutes_of_day(self.start) == minutes_of_day(self.end):
            raise ValidationError("pricing time window start and end must differ")
        return self


def _window_span(window: Any, day: int) -> tuple[int, int]:
    """Absolute-minute interval a window covers when it starts on ``day``."""
    start = minutes_of_day(read_field(window, "start"))
    end = minutes_of_day(read_field(window, "end"))
    if start is None or end is None:
        return (0, 0)
    start += day * 1440
    end += day * 1440
    if end <= start:
        end += 1440
    return (start, end)


def _window_timezone(window: Any) -> str:
    return str(read_field(window, "timezone") or "UTC")


#: Year the cross-timezone comparison reads each zone's UTC offsets from. A fixed
#: year keeps validation deterministic; a later DST rule can only move a boundary.
_OFFSET_REFERENCE_YEAR = 2025

#: Sampling step for the offset pairs two zones actually share. DST switches
#: happen at most twice a year, so a 6-hour grid cannot miss a regime.
_OFFSET_SAMPLE_STEP = timedelta(hours=6)


def _offset_minutes(moment: datetime, zone: ZoneInfo) -> int:
    """The zone's UTC offset at ``moment``, in minutes east of UTC."""
    offset = moment.astimezone(zone).utcoffset() or timedelta(0)
    return int(offset.total_seconds() // 60)


@lru_cache(maxsize=128)
def _coexisting_offsets(left_zone: str, right_zone: str) -> frozenset[tuple[int, int]]:
    """The UTC-offset pairs the two zones had at the same instant, over a year.

    Comparing windows across timezones in UTC needs each zone's offset, but a
    zone with DST has two. Testing every combination would invent pairings that
    never happen — London 09:00-10:00 and Berlin 09:00-10:00 would collide under a
    phantom (UTC+1, UTC+1) pairing that no date produces. Sampling the same
    instants keeps only the offsets that really coexisted, so the check rejects a
    genuine collision without rejecting disjoint regional bands.
    """
    left, right = ZoneInfo(left_zone), ZoneInfo(right_zone)
    moment = datetime(_OFFSET_REFERENCE_YEAR, 1, 1, tzinfo=UTC)
    end = datetime(_OFFSET_REFERENCE_YEAR + 1, 1, 1, tzinfo=UTC)
    pairs: set[tuple[int, int]] = set()
    while moment < end:
        pairs.add((_offset_minutes(moment, left), _offset_minutes(moment, right)))
        moment += _OFFSET_SAMPLE_STEP
    return frozenset(pairs)


def _windows_overlap(left: Any, right: Any) -> bool:
    """Whether two windows cover the same real instant (UTC).

    Windows in the same timezone are compared as they are written. Otherwise each
    span is shifted by its zone's offset into UTC, for every offset pair the two
    zones actually shared, which is what makes a Sunday 22:00-08:00 band in one
    zone collide with a Monday 00:00-02:00 band in another when they really do.
    """
    left_zone = _window_timezone(left)
    right_zone = _window_timezone(right)
    offset_pairs = (
        frozenset({(0, 0)})
        if left_zone == right_zone
        else _coexisting_offsets(left_zone, right_zone)
    )
    left_spans = _window_intervals(left)
    right_spans = _window_intervals(right)
    return any(
        left_start - left_offset < right_end - right_offset
        and right_start - right_offset < left_end - left_offset
        for left_offset, right_offset in offset_pairs
        for left_start, left_end in left_spans
        for right_start, right_end in right_spans
    )


def _window_intervals(window: Any) -> list[tuple[int, int]]:
    """Every absolute-minute span ``window`` covers, Monday 00:00 of week 0 = 0.

    A window is anchored on the weekday it *starts* on but may run past
    midnight, so two windows can collide even when their start-day sets are
    disjoint: a Sunday 22:00-08:00 band covers Monday 01:00, which a Monday
    00:00-02:00 band also covers. Enumerating each start day over two
    consecutive weeks turns that into an ordinary interval overlap and also
    covers a band that wraps past the week boundary.
    """
    days = read_field(window, "days")
    start_days = sorted(set(days)) if days else list(range(7))
    return [_window_span(window, anchor) for day in start_days for anchor in (day, day + 7)]


def normalize_pricing_time_windows(windows: Any) -> Any:
    """Validate and order the time windows of one price list.

    Windows are evaluated first-match-wins, so their order is the price list's
    own (sorted by timezone and start) and an overlap would make billing
    depend on list order — ambiguous, so it fails loudly. Overlaps are checked
    in UTC, over the spans each window actually covers: a window that crosses
    midnight also covers the next weekday's small hours, and two windows in
    different timezones can cover the same real instant even when their local
    clocks differ (a Sunday 22:00-08:00 band in one zone and a Monday
    00:00-02:00 band in another), so comparing only shared start days or shared
    timezones would miss it.

    A falsy value is returned unchanged — ``None`` keeps meaning "clear the
    field" — and a payload whose fields cannot be read is likewise returned
    unchanged, for field validation to report the offending path.
    """
    if not windows:
        return windows

    starts = [minutes_of_day(read_field(window, "start")) for window in windows]
    ends = [minutes_of_day(read_field(window, "end")) for window in windows]
    if any(start is None for start in starts) or any(end is None for end in ends):
        return windows

    # The cross-timezone comparison asks the tz database for an offset, so an
    # unreadable zone must be left to field validation instead of raising here.
    for window in windows:
        try:
            ZoneInfo(_window_timezone(window))
        except (ZoneInfoNotFoundError, ValueError) as _:
            return windows

    ordered = sorted(
        windows,
        key=lambda window: (
            _window_timezone(window),
            minutes_of_day(read_field(window, "start")) or 0,
        ),
    )

    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if _windows_overlap(left, right):
                raise ValidationError("pricing time windows must not overlap")

    return ordered


def normalize_service_tier_pricing(tiers: Any) -> Any:
    """Lowercase the service-tier keys so a lookup never misses on case.

    Keys are the mode names a provider publishes (``flex``, ``priority``,
    ``fast``); the request's ``service_tier`` is matched case-insensitively, so
    storing one spelling makes that contract hold for every writer — config load,
    the admin schemas and the models.dev sync alike.
    """
    if not tiers:
        return tiers
    if not isinstance(tiers, dict):
        return tiers
    return {str(key).strip().lower(): value for key, value in tiers.items() if str(key).strip()}


def _tier_threshold(tier: Any) -> int | None:
    """Threshold of a validated tier model or a raw tier payload."""
    return read_field(tier, "threshold")


def normalize_pricing_tiers(tiers: Any) -> Any:
    """Order tier bands ascending by threshold, rejecting duplicate thresholds.

    Duplicate thresholds are ambiguous (which band applies?) and would make
    billing order-dependent, so they fail loudly. Every boundary that accepts
    tiers runs this: config load validates models, the admin schemas validate
    raw payloads, and the pricing-apply schema validates its own tier options,
    so both shapes are accepted here.

    A falsy value is returned unchanged — ``None`` keeps meaning "clear the
    field" — and a payload whose thresholds cannot be read is likewise
    returned unchanged, for field validation to report the offending path.
    """
    if not tiers:
        return tiers

    thresholds: list[int] = []
    for tier in tiers:
        threshold = _tier_threshold(tier)
        if threshold is None:
            return tiers
        thresholds.append(threshold)

    if len(set(thresholds)) != len(thresholds):
        raise ValidationError("pricing_tiers thresholds must be unique")

    return [
        tier for _, tier in sorted(zip(thresholds, tiers, strict=True), key=lambda pair: pair[0])
    ]


class ModelProviderConfig(BaseModel):
    """Configuration for a provider associated with a model."""

    provider: str = Field(..., description="Provider name")
    priority: int = Field(
        default=0,
        ge=0,
        description="Priority for provider selection (higher = preferred)",
    )
    weight: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Relative share of traffic this mapping receives inside its priority group "
            "(1 = normal; 2 = twice as likely as a 1; 0 = only picked when nothing "
            "else in the group is left)"
        ),
    )
    provider_model_name: str | None = Field(
        default=None,
        description="Upstream model name for this specific provider",
    )
    input_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M input tokens in USD for this specific provider",
    )
    output_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M output tokens in USD for this specific provider",
    )
    cached_read_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M cached read tokens in USD for this specific provider",
    )
    cached_write_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M cached write tokens in USD for this specific provider",
    )
    audio_input_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M audio input tokens in USD for this specific provider",
    )
    audio_output_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M audio output tokens in USD for this specific provider",
    )
    image_input_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M image input tokens in USD for this specific provider",
    )
    cost_per_image: float | None = Field(
        default=None,
        ge=0,
        description="Cost per generated image in USD for this specific provider",
    )
    audio_cost_per_minute: float | None = Field(
        default=None,
        ge=0,
        description="Cost per minute of audio (STT) in USD for this specific provider",
    )
    tts_cost_per_1m_chars: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M characters (TTS) in USD for this specific provider",
    )
    web_search_cost_per_1k: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1k web search requests in USD for this specific provider",
    )
    reasoning_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M reasoning (thinking) tokens in USD for this provider",
    )
    pricing_tiers: list[PricingTierConfig] | None = Field(
        default=None,
        description=(
            "Context-based pricing tiers for this specific provider; null inherits the "
            "model-level list, an empty list disables it for this provider"
        ),
    )
    pricing_time_windows: list[PricingTimeWindowConfig] | None = Field(
        default=None,
        description=(
            "Time-of-day (peak/off-peak) price windows for this specific provider; null "
            "inherits the model-level list, an empty list disables it for this provider"
        ),
    )
    service_tier_pricing: dict[str, TokenRateOverrides] | None = Field(
        default=None,
        description=(
            "Per-service-tier price overrides keyed by mode name (flex, priority, ...); "
            "the request's service_tier selects one. Null inherits the model-level map, an "
            "empty map disables it for this provider"
        ),
    )
    parameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter overrides to enforce for requests via this provider",
    )

    @field_validator("pricing_tiers")
    @classmethod
    def validate_pricing_tiers(cls, v):
        return normalize_pricing_tiers(v)

    @field_validator("pricing_time_windows")
    @classmethod
    def validate_pricing_time_windows(cls, v):
        return normalize_pricing_time_windows(v)

    @field_validator("service_tier_pricing")
    @classmethod
    def validate_service_tier_pricing(cls, v):
        return normalize_service_tier_pricing(v)

    @field_validator("parameter_overrides")
    @classmethod
    def validate_parameter_overrides(cls, v):
        if not isinstance(v, dict):
            raise ValidationError("parameter_overrides must be a dictionary")
        for key in v:
            if not isinstance(key, str):
                raise ValidationError("Parameter override keys must be strings")
        return v


class ModelConfig(BaseModel):
    """Configuration for a single model with multi-provider support."""

    providers: list[ModelProviderConfig] = Field(
        ...,
        description="List of providers with priorities for fallback support",
        min_length=1,
    )
    model_name: str | None = Field(
        default=None,
        description="Default upstream provider's model name",
    )
    timeout: float | None = Field(default=None, description="Request timeout override")
    max_retries: int | None = Field(default=None, description="Max retries override")
    parameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter overrides to enforce for all requests to this model",
    )
    input_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M input tokens in USD",
    )
    output_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M output tokens in USD",
    )
    cached_read_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M cached read tokens in USD",
    )
    cached_write_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M cached write tokens in USD",
    )
    audio_input_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M audio input tokens in USD",
    )
    audio_output_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M audio output tokens in USD",
    )
    image_input_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M image input tokens in USD",
    )
    cost_per_image: float | None = Field(
        default=None,
        ge=0,
        description="Cost per generated image in USD",
    )
    audio_cost_per_minute: float | None = Field(
        default=None,
        ge=0,
        description="Cost per minute of audio (STT) in USD",
    )
    tts_cost_per_1m_chars: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M characters (TTS) in USD",
    )
    web_search_cost_per_1k: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1k web search requests in USD",
    )
    reasoning_cost_per_1m: float | None = Field(
        default=None,
        ge=0,
        description="Cost per 1M reasoning (thinking) tokens in USD",
    )
    pricing_tiers: list[PricingTierConfig] = Field(
        default_factory=list,
        description="Context-based pricing tiers (fallback when the provider sets none)",
    )
    pricing_time_windows: list[PricingTimeWindowConfig] = Field(
        default_factory=list,
        description=(
            "Time-of-day (peak/off-peak) price windows (fallback when the provider sets none)"
        ),
    )
    service_tier_pricing: dict[str, TokenRateOverrides] = Field(
        default_factory=dict,
        description=("Per-service-tier price overrides (fallback when the provider sets none)"),
    )
    auto_eligible: bool = Field(
        default=False, description="Eligible for smart routing candidate pool"
    )
    quality_tier: str | None = Field(
        default=None, description="Served quality tier: ECONOMY | BALANCED | PREMIUM"
    )
    routing_assignments: list[str] | None = Field(
        default=None,
        description=(
            "Virtual model modes this model participates in: auto, fast, best. "
            "Null means all modes."
        ),
    )
    supports_images: bool = Field(
        default=False,
        description="Whether this model supports image input",
    )
    supports_image_generation: bool = Field(
        default=False,
        description="Whether this is an image generation model",
    )
    supports_tts: bool = Field(
        default=False,
        description="Whether this is a text-to-speech model",
    )
    supports_stt: bool = Field(
        default=False,
        description="Whether this is a speech-to-text (transcription) model",
    )
    supports_embedding: bool = Field(
        default=False,
        description="Whether this is an embedding model",
    )
    supports_rerank: bool = Field(
        default=False,
        description="Whether this is a reranker model (e.g. /v1/rerank)",
    )
    supports_realtime: bool = Field(
        default=False,
        description="Whether this model is served through the Realtime WebSocket relay",
    )
    supports_systemone: bool = Field(
        default=False,
        description=SUPPORTS_SYSTEMONE_DESCRIPTION,
    )
    context_length: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Maximum context length in tokens; used for smart-routing context-capacity filtering"
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the model",
    )

    @field_validator("parameter_overrides")
    @classmethod
    def validate_parameter_overrides(cls, v):
        if not isinstance(v, dict):
            raise ValidationError("parameter_overrides must be a dictionary")
        for key in v:
            if not isinstance(key, str):
                raise ValidationError("Parameter override keys must be strings")
        return v

    @field_validator("pricing_tiers")
    @classmethod
    def validate_pricing_tiers(cls, v):
        return normalize_pricing_tiers(v)

    @field_validator("pricing_time_windows")
    @classmethod
    def validate_pricing_time_windows(cls, v):
        return normalize_pricing_time_windows(v)

    @field_validator("service_tier_pricing")
    @classmethod
    def validate_service_tier_pricing(cls, v):
        return normalize_service_tier_pricing(v)

    @field_validator("routing_assignments", mode="before")
    @classmethod
    def _normalize_empty_routing_assignments(cls, v):
        # An empty list means "no restriction" just like None, otherwise the
        # routing pool would exclude the model from every virtual mode.
        if v == []:
            return None
        return v

    def get_providers_by_priority(self) -> list[ModelProviderConfig]:
        return sorted(self.providers, key=lambda p: p.priority, reverse=True)
