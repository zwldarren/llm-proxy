"""Model configuration types."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from llm_proxy.core.exceptions import ValidationError


class ProviderSelectionStrategy(StrEnum):
    """Strategy for picking among same-priority providers of a model.

    Strategies only order candidates *within* the highest available priority
    group; higher-priority groups are always tried first. Per-request fallback
    walks the same ordering, skipping providers already tried.

    Attributes:
        RANDOM: Current default behavior — pick uniformly at random.
        SESSION_STICKY: Pin a conversation to one provider (cache affinity):
            same session keeps hitting the same provider so prompt caches stay
            warm. Uses Redis when available, rendezvous hashing otherwise.
        COST_OPTIMIZED: Pick the cheapest provider (per-mapping pricing with
            model-level fallback). Deterministic, so also cache-friendly.
        BALANCED: Score = 0.5 * normalized cost + 0.5 * normalized observed
            latency (in-memory EWMA); picks the best trade-off. Degrades to
            COST_OPTIMIZED while no latency samples exist.
    """

    RANDOM = "random"
    SESSION_STICKY = "session_sticky"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"


class ModelProviderConfig(BaseModel):
    """Configuration for a provider associated with a model."""

    provider: str = Field(..., description="Provider name")
    priority: int = Field(
        default=0,
        ge=0,
        description="Priority for provider selection (higher = preferred)",
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
    parameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter overrides to enforce for requests via this provider",
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
    supports_realtime: bool = Field(
        default=False,
        description="Whether this model is served through the Realtime WebSocket relay",
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
