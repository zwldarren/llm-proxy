"""Schemas for the models resource.

Three surfaces serve the same resource and share these shapes: the admin model
config (``api/routers/config/models.py``), the read-only model catalog
(``api/routers/catalog.py``) and the OpenAI-compatible model list
(``api/routers/models.py``, ``GET /v1/models``).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from llm_proxy.config.types.model import (
    SUPPORTS_SYSTEMONE_DESCRIPTION,
    normalize_pricing_tiers,
)

from .common import ValidatorMixin


class PricingTier(BaseModel):
    """Context-based pricing tier.

    Applies when a request's input token count reaches ``threshold``. Unset
    rates inherit the model/provider base rate for the same dimension.
    """

    threshold: int = Field(
        ..., ge=0, description="Input token count at which this tier starts applying"
    )
    input_cost_per_1m: float | None = Field(
        None, ge=0, description="Cost per 1M input tokens in USD"
    )
    output_cost_per_1m: float | None = Field(
        None, ge=0, description="Cost per 1M output tokens in USD"
    )
    cached_read_cost_per_1m: float | None = Field(
        None, ge=0, description="Cost per 1M cached read tokens in USD"
    )
    cached_write_cost_per_1m: float | None = Field(
        None, ge=0, description="Cost per 1M cached write tokens in USD"
    )
    audio_input_cost_per_1m: float | None = Field(
        None, ge=0, description="Cost per 1M audio input tokens in USD"
    )
    audio_output_cost_per_1m: float | None = Field(
        None, ge=0, description="Cost per 1M audio output tokens in USD"
    )
    image_input_cost_per_1m: float | None = Field(
        None, ge=0, description="Cost per 1M image input tokens in USD"
    )


class ModelProviderMapping(BaseModel, ValidatorMixin):
    """Schema for a provider mapping within a model configuration."""

    provider_name: str = Field(..., description="Name of the provider")
    priority: int = Field(
        default=0,
        ge=0,
        description="Priority for provider selection (higher = preferred)",
    )
    provider_model_name: str = Field(
        ...,
        description="The model name to use with this provider (e.g., 'gpt-4o', 'claude-3-opus')",
    )
    input_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description=(
            "Cost per 1M input tokens in USD for this specific provider "
            "(overrides model-level pricing)"
        ),
    )
    output_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description=(
            "Cost per 1M output tokens in USD for this specific provider "
            "(overrides model-level pricing)"
        ),
    )
    cached_read_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Cost per 1M cached read tokens in USD (overrides model pricing)",
    )
    cached_write_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Cost per 1M cached write tokens in USD (overrides model pricing)",
    )
    audio_input_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Cost per 1M audio input tokens in USD (overrides model pricing)",
    )
    audio_output_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Cost per 1M audio output tokens in USD (overrides model pricing)",
    )
    image_input_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Cost per 1M image input tokens in USD (overrides model pricing)",
    )
    cost_per_image: float | None = Field(
        None,
        ge=0,
        description="Cost per generated image in USD (overrides model pricing)",
    )
    audio_cost_per_minute: float | None = Field(
        None,
        ge=0,
        description="Cost per minute of audio (STT) in USD (overrides model pricing)",
    )
    tts_cost_per_1m_chars: float | None = Field(
        None,
        ge=0,
        description="Cost per 1M characters (TTS) in USD (overrides model pricing)",
    )
    web_search_cost_per_1k: float | None = Field(
        None,
        ge=0,
        description="Cost per 1k web search requests in USD (overrides model pricing)",
    )
    pricing_tiers: list[PricingTier] | None = Field(
        None,
        description=(
            "Context-based pricing tiers (overrides model-level tiers): each entry "
            "applies from its input-token threshold up, with unset rates inherited"
        ),
    )
    parameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter overrides to enforce for requests via this provider",
    )

    @field_validator("pricing_tiers", mode="before")
    @classmethod
    def validate_pricing_tiers(cls, v):
        return normalize_pricing_tiers(v)


ModelStatus = Literal["beta", "deprecated"]


def normalize_model_status(v: object) -> ModelStatus | None:
    """Normalize model status to the models.dev vocabulary: 'beta' or 'deprecated'.

    Empty strings collapse to ``None``; anything outside the vocabulary raises.
    Also narrows stored ``str`` values (validated at write time) to the
    ``ModelStatus`` alias for typed read paths.
    """
    if v is None:
        return None
    value = str(v).strip().lower()
    if value == "":
        return None
    if value not in ("beta", "deprecated"):
        raise ValueError("status must be 'beta' or 'deprecated'")
    return value


def derive_model_capabilities(model: Any) -> list[str]:
    """Collect the display capabilities configured for a model.

    Proxy-bound flags (vision/realtime/...) come first, then informational
    models.dev attributes. Order here defines plaza badge order; the admin
    ``ModelRead`` computed field and the catalog endpoint both derive from
    this single home.
    """
    capabilities: list[str] = []
    if model.supports_images:
        capabilities.append("vision")
    if model.supports_image_generation:
        capabilities.append("image_generation")
    if model.supports_tts:
        capabilities.append("tts")
    if model.supports_stt:
        capabilities.append("stt")
    if model.supports_embedding:
        capabilities.append("embedding")
    if model.supports_realtime:
        capabilities.append("realtime")
    if model.supports_systemone:
        capabilities.append("systemone")
    # Informational models.dev attributes (display-only).
    if model.reasoning:
        capabilities.append("reasoning")
    if model.tool_call:
        capabilities.append("tool_call")
    if model.structured_output:
        capabilities.append("structured_output")
    if model.attachment:
        capabilities.append("attachment")
    if model.temperature:
        capabilities.append("temperature")
    if model.open_weights:
        capabilities.append("open_weights")
    if model.experimental:
        capabilities.append("experimental")
    return capabilities


class ModelBase(BaseModel, ValidatorMixin):
    """Base schema for Model configuration."""

    name: str = Field(
        ...,
        description="Model name used by clients to request this model (e.g., 'gpt-4', 'my-claude')",
    )
    providers: list[ModelProviderMapping] = Field(
        ...,
        description="List of providers with priorities",
        min_length=1,
    )
    timeout: float | None = Field(None, description="Request timeout override")
    max_retries: int | None = Field(None, description="Max retries override")
    model_metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    parameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter overrides to enforce for all requests to this model",
    )
    input_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description=(
            "Default cost per 1M input tokens in USD (used when provider-level pricing not set)"
        ),
    )
    output_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description=(
            "Default cost per 1M output tokens in USD (used when provider-level pricing not set)"
        ),
    )
    cached_read_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M cached read tokens in USD",
    )
    cached_write_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M cached write tokens in USD",
    )
    audio_input_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M audio input tokens in USD",
    )
    audio_output_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M audio output tokens in USD",
    )
    image_input_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M image input tokens in USD",
    )
    cost_per_image: float | None = Field(
        None,
        ge=0,
        description="Default cost per generated image in USD",
    )
    audio_cost_per_minute: float | None = Field(
        None,
        ge=0,
        description="Default cost per minute of audio (STT) in USD",
    )
    tts_cost_per_1m_chars: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M characters (TTS) in USD",
    )
    web_search_cost_per_1k: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1k web search requests in USD",
    )
    pricing_tiers: list[PricingTier] | None = Field(
        None,
        description=(
            "Context-based pricing tiers: each entry applies from its input-token "
            "threshold up, with unset rates inherited from the base pricing above"
        ),
    )
    auto_eligible: bool = Field(default=False)
    quality_tier: str | None = Field(default=None)
    icon_url: str | None = Field(
        None,
        description="Optional URL to an icon image for this model",
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
        description="Whether this is an embedding model (e.g. /v1/embeddings)",
    )
    supports_realtime: bool = Field(
        default=False,
        description="Whether this model is served through the Realtime WebSocket relay",
    )
    supports_systemone: bool = Field(
        default=False,
        description=SUPPORTS_SYSTEMONE_DESCRIPTION,
    )
    # Display-only attributes, named after their models.dev counterparts so
    # operators familiar with models.dev can map entries 1:1. These do not
    # gate proxy behavior.
    attachment: bool = Field(
        default=False,
        description="models.dev: attachment — whether this model supports file attachments",
    )
    reasoning: bool = Field(
        default=False,
        description="models.dev: reasoning — whether this model produces reasoning/thinking output",
    )
    tool_call: bool = Field(
        default=False,
        description="models.dev: tool_call — whether this model supports tool/function calling",
    )
    structured_output: bool = Field(
        default=False,
        description="models.dev: structured_output — supports JSON-schema output",
    )
    temperature: bool = Field(
        default=False,
        description="models.dev: temperature — whether this model supports temperature sampling",
    )
    experimental: bool = Field(
        default=False,
        description="models.dev: experimental — whether this model is experimental",
    )
    open_weights: bool = Field(
        default=False,
        description="models.dev: open_weights — whether this model has openly available weights",
    )
    status: ModelStatus | None = Field(
        None,
        description="models.dev: status — lifecycle status, 'beta' or 'deprecated'",
    )
    family: str | None = Field(
        None,
        description="models.dev: family — model family identifier (e.g. 'claude-sonnet')",
    )
    knowledge: str | None = Field(
        None,
        description="models.dev: knowledge — knowledge cutoff date (YYYY-MM-DD)",
    )
    release_date: str | None = Field(
        None,
        description="models.dev: release_date — release date (YYYY-MM-DD)",
    )
    max_output_tokens: int | None = Field(
        None,
        ge=0,
        description="models.dev: limit.output — maximum output tokens in a single response",
    )
    description: str | None = Field(
        None,
        description="Human-readable description shown in the model catalog",
    )
    homepage_url: str | None = Field(
        None,
        description="URL to the model's homepage or Hugging Face page",
    )
    context_length: int | None = Field(
        None,
        ge=0,
        description="Maximum context length in tokens",
    )
    routing_assignments: list[str] | None = Field(
        None,
        description="Smart routing assignments (virtual model names like 'auto', 'fast', 'best')",
    )

    @field_validator("pricing_tiers", mode="before")
    @classmethod
    def validate_pricing_tiers(cls, v):
        return normalize_pricing_tiers(v)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: object) -> str | None:
        """Constrain status to the models.dev vocabulary: 'beta' or 'deprecated'."""
        return normalize_model_status(v)

    @field_validator("homepage_url", mode="before")
    @classmethod
    def normalize_homepage_url(cls, v: str | None) -> str | None:
        """Reject non-http(s) URLs to prevent stored XSS via ``javascript:``/``data:``.

        Empty strings collapse to ``None``. Only ``http``/``https`` schemes are
        accepted; anything else raises a validation error.
        """
        if v is None:
            return None
        v = v.strip()
        if v == "":
            return None
        lowered = v.lower()
        if not (lowered.startswith("http://") or lowered.startswith("https://")):
            raise ValueError("homepage_url must be an http:// or https:// URL")
        return v


class ModelCreate(ModelBase):
    """Schema for creating a new model."""


class ModelUpdate(BaseModel):
    """Schema for updating a model."""

    name: str | None = None
    providers: list[ModelProviderMapping] | None = None
    timeout: float | None = None
    max_retries: int | None = None
    model_metadata: dict[str, Any] | None = None
    parameter_overrides: dict[str, Any] | None = None
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    cached_read_cost_per_1m: float | None = None
    cached_write_cost_per_1m: float | None = None
    audio_input_cost_per_1m: float | None = None
    audio_output_cost_per_1m: float | None = None
    image_input_cost_per_1m: float | None = Field(None, ge=0)
    cost_per_image: float | None = Field(None, ge=0)
    audio_cost_per_minute: float | None = Field(None, ge=0)
    tts_cost_per_1m_chars: float | None = Field(None, ge=0)
    web_search_cost_per_1k: float | None = Field(None, ge=0)
    pricing_tiers: list[PricingTier] | None = Field(
        None,
        description="Context-based pricing tiers for this model",
    )
    supports_images: bool | None = None
    supports_image_generation: bool | None = None
    supports_tts: bool | None = None
    supports_stt: bool | None = None
    supports_embedding: bool | None = None
    supports_realtime: bool | None = None
    supports_systemone: bool | None = None
    attachment: bool | None = None
    reasoning: bool | None = None
    tool_call: bool | None = None
    structured_output: bool | None = None
    temperature: bool | None = None
    experimental: bool | None = None
    open_weights: bool | None = None
    status: ModelStatus | None = None
    family: str | None = None
    knowledge: str | None = None
    release_date: str | None = None
    max_output_tokens: int | None = Field(None, ge=0)
    auto_eligible: bool | None = None
    quality_tier: str | None = None
    routing_assignments: list[str] | None = None
    icon_url: str | None = None
    description: str | None = None
    homepage_url: str | None = None
    context_length: int | None = Field(None, ge=0)

    @field_validator("pricing_tiers", mode="before")
    @classmethod
    def validate_pricing_tiers(cls, v):
        return normalize_pricing_tiers(v)

    @field_validator("homepage_url", mode="before")
    @classmethod
    def normalize_homepage_url(cls, v: str | None) -> str | None:
        """Reject non-http(s) URLs to prevent stored XSS via ``javascript:``/``data:``."""
        if v is None:
            return None
        v = v.strip()
        if v == "":
            return None
        lowered = v.lower()
        if not (lowered.startswith("http://") or lowered.startswith("https://")):
            raise ValueError("homepage_url must be an http:// or https:// URL")
        return v

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: object) -> str | None:
        """Constrain status to the models.dev vocabulary: 'beta' or 'deprecated'."""
        return normalize_model_status(v)


class ModelRead(BaseModel):
    """Schema for reading model configuration."""

    id: int
    name: str = Field(
        ...,
        description="Model name used by clients",
    )
    providers: list[ModelProviderMapping] = Field(
        ...,
        description="List of providers with priorities",
    )
    timeout: float | None = Field(None, description="Request timeout override")
    max_retries: int | None = Field(None, description="Max retries override")
    model_metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    parameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter overrides to enforce for all requests to this model",
    )
    input_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M input tokens in USD",
    )
    output_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M output tokens in USD",
    )
    cached_read_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M cached read tokens in USD",
    )
    cached_write_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M cached write tokens in USD",
    )
    audio_input_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M audio input tokens in USD",
    )
    audio_output_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M audio output tokens in USD",
    )
    image_input_cost_per_1m: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M image input tokens in USD",
    )
    cost_per_image: float | None = Field(
        None,
        ge=0,
        description="Default cost per generated image in USD",
    )
    audio_cost_per_minute: float | None = Field(
        None,
        ge=0,
        description="Default cost per minute of audio (STT) in USD",
    )
    tts_cost_per_1m_chars: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1M characters (TTS) in USD",
    )
    web_search_cost_per_1k: float | None = Field(
        None,
        ge=0,
        description="Default cost per 1k web search requests in USD",
    )
    pricing_tiers: list[PricingTier] | None = Field(
        None,
        description="Context-based pricing tiers for this model",
    )
    auto_eligible: bool = Field(default=False)
    quality_tier: str | None = Field(default=None)
    routing_assignments: list[str] | None = Field(
        default=None,
        description=(
            "Virtual model modes this model participates in: auto, fast, best. "
            "Null means all modes."
        ),
    )
    icon_url: str | None = Field(
        None,
        description="Optional URL to an icon image for this model",
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
        description="Whether this is an embedding model (e.g. /v1/embeddings)",
    )
    supports_realtime: bool = Field(
        default=False,
        description="Whether this model is served through the Realtime WebSocket relay",
    )
    supports_systemone: bool = Field(
        default=False,
        description=SUPPORTS_SYSTEMONE_DESCRIPTION,
    )
    attachment: bool = Field(
        default=False,
        description="models.dev: attachment — whether this model supports file attachments",
    )
    reasoning: bool = Field(
        default=False,
        description="models.dev: reasoning — whether this model produces reasoning/thinking output",
    )
    tool_call: bool = Field(
        default=False,
        description="models.dev: tool_call — whether this model supports tool/function calling",
    )
    structured_output: bool = Field(
        default=False,
        description="models.dev: structured_output — supports JSON-schema output",
    )
    temperature: bool = Field(
        default=False,
        description="models.dev: temperature — whether this model supports temperature sampling",
    )
    experimental: bool = Field(
        default=False,
        description="models.dev: experimental — whether this model is experimental",
    )
    open_weights: bool = Field(
        default=False,
        description="models.dev: open_weights — whether this model has openly available weights",
    )
    status: ModelStatus | None = Field(
        None,
        description="models.dev: status — lifecycle status, 'beta' or 'deprecated'",
    )
    family: str | None = Field(
        None,
        description="models.dev: family — model family identifier (e.g. 'claude-sonnet')",
    )
    knowledge: str | None = Field(
        None,
        description="models.dev: knowledge — knowledge cutoff date (YYYY-MM-DD)",
    )
    release_date: str | None = Field(
        None,
        description="models.dev: release_date — release date (YYYY-MM-DD)",
    )
    max_output_tokens: int | None = Field(
        None,
        ge=0,
        description="models.dev: limit.output — maximum output tokens in a single response",
    )
    description: str | None = Field(
        None,
        description="Human-readable description shown in the model catalog",
    )
    homepage_url: str | None = Field(
        None,
        description="URL to the model's homepage or Hugging Face page",
    )
    context_length: int | None = Field(
        None,
        ge=0,
        description="Maximum context length in tokens",
    )

    @computed_field
    @property
    def capabilities(self) -> list[str]:
        """Display capabilities derived from the flags above (single derivation home)."""
        return derive_model_capabilities(self)

    model_config = ConfigDict(from_attributes=True)


class ModelCatalogEntry(BaseModel):
    """Display-oriented model entry for the public model catalog.

    Exposes only the fields needed to present a model in the model plaza to
    any authenticated user (including viewers). Sensitive pricing and admin
    configuration are intentionally excluded.
    """

    name: str = Field(..., description="Model name used by clients to request this model")
    icon_url: str | None = Field(None, description="Optional icon image URL")
    description: str | None = Field(None, description="Human-readable description")
    homepage_url: str | None = Field(
        None, description="URL to the model's homepage or Hugging Face page"
    )
    context_length: int | None = Field(None, ge=0, description="Maximum context length in tokens")
    max_output_tokens: int | None = Field(
        None,
        ge=0,
        description="models.dev: limit.output — maximum output tokens in a single response",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "Model capabilities configured by an admin: 'vision' (image input), "
            "'image_generation', 'tts', 'stt', 'embedding', 'realtime', plus informational "
            "models.dev attributes ('attachment', 'reasoning', 'tool_call', "
            "'structured_output', 'temperature', 'open_weights', 'experimental')."
        ),
    )
    # Informational models.dev attributes (display-only; not proxy behavior).
    status: ModelStatus | None = Field(
        None, description="models.dev: status — 'beta' or 'deprecated'"
    )
    family: str | None = Field(None, description="models.dev: family — model family identifier")
    knowledge: str | None = Field(
        None, description="models.dev: knowledge — knowledge cutoff date (YYYY-MM-DD)"
    )
    release_date: str | None = Field(None, description="models.dev: release_date")
    quality_tier: str | None = Field(
        None, description="Smart routing quality tier (ECONOMY | BALANCED | PREMIUM)"
    )
    provider_names: list[str] = Field(
        default_factory=list,
        description="Names of providers serving this model, ordered by priority",
    )

    model_config = ConfigDict(from_attributes=True)


class OpenAIModel(BaseModel):
    """Schema for OpenAI-compatible model info."""

    id: str
    provider: str | None = None
    object: str = "model"


class OpenAIModelList(BaseModel):
    """Schema for OpenAI-compatible model list."""

    object: str = "list"

    data: list[OpenAIModel]
