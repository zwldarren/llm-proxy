"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from llm_proxy.config.types.model import ProviderSelectionStrategy
from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models.provider import ProviderModelInfo
from llm_proxy.security.passwords import validate_password_strength
from llm_proxy.serialization.context import UnknownFieldsPolicy, UnsupportedBlockPolicy

from .logs import DailyUsage, UsageByModel, UsageSummary


class ValidatorMixin:
    """Shared field validators for Pydantic models.

    This mixin provides common validation logic to avoid duplication
    across multiple schema classes.
    """

    @field_validator(
        "model_metadata",
        "parameter_overrides",
        "server_metadata",
        "custom_headers",
        "provider_metadata",
        "endpoint_base_urls",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def convert_none_to_empty_dict(cls, v):
        return v if v is not None else {}

    @field_validator(
        "provider_models",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def convert_none_to_empty_list(cls, v):
        return v if v is not None else []

    @field_validator("base_url", mode="before", check_fields=False)
    @classmethod
    def convert_empty_base_url_to_none(cls, v):
        return None if v == "" else v


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
    parameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter overrides to enforce for requests via this provider",
    )


# --- Model Schemas ---


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
    supports_images: bool | None = None
    supports_image_generation: bool | None = None
    supports_tts: bool | None = None
    supports_stt: bool | None = None
    supports_embedding: bool | None = None
    supports_realtime: bool | None = None
    auto_eligible: bool | None = None
    quality_tier: str | None = None
    routing_assignments: list[str] | None = None
    icon_url: str | None = None
    description: str | None = None
    homepage_url: str | None = None
    context_length: int | None = Field(None, ge=0)

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
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "Model capabilities configured by an admin: 'vision' (image input), "
            "'image_generation', 'tts', 'stt', 'embedding'."
        ),
    )
    quality_tier: str | None = Field(
        None, description="Smart routing quality tier (ECONOMY | BALANCED | PREMIUM)"
    )
    provider_names: list[str] = Field(
        default_factory=list,
        description="Names of providers serving this model, ordered by priority",
    )

    model_config = ConfigDict(from_attributes=True)


class ProviderBase(BaseModel, ValidatorMixin):
    """Base schema for Provider configuration."""

    name: str = Field(..., description="Unique name of the provider")
    type: str = Field(..., description="Provider type (e.g., openai, anthropic)")
    base_url: str | None = Field(None, description="Base URL for the API")
    api_version: str | None = Field(None, description="API version")
    timeout: float = Field(300.0, description="Request timeout in seconds")
    rate_limit: int | None = Field(None, description="Rate limit (requests per minute)")
    custom_headers: dict[str, str] = Field(default_factory=dict, description="Custom headers")
    provider_models: list[str] = Field(default_factory=list, description="List of available models")
    enabled: bool = Field(default=True, description="Whether this provider is enabled")
    priority: int = Field(default=0, description="Priority for provider selection")
    provider_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    parameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter overrides to enforce for all requests to this provider",
    )
    endpoint_base_urls: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-endpoint base URL overrides. "
            "Use this when different endpoints need different base URLs. "
            "Keys are endpoint names (e.g., 'embeddings', 'chat_completion'), "
            "values are full URLs (e.g., 'https://model-specific.host.ai/v1/embeddings'). "
            "When set, the URL is used as-is without appending the endpoint path."
        ),
    )
    icon_url: str | None = Field(
        None,
        description="Optional URL to an icon image for this provider",
    )
    native_web_search: bool = Field(
        default=False,
        description="When True, web_search tools pass through to the upstream provider "
        "for native handling instead of being intercepted by the proxy.",
    )


class ProviderCreate(ProviderBase):
    """Schema for creating a new provider."""

    api_key: str = Field(
        default="",
        description="API key for the provider.",
    )

    @field_validator("name", mode="before")
    @classmethod
    def validate_name_not_empty(cls, v):
        if not v or (isinstance(v, str) and not v.strip()):
            raise ValidationError("Provider name cannot be empty")
        return v.strip() if isinstance(v, str) else v


class ProviderUpdate(BaseModel):
    """Schema for updating a provider."""

    type: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    timeout: float | None = None
    rate_limit: int | None = None
    custom_headers: dict[str, str] | None = None
    provider_models: list[str] | None = None
    enabled: bool | None = None
    priority: int | None = None
    provider_metadata: dict[str, Any] | None = None
    parameter_overrides: dict[str, Any] | None = None
    endpoint_base_urls: dict[str, str] | None = None
    icon_url: str | None = None
    native_web_search: bool | None = None


def mask_api_key(value: str | None) -> str:
    """Mask an API key for display (first 3 + last 4 chars).

    The plaintext key must never appear in admin API responses; this is the
    single masking helper for provider keys so the format stays consistent
    across list/detail/update responses.
    """
    if not value:
        return ""
    if len(value) > 8:
        return f"{value[:3]}...{value[-4:]}"
    return "***"


class ProviderRead(ProviderBase):
    """Schema for reading provider configuration (sanitized).

    ``api_key`` is never serialized: it only feeds the ``masked_api_key``
    computed field, so the plaintext key cannot leak into responses. The
    default keeps ``ProviderDetails(**model_dump())`` construction valid
    (the excluded field is absent from the dump).
    """

    id: int
    api_key: str = Field(default="", exclude=True)  # Never serialized

    @computed_field
    @property
    def masked_api_key(self) -> str:
        """Masked version of the API key for display purposes."""
        return mask_api_key(self.api_key)

    model_config = ConfigDict(from_attributes=True)


class ProviderDetails(ProviderRead):
    """Schema for reading full provider configuration including models."""

    models: list[ModelRead] = Field(default_factory=list, description="Configured models")


class ProviderKeyReveal(BaseModel):
    """Response for the explicit provider API key reveal endpoint.

    The plaintext key is only ever returned by this endpoint (never by
    list/detail/update responses, which carry ``masked_api_key``); every
    reveal is recorded in the audit log.
    """

    name: str
    api_key: str


class ProviderTypeRead(BaseModel):
    """Branding metadata for an available provider type (adapter).

    Served by the admin provider catalog (``GET /api/config/providers/provider-types``)
    so the frontend can render provider types without a per-provider static
    list. Names are localized server-side; ``icon_id``/``icon_variant`` feed
    the frontend's Lobe icon CDN URL builder (variant is "mono" or "color").

    Field names mirror ``core.adapter.ProviderTypeInfo`` one-to-one; the
    endpoint projects the dataclass onto this schema with ``model_validate``
    (``from_attributes``), keeping the core dataclass the single source of
    truth. Keep the two in sync when either changes.
    """

    model_config = ConfigDict(from_attributes=True)

    type: str
    name_en: str
    name_zh: str
    icon_id: str | None = None
    icon_variant: str = "color"


# --- Server Config Schemas ---


class LoggingConfigUpdate(BaseModel):
    """Schema for updating logging config."""

    log_input_output: bool = Field(default=True, description="Enable input/output logging")
    log_retention_days: int | None = Field(
        None, ge=0, description="Log retention days (0 = keep indefinitely)"
    )
    verbose_routing_logs: bool | None = Field(
        default=None,
        description="Include detailed per-candidate routing scorecards in request log metadata",
    )
    mask_sensitive_data: bool | None = Field(
        default=None, description="Mask sensitive fields in logs"
    )
    sampling_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Rate at which to log full request/response bodies",
    )
    audit_sampling_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling rate for audit logs (null = inherit sampling_rate)",
    )
    audit_retention_days: int | None = Field(
        default=None,
        ge=0,
        description="Retention days for audit logs (null = inherit log_retention_days)",
    )
    sensitive_keys: str | None = Field(
        default=None,
        description="Comma-separated extra key names to mask in logs",
    )


class SearXNGConfigUpdate(BaseModel):
    """Schema for updating SearXNG configuration."""

    url: str = Field(..., description="SearXNG instance URL")
    api_key: str | None = Field(None, description="API key if required")
    basic_auth_username: str | None = Field(None, description="Basic auth username")
    basic_auth_password: str | None = Field(None, description="Basic auth password")
    engines: list[str] | None = Field(None, description="Search engines to use")
    timeout: float = Field(30.0, ge=1.0, le=300.0, description="Request timeout in seconds")
    max_results: int = Field(10, ge=1, le=10, description="Maximum results per search")


class OllamaConfigUpdate(BaseModel):
    """Schema for updating Ollama web search configuration."""

    api_key: str = Field(..., description="Ollama API key")
    base_url: str = Field(
        "https://ollama.com",
        description="Ollama API base URL",
    )
    timeout: float = Field(30.0, ge=1.0, le=300.0, description="Request timeout in seconds")
    max_results: int = Field(10, ge=1, le=10, description="Maximum results per search")


class WebSearchConfigUpdate(BaseModel):
    """Schema for updating web search configuration."""

    enabled: bool = Field(False, description="Enable web search interception")
    provider: Literal["searxng", "ollama"] = Field("searxng", description="Search provider")
    searxng: SearXNGConfigUpdate | None = Field(None, description="SearXNG configuration")
    ollama: OllamaConfigUpdate | None = Field(None, description="Ollama configuration")


class RequestPolicyConfig(BaseModel):
    """Schema for global request policy configuration."""

    unknown_fields_policy: UnknownFieldsPolicy = Field(
        default="ignore",
        description=(
            "How to handle unknown request fields globally: "
            "'ignore' (strip fields silently), "
            "'passthrough' (keep unknown fields in body), "
            "'error' (reject request with validation error)"
        ),
    )
    unsupported_block_policy: UnsupportedBlockPolicy = Field(
        default="drop",
        description=(
            "How to handle content blocks the provider cannot serialize: "
            "'drop' (remove unsupported blocks silently), "
            "'degrade' (convert to a supported fallback representation), "
            "'error' (reject request with validation error)"
        ),
    )


class CircuitBreakerConfigSchema(BaseModel):
    """Schema for circuit breaker configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable circuit breaker to skip failing providers temporarily",
    )
    failure_threshold: int = Field(
        default=5,
        ge=1,
        description="Consecutive failures before a provider is skipped",
    )
    cooldown_seconds: float = Field(
        default=60.0,
        ge=1.0,
        description="Seconds before a skipped provider is probed again",
    )


class CorsConfig(BaseModel):
    """Schema for allowed CORS origins configuration.

    An empty list disables CORS (same-origin deployment). Origins should be
    full scheme+host(+port) values, e.g. "https://admin.example.com".
    """

    origins: list[str] = Field(
        default_factory=list,
        description="Allowed CORS origins; empty disables CORS",
    )


class RateLimitsConfig(BaseModel):
    """Schema for per-bucket rate limit overrides.

    Keys are bucket names (see DEFAULT_RATE_LIMITS in the rate limiting
    middleware); values are "N/period" specs, e.g. "5/minute".
    """

    limits: dict[str, str] = Field(
        default_factory=dict,
        description="Bucket name → 'N/period' rate limit spec",
    )


class KeepaliveConfig(BaseModel):
    """Schema for non-streaming response keepalive configuration."""

    enabled: bool = Field(default=False, description="Enable non-streaming keepalive heartbeats")
    grace_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Seconds to wait for normal completion before heartbeat mode",
    )
    interval_seconds: float = Field(
        default=15.0,
        gt=0,
        description="Interval between heartbeat bytes once in heartbeat mode",
    )


class SecurityConfig(BaseModel):
    """Schema for security / rate-limiting configuration (server_config ``security``)."""

    max_failed_login_attempts: int = Field(
        default=5, ge=1, description="Failed login attempts before account lockout"
    )
    lockout_duration_seconds: int = Field(
        default=900, ge=1, description="Account lockout duration in seconds"
    )
    max_failed_api_key_attempts: int = Field(
        default=10, ge=1, description="Failed API key attempts before IP lockout"
    )
    api_key_lockout_duration_seconds: int = Field(
        default=300, ge=1, description="API key lockout duration in seconds"
    )
    auth_failure_delay_ms: int = Field(
        default=100, ge=0, description="Artificial delay on failed authentication (ms)"
    )
    rate_limit_disabled: bool = Field(
        default=False, description="Disable all rate limiting (dangerous; testing only)"
    )
    redis_rate_limit_fail_closed: bool = Field(
        default=True,
        description="When the Redis rate limiter errors, block the request (true) "
        "or allow it through (false)",
    )
    hsts_enabled: bool = Field(
        default=True, description="Send the Strict-Transport-Security header"
    )
    hsts_max_age: int = Field(default=31536000, ge=0, description="HSTS max-age in seconds")
    max_request_body_size_bytes: int = Field(
        default=10 * 1024 * 1024, ge=0, description="Maximum request body size in bytes"
    )


class ResilienceConfig(BaseModel):
    """Schema for global resilience configuration (retry + fallback + circuit breaker)."""

    max_retries: int = Field(
        default=3,
        ge=0,
        description=(
            "Same-provider retry attempts for transient errors (rate limit, "
            "timeout, network) and server errors (5xx, 408). Client errors "
            "(401/403/400/422) are not retried in place -- they fall back "
            "directly. Overridable per-model via ModelConfig.max_retries"
        ),
    )
    max_fallback_attempts: int = Field(
        default=10,
        ge=0,
        description="Maximum number of fallback provider switches across all providers",
    )
    circuit_breaker: CircuitBreakerConfigSchema = Field(
        default_factory=CircuitBreakerConfigSchema,
        description="Circuit breaker configuration for provider fallback",
    )


class McpSecurityPolicyConfig(BaseModel):
    """Schema for MCP security policy configuration stored in the database.

    All policy fields (list-based rules and the permission switch) are
    UI-managed and persisted in the ``mcp_security_policy`` server_config key.
    """

    require_key_mcp_permissions: bool = Field(
        default=True,
        description="Require explicit MCP permissions on API keys for MCP access",
    )

    allowed_commands: list[str] = Field(
        default_factory=list,
        description="Commands permitted for stdio MCP servers",
    )
    blocked_commands: list[str] = Field(
        default_factory=lambda: [
            "bash",
            "sh",
            "zsh",
            "cmd.exe",
            "powershell.exe",
            "python",
            "python3",
            "node",
            "perl",
            "ruby",
        ],
        description="Commands always blocked even if in allowed list",
    )
    allowed_env_keys: list[str] = Field(
        default_factory=list,
        description="Environment variable keys permitted for MCP servers",
    )
    blocked_env_keys: list[str] = Field(
        default_factory=lambda: [
            "PATH",
            "LD_PRELOAD",
            "DYLD_INSERT_LIBRARIES",
            "PYTHONPATH",
            "NODE_OPTIONS",
            "SHELL",
            "HOME",
            "USER",
        ],
        description="Environment variable keys always blocked",
    )
    blocked_url_hosts: list[str] = Field(
        default_factory=list,
        description="URL hosts blocked for streamableHttp MCP servers",
    )
    blocked_url_ips: list[str] = Field(
        default_factory=lambda: [
            "127.0.0.0/8",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "169.254.169.254/32",
            "100.64.0.0/10",
            "::1/128",
            "fc00::/7",
            "fe80::/10",
            "::ffff:0:0/96",
        ],
        description="IP ranges blocked for streamableHttp MCP servers",
    )


# --- OpenAI Compatible Schemas ---


class OpenAIModel(BaseModel):
    """Schema for OpenAI-compatible model info."""

    id: str
    provider: str | None = None
    object: str = "model"


class OpenAIModelList(BaseModel):
    """Schema for OpenAI-compatible model list."""

    object: str = "list"

    data: list[OpenAIModel]


class LoginRequest(BaseModel):
    """Schema for admin login request."""

    username: str = Field(..., description="Admin username")
    password: str = Field(..., description="Admin password")


class LoginResponse(BaseModel):
    """Schema for admin login response."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    session_api_key: str = Field(default="", description="Session API key for /v1/* access")
    must_change_password: bool = Field(
        default=False,
        description="When true, the user must set a new password before any other API access",
    )


class SetupRequest(BaseModel):
    """Schema for first-run admin account creation."""

    username: str = Field(
        ..., min_length=1, max_length=100, description="Username for the admin account"
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="Password for the admin account (8-72 characters)",
    )

    _validate_password = field_validator("password")(validate_password_strength)


class SetupStatusResponse(BaseModel):
    """Schema indicating whether first-run setup is required."""

    needs_setup: bool = Field(
        ..., description="True if no admin account exists yet and setup is required"
    )


# --- Provider Models Schemas ---


class ProviderModelsResponse(BaseModel):
    """Schema for provider models list response."""

    provider_name: str = Field(..., description="Name of the provider")
    provider_type: str = Field(..., description="Type of the provider")
    models: list[ProviderModelInfo] = Field(
        default_factory=list, description="List of available models"
    )


# --- API Key Schemas ---


class ApiKeyCreate(BaseModel):
    """Schema for creating a new API key."""

    name: str = Field(..., min_length=1, max_length=255, description="Unique name for the API key")
    allowed_models: list[str] | None = Field(
        None, description="List of allowed model names. Empty/null means all models allowed."
    )
    allowed_mcp_servers: list[str] | None = Field(
        None,
        description="List of allowed MCP server names. Null means all MCP servers allowed; "
        "an empty list explicitly denies all.",
    )
    expires_at: datetime | None = Field(
        None,
        description="When the key expires (ISO 8601). Null means the key never expires.",
    )
    budget_usd: float | None = Field(
        None,
        gt=0,
        description="Spending cap in USD. Null means unlimited. Requests are rejected "
        "once the spend counted toward the budget reaches the cap.",
    )
    budget_period: Literal["daily", "weekly", "monthly"] | None = Field(
        None,
        description="Budget window (UTC calendar boundaries). Null means a lifetime "
        "budget: the cap applies to cumulative spend since the last manual reset.",
    )
    budget_reset_day: int | None = Field(
        None,
        ge=1,
        le=31,
        description="Day of the month a monthly budget window restarts on (UTC). "
        "Null means the 1st. Only valid with a monthly budget_period.",
    )
    rate_limit_rpm: int | None = Field(
        None,
        gt=0,
        description="Per-key request rate limit in requests per minute. Null means "
        "unlimited. Admin-only: non-admin users cannot set or change it.",
    )

    @model_validator(mode="after")
    def validate_budget_fields(self) -> ApiKeyCreate:
        """A period without a cap is meaningless; a reset day needs a monthly window."""
        if self.budget_usd is None and self.budget_period is not None:
            raise ValidationError("budget_period requires budget_usd to be set")
        if self.budget_reset_day is not None and self.budget_period != "monthly":
            raise ValidationError("budget_reset_day requires a monthly budget_period")
        return self


class ApiKeyRead(BaseModel):
    """Schema for reading API key metadata (without the key value)."""

    name: str = Field(..., description="Unique name for the API key")
    allowed_models: list[str] | None = Field(None, description="List of allowed model names")
    allowed_mcp_servers: list[str] | None = Field(
        None, description="List of allowed MCP servers. Null means all MCP servers allowed."
    )
    user_id: int = Field(..., description="ID of the user who owns this key")
    created_at: datetime = Field(..., description="When the API key was created")
    last_used_at: datetime | None = Field(None, description="When the API key was last used")
    is_active: bool = Field(..., description="Whether the API key is active")
    expires_at: datetime | None = Field(
        None, description="When the key expires. Null means it never expires."
    )
    budget_usd: float | None = Field(
        None, description="Spending cap in USD per budget period. Null means unlimited."
    )
    budget_period: str | None = Field(
        None, description="Budget window: 'daily', 'weekly', 'monthly', or null (lifetime)."
    )
    budget_reset_day: int | None = Field(
        None, description="Day of the month a monthly budget window restarts on (UTC)."
    )
    budget_reset_at: datetime | None = Field(
        None, description="Manual reset point for the current budget window."
    )
    rate_limit_rpm: int | None = Field(
        None, description="Per-key request rate limit (requests/minute). Null means unlimited."
    )

    model_config = ConfigDict(from_attributes=True)


class ApiKeyResponse(BaseModel):
    """Schema for API key creation response (includes plain text key, shown once)."""

    name: str = Field(..., description="Unique name for the API key")
    key: str = Field(..., description="The API key (only shown once on creation)")
    allowed_models: list[str] | None = Field(None, description="List of allowed model names")
    allowed_mcp_servers: list[str] | None = Field(
        None, description="List of allowed MCP servers. Null means all MCP servers allowed."
    )
    created_at: datetime = Field(..., description="When the API key was created")
    expires_at: datetime | None = Field(
        None, description="When the key expires. Null means it never expires."
    )
    budget_usd: float | None = Field(
        None, description="Spending cap in USD per budget period. Null means unlimited."
    )
    budget_period: str | None = Field(None, description="Budget window for the spending cap.")
    budget_reset_day: int | None = Field(
        None, description="Day of the month a monthly budget window restarts on (UTC)."
    )
    rate_limit_rpm: int | None = Field(
        None, description="Per-key request rate limit (requests/minute). Null means unlimited."
    )
    message: str = Field(default="Save this key now. It will not be shown again.")


class ApiKeyUpdateModels(BaseModel):
    """Schema for updating API key model restrictions."""

    allowed_models: list[str] | None = Field(
        None, description="List of allowed model names. Empty/null means all models allowed."
    )
    allowed_mcp_servers: list[str] | None = Field(
        None,
        description="List of allowed MCP server names. Null means all MCP servers allowed; "
        "an empty list explicitly denies all.",
    )


class ApiKeyUpdate(BaseModel):
    """Schema for updating an API key's name, restrictions, status, expiry, or budget.

    Only explicitly provided fields are changed (see ``exclude_unset`` usage in
    the router). Explicitly passing ``null`` for ``expires_at`` or ``budget_usd``
    clears the expiry / budget.
    """

    name: str | None = Field(
        None, min_length=1, max_length=255, description="New unique name for the API key"
    )
    allowed_models: list[str] | None = Field(
        None, description="List of allowed model names. Empty/null means all models allowed."
    )
    allowed_mcp_servers: list[str] | None = Field(
        None,
        description="List of allowed MCP server names. Null means all MCP servers allowed; "
        "an empty list explicitly denies all.",
    )
    is_active: bool | None = Field(
        None, description="Set to false to disable the key, true to re-enable it."
    )
    expires_at: datetime | None = Field(
        None, description="New expiry time. Explicit null clears the expiry."
    )
    budget_usd: float | None = Field(
        None,
        gt=0,
        description="New spending cap in USD. Explicit null clears the budget (and its "
        "window configuration).",
    )
    budget_period: Literal["daily", "weekly", "monthly"] | None = Field(
        None,
        description="New budget window. Explicit null makes the budget a lifetime cap "
        "(cumulative spend since the last manual reset).",
    )
    budget_reset_day: int | None = Field(
        None,
        ge=1,
        le=31,
        description="Day of the month a monthly budget window restarts on (UTC). Explicit "
        "null restores the 1st. Only valid when the effective window is monthly.",
    )
    rate_limit_rpm: int | None = Field(
        None,
        gt=0,
        description="New per-key request rate limit (requests/minute). Explicit null "
        "clears the limit. Admin-only: non-admin users cannot set or change it.",
    )

    @model_validator(mode="after")
    def validate_budget_fields(self) -> ApiKeyUpdate:
        """Reject contradictory budget fields within a single request.

        Clearing the cap while setting a window (or reset day) is contradictory.
        A reset day requires the effective window to be monthly: when the period
        is provided in the same request that is checked here, otherwise the
        router checks the stored period. Likewise, a non-null period without a
        cap in the same request is only valid when the stored key already has a
        cap — the router enforces that against the stored record.
        """
        provided = self.model_fields_set
        usd_cleared = "budget_usd" in provided and self.budget_usd is None
        period_set = "budget_period" in provided and self.budget_period is not None
        reset_day_set = "budget_reset_day" in provided and self.budget_reset_day is not None
        if usd_cleared and (period_set or reset_day_set):
            raise ValidationError("budget_usd and budget_period must be set together")
        if reset_day_set and "budget_period" in provided and self.budget_period != "monthly":
            raise ValidationError("budget_reset_day requires a monthly budget_period")
        return self


class ApiKeyDeleteResponse(BaseModel):
    """Schema for API key deletion response."""

    name: str = Field(..., description="Name of the deleted API key")
    message: str = Field(..., description="Human-readable message about the deletion")


class ApiKeySpendSummary(BaseModel):
    """Per-key spend summary for the API key list view.

    ``period_spend_usd`` / ``period_start`` are only present when the key has a
    budget configured (they describe the current budget window).
    """

    name: str = Field(..., description="API key name")
    total_spend_usd: float = Field(..., description="All-time endpoint spend in USD")
    total_requests: int = Field(..., description="All-time billable request count")
    period_spend_usd: float | None = Field(
        None, description="Spend in the current budget window. Null when no budget is set."
    )
    period_start: datetime | None = Field(
        None, description="Start of the current budget window. Null when no budget is set."
    )
    budget_usd: float | None = Field(None, description="Configured spending cap, if any")
    budget_period: str | None = Field(None, description="Configured budget window, if any")
    budget_reset_day: int | None = Field(None, description="Configured monthly reset day, if any")


class ApiKeyUsageResponse(BaseModel):
    """Detailed usage for a single API key over a date range."""

    summary: UsageSummary
    by_model: list[UsageByModel]
    daily_usage: list[DailyUsage]


# --- MCP Server Schemas ---


class McpServerBase(BaseModel, ValidatorMixin):
    """Base schema for MCP server configuration."""

    name: str = Field(..., description="Unique name of the MCP server")
    type: str = Field(..., description="Transport type: 'stdio' or 'streamableHttp'")
    command: str | None = Field(None, description="Command to execute (for stdio type)")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    base_url: str | None = Field(None, description="Base URL (for streamableHttp type)")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    enabled: bool = Field(default=True, description="Whether this server is enabled")


class McpServerCreate(McpServerBase):
    """Schema for creating an MCP server."""

    @model_validator(mode="after")
    def validate_type_requirement(self) -> McpServerCreate:
        """Validate that required fields are present based on type."""
        if self.type == "stdio" and not self.command:
            raise ValidationError("Command is required for stdio type")
        if self.type == "streamableHttp" and not self.base_url:
            raise ValidationError("Base URL is required for streamableHttp type")
        return self


class McpServerUpdate(BaseModel):
    """Schema for updating an MCP server."""

    type: str | None = None
    command: str | None = None
    args: list[str] | None = None
    base_url: str | None = None
    env: dict[str, str] | None = None
    enabled: bool | None = None

    @model_validator(mode="after")
    def validate_type_requirement(self) -> McpServerUpdate:
        """Validate that required fields are present based on type."""
        if self.type == "stdio" and self.command is not None and not self.command:
            raise ValidationError("Command is required for stdio type")
        if self.type == "streamableHttp" and self.base_url is not None and not self.base_url:
            raise ValidationError("Base URL is required for streamableHttp type")
        return self


class McpServerRead(McpServerBase):
    """Schema for reading MCP server configuration."""

    id: int
    proxy_url: str | None = None
    server_metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(..., description="When the server was created")
    updated_at: datetime = Field(..., description="When the server was last updated")
    status: str | None = Field(None, description="Runtime status: 'running', 'stopped', 'error'")

    model_config = ConfigDict(from_attributes=True)


class McpServerStatus(BaseModel):
    """Schema for MCP server runtime status."""

    name: str = Field(..., description="Server name")
    status: Literal["running", "stopped", "error"] = Field(..., description="Current status")
    proxy_url: str | None = Field(None, description="Proxy URL if running")
    uptime_seconds: float | None = Field(None, description="Uptime in seconds")
    error_message: str | None = Field(None, description="Error message if status is error")


class McpCapability(BaseModel):
    name: str = Field(..., description="Resource name")
    description: str | None = Field(None, description="Resource description")


class McpServerCapabilities(BaseModel):
    tools: list[McpCapability] = Field(default_factory=list)
    prompts: list[McpCapability] = Field(default_factory=list)
    resources: list[McpCapability] = Field(default_factory=list)


class SmartRoutingConfigUpdate(BaseModel):
    enabled: bool | None = Field(default=None)
    mode_weights: dict[str, float] | None = Field(
        default=None,
        description="Weights for each routing mode (fast, auto, best)",
    )


class ProviderSelectionConfigUpdate(BaseModel):
    strategy: ProviderSelectionStrategy | None = Field(
        default=None,
        description=(
            "How to pick among same-priority providers: 'random' (default), "
            "'session_sticky' (pin a conversation to one provider for cache affinity), "
            "'cost_optimized' (cheapest first), 'balanced' (cost + observed latency)"
        ),
    )
