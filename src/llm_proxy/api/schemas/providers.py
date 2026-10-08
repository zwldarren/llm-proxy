"""Schemas for the providers resource.

Serves the admin provider config (``api/routers/config/providers.py``) and the
provider model listing (``api/routers/config/provider_models.py``).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from llm_proxy.config.types.provider import AppAttributionValidators
from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models.provider import ProviderModelInfo

from .common import ValidatorMixin
from .models import ModelRead


class AppAttributionSchema(BaseModel, AppAttributionValidators):
    """App-attribution identity for upstreams that track client applications.

    OpenRouter reads these to credit usage to an application on its public
    rankings; other upstreams ignore them. Unset values fall back to the
    adapter defaults. Values become raw HTTP header values, so they are
    validated as printable ASCII with an absolute http(s) URL and at most two
    categories (see :class:`AppAttributionValidators`).
    """

    url: str | None = Field(None, description="Application URL (OpenRouter's HTTP-Referer header)")
    title: str | None = Field(
        None, description="Application display name (OpenRouter's X-OpenRouter-Title header)"
    )
    categories: list[str] = Field(
        default_factory=list, description="Marketplace categories (OpenRouter only)"
    )
    visibility: Literal["public", "hidden"] | None = Field(
        None, description="Public listing for a newly created app (OpenRouter only)"
    )


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
    app_attribution: AppAttributionSchema = Field(
        default_factory=AppAttributionSchema,
        description="App-attribution identity for upstreams that track client applications",
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
    app_attribution: AppAttributionSchema | None = None


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


class ProviderModelsResponse(BaseModel):
    """Schema for provider models list response."""

    provider_name: str = Field(..., description="Name of the provider")
    provider_type: str = Field(..., description="Type of the provider")
    models: list[ProviderModelInfo] = Field(
        default_factory=list, description="List of available models"
    )
