"""Provider configuration types."""

from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator

from llm_proxy.core.exceptions import ValidationError


class ProviderConfig(BaseModel):
    """Enhanced configuration for a single provider with extensibility support."""

    type: str = Field(..., description="Provider type")
    api_key: SecretStr = Field(default=SecretStr(""), description="API key for the provider.")

    def get_api_key(self) -> str:
        """Return the decrypted provider API key as a plain string."""
        return self.api_key.get_secret_value()

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if not v or not v.strip():
            raise ValidationError("Provider type cannot be empty")
        return v.strip()

    base_url: str | None = Field(None, description="Custom base URL")
    api_version: str | None = Field(None, description="API version")
    timeout: float = Field(600.0, description="Request timeout in seconds")
    rate_limit: int | None = Field(None, description="Rate limit (requests per minute)")
    custom_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers to include in requests to this provider",
    )
    provider_models: list[str] = Field(
        default_factory=list, description="List of available models for this provider"
    )
    enabled: bool = Field(default=True, description="Whether this provider is enabled")
    priority: int = Field(default=0, description="Priority for provider selection")
    parameter_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter overrides to enforce for all requests to this provider",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the provider"
    )
    endpoint_base_urls: dict[str, str] = Field(
        default_factory=dict,
        description="Per-endpoint base URL overrides",
    )
    definition_path: str | None = Field(
        default=None,
        description="Path to provider definition YAML/JSON file (optional)",
    )
    native_web_search: bool = Field(
        default=False,
        description="When True, web_search tools pass through to the upstream provider "
        "for native handling instead of being intercepted by the proxy.",
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v, info):
        if info.data.get("type") == "openai-compatible" and not v:
            raise ValidationError("base_url is required for openai-compatible providers")
        return v

    @field_validator("parameter_overrides")
    @classmethod
    def validate_parameter_overrides(cls, v):
        if not isinstance(v, dict):
            raise ValidationError("parameter_overrides must be a dictionary")
        for key in v:
            if not isinstance(key, str):
                raise ValidationError("Parameter override keys must be strings")
        return v
