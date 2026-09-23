"""Provider configuration types."""

import re
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, field_validator

from llm_proxy.core.exceptions import ValidationError

#: App-attribution values are sent as raw HTTP header values, whose encoding
#: only accepts printable ASCII. Reject control characters and non-ASCII at
#: config time instead of letting every request fail at runtime.
_HEADER_VALUE_UNSAFE = re.compile(r"[^\x20-\x7e]")

#: OpenRouter accepts at most two marketplace categories per application.
_MAX_ATTRIBUTION_CATEGORIES = 2


class AppAttributionValidators:
    """Shared field validation for app-attribution models.

    Mixed into both the config model (:class:`AppAttribution`) and the admin
    API schema so the two layers accept exactly the same values.
    """

    @field_validator("url", "title", mode="before", check_fields=False)
    @classmethod
    def _normalize_attribution_text(cls, v: Any) -> Any:
        if not isinstance(v, str):
            return v
        v = v.strip()
        if not v:
            return None
        if _HEADER_VALUE_UNSAFE.search(v):
            raise ValidationError(
                "App attribution values must be printable ASCII: they are sent "
                "as HTTP header values and cannot contain control or non-ASCII "
                "characters."
            )
        return v

    @field_validator("url", mode="after", check_fields=False)
    @classmethod
    def _validate_attribution_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if re.search(r"\s", v):
            raise ValidationError("App attribution URL must not contain whitespace.")
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValidationError("App attribution URL must be an absolute http(s) URL.")
        return v

    @field_validator("categories", mode="before", check_fields=False)
    @classmethod
    def _normalize_attribution_categories(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            v = [part.strip() for part in v.split(",") if part.strip()]
        if not isinstance(v, list):
            raise ValidationError("App attribution categories must be a list of strings.")
        categories: list[str] = []
        for item in v:
            if not isinstance(item, str):
                raise ValidationError("App attribution categories must be strings.")
            item = item.strip()
            if not item:
                continue
            if _HEADER_VALUE_UNSAFE.search(item):
                raise ValidationError(
                    "App attribution categories must be printable ASCII: they are "
                    "sent as HTTP header values and cannot contain control or "
                    "non-ASCII characters."
                )
            categories.append(item)
        if len(categories) > _MAX_ATTRIBUTION_CATEGORIES:
            raise ValidationError(
                f"At most {_MAX_ATTRIBUTION_CATEGORIES} app attribution categories are allowed."
            )
        return categories


class AppAttribution(BaseModel, AppAttributionValidators):
    """App-attribution identity sent to upstreams that track client apps.

    OpenRouter uses these to credit usage to an application on its public
    rankings (``HTTP-Referer`` + ``X-OpenRouter-Title``) and to categorise it
    in the app marketplace. Adapters for upstreams without an equivalent
    ignore the values. Every field is optional: unset values fall back to the
    adapter's defaults, and headers set explicitly in ``custom_headers`` win
    over both.
    """

    url: str | None = Field(
        None,
        description=(
            "Canonical URL of the application (OpenRouter's ``HTTP-Referer``). "
            "Required by OpenRouter to create an app page and appear in rankings."
        ),
    )
    title: str | None = Field(
        None, description="Application display name (OpenRouter's ``X-OpenRouter-Title``)"
    )
    categories: list[str] = Field(
        default_factory=list,
        description=(
            "OpenRouter marketplace categories (sent comma-separated as "
            "``X-OpenRouter-Categories``); at most two per request."
        ),
    )
    visibility: Literal["public", "hidden"] | None = Field(
        None,
        description=(
            "Whether a newly created OpenRouter app is listed publicly "
            "(``X-OpenRouter-App-Visibility``). Only applies when the app is "
            "first created; ignored for apps that already exist."
        ),
    )


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
    app_attribution: AppAttribution = Field(
        default_factory=AppAttribution,
        description="App-attribution identity for upstreams that track client applications",
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
