"""Pydantic schemas for tracing configuration endpoints."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TracingProviderRead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "langfuse"
    name: str = "langfuse"
    enabled: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)
    masked_settings: dict[str, Any] = Field(default_factory=dict)


class TracingProviderWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    name: str
    enabled: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)


class TracingProviderStatus(BaseModel):
    """Status of a single tracing provider. Fields are intentionally required — a provider
    entry without name/provider is invalid and should not be silently accepted."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str


class TracingStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    providers: list[TracingProviderStatus] = Field(default_factory=list)
    is_configured: bool = False


def _mask_key(v: str | None) -> str | None:
    """Mask a key in responses (show first 4 and last 4 chars)."""
    if v is None:
        return None
    if len(v) > 8:
        return v[:4] + "****" + v[-4:]
    return "****"


def _mask_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Mask sensitive settings in responses."""
    if settings is None:
        return {}

    sensitive_keys = {"public_key", "secret_key", "api_key", "token"}
    result = {}
    for key, value in settings.items():
        if key == "headers" and isinstance(value, dict):
            result[key] = _mask_headers(value)
        elif isinstance(value, dict):
            result[key] = _mask_settings(value)
        elif key in sensitive_keys and isinstance(value, str):
            result[key] = _mask_key(value)
        else:
            result[key] = value
    return result


def _mask_headers(headers: dict[str, Any]) -> dict[str, Any]:
    """Mask sensitive HTTP header values (Authorization, keys, tokens, secrets)."""
    masked: dict[str, Any] = {}
    for key, value in headers.items():
        lower_key = key.lower()
        if lower_key in {"authorization", "proxy-authorization", "cookie"} or any(
            term in lower_key
            for term in {"api-key", "apikey", "auth-token", "access-token", "secret"}
        ):
            if isinstance(value, str):
                masked[key] = _mask_key(value)
            else:
                masked[key] = "****"
        else:
            masked[key] = value
    return masked


class TracingConfigRead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    providers: list[TracingProviderRead] = Field(default_factory=list)


class TracingConfigWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    providers: list[TracingProviderWrite] = Field(default_factory=list)


class TracingGetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: TracingConfigRead
    status: TracingStatus


class TracingUpdateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: TracingConfigRead
    status: TracingStatus
    message: str


class TracingProviderField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = "text"
    required: bool = False
    choices: list[str] | None = None
    default: Any | None = None
    description: str | None = None


class TracingProviderDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    required_fields: list[str] = Field(default_factory=list)
    optional_fields: list[str] = Field(default_factory=list)
    description: str | None = None
    fields: list[TracingProviderField] = Field(default_factory=list)


class TracingProvidersResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: list[TracingProviderDetails] = Field(default_factory=list)


__all__ = [
    "TracingProviderRead",
    "TracingProviderWrite",
    "TracingProviderStatus",
    "TracingStatus",
    "TracingConfigRead",
    "TracingConfigWrite",
    "TracingGetResponse",
    "TracingUpdateResponse",
    "TracingProviderDetails",
    "TracingProviderField",
    "TracingProvidersResponse",
    "_mask_settings",
]
