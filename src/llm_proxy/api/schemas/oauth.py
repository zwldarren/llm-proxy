"""Schemas for external OIDC provider login and account linking."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from llm_proxy.api.schemas.common import validate_username_format
from llm_proxy.security.passwords import validate_password_strength


class OAuthProviderRead(BaseModel):
    id: str
    name: str


class OAuthProviderListResponse(BaseModel):
    providers: list[OAuthProviderRead]


class OAuthIdentityRead(BaseModel):
    id: int
    provider: str
    email: str
    created_at: datetime
    last_login_at: datetime | None = None


class OAuthIdentityListResponse(BaseModel):
    identities: list[OAuthIdentityRead]


class OAuthCompleteRequest(BaseModel):
    """Finish an OAuth invitation by choosing the canonical username."""

    completion_token: str = Field(..., min_length=1)
    username: str | None = Field(default=None, min_length=1, max_length=64)
    password: str | None = Field(default=None, min_length=8, max_length=72)

    @field_validator("username")
    @classmethod
    def _validate_username(cls, value: str | None) -> str | None:
        return validate_username_format(value) if value is not None else None

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str | None) -> str | None:
        return validate_password_strength(value) if value is not None else None


class OAuthLinkCompleteRequest(BaseModel):
    """Finish linking a provider identity that the callback already verified."""

    link_token: str = Field(..., min_length=1)
