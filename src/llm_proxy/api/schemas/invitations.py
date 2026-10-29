"""Schemas for administrator-created email invitations."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from llm_proxy.api.schemas.common import validate_email, validate_username_format
from llm_proxy.security.passwords import validate_password_strength


class InvitationCreate(BaseModel):
    """Create a single-use invitation for an email address."""

    email: str = Field(..., max_length=320)
    role: Literal["admin", "viewer"] = "viewer"
    allowed_models: list[str] | None = None
    budget_usd: float | None = Field(default=None, gt=0)
    budget_period: Literal["daily", "weekly", "monthly"] | None = None
    budget_reset_day: int | None = Field(default=None, ge=1, le=31)

    _validate_email = field_validator("email")(validate_email)


class InvitationRead(BaseModel):
    """An invitation row as shown in the team view."""

    id: str
    email: str
    role: Literal["admin", "viewer"]
    status: Literal["pending", "accepted", "revoked", "expired"]
    allowed_models: list[str] | None = None
    budget_usd: float | None = None
    budget_period: str | None = None
    budget_reset_day: int | None = None
    created_at: datetime
    expires_at: datetime
    accepted_at: datetime | None = None
    revoked_at: datetime | None = None
    last_sent_at: datetime | None = None


class InvitationListResponse(BaseModel):
    items: list[InvitationRead]
    total: int


class InvitationCreatedResponse(BaseModel):
    """The invitation plus the one-time plaintext token and ready-to-copy URL."""

    invitation: InvitationRead
    token: str
    invite_url: str
    email_sent: bool = False


class InvitationPublic(BaseModel):
    """What the invitee may see before accepting (no token hash, no role data leak)."""

    email: str
    role: Literal["admin", "viewer"]
    expires_at: datetime


class InvitationAccept(BaseModel):
    """Accept an invitation and create the account."""

    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=8, max_length=72)

    _validate_username = field_validator("username")(validate_username_format)
    _validate_password = field_validator("password")(validate_password_strength)
