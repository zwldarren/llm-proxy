"""Schemas for team member management."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from llm_proxy.api.schemas.me import validate_username_format
from llm_proxy.security.passwords import validate_password_strength


class TeamMemberCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=8, max_length=72)
    role: Literal["admin", "viewer"] = Field(default="viewer", description="User role")
    allowed_models: list[str] | None = Field(
        default=None,
        description="Model allowlist for this user (null = unrestricted)",
    )

    _validate_password = field_validator("password")(validate_password_strength)


class TeamMemberModelsUpdate(BaseModel):
    allowed_models: list[str] | None = Field(
        ...,
        description="Model allowlist for this user (null = unrestricted, [] = deny all)",
    )


class TeamMemberRoleUpdate(BaseModel):
    role: Literal["admin", "viewer"] = Field(..., description="New role for the member")


class TeamMemberBudgetUpdate(BaseModel):
    """Set or clear a member's account-level budget (admin-only).

    Only explicitly provided fields are changed. Explicit ``null`` for
    ``budget_usd`` clears the budget (and its window configuration).
    Contradictory combinations — window fields alongside a cap clear, or a
    reset day without a monthly window — are rejected by the endpoint with
    400, against the *effective* values (request merged over stored).
    """

    budget_usd: float | None = Field(
        None,
        gt=0,
        description="Account-level spending cap in USD, aggregated across all of the "
        "member's API keys. Explicit null clears the budget (unlimited).",
    )
    budget_period: Literal["daily", "weekly", "monthly"] | None = Field(
        None,
        description="Budget window (UTC calendar boundaries). Explicit null makes the "
        "budget a lifetime cap (cumulative spend since the last manual reset).",
    )
    budget_reset_day: int | None = Field(
        None,
        ge=1,
        le=31,
        description="Day of the month a monthly budget window restarts on (UTC). "
        "Explicit null restores the 1st. Only valid with a monthly budget_period.",
    )


class TeamMemberRead(BaseModel):
    id: int
    username: str
    role: Literal["admin", "viewer"]
    is_active: bool
    must_change_password: bool = False
    allowed_models: list[str] | None = None
    budget_usd: float | None = Field(
        None, description="Account-level spending cap in USD. Null means unlimited."
    )
    budget_period: str | None = Field(
        None, description="Budget window: 'daily', 'weekly', 'monthly', or null (lifetime)."
    )
    budget_reset_day: int | None = Field(
        None, description="Day of the month a monthly budget window restarts on (UTC)."
    )
    # Spend enrichment, filled in by the list endpoint for budgeted members.
    budget_spend_usd: float | None = Field(
        None, description="Spend in the current budget window. Null when no budget is set."
    )
    budget_period_start: datetime | None = Field(
        None, description="Start of the current budget window. Null when no budget is set."
    )
    created_at: datetime

    model_config = {"from_attributes": True}


class TeamMemberPasswordReset(BaseModel):
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
    )

    _validate_password = field_validator("password")(validate_password_strength)


class TeamMemberUsernameUpdate(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)

    _validate_username = field_validator("username")(validate_username_format)


class TeamMemberUsernameUpdateResponse(TeamMemberRead):
    """The renamed member, plus a fresh JWT when the admin renamed themselves.

    A self-rename invalidates the admin's current token (its `sub` claim still
    references the old username), so a replacement token is included in that
    case. It is None when renaming another member.
    """

    access_token: str | None = None
    token_type: str = "bearer"
