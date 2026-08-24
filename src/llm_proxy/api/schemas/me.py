"""Schemas for self-service endpoints."""

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from llm_proxy.security.passwords import validate_password_strength

# Matches the username format enforced by the team-management UI.
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_username_format(username: str) -> str:
    """Ensure a username only contains letters, digits, underscores, and dashes."""
    if not USERNAME_PATTERN.match(username):
        raise ValueError("Username may only contain letters, digits, underscores, and hyphens")
    return username


class MeProfile(BaseModel):
    id: int
    username: str
    role: str
    must_change_password: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


class MePasswordChange(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=72)

    _validate_password = field_validator("new_password")(validate_password_strength)


class MeUsernameChange(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_username: str = Field(..., min_length=1, max_length=64)

    _validate_username = field_validator("new_username")(validate_username_format)


class MeUsernameChangeResponse(BaseModel):
    message: str
    username: str
    # A fresh JWT is returned because the old token's `sub` claim still
    # references the previous username and would no longer resolve to a user.
    access_token: str
    token_type: str = "bearer"


class MeBudget(BaseModel):
    """The current user's account-level budget and current-window spend.

    All fields are null when the account has no budget configured.
    """

    budget_usd: float | None = Field(
        None, description="Account-level spending cap in USD. Null means unlimited."
    )
    budget_period: str | None = Field(
        None, description="Budget window: 'daily', 'weekly', 'monthly', or null (lifetime)."
    )
    budget_reset_day: int | None = Field(
        None, description="Day of the month a monthly budget window restarts on (UTC)."
    )
    period_start: datetime | None = Field(
        None, description="Start of the current budget window. Null when no budget is set."
    )
    period_spend_usd: float | None = Field(
        None, description="Spend in the current budget window. Null when no budget is set."
    )


class MeFeedback(BaseModel):
    """Explicit feedback on a smart-routed request.

    ``signal`` semantics: "ok" = routing was adequate; "weak" = the response
    was unsatisfactory (route higher next time); "strong" = the request
    deserved a stronger tier than routed to.
    """

    request_id: str = Field(..., min_length=1, max_length=64)
    signal: Literal["ok", "weak", "strong"]
