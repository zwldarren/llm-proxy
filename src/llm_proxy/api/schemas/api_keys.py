"""Schemas for the API keys resource (``api/routers/api_keys.py``)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_proxy.core.exceptions import ValidationError

from .logs import DailyUsage, UsageByModel, UsageSummary


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
