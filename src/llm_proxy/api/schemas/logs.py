"""Pydantic schemas for request log APIs."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LogRead(BaseModel):
    """Schema for reading a request log entry."""

    id: int
    timestamp: float
    request_id: str

    endpoint: str
    log_type: str | None = None
    method: str
    status_code: int | None = None
    response_time_ms: int | None = None
    ttft_ms: int | None = None

    user_identity: str | None = None
    model: str | None = None
    provider: str | None = None
    api_key_name: str | None = None

    request_headers: dict[str, Any] = Field(default_factory=dict)
    request_body: Any = Field(default_factory=dict)
    response_headers: dict[str, Any] = Field(default_factory=dict)
    response_body: Any = Field(default_factory=dict)

    error_message: str | None = None
    error_stack_trace: str | None = None

    log_metadata: dict[str, Any] = Field(default_factory=dict)

    # Computed fields from metadata
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None

    # Cache token fields
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cached_prompt_tokens: int | None = None

    # Audit fields - Who
    client_ip: str | None = None
    user_agent: str | None = None
    session_id: str | None = None
    auth_method: str | None = None

    # Audit fields - Where
    server_hostname: str | None = None
    service_name: str | None = None

    # Audit fields - What
    event_type: str | None = None
    action_category: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    outcome: str | None = None

    # Audit fields - Integrity
    sequence_number: int | None = None
    content_hash: str | None = None
    previous_hash: str | None = None

    model_config = ConfigDict(from_attributes=True)


class LogListItem(BaseModel):
    """Schema for list view - excludes large body fields."""

    id: int
    timestamp: float
    request_id: str

    endpoint: str
    log_type: str | None = None
    method: str
    status_code: int | None = None
    response_time_ms: int | None = None
    ttft_ms: int | None = None

    user_identity: str | None = None
    model: str | None = None
    provider: str | None = None
    api_key_name: str | None = None
    auth_method: str | None = None
    client_ip: str | None = None

    error_message: str | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None

    event_type: str | None = None
    action_category: str | None = None

    # Lightweight metadata for list display (e.g., request_type, streaming)
    log_metadata: dict[str, Any] | None = None

    model_config = ConfigDict(from_attributes=True)


class LogListResponse(BaseModel):
    """Paginated logs list response."""

    items: list[LogListItem]
    total: int | None = None
    page: int | None = None
    page_size: int | None = None
    next_cursor: str | None = None
    has_more: bool | None = None


class UsageSummary(BaseModel):
    """Summary statistics for usage."""

    total_cost: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    avg_response_time_ms: float
    success_rate: float

    # Performance and speed
    avg_ttft_ms: float = 0.0
    avg_tokens_per_second: float = 0.0

    # Cache token statistics
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cached_prompt_tokens: int = 0
    cache_savings_usd: float = 0.0


class UsageByProvider(BaseModel):
    """Usage statistics grouped by provider."""

    provider: str
    requests: int
    cost: float
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cached_prompt_tokens: int = 0


class UsageByModel(BaseModel):
    """Usage statistics grouped by model."""

    model: str
    provider: str
    requests: int
    cost: float


class DailyModelUsage(BaseModel):
    """Model-specific daily usage statistics."""

    model: str
    requests: int
    cost: float
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cached_prompt_tokens: int = 0


class DailyUsage(BaseModel):
    """Daily usage statistics."""

    date: str
    requests: int
    cost: float
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cached_prompt_tokens: int = 0
    by_model: list[DailyModelUsage] = Field(default_factory=list)


class UsageStatsResponse(BaseModel):
    """Complete usage statistics response."""

    summary: UsageSummary
    by_provider: list[UsageByProvider]
    by_model: list[UsageByModel]
    daily_usage: list[DailyUsage]
