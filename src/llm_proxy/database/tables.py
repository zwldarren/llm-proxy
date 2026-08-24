"""SQLAlchemy ORM models for LLM Proxy configuration and logging."""

import time as _time
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    false,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from llm_proxy.database.base import Base


class ProviderRecord(Base):
    """Database model for provider configuration."""

    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    type: Mapped[str] = mapped_column(String(50))
    api_key: Mapped[str] = mapped_column(String(500))
    base_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    api_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    timeout: Mapped[float] = mapped_column(default=300.0)
    rate_limit: Mapped[int | None] = mapped_column(nullable=True)
    custom_headers: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    provider_models: Mapped[list[str]] = mapped_column(JSON, default=list)
    enabled: Mapped[bool] = mapped_column(default=True)
    priority: Mapped[int] = mapped_column(default=0)
    provider_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    definition_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    icon_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Relationships via join table
    model_provider_mappings: Mapped[list[ModelProviderRecord]] = relationship(
        "ModelProviderRecord", back_populates="provider", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ProviderRecord(id={self.id}, name='{self.name}', type='{self.type}')>"


class ModelProviderRecord(Base):
    """Database model for model-provider relationship with priority.

    This join table allows a model to have multiple providers with different
    priorities for fallback support. Each provider mapping can have its own
    pricing (input_cost_per_1m, output_cost_per_1m).
    """

    __tablename__ = "model_providers"
    __table_args__ = (
        Index("ix_model_providers_model_priority", "model_id", "priority"),
        Index("ix_model_providers_provider_model_name", "provider_model_name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id", ondelete="CASCADE"), nullable=False
    )
    priority: Mapped[int] = mapped_column(default=0)
    provider_model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # Per-provider pricing (overrides model-level pricing when set)
    input_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    output_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    cached_read_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    cached_write_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_input_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_output_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_input_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Unit-based pricing (non-token dimensions)
    cost_per_image: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_cost_per_minute: Mapped[float | None] = mapped_column(Float, nullable=True)
    tts_cost_per_1m_chars: Mapped[float | None] = mapped_column(Float, nullable=True)
    web_search_cost_per_1k: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Per-provider parameter overrides (applied to all requests via this provider)
    parameter_overrides: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Relationships
    model: Mapped[ModelRecord] = relationship("ModelRecord", back_populates="provider_mappings")
    provider: Mapped[ProviderRecord] = relationship(
        "ProviderRecord", back_populates="model_provider_mappings"
    )

    def __repr__(self) -> str:
        return (
            f"<ModelProviderRecord(id={self.id}, model_id={self.model_id}, "
            f"provider_id={self.provider_id}, priority={self.priority})>"
        )


class ModelRecord(Base):
    """Database model for model configuration."""

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    timeout: Mapped[float | None] = mapped_column(nullable=True)
    max_retries: Mapped[int | None] = mapped_column(nullable=True)
    input_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    output_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    cached_read_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    cached_write_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_input_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_output_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_input_cost_per_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Unit-based pricing (non-token dimensions)
    cost_per_image: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_cost_per_minute: Mapped[float | None] = mapped_column(Float, nullable=True)
    tts_cost_per_1m_chars: Mapped[float | None] = mapped_column(Float, nullable=True)
    web_search_cost_per_1k: Mapped[float | None] = mapped_column(Float, nullable=True)
    icon_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    homepage_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    context_length: Mapped[int | None] = mapped_column(nullable=True)
    model_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    auto_eligible: Mapped[bool] = mapped_column(
        default=False, nullable=False, server_default=false()
    )
    quality_tier: Mapped[str | None] = mapped_column(String(20), nullable=True)
    routing_assignments: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    supports_images: Mapped[bool] = mapped_column(
        default=False, nullable=False, server_default=false()
    )
    # Manually configured model-type capabilities (shown in the model catalog).
    supports_image_generation: Mapped[bool] = mapped_column(
        default=False, nullable=False, server_default=false()
    )
    supports_tts: Mapped[bool] = mapped_column(
        default=False, nullable=False, server_default=false()
    )
    supports_stt: Mapped[bool] = mapped_column(
        default=False, nullable=False, server_default=false()
    )
    supports_embedding: Mapped[bool] = mapped_column(
        default=False, nullable=False, server_default=false()
    )
    supports_realtime: Mapped[bool] = mapped_column(
        default=False, nullable=False, server_default=false()
    )

    # Relationships via join table
    provider_mappings: Mapped[list[ModelProviderRecord]] = relationship(
        "ModelProviderRecord",
        back_populates="model",
        cascade="all, delete-orphan",
        order_by="desc(ModelProviderRecord.priority)",
    )

    @property
    def provider_name(self) -> str:
        """Get provider name from the highest priority provider mapping."""
        if self.provider_mappings:
            return self.provider_mappings[0].provider.name
        return ""

    @property
    def model_name(self) -> str | None:
        """Get model name from the highest priority provider mapping."""
        if self.provider_mappings:
            return self.provider_mappings[0].provider_model_name
        return None

    def __repr__(self) -> str:
        return f"<ModelRecord(id={self.id}, name='{self.name}')>"


class ServerConfigRecord(Base):
    """Database model for server configuration."""

    __tablename__ = "server_config"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    value: Mapped[Any] = mapped_column(JSON)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<ServerConfigRecord(id={self.id}, key='{self.key}')>"


class ApiKeyRecord(Base):
    """Database model for API keys used for authentication."""

    __tablename__ = "api_keys"

    name: Mapped[str] = mapped_column(String(255), primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    allowed_models: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    allowed_mcp_servers: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    # NULL means the key never expires. Checked at request-auth time against the
    # cached record, so an expired key keeps working at most for the cache TTL.
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Spending cap in USD. NULL means unlimited. When the current-period spend
    # reaches the cap, requests are rejected (429) until the budget is raised or
    # the period counter is manually reset.
    budget_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Budget window: 'daily' | 'weekly' | 'monthly' (UTC calendar boundaries).
    # NULL means a lifetime budget: the cap applies to cumulative spend since
    # the last manual reset. Only meaningful when budget_usd is set.
    budget_period: Mapped[str | None] = mapped_column(String(10), nullable=True)
    # Day of the month (1-31) on which a monthly budget window restarts. NULL
    # means the 1st. Only meaningful when budget_period == 'monthly'.
    budget_reset_day: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Manual reset point: current-period spend counts usage at or after
    # max(period_start, budget_reset_at).
    budget_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Per-key request rate limit in requests per minute. NULL means unlimited.
    # Enforced at request-auth time against the cached record; only writable by
    # admins (it is a quota tool, not a self-service preference).
    rate_limit_rpm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<ApiKeyRecord(name='{self.name}', is_active={self.is_active})>"


class UserRecord(Base):
    """Database model for admin/UI users.

    Stores the credentials used to authenticate into the admin UI. The admin
    account is created on first run via the frontend setup screen rather than
    via environment variables or auto-generation.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    # Least-privilege default: every creation path passes a role explicitly;
    # the DB-level default must never silently grant admin.
    role: Mapped[str] = mapped_column(
        String(20), default="viewer", server_default="viewer", nullable=False
    )
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    # When true, the user must set a new password before accessing anything
    # beyond the password-change endpoint. Set by admin-created accounts and
    # admin password resets; cleared by a successful self-service change.
    must_change_password: Mapped[bool] = mapped_column(
        default=False, nullable=False, server_default=false()
    )
    # Monotonic token version embedded in issued JWTs ("tv" claim). Bumping it
    # (password change / admin reset) immediately invalidates all previously
    # issued tokens for this user.
    token_version: Mapped[int] = mapped_column(default=0, server_default="0", nullable=False)
    # Per-user model allowlist. NULL means unrestricted. Non-admin users may
    # only create API keys within this set, and the effective permission of
    # any of their keys is intersected with it at request time.
    allowed_models: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    # Admin-set account-level spending cap in USD, aggregated across all of
    # the user's API keys. NULL means unlimited. When the current-period spend
    # reaches the cap, all of the user's keys are rejected (429) until the
    # budget is raised or the period counter is manually reset. Key-level
    # budgets are purely self-service; this is the admin-controlled envelope.
    budget_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Budget window: 'daily' | 'weekly' | 'monthly' (UTC calendar boundaries).
    # NULL means a lifetime budget: the cap applies to cumulative spend since
    # the last manual reset. Only meaningful when budget_usd is set.
    budget_period: Mapped[str | None] = mapped_column(String(10), nullable=True)
    # Day of the month (1-31) on which a monthly budget window restarts. NULL
    # means the 1st. Only meaningful when budget_period == 'monthly'.
    budget_reset_day: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Manual reset point: current-period spend counts usage at or after
    # max(period_start, budget_reset_at).
    budget_reset_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Per-user tracing/observability configuration. When absent (NULL) the
    # user's requests fall back to the admin-managed global tracing config.
    tracing_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    def __repr__(self) -> str:
        return f"<UserRecord(id={self.id}, username='{self.username}', role='{self.role}')>"


class UserSessionRecord(Base):
    """Database model for admin UI session API keys.

    Auto-generated on login, used by the frontend to authenticate /v1/* requests.
    """

    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    token_prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<UserSessionRecord(id='{self.id}', user_id={self.user_id}, active={self.is_active})>"
        )


class McpServerRecord(Base):
    """Database model for MCP server configurations."""

    __tablename__ = "mcp_servers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    type: Mapped[str] = mapped_column(String(50))
    command: Mapped[str | None] = mapped_column(String(500), nullable=True)
    args: Mapped[list[str]] = mapped_column(JSON, default=list)
    base_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    env: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(default=True)
    proxy_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    server_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<McpServerRecord(id={self.id}, name='{self.name}', type='{self.type}')>"


class RequestLog(Base):
    """Database model for request/response logs."""

    __tablename__ = "request_logs"

    __table_args__ = (
        Index("ix_request_logs_timestamp", "timestamp"),
        Index("ix_request_logs_request_id", "request_id"),
        Index("ix_request_logs_status_code", "status_code"),
        Index("ix_request_logs_model", "model"),
        Index("ix_request_logs_provider", "provider"),
        Index("ix_request_logs_timestamp_status", "timestamp", "status_code"),
        # Composite index for common query pattern: timestamp + status_code + model
        Index("ix_request_logs_ts_status_model", "timestamp", "status_code", "model"),
        Index("ix_request_logs_cost_usd", "cost_usd"),
        Index("ix_request_logs_total_tokens", "total_tokens"),
        # Audit field indexes
        Index("ix_request_logs_event_type", "event_type"),
        Index("ix_request_logs_outcome", "outcome"),
        Index("ix_request_logs_client_ip", "client_ip"),
        Index("ix_request_logs_timestamp_sequence", "timestamp", "sequence_number"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Unix timestamp in seconds (float) for easy range filtering in SQLite
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)

    request_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    endpoint: Mapped[str] = mapped_column(String(500), nullable=False)
    log_type: Mapped[str] = mapped_column(String(20), nullable=True, default="endpoint")
    method: Mapped[str] = mapped_column(String(16), nullable=False)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    user_identity: Mapped[str | None] = mapped_column(String(200), nullable=True)
    model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)

    request_headers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    request_body: Mapped[Any] = mapped_column(JSON, default=dict)
    response_headers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    response_body: Mapped[Any] = mapped_column(JSON, default=dict)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_stack_trace: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Dedicated columns for frequently-queried metrics (extracted from log_metadata)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Cache token tracking
    cache_creation_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_read_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cached_prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_savings_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Audio tokens (Realtime / audio-capable models)
    audio_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    audio_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Keep log_metadata for additional fields, but exclude the extracted fields
    log_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    api_key_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Time to first token

    # Compression support for large request/response bodies
    request_body_compressed: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    response_body_compressed: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    request_body_compression: Mapped[str | None] = mapped_column(String(20), nullable=True)
    response_body_compression: Mapped[str | None] = mapped_column(String(20), nullable=True)
    request_body_original_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_body_original_size: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Audit fields - Who (user/client identification)
    client_ip: Mapped[str | None] = mapped_column(String(45), nullable=True)  # IPv6 max length
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    auth_method: Mapped[str | None] = mapped_column(String(20), nullable=True)  # api_key, jwt, none

    # Audit fields - Where (service identification)
    server_hostname: Mapped[str | None] = mapped_column(String(255), nullable=True)
    service_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    service_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Audit fields - What (event classification)
    event_type: Mapped[str | None] = mapped_column(String(30), nullable=True)  # EventType enum
    action_category: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # ActionCategory enum
    resource_type: Mapped[str | None] = mapped_column(
        String(30), nullable=True
    )  # ResourceType enum
    resource_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(20), nullable=True)  # Outcome enum

    # Audit fields - Integrity (hash chain)
    sequence_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)  # SHA-256 hex
    previous_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)  # Previous entry
    content_hash_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=2, server_default=text("'2'")
    )  # Audit hash algorithm version

    def __repr__(self) -> str:
        return (
            "<RequestLog("
            f"id={self.id}, request_id='{self.request_id}', method='{self.method}', "
            f"endpoint='{self.endpoint}', log_type='{self.log_type}', "
            f"status_code={self.status_code})>"
        )


class UsageRecord(Base):
    """Database model for usage statistics - independent from logs.

    This table stores usage data (tokens, costs, cache metrics) separately from
    request logs, ensuring usage statistics persist even when logs are deleted
    or logging is disabled.
    """

    __tablename__ = "usage_records"

    __table_args__ = (
        Index("ix_usage_records_timestamp", "timestamp"),
        Index("ix_usage_records_model", "model"),
        Index("ix_usage_records_provider", "provider"),
        Index("ix_usage_records_timestamp_model", "timestamp", "model"),
        Index("ix_usage_records_timestamp_provider", "timestamp", "provider"),
        Index("ix_usage_records_api_key", "api_key_name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Unix timestamp in seconds (float) for easy range filtering
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    # Optional link to request log (may be None if log was deleted)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Model and provider info
    model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Token counts
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Cost
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Cache tokens (Anthropic-style)
    cache_creation_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_read_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Cache tokens (OpenAI-style)
    cached_prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_savings_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Audio tokens (Realtime / audio-capable models)
    audio_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    audio_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Request metadata
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    user_identity: Mapped[str | None] = mapped_column(String(200), nullable=True)
    api_key_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Streaming metrics
    is_streaming: Mapped[bool] = mapped_column(default=False)
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)  # Time to first token

    # Log type for filtering (endpoint, audit, etc.)
    log_type: Mapped[str | None] = mapped_column(String(20), nullable=True)

    def __repr__(self) -> str:
        return (
            "<UsageRecord("
            f"id={self.id}, model='{self.model}', provider='{self.provider}', "
            f"prompt_tokens={self.prompt_tokens}, completion_tokens={self.completion_tokens}, "
            f"cost_usd={self.cost_usd})>"
        )


class AuditSequence(Base):
    """Sequence counter for audit log hash chain.

    Maintains the current sequence number and last hash for atomic
    hash chain updates. Only used for audit logs (log_type='audit').

    This table has a single row (id=1) that tracks the current state
    of the audit log chain. All updates use SELECT FOR UPDATE to ensure
    atomicity when computing the next hash in the chain.
    """

    __tablename__ = "audit_sequence"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    current_sequence: Mapped[int] = mapped_column(Integer, default=0)
    last_hash: Mapped[str] = mapped_column(String(64), default="GENESIS")
    updated_at: Mapped[float] = mapped_column(Float, default=_time.time)

    def __repr__(self) -> str:
        return (
            f"<AuditSequence(id={self.id}, current_sequence={self.current_sequence}, "
            f"last_hash='{self.last_hash[:16]}...')>"
        )


class ModelExperienceRecord(Base):
    """Per-model EWMA stats for Thompson sampling in smart routing."""

    __tablename__ = "model_experience"

    name: Mapped[str] = mapped_column(
        ForeignKey("models.name", ondelete="CASCADE"), primary_key=True
    )
    samples: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reward_mean: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    latency: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    reliability: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    feedback: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    cache_affinity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, default=_time.time)

    def __repr__(self) -> str:
        return f"<ModelExperienceRecord(name='{self.name}', samples={self.samples})>"


class FeedbackRecord(Base):
    """Explicit user feedback (ok/weak/strong) on a routed request.

    One row per request_id: the primary key enforces feedback idempotency.
    Rows double as calibration eval samples (joined with the request log's
    routing_confidence) for future Platt temperature re-fitting.
    """

    __tablename__ = "feedback_records"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    signal: Mapped[str] = mapped_column(String(16), nullable=False)
    # Resolved concrete model the feedback applies to (snapshot at submit time).
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=_time.time)

    def __repr__(self) -> str:
        return f"<FeedbackRecord(request_id='{self.request_id}', signal='{self.signal}')>"
