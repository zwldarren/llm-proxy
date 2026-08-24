"""Server configuration types."""

from typing import Literal

from pydantic import BaseModel, Field

from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.config.types.web_search import WebSearchConfig


class CircuitBreakerParams(BaseModel):
    """Circuit breaker configuration for provider fallback protection."""

    enabled: bool = Field(
        default=True,
        description="Enable circuit breaker to skip failing providers temporarily",
    )
    failure_threshold: int = Field(
        default=5,
        ge=1,
        description="Consecutive failures before a provider is skipped",
    )
    cooldown_seconds: float = Field(
        default=60.0,
        ge=1.0,
        description="Seconds before a skipped provider is probed again",
    )


class SecurityParams(BaseModel):
    """Security / rate-limiting parameters (UI-managed, server_config ``security``).

    Hot-reloaded via the config manager: consumers read these from the cached
    ``ProxyConfig`` on every request, so changes take effect immediately.
    """

    max_failed_login_attempts: int = Field(
        default=5, ge=1, description="Failed login attempts before account lockout"
    )
    lockout_duration_seconds: int = Field(
        default=900, ge=1, description="Account lockout duration in seconds"
    )
    max_failed_api_key_attempts: int = Field(
        default=10, ge=1, description="Failed API key attempts before IP lockout"
    )
    api_key_lockout_duration_seconds: int = Field(
        default=300, ge=1, description="API key lockout duration in seconds"
    )
    auth_failure_delay_ms: int = Field(
        default=100, ge=0, description="Artificial delay on failed authentication (ms)"
    )
    rate_limit_disabled: bool = Field(
        default=False, description="Disable all rate limiting (dangerous; testing only)"
    )
    redis_rate_limit_fail_closed: bool = Field(
        default=True,
        description="When the Redis rate limiter errors, block the request (true) "
        "or allow it through (false)",
    )
    hsts_enabled: bool = Field(
        default=True, description="Send the Strict-Transport-Security header"
    )
    hsts_max_age: int = Field(default=31536000, ge=0, description="HSTS max-age in seconds")
    max_request_body_size_bytes: int = Field(
        default=10 * 1024 * 1024, ge=0, description="Maximum request body size in bytes"
    )


class KeepaliveParams(BaseModel):
    """Non-streaming response keepalive (heartbeat) parameters.

    CDNs such as Cloudflare (free/pro plans) terminate proxied HTTP requests
    that produce no bytes for ~100 seconds (error 524). Slow non-streaming
    LLM requests (e.g. long-reasoning models) can exceed that. When enabled,
    a non-streaming JSON response that outlasts the grace period is started
    early and kept alive with RFC 8259-insignificant whitespace heartbeats
    until the real JSON body is ready. Note: in heartbeat mode the status is
    committed to 200 early, so errors surfacing afterwards are delivered as
    a 200 + error JSON body.
    """

    enabled: bool = Field(default=False, description="Enable non-streaming keepalive heartbeats")
    grace_seconds: float = Field(
        default=30.0,
        gt=0,
        description="How long to wait for normal completion before switching to "
        "heartbeat mode. Must stay comfortably below the CDN's timeout.",
    )
    interval_seconds: float = Field(
        default=15.0,
        gt=0,
        description="Interval between heartbeat bytes once in heartbeat mode.",
    )


class ServerParams(BaseModel):
    """Server configuration parameters."""

    max_fallback_attempts: int = Field(
        default=10,
        ge=0,
        description="Maximum number of fallback provider switches across all providers",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description=(
            "Default retry count per provider; overridable per-model via ModelConfig.max_retries"
        ),
    )
    auth: ProxyAuthConfig = Field(
        default_factory=ProxyAuthConfig,
        description="Authentication configuration",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Request/response logging configuration",
    )
    web_search: WebSearchConfig | None = Field(
        default=None,
        description="Web search tool configuration for non-Anthropic providers",
    )
    unknown_fields_policy: Literal["ignore", "passthrough", "error"] = Field(
        default="ignore",
        description=(
            "How to handle unknown request fields globally: "
            "'ignore' (strip fields silently), "
            "'passthrough' (keep unknown fields in body), "
            "'error' (reject request with validation error). "
            "Note: native passthrough requests (client protocol identical to the "
            "upstream's wire format) are forwarded verbatim and bypass this policy."
        ),
    )
    unsupported_block_policy: Literal["drop", "degrade", "error"] = Field(
        default="drop",
        description=(
            "How to handle content blocks the provider cannot serialize: "
            "'drop' (remove unsupported blocks silently), "
            "'degrade' (convert to a supported fallback representation), "
            "'error' (reject request with validation error)"
        ),
    )
    circuit_breaker: CircuitBreakerParams = Field(
        default_factory=CircuitBreakerParams,
        description="Circuit breaker configuration for provider fallback",
    )
    security: SecurityParams = Field(
        default_factory=SecurityParams,
        description="Security / rate-limiting parameters (server_config: security)",
    )
    keepalive: KeepaliveParams = Field(
        default_factory=KeepaliveParams,
        description="Non-streaming response keepalive parameters (server_config: keepalive)",
    )
    rate_limits: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-bucket rate limit overrides as 'N/period' specs "
            "(server_config: rate_limits). Missing buckets use code defaults."
        ),
    )
    cors_origins: list[str] = Field(
        default_factory=list,
        description=(
            "Allowed CORS origins (server_config: cors_origins). Empty means "
            "CORS is disabled (same-origin deployment). Hot-reloaded."
        ),
    )
