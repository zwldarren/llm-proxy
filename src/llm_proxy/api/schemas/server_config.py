"""Schemas for the server configuration sections.

One schema per section of the settings UI, each mirroring the runtime config
type it updates, all served by ``api/routers/config/server.py``: logging, web
search, smart routing, provider selection, request policy, resilience
(circuit breaker) and security (cookies, MCP policy).
"""

from typing import Literal

from pydantic import BaseModel, Field

from llm_proxy.config.types.model import ProviderSelectionStrategy
from llm_proxy.config.types.smart_routing import RoutingJudgeConfig
from llm_proxy.serialization.context import UnknownFieldsPolicy, UnsupportedBlockPolicy

#: Global request-policy selection values. ``"default"`` means "not configured":
#: the per-adapter default applies (e.g. vLLM/SGLang pass unknown fields
#: through). Any other value is an explicit operator choice. The concrete values
#: come from the serialization-layer policy Literals so the two layers cannot
#: drift apart.
UnknownFieldsPolicySelection = Literal["default", UnknownFieldsPolicy]

UnsupportedBlockPolicySelection = Literal["default", UnsupportedBlockPolicy]


class LoggingConfigUpdate(BaseModel):
    """Schema for updating logging config."""

    log_input_output: bool = Field(
        default=False,
        description=(
            "Log request/response bodies; off by default so rows persist with bodies "
            "scrubbed. Master switch: it cannot be overridden per request"
        ),
    )
    log_raw_stream: bool | None = Field(
        default=None,
        description=(
            "Store raw SSE text for streaming responses; when off, the reassembled "
            "non-streaming response body is stored instead"
        ),
    )
    log_retention_days: int | None = Field(
        None, ge=0, description="Log retention days (0 = keep indefinitely)"
    )
    max_logged_body_bytes: int | None = Field(
        None,
        ge=0,
        description=(
            "Cap on the serialized size of a stored request/response body in bytes; "
            "larger bodies are stored as a truncation marker (0 = no cap)"
        ),
    )
    verbose_routing_logs: bool | None = Field(
        default=None,
        description="Include detailed per-candidate routing scorecards in request log metadata",
    )
    mask_sensitive_data: bool | None = Field(
        default=None, description="Mask sensitive fields in logs"
    )
    sampling_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Rate at which to log full request/response bodies",
    )
    audit_sampling_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling rate for audit logs (null = inherit sampling_rate)",
    )
    audit_retention_days: int | None = Field(
        default=None,
        ge=0,
        description="Retention days for audit logs (null = inherit log_retention_days)",
    )
    sensitive_keys: str | None = Field(
        default=None,
        description="Comma-separated extra key names to mask in logs",
    )


class SearXNGConfigUpdate(BaseModel):
    """Schema for updating SearXNG configuration."""

    url: str = Field(..., description="SearXNG instance URL")
    api_key: str | None = Field(None, description="API key if required")
    basic_auth_username: str | None = Field(None, description="Basic auth username")
    basic_auth_password: str | None = Field(None, description="Basic auth password")
    engines: list[str] | None = Field(None, description="Search engines to use")
    timeout: float = Field(30.0, ge=1.0, le=300.0, description="Request timeout in seconds")
    max_results: int = Field(10, ge=1, le=10, description="Maximum results per search")


class OllamaConfigUpdate(BaseModel):
    """Schema for updating Ollama web search configuration."""

    api_key: str = Field(..., description="Ollama API key")
    base_url: str = Field(
        "https://ollama.com",
        description="Ollama API base URL",
    )
    timeout: float = Field(30.0, ge=1.0, le=300.0, description="Request timeout in seconds")
    max_results: int = Field(10, ge=1, le=10, description="Maximum results per search")


class WebSearchConfigUpdate(BaseModel):
    """Schema for updating web search configuration."""

    enabled: bool = Field(False, description="Enable web search interception")
    provider: Literal["searxng", "ollama"] = Field("searxng", description="Search provider")
    searxng: SearXNGConfigUpdate | None = Field(None, description="SearXNG configuration")
    ollama: OllamaConfigUpdate | None = Field(None, description="Ollama configuration")


class RequestPolicyConfig(BaseModel):
    """Schema for global request policy configuration.

    ``"default"`` is stored verbatim and resolves to "not configured" at
    runtime, letting each provider apply its own default. Concrete values
    override every provider.
    """

    unknown_fields_policy: UnknownFieldsPolicySelection = Field(
        default="default",
        description=(
            "How to handle unknown request fields globally: "
            "'default' (per-provider default), "
            "'ignore' (strip fields silently), "
            "'passthrough' (keep unknown fields in body), "
            "'error' (reject request with validation error)"
        ),
    )
    unsupported_block_policy: UnsupportedBlockPolicySelection = Field(
        default="default",
        description=(
            "How to handle content blocks the provider cannot serialize: "
            "'default' (per-provider default), "
            "'drop' (remove unsupported blocks silently), "
            "'degrade' (convert to a supported fallback representation), "
            "'error' (reject request with validation error)"
        ),
    )


class CircuitBreakerConfigSchema(BaseModel):
    """Schema for circuit breaker configuration."""

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


class CorsConfig(BaseModel):
    """Schema for allowed CORS origins configuration.

    An empty list disables CORS (same-origin deployment). Origins should be
    full scheme+host(+port) values, e.g. "https://admin.example.com".
    """

    origins: list[str] = Field(
        default_factory=list,
        description="Allowed CORS origins; empty disables CORS",
    )


class RateLimitsConfig(BaseModel):
    """Schema for per-bucket rate limit overrides.

    Keys are bucket names (see DEFAULT_RATE_LIMITS in the rate limiting
    middleware); values are "N/period" specs, e.g. "5/minute".
    """

    limits: dict[str, str] = Field(
        default_factory=dict,
        description="Bucket name → 'N/period' rate limit spec",
    )


class KeepaliveConfig(BaseModel):
    """Schema for non-streaming response keepalive configuration.

    Defaults must stay in step with
    :class:`~llm_proxy.config.types.server.KeepaliveParams` (the runtime type the
    config manager actually uses): this schema is what the settings UI reads, so
    a divergence would make the panel report a state that is not in effect.
    """

    enabled: bool = Field(default=True, description="Enable non-streaming keepalive heartbeats")
    grace_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Seconds to wait for normal completion before heartbeat mode",
    )
    interval_seconds: float = Field(
        default=15.0,
        gt=0,
        description="Interval between heartbeat bytes once in heartbeat mode",
    )


class SecurityConfig(BaseModel):
    """Schema for security / rate-limiting configuration (server_config ``security``)."""

    login_lockout_enabled: bool = Field(
        default=False,
        description=(
            "Lock an account after repeated failed logins (keyed by username). "
            "Off by default: a hard per-account lockout lets anyone who knows the "
            "admin username lock the account out on purpose. Per-IP throttling "
            "(auth.login bucket) and the auth failure delay still apply when off."
        ),
    )
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
        default=64 * 1024 * 1024,
        ge=0,
        description=(
            "Maximum request body size in bytes. 64 MiB leaves headroom for "
            "multimodal and batch payloads (base64 images/audio/PDFs) while "
            "bounding the memory a single request can pin. 0 disables the limit."
        ),
    )


class ResilienceConfig(BaseModel):
    """Schema for global resilience configuration (retry + fallback + circuit breaker)."""

    max_retries: int = Field(
        default=3,
        ge=0,
        description=(
            "Same-provider retry attempts for transient errors (rate limit, "
            "timeout, network) and server errors (5xx, 408). Client errors "
            "(401/403/400/422) are not retried in place -- they fall back "
            "directly. Overridable per-model via ModelConfig.max_retries"
        ),
    )
    max_fallback_attempts: int = Field(
        default=10,
        ge=0,
        description="Maximum number of fallback provider switches across all providers",
    )
    circuit_breaker: CircuitBreakerConfigSchema = Field(
        default_factory=CircuitBreakerConfigSchema,
        description="Circuit breaker configuration for provider fallback",
    )


class McpSecurityPolicyConfig(BaseModel):
    """Schema for MCP security policy configuration stored in the database.

    All policy fields (list-based rules and the permission switch) are
    UI-managed and persisted in the ``mcp_security_policy`` server_config key.
    """

    require_key_mcp_permissions: bool = Field(
        default=True,
        description="Require explicit MCP permissions on API keys for MCP access",
    )

    allowed_commands: list[str] = Field(
        default_factory=list,
        description="Commands permitted for stdio MCP servers",
    )
    blocked_commands: list[str] = Field(
        default_factory=lambda: [
            "bash",
            "sh",
            "zsh",
            "cmd.exe",
            "powershell.exe",
            "python",
            "python3",
            "node",
            "perl",
            "ruby",
        ],
        description="Commands always blocked even if in allowed list",
    )
    allowed_env_keys: list[str] = Field(
        default_factory=list,
        description="Environment variable keys permitted for MCP servers",
    )
    blocked_env_keys: list[str] = Field(
        default_factory=lambda: [
            "PATH",
            "LD_PRELOAD",
            "DYLD_INSERT_LIBRARIES",
            "PYTHONPATH",
            "NODE_OPTIONS",
            "SHELL",
            "HOME",
            "USER",
        ],
        description="Environment variable keys always blocked",
    )
    blocked_url_hosts: list[str] = Field(
        default_factory=list,
        description="URL hosts blocked for streamableHttp MCP servers",
    )
    blocked_url_ips: list[str] = Field(
        default_factory=lambda: [
            "127.0.0.0/8",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "169.254.169.254/32",
            "100.64.0.0/10",
            "::1/128",
            "fc00::/7",
            "fe80::/10",
            "::ffff:0:0/96",
        ],
        description="IP ranges blocked for streamableHttp MCP servers",
    )


class SmartRoutingConfigUpdate(BaseModel):
    enabled: bool | None = Field(default=None)
    mode_weights: dict[str, float] | None = Field(
        default=None,
        description="Weights for each routing mode (fast, auto, best)",
    )
    judge: RoutingJudgeConfig | None = Field(
        default=None,
        description=(
            "System One routing judge: consulted for ambiguous first turns when enabled, "
            "and recorded without changing the decision while shadow is on (ADR-0018)"
        ),
    )


class ProviderSelectionConfigUpdate(BaseModel):
    strategy: ProviderSelectionStrategy | None = Field(
        default=None,
        description=(
            "How to pick among same-priority providers: 'random' (default), "
            "'session_sticky' (pin a conversation to one provider for cache affinity), "
            "'cost_optimized' (cheapest first), 'balanced' (cost + observed latency)"
        ),
    )
