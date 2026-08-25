"""Global application settings backed by pydantic-settings.

All startup-time configuration is centralized here. Environment variables,
``.env`` files, and defaults are unified into a single typed hierarchy.
Dynamic runtime configuration (providers, models, server config from DB)
continues to live in ``DatabaseConfigManager``.
"""

import threading
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

if TYPE_CHECKING:
    from .types import LoggingConfig, ProxyAuthConfig, RedisConfig


class AuthSettings(BaseSettings):
    """Authentication-related environment variables.

    Authentication is always enabled, so there is no toggle. The only env var
    retained is ``JWT_SECRET``, which acts as an explicit override: when set
    (>= 32 characters) it wins over the database-persisted secret managed by
    ``config.secrets``. Admin credentials and API keys are managed from the
    admin UI, not via environment variables.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    jwt_secret: SecretStr | None = Field(default=None, alias="JWT_SECRET")


class LoggingSettings(BaseSettings):
    """Request/response logging environment variables.

    Only ``LOG_LEVEL`` remains an env var: the logging pipeline is built at
    startup before the database is available. All behavioral logging fields
    (masking, sampling, retention, sensitive keys) are UI-managed and stored
    in the ``logging`` server_config key — see ``load_logging_config``.
    """

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="forbid")

    level: str = "INFO"


class RedisSettings(BaseSettings):
    """Redis-related environment variables."""

    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="forbid")

    enabled: bool = False
    url: str = "redis://localhost:6379"
    pool_size: int = Field(default=10, ge=1)
    timeout: float = Field(default=5.0, gt=0)
    rate_limit_enabled: bool = False
    rate_limit_prefix: str = "rate_limit:"
    cache_enabled: bool = False
    cache_prefix: str = "cache:"
    cache_ttl_provider_config: int = Field(default=300, ge=0)
    cache_ttl_model_mapping: int = Field(default=300, ge=0)
    logging_enabled: bool = False
    logging_prefix: str = "logs:"
    logging_ttl_days: int = Field(default=30, ge=0)
    logging_batch_size: int = Field(default=50, ge=1)
    logging_flush_interval_ms: int = Field(default=250, ge=1)


class DBSettings(BaseSettings):
    """Database-related environment variables."""

    model_config = SettingsConfigDict(env_prefix="", extra="forbid")

    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    db_path: str | None = Field(default=None, alias="LLM_PROXY_DB_PATH")
    pool_size: int | None = Field(default=None, alias="DB_POOL_SIZE", ge=1)
    max_overflow: int | None = Field(default=None, alias="DB_MAX_OVERFLOW", ge=0)
    pool_recycle_seconds: int = Field(default=3600, alias="DB_POOL_RECYCLE_SECONDS", ge=60)
    pool_timeout_seconds: int = Field(default=30, alias="DB_POOL_TIMEOUT_SECONDS", ge=1)


class HTTPSettings(BaseSettings):
    """HTTP client-related environment variables."""

    model_config = SettingsConfigDict(env_prefix="HTTP_", extra="forbid")

    max_connections: int = Field(default=200, alias="HTTP_MAX_CONNECTIONS", ge=1)
    max_keepalive: int = Field(
        default=200,
        alias="HTTP_MAX_KEEPALIVE",
        ge=0,
    )
    disable_http2: bool = Field(default=True, alias="HTTP_DISABLE_HTTP2")


# Networks trusted to set forwarded headers (X-Forwarded-For / X-Real-IP).
# By default we trust the common private ranges plus loopback and link-local,
# mirroring the posture of mature reverse-proxy-aware servers (e.g. Tomcat's
# RemoteIpValve `internalProxies`). This makes the typical Docker / traefik /
# same-host deployment resolve the real client IP out of the box, while public
# peers (not in these ranges) are never trusted so IP spoofing for rate-limiting
# and lockout attribution is still prevented.
#
# Set ``TRUSTED_PROXIES=`` (empty) to trust nobody and always use the TCP peer
# IP — the strictest posture, suitable for direct public exposure.
DEFAULT_TRUSTED_PROXIES: list[str] = [
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
]


class SecuritySettings(BaseSettings):
    """Security environment variables.

    Only ``TRUSTED_PROXIES`` remains an env var: it feeds uvicorn's
    ``forwarded_allow_ips`` startup parameter, so it cannot be hot-reloaded.
    All other security / rate-limiting parameters (lockout policy, HSTS, body
    size limit, failure delays) are UI-managed via the ``security``
    server_config key — see ``SecurityParams``.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="forbid")

    trusted_proxies: list[str] = Field(
        default_factory=lambda: list(DEFAULT_TRUSTED_PROXIES),
        alias="TRUSTED_PROXIES",
    )

    @field_validator("trusted_proxies", mode="before")
    @classmethod
    def _parse_trusted_proxies(cls, v: Any) -> list[str]:
        if v is None or v == "":
            return []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return [str(v).strip()]

    @field_validator("trusted_proxies")
    @classmethod
    def _validate_trusted_proxies(cls, v: list[str]) -> list[str]:
        import ipaddress

        errors: list[str] = []
        for item in v:
            try:
                ipaddress.ip_network(item, strict=False)
            except ValueError as exc:
                errors.append(f"{item}: {exc}")
        if errors:
            raise ValueError(f"Invalid TRUSTED_PROXIES entries: {', '.join(errors)}")
        return v


class BatchWriterSettings(BaseSettings):
    """Base for background batch-writer environment variables.

    Concrete subclasses (``LogBatchWriterSettings``, ``UsageBatchWriterSettings``)
    declare their own ``env_prefix`` via ``model_config``.
    """

    model_config = SettingsConfigDict(extra="forbid")

    queue_size: int = Field(default=2000, ge=1)
    batch_size: int = Field(default=100, ge=1)
    flush_interval_ms: int = Field(default=500, ge=1)


class LogBatchWriterSettings(BatchWriterSettings):
    """Log-specific batch-writer environment variables (``LOG_WRITE_*``)."""

    model_config = SettingsConfigDict(env_prefix="LOG_WRITE_", extra="forbid")


class UsageBatchWriterSettings(BatchWriterSettings):
    """Usage-specific batch-writer environment variables (``USAGE_WRITE_*``)."""

    model_config = SettingsConfigDict(env_prefix="USAGE_WRITE_", extra="forbid")
    batch_size: int = Field(default=200, ge=1)  # usage batches are larger


class UvicornSettings(BaseSettings):
    """Uvicorn server environment variables."""

    model_config = SettingsConfigDict(env_prefix="")

    timeout_keepalive: int = Field(default=600, alias="UVICORN_TIMEOUT_KEEPALIVE")


class UpdateCheckSettings(BaseSettings):
    """GitHub-based update-check environment variables.

    When enabled, ``GET /api/system/info`` compares the running version
    against the highest tag of the GitHub repository (cached in memory).
    Disable to guarantee the server never calls out to GitHub.
    """

    model_config = SettingsConfigDict(env_prefix="UPDATE_CHECK__", extra="forbid")

    enabled: bool = True


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    auth: AuthSettings = Field(default_factory=AuthSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    db: DBSettings = Field(default_factory=DBSettings)
    http: HTTPSettings = Field(default_factory=HTTPSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    log_batch: LogBatchWriterSettings = Field(default_factory=LogBatchWriterSettings)
    usage_batch: UsageBatchWriterSettings = Field(default_factory=UsageBatchWriterSettings)
    uvicorn: UvicornSettings = Field(default_factory=UvicornSettings)
    update_check: UpdateCheckSettings = Field(default_factory=UpdateCheckSettings)

    encryption_key: SecretStr | None = Field(default=None, alias="ENCRYPTION_KEY")


_settings: Settings | None = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    """Get or create the global ``Settings`` singleton (thread-safe)."""
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = Settings()
    return _settings


def get_auth_config() -> ProxyAuthConfig:
    """Get validated authentication configuration.

    This is a convenience function that combines settings with validation logic.
    It raises ConfigurationError if required fields are missing or invalid.

    Use this instead of manually unwrapping SecretStr values.
    """
    from .manager import load_auth_config

    return load_auth_config()


def get_logging_config(
    overrides: dict[str, str | int | float | bool] | None = None,
) -> LoggingConfig:
    """Get logging configuration with optional database overrides.

    This is a convenience wrapper around load_logging_config().
    """
    from .manager import load_logging_config

    return load_logging_config(overrides)


def get_redis_config() -> RedisConfig:
    """Get Redis configuration from settings."""
    from .manager import load_redis_config

    return load_redis_config()


def set_settings(s: Settings) -> None:
    """Replace the global ``Settings`` singleton (useful in tests)."""
    global _settings
    with _settings_lock:
        _settings = s


def reset_settings() -> None:
    """Clear the global ``Settings`` singleton so it re-reads on next call."""
    global _settings
    with _settings_lock:
        _settings = None
