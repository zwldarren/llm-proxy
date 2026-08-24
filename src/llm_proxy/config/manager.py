"""Database-backed configuration manager with optional Redis caching."""

import asyncio
from typing import TYPE_CHECKING, Any

from llm_proxy.core.constants import DEFAULT_MAX_FALLBACK_ATTEMPTS, DEFAULT_MAX_RETRIES
from llm_proxy.core.exceptions import ConfigurationError, ProviderNotConfiguredError
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.cache.redis_cache import RedisCache
    from llm_proxy.config.types.smart_routing import SmartRoutingConfig

from .mappers import map_model_record, map_provider_record
from .settings import get_settings
from .types import (
    CircuitBreakerParams,
    KeepaliveParams,
    LoggingConfig,
    ModelConfig,
    ProviderConfig,
    ProxyAuthConfig,
    ProxyConfig,
    RedisCacheConfig,
    RedisConfig,
    RedisLoggingConfig,
    RedisRateLimitConfig,
    SecurityParams,
    ServerParams,
)

logger = get_logger(__name__)

# Async lock to protect concurrent DB read-modify-write sequences
_config_lock_obj: asyncio.Lock | None = None
_config_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_config_lock() -> asyncio.Lock:
    global _config_lock_obj, _config_lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _config_lock_obj is None or _config_lock_loop != loop:
        _config_lock_obj = asyncio.Lock()
        _config_lock_loop = loop
    return _config_lock_obj


def load_auth_config() -> ProxyAuthConfig:
    """Load authentication configuration from the secrets holder.

    Authentication is always enabled. The JWT secret is resolved at startup
    by ``ensure_secrets()`` (env override > database > auto-generate) and read
    here from the in-memory cache.
    """
    from .secrets import get_jwt_secret

    return ProxyAuthConfig(jwt_secret=get_jwt_secret())


def _parse_bool(value: Any) -> bool:
    """Parse a value as boolean, supporting string and non-string types."""
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _parse_optional_float(value: Any, name: str) -> float | None:
    """Parse an optional float config value (None/empty string → None)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid {name}: {exc}") from exc


def _parse_optional_int(value: Any, name: str) -> int | None:
    """Parse an optional int config value (None/empty string → None)."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid {name}: {exc}") from exc


def load_logging_config(
    overrides: dict[str, Any] | None = None,
) -> LoggingConfig:
    """Build the logging configuration from database overrides.

    All behavioral fields (masking, sampling, retention, sensitive keys) are
    UI-managed and stored in the ``logging`` server_config key; this function
    merges those stored values with code defaults. The only remaining env var
    is ``LOG_LEVEL`` (startup-time logging pipeline verbosity).
    """
    settings = get_settings().logging
    overrides = overrides or {}

    enable_database_logging = _parse_bool(overrides.get("log_input_output", True))

    try:
        retention_days = int(overrides.get("log_retention_days", 30))
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid log_retention_days: {exc}") from exc

    verbose_routing_logs = _parse_bool(overrides.get("verbose_routing_logs", False))
    mask_sensitive_data = _parse_bool(overrides.get("mask_sensitive_data", True))

    try:
        sampling_rate = float(overrides.get("sampling_rate", 1.0))
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid sampling_rate: {exc}") from exc

    audit_sampling_rate = _parse_optional_float(
        overrides.get("audit_sampling_rate"), "audit_sampling_rate"
    )
    audit_retention_days = _parse_optional_int(
        overrides.get("audit_retention_days"), "audit_retention_days"
    )

    sensitive_keys = [
        "authorization",
        "api_key",
        "apikey",
        "password",
        "passwd",
        "token",
        "access_token",
        "refresh_token",
        "jwt_secret",
    ]
    extra_keys_raw = overrides.get("sensitive_keys", "")
    if isinstance(extra_keys_raw, str):
        extra_keys = [k.strip().lower() for k in extra_keys_raw.split(",") if k.strip()]
    elif isinstance(extra_keys_raw, list):
        extra_keys = [str(k).strip().lower() for k in extra_keys_raw if str(k).strip()]
    else:
        extra_keys = []
    if extra_keys:
        sensitive_keys = [*dict.fromkeys(sensitive_keys + extra_keys)]

    # Env-only value (startup-time logging pipeline verbosity):
    log_level = settings.level.upper()

    return LoggingConfig(
        enable_database_logging=enable_database_logging,
        retention_days=retention_days,
        mask_sensitive_data=mask_sensitive_data,
        log_level=log_level,
        sampling_rate=sampling_rate,
        audit_sampling_rate=audit_sampling_rate,
        audit_retention_days=audit_retention_days,
        sensitive_keys=sensitive_keys,
        verbose_routing_logs=verbose_routing_logs,
    )


def load_redis_config() -> RedisConfig:
    """Load Redis configuration from settings."""
    settings = get_settings().redis

    if not settings.enabled:
        return RedisConfig(enabled=False)

    return RedisConfig(
        enabled=settings.enabled,
        url=settings.url,
        pool_size=settings.pool_size,
        timeout=settings.timeout,
        rate_limit=RedisRateLimitConfig(
            enabled=settings.rate_limit_enabled,
            prefix=settings.rate_limit_prefix,
        ),
        cache=RedisCacheConfig(
            enabled=settings.cache_enabled,
            prefix=settings.cache_prefix,
            ttl_provider_config=settings.cache_ttl_provider_config,
            ttl_model_mapping=settings.cache_ttl_model_mapping,
        ),
        logging=RedisLoggingConfig(
            enabled=settings.logging_enabled,
            prefix=settings.logging_prefix,
            ttl_days=settings.logging_ttl_days,
            batch_size=settings.logging_batch_size,
            flush_interval_ms=settings.logging_flush_interval_ms,
        ),
    )


def resolve_logging_config(
    config_manager: DatabaseConfigManager | None,
) -> LoggingConfig:
    """Resolve the effective logging configuration.

    Prefers the config manager's cached :class:`ProxyConfig` (which applies
    the UI-managed ``logging`` server_config overrides and is refreshed on
    every settings change); falls back to code defaults when the manager or
    its cache is not available yet (early startup, tests).
    """
    cached = config_manager.get_cached_config() if config_manager is not None else None
    if isinstance(cached, ProxyConfig):
        return cached.server_params.logging
    return load_logging_config()


def resolve_security_params(
    config_manager: DatabaseConfigManager | None,
) -> SecurityParams:
    """Resolve the effective security parameters.

    Prefers the config manager's cached :class:`ProxyConfig` (refreshed on
    every settings change); falls back to code defaults when unavailable
    (early startup, tests).
    """
    cached = config_manager.get_cached_config() if config_manager is not None else None
    if isinstance(cached, ProxyConfig):
        return cached.server_params.security
    return SecurityParams()


def resolve_keepalive_params(
    config_manager: DatabaseConfigManager | None,
) -> KeepaliveParams:
    """Resolve the effective non-streaming keepalive parameters.

    Prefers the config manager's cached :class:`ProxyConfig` (refreshed on
    every settings change); falls back to code defaults when unavailable
    (early startup, tests).
    """
    cached = config_manager.get_cached_config() if config_manager is not None else None
    if isinstance(cached, ProxyConfig):
        return cached.server_params.keepalive
    return KeepaliveParams()


class DatabaseConfigManager:
    """Configuration manager using database storage with optional Redis caching.

    This class provides a unified interface for configuration retrieval,
    combining database access with optional Redis caching for improved performance.

    The caching layer is optional and gracefully degrades if Redis is unavailable.
    Cache operations never block or fail the main configuration retrieval.
    """

    def __init__(self):
        self._config: ProxyConfig | None = None
        self._redis_cache: RedisCache | None = None
        self._cache_enabled: bool = False

    async def load(self) -> ProxyConfig:
        """Load configuration from database."""
        from llm_proxy.database.connection import get_async_session_context
        from llm_proxy.database.repositories import ConfigRepository

        async with get_async_session_context() as session:
            repo = ConfigRepository(session)

            provider_records = await repo.get_all_providers()
            provider_configs = {}
            for record in provider_records:
                provider_configs[record.name] = map_provider_record(record)

            model_records = await repo.get_all_models()
            models = {}
            for record in model_records:
                model_config = map_model_record(record)
                if model_config is None:
                    continue
                models[record.name] = model_config

            server_configs = await repo.get_all_server_config()
            server_config_dict = {}
            for config in server_configs:
                server_config_dict[config.key] = config.value

            auth_config = load_auth_config()
            logging_overrides = server_config_dict.get("logging")
            logging_config = load_logging_config(
                logging_overrides if isinstance(logging_overrides, dict) else None
            )
            redis_config = load_redis_config()

            web_search_config = None
            if "web_search_config" in server_config_dict:
                try:
                    from llm_proxy.config.types.web_search import WebSearchConfig

                    web_search_config = WebSearchConfig.model_validate(
                        server_config_dict["web_search_config"]
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse web_search_config: {e}")

            # Read global request policies from the request_policy config key
            request_policy = server_config_dict.get("request_policy", {})
            if not isinstance(request_policy, dict):
                request_policy = {}

            unknown_fields_policy = request_policy.get("unknown_fields_policy", "ignore")
            if unknown_fields_policy not in {"ignore", "passthrough", "error"}:
                logger.warning(
                    f"Invalid unknown_fields_policy '{unknown_fields_policy}' in server config; "
                    "falling back to 'ignore'"
                )
                unknown_fields_policy = "ignore"

            unsupported_block_policy = request_policy.get("unsupported_block_policy", "drop")
            if unsupported_block_policy not in {"drop", "degrade", "error"}:
                logger.warning(
                    f"Invalid unsupported_block_policy "
                    f"'{unsupported_block_policy}' in server config; "
                    "falling back to 'drop'"
                )
                unsupported_block_policy = "drop"

            # Read global resilience policies from the resilience config key
            resilience = server_config_dict.get("resilience", {})
            if not isinstance(resilience, dict):
                resilience = {}

            max_fallback_attempts = resilience.get(
                "max_fallback_attempts", DEFAULT_MAX_FALLBACK_ATTEMPTS
            )
            if not isinstance(max_fallback_attempts, int) or max_fallback_attempts < 0:
                logger.warning(
                    "Invalid max_fallback_attempts in server config; falling back to default"
                )
                max_fallback_attempts = DEFAULT_MAX_FALLBACK_ATTEMPTS

            global_max_retries = resilience.get("max_retries", DEFAULT_MAX_RETRIES)
            if not isinstance(global_max_retries, int) or global_max_retries < 0:
                logger.warning("Invalid max_retries in server config; falling back to default")
                global_max_retries = DEFAULT_MAX_RETRIES

            cb = resilience.get("circuit_breaker", {})
            if not isinstance(cb, dict):
                cb = {}
            circuit_breaker_params = CircuitBreakerParams(
                enabled=cb.get("enabled", True),
                failure_threshold=cb.get("failure_threshold", 5),
                cooldown_seconds=cb.get("cooldown_seconds", 60.0),
            )

            # Read global security / rate-limiting parameters from the
            # ``security`` server_config key (UI-managed, hot-reloaded).
            security_value = server_config_dict.get("security")
            try:
                security_params = SecurityParams.model_validate(
                    security_value if isinstance(security_value, dict) else {}
                )
            except Exception as e:
                logger.warning(f"Failed to parse security config: {e}")
                security_params = SecurityParams()

            # Non-streaming keepalive parameters (UI-managed, hot-reloaded).
            keepalive_value = server_config_dict.get("keepalive")
            try:
                keepalive_params = KeepaliveParams.model_validate(
                    keepalive_value if isinstance(keepalive_value, dict) else {}
                )
            except Exception as e:
                logger.warning(f"Failed to parse keepalive config: {e}")
                keepalive_params = KeepaliveParams()

            # Per-bucket rate limit overrides (UI-managed, hot-reloaded).
            rate_limits_value = server_config_dict.get("rate_limits")
            rate_limits: dict[str, str] = {}
            if isinstance(rate_limits_value, dict):
                rate_limits = {str(k): str(v) for k, v in rate_limits_value.items()}

            # Allowed CORS origins (UI-managed, hot-reloaded).
            cors_origins_value = server_config_dict.get("cors_origins")
            cors_origins: list[str] = []
            if isinstance(cors_origins_value, list):
                cors_origins = [str(o).strip() for o in cors_origins_value if str(o).strip()]

            server_params = ServerParams(
                auth=auth_config,
                logging=logging_config,
                web_search=web_search_config,
                unknown_fields_policy=unknown_fields_policy,
                unsupported_block_policy=unsupported_block_policy,
                max_fallback_attempts=max_fallback_attempts,
                max_retries=global_max_retries,
                circuit_breaker=circuit_breaker_params,
                security=security_params,
                keepalive=keepalive_params,
                rate_limits=rate_limits,
                cors_origins=cors_origins,
            )

            # Smart-routing global config, read from the same server_config
            # batch as logging/web_search/request_policy/resilience so the
            # hot path can read it from the cached ProxyConfig without a
            # per-request DB session or the global config lock.
            smart_routing_value = server_config_dict.get("smart_routing")
            try:
                from llm_proxy.config.types.smart_routing import SmartRoutingConfig

                smart_routing_config = SmartRoutingConfig.from_row(
                    smart_routing_value if isinstance(smart_routing_value, dict) else None
                )
            except Exception as e:
                logger.warning(f"Failed to parse smart_routing config: {e}")
                from llm_proxy.config.types.smart_routing import SmartRoutingConfig

                smart_routing_config = SmartRoutingConfig()

            # Global provider-selection strategy, loaded the same way so the
            # hot path reads it from the cached ProxyConfig.
            provider_selection_value = server_config_dict.get("provider_selection")
            from llm_proxy.config.types.provider_selection import ProviderSelectionConfig

            try:
                provider_selection_config = ProviderSelectionConfig.from_row(
                    provider_selection_value if isinstance(provider_selection_value, dict) else None
                )
            except Exception as e:
                logger.warning(f"Failed to parse provider_selection config: {e}")
                provider_selection_config = ProviderSelectionConfig()

            config = ProxyConfig(
                server_params=server_params,
                provider_configs=provider_configs,
                models=models,
                redis=redis_config,
                smart_routing=smart_routing_config,
                provider_selection=provider_selection_config,
            )
            self._config = config
            return config

    def enable_cache(self, redis_cache: RedisCache) -> None:
        """Enable Redis caching for configuration lookups.

        Args:
            redis_cache: The Redis cache instance to use
        """
        self._redis_cache = redis_cache
        self._cache_enabled = True
        logger.info("Redis cache enabled for configuration manager")

    def get_cached_config(self) -> ProxyConfig | None:
        """Return the in-memory cached config without triggering a load.

        Useful on sync hot paths (middleware, exception handlers) where the
        config has already been loaded at startup and is refreshed by
        :meth:`reload` whenever UI-managed settings change.
        """
        return self._config

    async def get_config(self) -> ProxyConfig:
        """Get the current configuration."""
        if not self._config:
            return await self.load()
        return self._config

    async def get_provider_config(self, provider: str) -> ProviderConfig:
        """Get configuration for a specific provider.

        This method first checks the Redis cache (if enabled), then falls
        back to the in-memory/database configuration.

        Args:
            provider: Provider name

        Returns:
            Provider configuration

        Raises:
            ProviderNotConfiguredError: If provider is not configured
        """
        if self._cache_enabled and self._redis_cache:
            try:
                cached = await self._redis_cache.get_provider_config(provider)
                if cached is not None:
                    return cached
            except Exception as e:
                logger.warning(f"Cache lookup failed for provider {provider}: {e}")

        # Fall back to in-memory/database config
        config = await self.get_config()
        if provider not in config.provider_configs:
            raise ProviderNotConfiguredError(provider)

        provider_config = config.provider_configs[provider]

        if self._cache_enabled and self._redis_cache:
            try:
                await self._redis_cache.set_provider_config(provider, provider_config)
            except Exception as e:
                logger.warning(f"Failed to cache provider config {provider}: {e}")

        return provider_config

    async def get_model_config(self, model: str) -> ModelConfig | None:
        """Get configuration for a specific model.

        This method first checks the Redis cache (if enabled), then falls
        back to the in-memory/database configuration.

        Args:
            model: Model name

        Returns:
            Model configuration or None if not found
        """
        if self._cache_enabled and self._redis_cache:
            try:
                cached = await self._redis_cache.get_model_config(model)
                if cached is not None:
                    return cached
            except Exception as e:
                logger.warning(f"Cache lookup failed for model {model}: {e}")

        config = await self.get_config()
        model_config = config.models.get(model)

        if model_config is not None and self._cache_enabled and self._redis_cache:
            try:
                await self._redis_cache.set_model_config(model, model_config)
            except Exception as e:
                logger.warning(f"Failed to cache model config {model}: {e}")

        return model_config

    async def get_all_models(self) -> dict[str, ModelConfig]:
        """Get all model configurations."""
        return (await self.get_config()).models

    async def get_smart_routing_config(self) -> SmartRoutingConfig:
        """Get the global smart-routing configuration.

        Reads from the cached :class:`ProxyConfig` populated during
        :meth:`load` (the same server_config batch as logging/resilience),
        avoiding a per-request DB session and the global config lock on the
        routing hot path.
        """
        return (await self.get_config()).smart_routing

    async def reload(self) -> None:
        """Reload configuration from database and invalidate cache."""
        self._config = None
        await self.load()
        await self.invalidate_all_cache()

    async def invalidate_all_cache(self) -> None:
        """Invalidate all cached configurations."""
        if self._cache_enabled and self._redis_cache:
            try:
                await self._redis_cache.invalidate_all()
            except Exception as e:
                logger.warning(f"Failed to invalidate all cache: {e}")


__all__ = [
    "DatabaseConfigManager",
    "load_auth_config",
    "load_logging_config",
    "load_redis_config",
    "resolve_logging_config",
    "resolve_keepalive_params",
    "resolve_security_params",
]
