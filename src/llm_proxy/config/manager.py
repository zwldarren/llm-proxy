"""Database-backed configuration manager with optional Redis caching."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

# Re-exported: the implementation moved to the server settings registry
# (ADR-0026), which owns both the admin endpoints and this runtime read.
from llm_proxy.config.server_settings import load_logging_config, load_server_settings
from llm_proxy.core.exceptions import ProviderNotConfiguredError
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.cache.redis_cache import RedisCache
    from llm_proxy.config.types.smart_routing import SmartRoutingConfig

from .mappers import map_model_record, map_provider_record
from .settings import get_settings
from .types import (
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

#: Upper bound on how long a worker may serve its in-memory snapshot before a
#: normal lookup re-checks the shared configuration generation. Additions land
#: immediately through the miss path (``get_model_config``); this interval
#: bounds how long edits and deletions can stay invisible on a peer worker.
_CONFIG_GENERATION_CHECK_INTERVAL = 1.0

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
        # Shared Redis channel for peer-visible change tokens. Set whenever
        # Redis is connected, *independently* of ``_cache_enabled``: the
        # generation token is a change *signal* for process-local runtime
        # state (web search, circuit breaker, MCP processes, API-key and role
        # caches), not part of the provider/model cache it reads through.
        self._generation_cache: RedisCache | None = None
        # Cross-process config freshness. ``_generation`` identifies the
        # snapshot this process holds; peers publish a new token on every
        # mutation so a stale worker reloads instead of serving an outdated
        # provider/model set. See ``_refresh_if_generation_changed``.
        self._generation: str | None = None
        self._generation_checked_at: float = 0.0
        self._refresh_lock = asyncio.Lock()
        # Listeners notified after the snapshot is swapped, on both a local
        # ``reload()`` and a peer-triggered generation refresh. They own the
        # process-local runtime state derived from config (web-search
        # interceptor, circuit-breaker thresholds, MCP server processes) so a
        # peer's settings change is applied by *every* worker, not just the one
        # that handled the admin request. Listeners must not call back into
        # this manager (``get_config``/``reload``): they receive the fresh
        # ``ProxyConfig`` they need as an argument.
        self._reload_listeners: list[Callable[[ProxyConfig], Awaitable[None]]] = []

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

            # UI-managed server settings: one tolerant typed parse over the
            # same registry that drives the admin endpoints (ADR-0026).
            settings = load_server_settings(server_config_dict)

            server_params = ServerParams(
                auth=load_auth_config(),
                console_auth=settings.console_auth,
                logging=settings.logging,
                web_search=settings.web_search,
                unknown_fields_policy=settings.request_policy_unknown_fields,
                unsupported_block_policy=settings.request_policy_unsupported_block,
                max_fallback_attempts=settings.resilience.max_fallback_attempts,
                max_retries=settings.resilience.max_retries,
                circuit_breaker=settings.resilience.circuit_breaker,
                security=settings.security,
                keepalive=settings.keepalive,
                rate_limits=settings.rate_limits,
                cors_origins=settings.cors_origins,
            )

            config = ProxyConfig(
                server_params=server_params,
                provider_configs=provider_configs,
                models=models,
                redis=load_redis_config(),
                smart_routing=settings.smart_routing,
                provider_selection=settings.provider_selection,
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
        if self._generation_cache is None:
            self._generation_cache = redis_cache
        logger.info("Redis cache enabled for configuration manager")

    def set_generation_channel(self, redis_cache: RedisCache) -> None:
        """Enable the shared config-generation channel without config caching.

        The generation token is how peers learn that *any* worker mutated
        configuration (including tables read outside ``ProxyConfig``), and it
        drives the reload listeners that own process-local runtime state. It
        must therefore be available whenever Redis is connected, not only when
        ``REDIS_CACHE_ENABLED`` turns on the provider/model cache.
        """
        self._generation_cache = redis_cache
        logger.info("Redis config-generation channel enabled")

    def add_reload_listener(self, listener: Callable[[ProxyConfig], Awaitable[None]]) -> None:
        """Register a coroutine called after every config snapshot swap.

        Runs on a local :meth:`reload` and on a peer-triggered generation
        refresh, which is how process-local runtime state stays in sync across
        workers. A listener must not call back into this manager.
        """
        self._reload_listeners.append(listener)

    async def _notify_reload_listeners(self, config: ProxyConfig) -> None:
        """Fan the fresh snapshot out to registered listeners.

        A failing listener is logged and skipped: it must never abort config
        loading or block the other listeners.
        """
        for listener in self._reload_listeners:
            try:
                await listener(config)
            except Exception as e:
                logger.warning(f"Config reload listener failed: {e}", exc_info=e)

    def get_cached_config(self) -> ProxyConfig | None:
        """Return the in-memory cached config without triggering a load.

        Useful on sync hot paths (middleware, exception handlers) where the
        config has already been loaded at startup and is refreshed by
        :meth:`reload` whenever UI-managed settings change.
        """
        return self._config

    async def get_config(self) -> ProxyConfig:
        """Get the current configuration.

        When Redis caching is enabled, this also re-checks the shared
        configuration generation (at most once per
        ``_CONFIG_GENERATION_CHECK_INTERVAL``) so a peer worker's changes are
        picked up without a restart.
        """
        if not self._config:
            return await self.load()
        await self._refresh_if_generation_changed()
        return self._config

    async def get_provider_config(self, provider: str) -> ProviderConfig:
        """Get configuration for a specific provider.

        Reads the in-memory :class:`ProxyConfig` first — it is loaded in full at
        startup and swapped on every settings change, so it is authoritative.
        The Redis cache is consulted only when the in-memory snapshot does not
        have the entry, which keeps a per-request Redis round trip off the hot
        path for data that ``get_config()`` already holds.

        Args:
            provider: Provider name

        Returns:
            Provider configuration

        Raises:
            ProviderNotConfiguredError: If provider is not configured
        """
        config = await self.get_config()
        if provider in config.provider_configs:
            return config.provider_configs[provider]

        if self._cache_enabled and self._redis_cache:
            try:
                cached = await self._redis_cache.get_provider_config(provider)
                if cached is not None:
                    return cached
            except Exception as e:
                logger.warning(f"Cache lookup failed for provider {provider}: {e}")

        # A peer worker may have added the provider after this process loaded
        # its snapshot. Bypass the refresh interval so the first request for a
        # newly added provider succeeds on every worker, then retry once.
        if await self._refresh_if_generation_changed(force=True):
            refreshed = (await self.get_config()).provider_configs
            if provider in refreshed:
                return refreshed[provider]

        raise ProviderNotConfiguredError(provider)

    async def get_model_config(self, model: str) -> ModelConfig | None:
        """Get configuration for a specific model.

        In-memory :class:`ProxyConfig` first (authoritative, loaded in full and
        swapped on reload); Redis only as a fallback for models the in-memory
        snapshot does not carry.

        Args:
            model: Model name

        Returns:
            Model configuration or None if not found
        """
        config = await self.get_config()
        model_config = config.models.get(model)
        if model_config is not None:
            return model_config

        if self._cache_enabled and self._redis_cache:
            try:
                cached = await self._redis_cache.get_model_config(model)
                if cached is not None:
                    return cached
            except Exception as e:
                logger.warning(f"Cache lookup failed for model {model}: {e}")

        # A peer worker may have added the model after this process loaded its
        # snapshot. Bypass the refresh interval so the very first request for a
        # newly added model succeeds on every worker, then retry once.
        if await self._refresh_if_generation_changed(force=True):
            return (await self.get_config()).models.get(model)

        return None

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
        """Reload configuration from database and invalidate cache.

        Notifies reload listeners so this worker's process-local runtime state
        (web search, circuit breaker, MCP servers) follows the new snapshot,
        then publishes a fresh configuration generation so peer worker
        processes (multi-worker deployments share Redis) drop their stale
        in-memory snapshots on their next lookup.
        """
        self._config = None
        config = await self.load()
        await self._notify_reload_listeners(config)
        await self.invalidate_all_cache()
        await self._publish_generation()

    async def publish_generation(self) -> None:
        """Publish a fresh generation token for changes outside server_config.

        Provider/model tables, the MCP server table, API keys and user rows are
        read outside :class:`ProxyConfig`; an admin mutation there still needs
        peer workers to refresh their snapshot and re-run the reload listeners.
        A no-op when Redis is not connected — then this process is the only
        worker, and every mutation already reloads its own state.
        """
        await self._publish_generation()

    async def sync_generation(self) -> None:
        """Adopt the shared generation token without reloading.

        Called once at startup, after Redis caching is enabled, so the first
        request does not trigger a redundant reload for a generation this
        process already loaded from the database.
        """
        cache = self._generation_cache
        if cache is None:
            return
        try:
            self._generation = await cache.get_config_generation()
        except Exception as e:
            logger.warning(f"Failed to sync config generation: {e}")

    async def _publish_generation(self) -> None:
        """Publish a new configuration generation token for peer workers."""
        cache = self._generation_cache
        if cache is None:
            return
        token = uuid.uuid4().hex
        try:
            published = await cache.set_config_generation(token)
        except Exception as e:
            logger.warning(f"Failed to publish config generation: {e}")
            return
        if published:
            self._generation = token

    async def _refresh_if_generation_changed(self, *, force: bool = False) -> bool:
        """Reload the snapshot when a peer worker published a new generation.

        Returns ``True`` when the config was reloaded. Checks are throttled to
        ``_CONFIG_GENERATION_CHECK_INTERVAL`` seconds unless ``force`` is set,
        so the hot path pays at most one shared-cache read per interval. A
        no-op when Redis is not connected: a single process already reloads
        on every mutation.
        """
        cache = self._generation_cache
        if cache is None:
            return False

        now = time.monotonic()
        if not force and now - self._generation_checked_at < _CONFIG_GENERATION_CHECK_INTERVAL:
            return False

        refreshed: ProxyConfig | None = None
        async with self._refresh_lock:
            now = time.monotonic()
            if not force and now - self._generation_checked_at < _CONFIG_GENERATION_CHECK_INTERVAL:
                return False
            self._generation_checked_at = now

            try:
                generation = await cache.get_config_generation()
            except Exception as e:
                logger.warning(f"Config generation check failed: {e}")
                return False

            if generation is None or generation == self._generation:
                return False

            refreshed = await self.load()
            self._generation = generation

        # Notify outside the refresh lock: a listener may open a database
        # session (MCP reconcile), and must never risk a re-entrant lock wait.
        if refreshed is not None:
            await self._notify_reload_listeners(refreshed)
            logger.debug("Config refreshed after a peer worker changed it")
            return True
        return False

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
