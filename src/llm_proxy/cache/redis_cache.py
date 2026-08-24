"""Redis caching layer for provider configurations and model mappings.

This module provides a Redis-based caching layer for frequently accessed
configuration data like provider configurations and model mappings.
"""

import asyncio
from typing import Any

import orjson
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from llm_proxy.config.types.model import ModelConfig
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.config.types.redis import RedisCacheConfig
from llm_proxy.database.redis_client import RedisClient, get_redis_client_async
from llm_proxy.database.redis_utils import delete_keys_pipeline, scan_keys
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


_CONNECTION_ERRORS = (
    ConnectionError,
    TimeoutError,
    RedisConnectionError,
    RedisTimeoutError,
    OSError,
)


def _is_connection_error(exc: Exception) -> bool:
    if isinstance(exc, _CONNECTION_ERRORS):
        return True
    if exc.__cause__ is not None and isinstance(exc.__cause__, Exception):
        return _is_connection_error(exc.__cause__)
    return False


def _empty_cache_stats() -> dict[str, int]:
    return {
        "hits": 0,
        "misses": 0,
        "sets": 0,
        "deletes": 0,
        "errors": 0,
    }


class RedisCache:
    """Redis-based cache for provider configurations and model mappings.

    This cache provides:
    - Provider configuration caching with configurable TTL
    - Model mapping caching with configurable TTL
    - Cache invalidation on configuration changes
    - Metrics and monitoring support
    - Cached Redis async client for connection reuse
    """

    def __init__(
        self,
        redis_client: RedisClient | None = None,
        config: RedisCacheConfig | None = None,
    ):
        self._redis_client = redis_client
        self.config = config or RedisCacheConfig()
        self._stats = _empty_cache_stats()

        self._cached_async_client = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> Any:
        if self._cached_async_client is not None:
            return self._cached_async_client

        async with self._client_lock:
            if self._cached_async_client is not None:
                return self._cached_async_client

            self._cached_async_client = await get_redis_client_async(self._redis_client)
            return self._cached_async_client

    async def _invalidate_cached_client(self) -> None:
        async with self._client_lock:
            self._cached_async_client = None

    def _get_key(self, key_type: str, key: str) -> str:
        return f"{self.config.prefix}{key_type}:{key}"

    async def get_provider_config(self, provider_name: str) -> ProviderConfig | None:
        """Get cached provider configuration."""
        try:
            redis_client = await self._get_client()
            key = self._get_key("provider", provider_name)
            data = await redis_client.get(key)

            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            config_dict = orjson.loads(data)
            return ProviderConfig(**config_dict)

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Failed to get provider config from cache: {e}")
            if _is_connection_error(e):
                await self._invalidate_cached_client()
            return None

    async def set_provider_config(
        self,
        provider_name: str,
        config: ProviderConfig,
        ttl: int | None = None,
    ) -> bool:
        """Cache provider configuration.

        Args:
            provider_name: Name of the provider
            config: Provider configuration to cache
            ttl: TTL in seconds. If None, uses configured default.

        Returns:
            True if cached successfully, False otherwise
        """
        try:
            redis_client = await self._get_client()
            key = self._get_key("provider", provider_name)

            data = orjson.dumps(config.model_dump())

            cache_ttl = ttl or self.config.ttl_provider_config
            await redis_client.setex(key, cache_ttl, data)

            self._stats["sets"] += 1
            return True

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Failed to cache provider config: {e}")
            if _is_connection_error(e):
                await self._invalidate_cached_client()
            return False

    async def invalidate_provider_config(self, provider_name: str) -> bool:
        """Invalidate cached provider configuration.

        Args:
            provider_name: Name of the provider

        Returns:
            True if invalidated successfully, False otherwise
        """
        try:
            redis_client = await self._get_client()
            key = self._get_key("provider", provider_name)
            result = await redis_client.delete(key)

            self._stats["deletes"] += 1
            return result > 0

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Failed to invalidate provider config cache: {e}")
            if _is_connection_error(e):
                await self._invalidate_cached_client()
            return False

    async def get_model_config(self, model_name: str) -> ModelConfig | None:
        """Get cached model configuration."""
        try:
            redis_client = await self._get_client()
            key = self._get_key("model", model_name)
            data = await redis_client.get(key)

            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            config_dict = orjson.loads(data)
            return ModelConfig(**config_dict)

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Failed to get model config from cache: {e}")
            if _is_connection_error(e):
                await self._invalidate_cached_client()
            return None

    async def set_model_config(
        self,
        model_name: str,
        config: ModelConfig,
        ttl: int | None = None,
    ) -> bool:
        """Cache model configuration.

        Args:
            model_name: Name of the model
            config: Model configuration to cache
            ttl: TTL in seconds. If None, uses configured default.

        Returns:
            True if cached successfully, False otherwise
        """
        try:
            redis_client = await self._get_client()
            key = self._get_key("model", model_name)

            data = orjson.dumps(config.model_dump())

            cache_ttl = ttl or self.config.ttl_model_mapping
            await redis_client.setex(key, cache_ttl, data)

            self._stats["sets"] += 1
            return True

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Failed to cache model config: {e}")
            if _is_connection_error(e):
                await self._invalidate_cached_client()
            return False

    async def invalidate_model_config(self, model_name: str) -> bool:
        """Invalidate cached model configuration.

        Args:
            model_name: Name of the model

        Returns:
            True if invalidated successfully, False otherwise
        """
        try:
            redis_client = await self._get_client()
            key = self._get_key("model", model_name)
            result = await redis_client.delete(key)

            self._stats["deletes"] += 1
            return result > 0

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Failed to invalidate model config cache: {e}")
            if _is_connection_error(e):
                await self._invalidate_cached_client()
            return False

    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        try:
            redis_client = await self._get_client()
            full_key = f"{self.config.prefix}{key}"
            data = await redis_client.get(full_key)

            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return orjson.loads(data)

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Failed to get from cache: {e}")
            if _is_connection_error(e):
                await self._invalidate_cached_client()
            return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set a value in cache."""
        try:
            redis_client = await self._get_client()
            full_key = f"{self.config.prefix}{key}"
            data = orjson.dumps(value)
            await redis_client.setex(full_key, ttl, data)

            self._stats["sets"] += 1
            return True

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Failed to set cache: {e}")
            if _is_connection_error(e):
                await self._invalidate_cached_client()
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        try:
            redis_client = await self._get_client()
            full_key = f"{self.config.prefix}{key}"
            result = await redis_client.delete(full_key)

            self._stats["deletes"] += 1
            return result > 0

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Failed to delete from cache: {e}")
            if _is_connection_error(e):
                await self._invalidate_cached_client()
            return False

    async def invalidate_all(self) -> int:
        """Invalidate all cached entries. Returns number of keys deleted."""
        try:
            redis_client = await self._get_client()
            pattern = f"{self.config.prefix}*"

            all_keys = await scan_keys(redis_client, pattern, count=1000)
            deleted_count = 0
            if all_keys:
                deleted_count = await delete_keys_pipeline(redis_client, all_keys)

            self._stats["deletes"] += deleted_count
            logger.info(f"Invalidated {deleted_count} cache entries")
            return deleted_count

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Failed to invalidate all cache entries: {e}")
            if _is_connection_error(e):
                await self._invalidate_cached_client()
            return 0


# Global Redis cache instance
_redis_cache: RedisCache | None = None


async def get_redis_cache(
    redis_client: RedisClient | None = None,
    config: RedisCacheConfig | None = None,
) -> RedisCache:
    """Get global Redis cache instance.

    Args:
        redis_client: Optional Redis client to use
        config: Optional cache configuration

    Returns:
        RedisCache instance
    """
    global _redis_cache

    if _redis_cache is None:
        _redis_cache = RedisCache(
            redis_client=redis_client,
            config=config,
        )

    return _redis_cache


async def invalidate_cache(
    provider_name: str | None = None,
    model_name: str | None = None,
) -> None:
    """Invalidate cache entries.

    Args:
        provider_name: If provided, invalidate this provider's cache
        model_name: If provided, invalidate this model's cache

    If neither is provided, invalidates all cache entries.
    """
    global _redis_cache

    if _redis_cache is None:
        return

    if provider_name:
        await _redis_cache.invalidate_provider_config(provider_name)

    if model_name:
        await _redis_cache.invalidate_model_config(model_name)

    if not provider_name and not model_name:
        await _redis_cache.invalidate_all()


__all__ = [
    "RedisCache",
    "get_redis_cache",
    "invalidate_cache",
]
