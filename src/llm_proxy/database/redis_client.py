"""Redis client for LLM Proxy."""

from typing import Any, cast

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

from llm_proxy.config.types.redis import RedisConfig
from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Async Redis client with connection pooling and health checks."""

    def __init__(self, config: RedisConfig | None = None):
        """Initialize Redis client.

        Args:
            config: Redis configuration. If None, Redis will be disabled.
        """
        self.config = config
        self._client: redis.Redis | None = None
        self._pool: ConnectionPool | None = None

    async def initialize(self) -> None:
        """Initialize Redis connection if enabled."""
        if not self.config or not self.config.enabled:
            logger.info("Redis is disabled, skipping initialization")
            return

        try:
            # Create connection pool
            self._pool = ConnectionPool.from_url(
                self.config.url,
                max_connections=self.config.pool_size,
                socket_timeout=self.config.timeout,
                socket_connect_timeout=self.config.timeout,
            )

            # Create Redis client
            self._client = redis.Redis(
                connection_pool=self._pool,
                decode_responses=True,
            )

            try:
                await cast(Any, self._client.ping())
                logger.info(f"Redis connected to {self.config.url}")
            except Exception as ping_error:
                raise ConfigurationError(f"Redis ping failed: {ping_error}") from ping_error

        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self._client = None
            self._pool = None
            raise ConfigurationError(f"Redis initialization failed: {e}") from e

    async def close(self) -> None:
        """Close Redis connection pool."""
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
            self._client = None
            logger.info("Redis connection pool closed")

    @property
    def client(self) -> redis.Redis | None:
        """Get Redis client instance."""
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected and ready."""
        return self._client is not None

    async def health_check(self) -> bool:
        """Perform health check on Redis connection."""
        if not self.is_connected or not self._client:
            return False

        try:
            await cast(Any, self._client.ping())
            return True
        except RedisError as e:
            logger.warning(f"Redis health check failed: {e}")
            return False


_redis_client: RedisClient | None = None


async def get_redis_client(config: RedisConfig | None = None) -> RedisClient:
    """Get or create Redis client singleton.

    Args:
        config: Redis configuration. Required for first call.

    Returns:
        RedisClient instance
    """
    global _redis_client

    if _redis_client is None:
        if config is None:
            raise ConfigurationError("Redis configuration is required for first call")
        _redis_client = RedisClient(config)
        await _redis_client.initialize()

    return _redis_client


async def close_redis_client() -> None:
    """Close Redis client singleton."""
    global _redis_client

    if _redis_client:
        await _redis_client.close()
        _redis_client = None


async def get_redis_client_async(
    redis_client: RedisClient | None = None,
) -> redis.Redis:
    """Get Redis client instance from injected client or global singleton.

    Raises:
        RuntimeError: If no Redis client is available
    """
    try:
        client = redis_client if redis_client is not None else await get_redis_client()
    except Exception as e:
        logger.warning(f"Failed to get Redis client: {e}")
        raise ConfigurationError(
            "Redis client is not available. Make sure Redis is enabled and configured."
        ) from e

    if client is None or client.client is None:
        raise ConfigurationError(
            "Redis client is not available. Make sure Redis is enabled and configured."
        )
    return client.client


__all__ = [
    "RedisClient",
    "get_redis_client",
    "close_redis_client",
    "get_redis_client_async",
]
