"""Shared Redis utility functions for common operations across the codebase."""

from typing import TYPE_CHECKING

from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    import redis.asyncio.client

logger = get_logger(__name__)


async def scan_keys(
    redis_client: redis.asyncio.client.Redis,
    pattern: str,
    count: int = 100,
) -> list[str]:
    """Scan Redis keys matching a pattern.

    This is a shared utility used by both RedisCache and RedisRateLimiter
    to avoid code duplication.

    Args:
        redis_client: Redis client instance
        pattern: Key pattern to match
        count: Number of keys to return per scan iteration

    Returns:
        List of matching keys
    """
    all_keys: list[str] = []
    cursor = 0
    while True:
        cursor, keys = await redis_client.scan(
            cursor=cursor,
            match=pattern,
            count=count,
        )
        all_keys.extend([k.decode() if isinstance(k, bytes) else k for k in keys])
        if cursor == 0:
            break
    return all_keys


async def delete_keys_pipeline(
    redis_client: redis.asyncio.client.Redis,
    keys: list[str],
    batch_size: int = 100,
) -> int:
    """Delete keys using pipeline for better performance.

    Uses pipeline to batch delete operations, reducing round-trips.

    Args:
        redis_client: Redis client instance
        keys: List of keys to delete
        batch_size: Number of keys per pipeline batch

    Returns:
        Total number of keys deleted
    """
    if not keys:
        return 0

    total_deleted = 0

    for i in range(0, len(keys), batch_size):
        batch = keys[i : i + batch_size]
        async with redis_client.pipeline(transaction=False) as pipe:
            for key in batch:
                pipe.delete(key)
            results = await pipe.execute()
            total_deleted += sum(1 for r in results if r)

    return total_deleted


__all__ = ["scan_keys", "delete_keys_pipeline"]
