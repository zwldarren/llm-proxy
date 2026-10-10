"""Tests for the Redis rate-limit backend.

The limiter's check-and-record runs as one Lua script so concurrent requests
cannot all read a stale count and overshoot the limit (the old two-pipeline
implementation's check-then-act race). fakeredis with the ``lua`` extra
executes that script with lupa, so a burst of concurrent task calls exercises
the real atomicity semantics without a live Redis server.
"""

import asyncio
import time
from typing import Any

import pytest
from fakeredis import aioredis as fakeredis_aioredis

from llm_proxy.api.middleware.rate_limiting import RedisRateLimiter


class _StubRedisClient:
    """Stand-in for ``database.redis_client.RedisClient``.

    ``get_redis_client_async`` only needs ``.client`` to be non-``None``; the
    real wrapper would insist on a real Redis connection.
    """

    def __init__(self, fake: Any):
        self.client = fake


@pytest.fixture
def fake_redis() -> Any:
    """A fresh fake Redis server (Lua enabled via lupa) per test."""
    return fakeredis_aioredis.FakeRedis()


@pytest.fixture
def redis_limiter(fake_redis: Any) -> RedisRateLimiter:
    return RedisRateLimiter(redis_client=_StubRedisClient(fake_redis))


@pytest.mark.asyncio
async def test_allowed_requests_record_entries(redis_limiter, fake_redis) -> None:
    """Each allowed request lands exactly one entry and refreshes the TTL."""
    for _ in range(3):
        is_limited, metadata = await redis_limiter.is_rate_limited("client-1", limit=5)
        assert is_limited is False
        assert metadata["remaining"] == 5 - _ - 1

    key = f"{redis_limiter.prefix}client-1"
    assert await fake_redis.zcard(key) == 3
    # Entries are unique members: same-second requests must not collapse.
    members = await fake_redis.zrange(key, 0, -1)
    assert len({m.decode() if isinstance(m, bytes) else m for m in members}) == 3
    assert await fake_redis.ttl(key) > 0


@pytest.mark.asyncio
async def test_concurrent_burst_never_overshoots_limit(redis_limiter) -> None:
    """N concurrent requests against a limit of L allow exactly L of them.

    This is the regression test for the old implementation's race: the ZCARD
    check and the ZADD record ran as two pipelines, so every concurrent
    request could read the same pre-burst count and all record themselves.
    """
    limit = 5
    concurrency = 12
    results = await asyncio.gather(
        *(redis_limiter.is_rate_limited("burst-client", limit=limit) for _ in range(concurrency))
    )
    allowed = sum(1 for is_limited, _ in results if not is_limited)
    denied = sum(1 for is_limited, _ in results if is_limited)
    assert allowed == limit
    assert denied == concurrency - limit


@pytest.mark.asyncio
async def test_denied_request_reports_oldest_expiry(redis_limiter, fake_redis) -> None:
    """A denied request's reset time tracks the oldest surviving entry."""
    limit = 3
    for _ in range(limit):
        await redis_limiter.is_rate_limited("client-2", limit=limit)

    before = time.time()
    is_limited, metadata = await redis_limiter.is_rate_limited("client-2", limit=limit)
    assert is_limited is True
    assert metadata["remaining"] == 0

    key = f"{redis_limiter.prefix}client-2"
    oldest = float((await fake_redis.zrange(key, 0, 0, withscores=True))[0][1])
    assert metadata["reset_time"] == pytest.approx(oldest + 60)
    assert metadata["reset_time"] >= before  # not a truncated integer


@pytest.mark.asyncio
async def test_denied_requests_do_not_grow_the_window(redis_limiter, fake_redis) -> None:
    """Hits against a full window record nothing."""
    limit = 3
    for _ in range(limit):
        await redis_limiter.is_rate_limited("client-3", limit=limit)

    for _ in range(5):
        await redis_limiter.is_rate_limited("client-3", limit=limit)

    assert await fake_redis.zcard(f"{redis_limiter.prefix}client-3") == limit


@pytest.mark.asyncio
async def test_window_slides(redis_limiter, fake_redis) -> None:
    """Entries older than the window are cleared before the count is read."""
    key = f"{redis_limiter.prefix}client-4"
    now = time.time()
    window = 60
    # Seed the key with entries that are just over the window old.
    await fake_redis.zadd(
        key,
        {f"{now - window - 5:.17g}:stale-{i}": now - window - 5 for i in range(3)},
    )
    await fake_redis.expire(key, window)

    is_limited, metadata = await redis_limiter.is_rate_limited(
        "client-4", limit=3, window_size=window
    )
    assert is_limited is False
    # The stale entries were swept before this request was recorded.
    assert await fake_redis.zcard(key) == 1
    assert metadata["remaining"] == 2


@pytest.mark.asyncio
async def test_identifiers_are_isolated(redis_limiter) -> None:
    """Different identifiers have independent windows."""
    for _ in range(3):
        await redis_limiter.is_rate_limited("client-a", limit=3)
    is_limited, _ = await redis_limiter.is_rate_limited("client-b", limit=3)
    assert is_limited is False


@pytest.mark.asyncio
async def test_separate_limiter_instances_share_server_state(fake_redis) -> None:
    """Two limiter objects on one server see the same window.

    Simulates multiple uvicorn workers: the limit must hold per server, not
    per limiter instance, which is the whole point of the Redis backend.
    """
    limit = 4
    limiters = [RedisRateLimiter(redis_client=_StubRedisClient(fake_redis)) for _ in range(2)]

    results = await asyncio.gather(
        *(limiters[i % 2].is_rate_limited("shared-client", limit=limit) for i in range(limit))
    )
    assert all(not is_limited for is_limited, _ in results)

    fifth, _ = await limiters[0].is_rate_limited("shared-client", limit=limit)
    assert fifth is True
