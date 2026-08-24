"""Tests for the in-memory rate-limit backend."""

import asyncio

import pytest

from llm_proxy.api.middleware.rate_limiting import RateLimitManager


@pytest.fixture
def memory_manager():
    """Return a fresh in-memory rate limit manager."""
    return RateLimitManager(use_redis=False)


@pytest.mark.asyncio
async def test_memory_backend_allows_requests_under_limit(memory_manager) -> None:
    """Requests within the limit are allowed."""
    for _ in range(3):
        is_limited, metadata = await memory_manager.check_rate_limit(
            "client-1", limit=5, window_size=60
        )
        assert is_limited is False
        assert metadata["remaining"] == 5 - _ - 1


@pytest.mark.asyncio
async def test_memory_backend_blocks_requests_over_limit(memory_manager) -> None:
    """Requests beyond the limit are blocked."""
    for _ in range(5):
        is_limited, _ = await memory_manager.check_rate_limit("client-2", limit=5, window_size=60)
        assert is_limited is False

    is_limited, metadata = await memory_manager.check_rate_limit(
        "client-2", limit=5, window_size=60
    )
    assert is_limited is True
    assert metadata["remaining"] == 0
    assert metadata["reset_time"] > 0


@pytest.mark.asyncio
async def test_memory_backend_isolates_identifiers(memory_manager) -> None:
    """Different identifiers have independent windows."""
    for _ in range(5):
        await memory_manager.check_rate_limit("client-a", limit=5, window_size=60)
    is_limited, _ = await memory_manager.check_rate_limit("client-b", limit=5, window_size=60)
    assert is_limited is False


@pytest.mark.asyncio
async def test_memory_backend_window_expires(memory_manager) -> None:
    """Old entries are removed after the window expires."""
    for _ in range(5):
        await memory_manager.check_rate_limit("client-3", limit=5, window_size=0.1)

    # Wait for the window to expire with a comfortable margin for slow CI.
    await asyncio.sleep(0.3)

    is_limited, _ = await memory_manager.check_rate_limit("client-3", limit=5, window_size=0.1)
    assert is_limited is False
