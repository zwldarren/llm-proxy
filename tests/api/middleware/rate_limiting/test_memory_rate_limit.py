"""Tests for the in-memory rate-limit backend."""

import asyncio
import threading

import pytest

from llm_proxy.api.middleware.rate_limiting import RateLimitManager


@pytest.fixture
def memory_manager():
    """Return a fresh in-memory rate limit manager."""
    return RateLimitManager(use_redis=False)


def test_backend_names_the_store():
    """The store decides whether a configured limit is per deployment or per process."""
    assert RateLimitManager(use_redis=False).backend == "memory"
    assert RateLimitManager(use_redis=True).backend == "redis"


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


@pytest.mark.asyncio
async def test_memory_backend_isolates_buckets(memory_manager) -> None:
    """Two buckets watching one identifier keep separate counters.

    Keyed by identifier alone, the 10/minute ``auth.setup_status`` polls a
    dashboard makes would consume the 5/minute ``auth.login`` budget for the same
    IP, so a login is refused before its credentials are ever checked.
    """
    for _ in range(10):
        is_limited, _ = await memory_manager.check_rate_limit(
            "127.0.0.1", limit=10, window_size=60, bucket="auth.setup_status"
        )
        assert is_limited is False

    for _ in range(5):
        is_limited, _ = await memory_manager.check_rate_limit(
            "127.0.0.1", limit=5, window_size=60, bucket="auth.login"
        )
        assert is_limited is False

    assert set(memory_manager._memory_windows) == {
        "auth.setup_status:127.0.0.1",
        "auth.login:127.0.0.1",
    }


@pytest.mark.asyncio
async def test_unbucketed_callers_share_one_window(memory_manager) -> None:
    """A caller that passes no bucket deliberately gets one shared counter.

    This is how the per-API-key requests/minute cap works: one window per key,
    however many entry points spend it.
    """
    for _ in range(2):
        is_limited, _ = await memory_manager.check_rate_limit(
            "apikey:dev-key", limit=2, window_size=60
        )
        assert is_limited is False
    is_limited, _ = await memory_manager.check_rate_limit("apikey:dev-key", limit=2, window_size=60)
    assert is_limited is True


@pytest.mark.asyncio
async def test_memory_backend_sweeps_quiet_identifiers(memory_manager) -> None:
    """Identifiers that stop being used do not accumulate forever.

    Each window list expires on its own, but only a sweep removes the dict entry
    underneath it — Redis gets a TTL per key, the memory backend needs a sweep.
    """
    for index in range(20):
        await memory_manager.check_rate_limit(f"client-sweep-{index}", limit=5, window_size=0.05)
    assert len(memory_manager._memory_windows) == 20

    # Beyond the sweep interval, with every stored timestamp outside the horizon.
    memory_manager._memory_next_cleanup = 0.0
    await asyncio.sleep(0.1)
    await memory_manager.check_rate_limit("client-fresh", limit=5, window_size=0.05)

    assert "client-sweep-0" not in memory_manager._memory_windows
    assert "client-fresh" in memory_manager._memory_windows


def test_memory_backend_shares_state_across_event_loops() -> None:
    """A second event loop in the same process counts in the same windows.

    The window dict is process-wide, so it needs a loop-agnostic lock. Guarding it
    with an ``asyncio.Lock`` (bound to whichever loop created it) left a second
    loop free to mutate the dict concurrently — a real setup, because
    ``uvicorn --reload``, ``asyncio.to_thread`` and ``TestClient`` all put an app
    on more than one loop over a process's life.
    """
    manager = RateLimitManager(use_redis=False)
    window = asyncio.run(_spend(manager, "client-loops", 3))

    # Two more fit the 5-request budget when counted from an unrelated loop in
    # another thread; the third is refused.
    assert window == 3
    assert _from_another_loop(manager, "client-loops") is False
    assert _from_another_loop(manager, "client-loops") is False
    assert _from_another_loop(manager, "client-loops") is True


async def _spend(manager: RateLimitManager, identifier: str, times: int) -> int:
    """Record ``times`` requests and return how many were allowed."""
    results = [
        await manager.check_rate_limit(identifier, limit=5, window_size=60) for _ in range(times)
    ]
    return sum(1 for limited, _ in results if not limited)


def _from_another_loop(manager: RateLimitManager, identifier: str) -> bool:
    """One request recorded on a fresh event loop in a fresh thread."""
    outcome: list[bool] = []

    def run() -> None:
        limited, _ = asyncio.run(manager.check_rate_limit(identifier, limit=5, window_size=60))
        outcome.append(limited)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return outcome[0]
