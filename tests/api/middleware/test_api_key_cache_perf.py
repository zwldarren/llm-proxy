"""Regression tests for API-key cache hot-path performance guards.

Covers the three fixes that removed the load-test death spiral:

- ``claim_last_used_update`` throttles the per-request ``last_used_at`` write
  that previously exhausted the DB connection pool.
- ``get_cached_api_keys`` is single-flight, so a TTL rollover refreshes the DB
  once per worker instead of once per in-flight request.
- A TTL-expired verified-key entry is revalidated by its recorded bcrypt hash
  (no bcrypt), so the cache rollover no longer stalls the worker.
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.api.middleware import api_key_cache as cache_module
from llm_proxy.api.middleware import mcp_proxy as mcp_proxy_module
from llm_proxy.api.middleware.api_key_cache import (
    ApiKeyCache,
    CachedApiKey,
    claim_last_used_update,
    get_api_key_cache,
    get_budget_spend_cache,
    hash_api_key_for_cache,
)
from llm_proxy.api.middleware.mcp_proxy import verify_api_key_for_mcp
from llm_proxy.core.budget import BudgetEnvelope


def _cached_key(**overrides: Any) -> CachedApiKey:
    defaults: dict[str, Any] = {
        "name": "test-key",
        "key_hash": "hash123",
        "is_active": True,
        "allowed_models": None,
        "allowed_mcp_servers": None,
        "user_id": 1,
        "user_allowed_models": None,
        "user_is_active": True,
        "expires_at": None,
        "budget": BudgetEnvelope(),
        "user_budget": BudgetEnvelope(),
    }
    defaults.update(overrides)
    return CachedApiKey(**defaults)


@pytest.fixture(autouse=True)
def _reset_caches():
    """Reset global caches and single-flight locks between tests.

    The asyncio locks are process-global; clearing them keeps each test's
    lock bound to that test's event loop.
    """
    get_api_key_cache().invalidate()
    get_budget_spend_cache().invalidate()
    cache_module._api_key_refresh_lock = None
    cache_module._key_verification_lock = None
    yield
    get_api_key_cache().invalidate()
    get_budget_spend_cache().invalidate()
    cache_module._api_key_refresh_lock = None
    cache_module._key_verification_lock = None


def _make_stale(cache: ApiKeyCache, sha: str) -> None:
    """Backdate a verified entry so it reads as past its TTL."""
    info = cache._verified_keys[sha]
    info.verified_at = time.time() - (cache.ttl + 5)


def test_claim_last_used_update_throttles_per_key():
    """Only the first call for a key inside the interval is admitted."""
    unique = "throttle-test-key-1"
    assert claim_last_used_update(unique) is True
    assert claim_last_used_update(unique) is False
    # A different key is tracked independently.
    assert claim_last_used_update("throttle-test-key-2") is True


def test_claim_last_used_update_interval_is_not_trivial():
    """Guard against the interval being silently set to 0 (no throttling)."""
    assert cache_module.LAST_USED_UPDATE_INTERVAL_S >= 1.0


@pytest.mark.asyncio
async def test_get_cached_api_keys_refreshes_once_under_concurrency():
    """Concurrent cold-cache callers must trigger a single DB refresh."""
    cache = get_api_key_cache()
    assert cache.get_all_keys() is None  # cold
    calls = 0

    async def fake_refresh(inner_cache: ApiKeyCache) -> list[CachedApiKey]:
        nonlocal calls
        calls += 1
        # Yield so the other waiters actually queue on the lock.
        await asyncio.sleep(0.05)
        inner_cache.set_all_keys([])
        return []

    with patch.object(cache_module, "_refresh_cached_api_keys", side_effect=fake_refresh):
        results = await asyncio.gather(*(cache_module.get_cached_api_keys() for _ in range(20)))

    assert results == [[] for _ in range(20)]
    assert calls == 1


@pytest.mark.asyncio
async def test_stale_verified_entry_revalidates_without_bcrypt():
    """A lapsed verified entry whose bcrypt hash is still valid skips bcrypt."""
    cache = get_api_key_cache()
    sha = hash_api_key_for_cache("sk-secret")
    record = _cached_key(name="k", key_hash="hash123")
    cache.set_verified_key(sha, "k", None, key_hash="hash123")
    _make_stale(cache, sha)

    with (
        patch.object(mcp_proxy_module, "get_cached_api_keys", new=AsyncMock(return_value=[record])),
        patch.object(mcp_proxy_module, "verify_api_key", return_value=True) as bcrypt,
    ):
        auth = await verify_api_key_for_mcp("sk-secret")

    assert auth is not None
    assert auth["principal_id"] == "k"
    bcrypt.assert_not_called()
    # Revalidation refreshed the entry, so it is fresh again.
    assert cache.get_verified_key(sha) is not None


@pytest.mark.asyncio
async def test_revalidation_skipped_when_hash_rotated():
    """A rotated key hash forces the full bcrypt search (and can fail)."""
    cache = get_api_key_cache()
    sha = hash_api_key_for_cache("sk-secret")
    cache.set_verified_key(sha, "k", None, key_hash="old-hash")
    _make_stale(cache, sha)
    record = _cached_key(name="k", key_hash="new-hash")

    with (
        patch.object(mcp_proxy_module, "get_cached_api_keys", new=AsyncMock(return_value=[record])),
        patch.object(mcp_proxy_module, "verify_api_key", return_value=False) as bcrypt,
        patch.object(mcp_proxy_module, "_verify_session_api_key", new=AsyncMock(return_value=None)),
    ):
        auth = await verify_api_key_for_mcp("sk-secret")

    assert auth is None
    bcrypt.assert_called_once()


@pytest.mark.asyncio
async def test_stale_entry_without_hash_falls_back_to_bcrypt():
    """Entries cached before key_hash tracking still authenticate via bcrypt."""
    cache = get_api_key_cache()
    sha = hash_api_key_for_cache("sk-secret")
    cache.set_verified_key(sha, "k", None)  # no key_hash recorded
    _make_stale(cache, sha)
    record = _cached_key(name="k", key_hash="hash123")

    with (
        patch.object(mcp_proxy_module, "get_cached_api_keys", new=AsyncMock(return_value=[record])),
        patch.object(mcp_proxy_module, "verify_api_key", return_value=True) as bcrypt,
    ):
        auth = await verify_api_key_for_mcp("sk-secret")

    assert auth is not None
    bcrypt.assert_called_once()
    # The re-verified entry now records the hash for future revalidation.
    assert cache.get_verified_key(sha).key_hash == "hash123"


@pytest.mark.asyncio
async def test_expired_key_is_evicted_even_when_verified_entry_is_stale():
    """A key past its own expires_at is rejected before any revalidation."""
    cache = get_api_key_cache()
    sha = hash_api_key_for_cache("sk-secret")
    past = datetime.now(UTC) - timedelta(hours=1)
    cache.set_verified_key(sha, "k", None, expires_at=past, key_hash="hash123")
    record = _cached_key(name="k", key_hash="hash123", expires_at=past)

    with (
        patch.object(mcp_proxy_module, "get_cached_api_keys", new=AsyncMock(return_value=[record])),
        patch.object(mcp_proxy_module, "verify_api_key", return_value=True) as bcrypt,
    ):
        auth = await verify_api_key_for_mcp("sk-secret")

    assert auth is None
    bcrypt.assert_not_called()
    assert cache.get_verified_key(sha, allow_stale=True) is None
