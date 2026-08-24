"""API key cache module.

In-memory cache for API keys with TTL to reduce database lookups.
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

from llm_proxy.core.budget import BudgetEnvelope
from llm_proxy.core.constants import API_KEY_CACHE_TTL_SECONDS

# TTL for the per-key budget-spend cache. Budget enforcement queries the
# usage table on cache miss only, bounding the extra load to one indexed SUM
# per key per TTL window. A short TTL keeps the enforced spend close to the
# real (asynchronously written) usage.
BUDGET_SPEND_CACHE_TTL_SECONDS = 15.0


@dataclass
class CachedApiKey:
    """Cached API key record."""

    name: str
    key_hash: str
    is_active: bool
    allowed_models: list[str] | None
    allowed_mcp_servers: list[str] | None = None
    user_id: int | None = None
    # Owner-user snapshot used to enforce account status and user-level model
    # constraints at request time without a per-request database join.
    user_allowed_models: list[str] | None = None
    user_is_active: bool = True
    # Expiry and budget configuration, enforced at request time.
    expires_at: datetime | None = None
    budget: BudgetEnvelope = field(default_factory=BudgetEnvelope)
    # Admin-set account-level budget envelope for the key's owner, enforced
    # at request time across all of the owner's keys (empty = unlimited).
    user_budget: BudgetEnvelope = field(default_factory=BudgetEnvelope)
    # Per-key requests-per-minute cap (None = unlimited), enforced at
    # request-auth time from this snapshot.
    rate_limit_rpm: int | None = None
    cached_at: float = field(default_factory=time.time)


@dataclass
class VerifiedKeyInfo:
    """Information about a verified API key."""

    name: str
    allowed_models: list[str] | None
    allowed_mcp_servers: list[str] | None = None
    user_id: int | None = None
    expires_at: datetime | None = None
    budget: BudgetEnvelope = field(default_factory=BudgetEnvelope)
    # Account-level budget envelope of the key's owner (empty = unlimited).
    user_budget: BudgetEnvelope = field(default_factory=BudgetEnvelope)
    rate_limit_rpm: int | None = None
    verified_at: float = field(default_factory=time.time)


@dataclass
class ApiKeyCache:
    """In-memory cache for API keys with TTL.

    Thread-safe with two-level caching:
    1. _all_keys: All API keys from database (for bcrypt verification)
    2. _verified_keys: SHA-256 hash -> verified key info (fast path for repeat requests)
    """

    ttl: float = API_KEY_CACHE_TTL_SECONDS
    _all_keys: list[CachedApiKey] = field(default_factory=list)
    _verified_keys: dict[str, VerifiedKeyInfo] = field(default_factory=dict)
    _last_refresh: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def is_expired(self) -> bool:
        """Check if the cache has expired."""
        return time.time() - self._last_refresh > self.ttl

    def get_all_keys(self) -> list[CachedApiKey] | None:
        """Get all cached keys if cache is valid."""
        with self._lock:
            if self.is_expired():
                return None
            return self._all_keys.copy()

    def set_all_keys(self, keys: list[CachedApiKey]) -> None:
        """Update the cache with new keys."""
        with self._lock:
            self._all_keys = keys
            self._last_refresh = time.time()

    def get_verified_key(self, api_key_sha256: str) -> VerifiedKeyInfo | None:
        """Get verified key info from cache if available (O(1) lookup)."""
        with self._lock:
            info = self._verified_keys.get(api_key_sha256)
            if info is None:
                return None
            if time.time() - info.verified_at > self.ttl:
                del self._verified_keys[api_key_sha256]
                return None
            return info

    def set_verified_key(
        self,
        api_key_sha256: str,
        name: str,
        allowed_models: list[str] | None,
        allowed_mcp_servers: list[str] | None = None,
        *,
        user_id: int | None = None,
        expires_at: datetime | None = None,
        budget: BudgetEnvelope | None = None,
        user_budget: BudgetEnvelope | None = None,
        rate_limit_rpm: int | None = None,
    ) -> None:
        """Cache a verified API key for fast future lookups."""
        with self._lock:
            self._verified_keys[api_key_sha256] = VerifiedKeyInfo(
                name=name,
                allowed_models=allowed_models,
                allowed_mcp_servers=allowed_mcp_servers,
                user_id=user_id,
                expires_at=expires_at,
                budget=BudgetEnvelope() if budget is None else budget,
                user_budget=BudgetEnvelope() if user_budget is None else user_budget,
                rate_limit_rpm=rate_limit_rpm,
            )

    def evict_verified_key(self, api_key_sha256: str) -> None:
        """Drop a verified-key entry (e.g., when the key has expired)."""
        with self._lock:
            self._verified_keys.pop(api_key_sha256, None)

    def invalidate(self) -> None:
        """Invalidate the cache."""
        with self._lock:
            self._all_keys = []
            self._verified_keys = {}
            self._last_refresh = 0.0


# Global API key cache
_api_key_cache: ApiKeyCache | None = None
_api_key_cache_lock = Lock()


def get_api_key_cache() -> ApiKeyCache:
    """Get the global API key cache instance."""
    global _api_key_cache
    if _api_key_cache is None:
        with _api_key_cache_lock:
            if _api_key_cache is None:
                _api_key_cache = ApiKeyCache()
    assert _api_key_cache is not None
    return _api_key_cache


def invalidate_api_key_cache() -> None:
    """Invalidate the global API key cache.

    Call this when API keys are created, updated, or deleted. Also drops the
    budget spend cache so budget edits / resets take effect immediately.
    """
    cache = get_api_key_cache()
    cache.invalidate()
    get_budget_spend_cache().invalidate()


@dataclass
class UserSnapshot:
    """Owner-user fields the key cache snapshots for request-time checks."""

    allowed_models: list[str] | None
    is_active: bool = True
    # Admin-set account-level budget envelope (empty = unlimited).
    budget: BudgetEnvelope = field(default_factory=BudgetEnvelope)


def _user_snapshot(user_map: dict[int, UserSnapshot], user_id: int | None) -> UserSnapshot:
    """Resolve a key owner's snapshot; orphaned keys get permissive defaults."""
    return user_map.get(user_id) or UserSnapshot(None)


async def get_cached_api_keys() -> list[CachedApiKey]:
    """Get API keys from cache or database."""
    from llm_proxy.database import ApiKeyRepository, get_async_session_context
    from llm_proxy.observability.logger import get_logger

    logger = get_logger(__name__)

    cache = get_api_key_cache()
    cached_keys = cache.get_all_keys()

    if cached_keys is not None:
        return cached_keys

    async with get_async_session_context() as session:
        repo = ApiKeyRepository(session)
        all_keys = await repo.list_api_keys()

        # Snapshot the owning users' status and model constraints so request
        # authentication can enforce them without an extra per-request join.
        from sqlalchemy import select

        from llm_proxy.database.tables import UserRecord

        result = await session.execute(
            select(
                UserRecord.id,
                UserRecord.allowed_models,
                UserRecord.is_active,
                UserRecord.budget_usd,
                UserRecord.budget_period,
                UserRecord.budget_reset_day,
                UserRecord.budget_reset_at,
            )
        )
        user_map: dict[int, UserSnapshot] = {
            row.id: UserSnapshot(
                row.allowed_models,
                row.is_active,
                BudgetEnvelope.from_orm_fields(row),
            )
            for row in result
        }

        cached_keys = []
        for k in all_keys:
            snapshot = _user_snapshot(user_map, k.user_id)
            cached_keys.append(
                CachedApiKey(
                    name=k.name,
                    key_hash=k.key_hash,
                    is_active=k.is_active,
                    allowed_models=k.allowed_models,
                    allowed_mcp_servers=k.allowed_mcp_servers,
                    user_id=k.user_id,
                    user_allowed_models=snapshot.allowed_models,
                    user_is_active=snapshot.is_active,
                    expires_at=k.expires_at,
                    budget=BudgetEnvelope.from_orm_fields(k),
                    user_budget=snapshot.budget,
                    rate_limit_rpm=k.rate_limit_rpm,
                )
            )
        cache.set_all_keys(cached_keys)
        logger.debug(f"API key cache refreshed with {len(cached_keys)} keys")
        return cached_keys


# --- Budget spend cache ------------------------------------------------------


@dataclass
class BudgetSpendCache:
    """Short-TTL spend cache for budget enforcement.

    Avoids running a SUM query against the usage table on every request for
    budget-limited keys/accounts. Staleness up to ``ttl`` seconds is
    acceptable: usage records themselves are written asynchronously, so
    enforcement is best-effort either way.
    """

    ttl: float = BUDGET_SPEND_CACHE_TTL_SECONDS
    # key -> (spend, cached_at, window_start); window_start may be None for
    # entries recorded without a window identity (treated as wildcard).
    _spend: dict[str, tuple[float, float, float | None]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def get(self, key_name: str, window_start: float | None = None) -> float | None:
        """Return the cached spend for ``key_name`` if still fresh.

        ``window_start`` is the unix timestamp of the budget window the caller
        is enforcing. An entry recorded for a *different* window (a calendar
        rollover or a manual reset since it was cached) is treated as a miss
        and dropped, so a previous window's spend is never enforced against
        the current one. Entries cached without a window identity match any
        window.
        """
        with self._lock:
            entry = self._spend.get(key_name)
            if entry is None:
                return None
            spend, cached_at, entry_window = entry
            if time.time() - cached_at > self.ttl:
                del self._spend[key_name]
                return None
            if entry_window is not None and entry_window != window_start:
                del self._spend[key_name]
                return None
            return spend

    def set(self, key_name: str, spend: float, window_start: float | None = None) -> None:
        """Cache the current-period spend for ``key_name``."""
        with self._lock:
            self._spend[key_name] = (spend, time.time(), window_start)

    def invalidate(self, key_name: str | None = None) -> None:
        """Drop cached spend for one key (or all keys)."""
        with self._lock:
            if key_name is None:
                self._spend.clear()
            else:
                self._spend.pop(key_name, None)


_budget_spend_cache: BudgetSpendCache | None = None
_budget_spend_cache_lock = Lock()


def get_budget_spend_cache() -> BudgetSpendCache:
    """Get the global budget spend cache instance."""
    global _budget_spend_cache
    if _budget_spend_cache is None:
        with _budget_spend_cache_lock:
            if _budget_spend_cache is None:
                _budget_spend_cache = BudgetSpendCache()
    assert _budget_spend_cache is not None
    return _budget_spend_cache


def hash_api_key_for_cache(api_key: str) -> str:
    """Create a fast hash of the API key for cache lookup.

    Uses SHA-256 which is much faster than bcrypt for cache lookups.
    This is safe because we're only using it as a cache key, not for
    security verification.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()
