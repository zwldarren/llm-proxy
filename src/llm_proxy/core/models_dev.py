"""Shared access to the models.dev upstream model catalog.

The pricing sync endpoint (``api/routers/config/pricing.py``) reads from
https://models.dev/api.json. The payload is several MB, so the parsed catalog
is cached in-process for a short TTL; concurrent callers share a single
in-flight request via an asyncio lock.
"""

import asyncio
import time
from typing import Any

import httpx2

from llm_proxy.http.client import AsyncSession, fetch_json
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

MODELS_DEV_API_URL = "https://models.dev/api.json"

# Freshness window for the cached catalog. models.dev updates roughly daily,
# so one hour keeps pricing/metadata current without re-downloading MBs on
# every admin action.
_CACHE_TTL_SECONDS = 3600.0

# When the upstream is unreachable but a stale copy exists, serve it for a
# short grace period before retrying the network again.
_STALE_GRACE_SECONDS = 300.0

_cache_data: dict[str, Any] | None = None
_cache_expires_at: float = 0.0
_fetch_lock = asyncio.Lock()


def coerce_float(value: Any) -> float | None:
    """Best-effort float coercion for optional models.dev payload fields."""
    try:
        return float(value) if value is not None else None
    except ValueError, TypeError:
        return None


def build_model_index(data: dict[str, Any]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Index catalog model entries by model id (plus ``/``-suffix alias).

    models.dev nests entries under provider keys (``data[provider]["models"]``);
    the index inverts that to ``model_id -> [(provider_key, entry), ...]`` so a
    configured provider model name resolves to every catalog entry claiming
    that id, regardless of which provider published it. IDs containing ``/``
    are additionally indexed by their suffix, mirroring the pricing sync
    aliasing (``org/gpt-4o`` also resolves as ``gpt-4o``).
    """
    index: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for provider_key, provider_data in data.items():
        if not isinstance(provider_data, dict):
            continue
        models = provider_data.get("models")
        if not isinstance(models, dict):
            continue
        for model_id, entry in models.items():
            if not isinstance(entry, dict):
                continue
            index.setdefault(model_id, []).append((provider_key, entry))
            if "/" in model_id:
                _, suffix = model_id.rsplit("/", 1)
                index.setdefault(suffix, []).append((provider_key, entry))
    return index


async def fetch_models_dev_data(client: AsyncSession) -> dict[str, Any]:
    """Return the parsed models.dev catalog, using the in-process cache.

    On fetch failure, stale cached data is served when available (with a
    short grace extension so a flapping upstream is not hammered); otherwise
    the exception propagates to the caller.
    """
    global _cache_data, _cache_expires_at

    if _cache_data is not None and time.monotonic() < _cache_expires_at:
        return _cache_data

    async with _fetch_lock:
        # Re-check inside the lock: a concurrent caller may have populated it.
        if _cache_data is not None and time.monotonic() < _cache_expires_at:
            return _cache_data

        try:
            data = await fetch_json(client, MODELS_DEV_API_URL)
        except httpx2.HTTPError:
            if _cache_data is not None:
                logger.warning("models.dev fetch failed; serving stale cached catalog")
                _cache_expires_at = time.monotonic() + _STALE_GRACE_SECONDS
                return _cache_data
            raise

        _cache_data = data
        _cache_expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        return data


def clear_models_dev_cache() -> None:
    """Reset the in-process cache (used by tests)."""
    global _cache_data, _cache_expires_at
    _cache_data = None
    _cache_expires_at = 0.0
