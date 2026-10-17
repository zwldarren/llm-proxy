"""The models.dev catalog — the OpenCode team's own per-model dialect data.

The OpenCode gateway's ``/v1/models`` endpoint answers with bare ids, so it
cannot say which wire dialect a model speaks. The OpenCode team publishes
that fact — together with pricing, context windows and modalities — in the
models.dev catalog (``https://models.dev/api.json``), the same machine-readable
source the OpenCode CLI configures its providers from. Each provider entry
(``opencode``, ``opencode-go``) carries one record per model with a
``provider.npm`` marker naming the AI SDK package that speaks the model's
wire format — which is exactly the dialect the gateway serves the model on.

Reading the catalog at runtime is what keeps the adapter maintenance-free:
new model families appear there with their dialect attached and start
routing correctly without a code change. When the catalog is unreachable the
adapter degrades to the prefix heuristics in :mod:`...opencode.dialect`.

Caching: the payload is several MB, so it is fetched with a process-wide TTL
cache and a single-flight lock (concurrent first requests share one fetch).
Failures are negatively cached briefly so a dead network does not turn into
a per-request retry storm; every path degrades to heuristics instead.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from llm_proxy.models.provider import ProviderModelPricing
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.opencode.dialect import Dialect, dialect_from_npm

logger = get_logger(__name__)

#: Default location of the catalog. Overridable per provider via the
#: ``opencode_catalog_url`` metadata key (air-gapped deployments point it at
#: a mirror; an empty value disables the catalog entirely).
DEFAULT_CATALOG_URL = "https://models.dev/api.json"

#: How long a successful fetch stays authoritative.
_POSITIVE_TTL_SECONDS = 6 * 60 * 60
#: How long a failed fetch suppresses retries.
_NEGATIVE_TTL_SECONDS = 5 * 60
#: Hard ceiling on the catalog fetch: a live completion must never stall
#: behind a dead network; the fallback heuristics take over.
_FETCH_TIMEOUT_SECONDS = 10.0

# Process-wide cache. ``None`` until the first attempt; ``(fetched_at, tables)``
# after a success and ``(fetched_at, None)`` after a failure (negative cache).
# Writes happen under the single-flight lock; reads are plain attribute reads.
_cache: tuple[float, dict[str, dict[str, CatalogModel]] | None] | None = None
_fetch_lock = asyncio.Lock()


@dataclass(frozen=True)
class CatalogModel:
    """One model's catalog facts — the subset the adapter consumes."""

    dialect: Dialect
    name: str | None = None
    description: str | None = None
    context_length: int | None = None
    pricing: ProviderModelPricing | None = None


def _parse_pricing(raw: Any) -> ProviderModelPricing | None:
    """models.dev prices are USD per 1M tokens; the proxy stores per token."""
    if not isinstance(raw, dict):
        return None

    def _per_token(key: str) -> float | None:
        try:
            return float(raw[key]) / 1_000_000
        except KeyError, TypeError, ValueError:
            return None

    pricing = ProviderModelPricing(
        prompt=_per_token("input"),
        completion=_per_token("output"),
    )
    if pricing.prompt is None and pricing.completion is None:
        return None
    return pricing


def _parse_model(model_id: str, raw: Any) -> CatalogModel | None:
    if not isinstance(raw, dict):
        return None
    provider_meta = raw.get("provider")
    npm = provider_meta.get("npm") if isinstance(provider_meta, dict) else None
    limit = raw.get("limit") if isinstance(raw.get("limit"), dict) else {}
    context_length = limit.get("context")
    cost = raw.get("cost")
    return CatalogModel(
        dialect=dialect_from_npm(npm),
        name=raw.get("name"),
        description=raw.get("description"),
        context_length=context_length if isinstance(context_length, int) else None,
        pricing=_parse_pricing(cost if isinstance(cost, dict) else None),
    )


def parse_catalog(data: dict[str, Any]) -> dict[str, dict[str, CatalogModel]]:
    """Parse the models.dev payload into per-entry model tables.

    Only the OpenCode entries are kept; the rest of the catalog is other
    providers' data and is dropped here rather than carried around. A payload
    without any OpenCode entry raises — the caller treats that like any other
    catalog failure and falls back to heuristics.
    """
    if not isinstance(data, dict):
        raise ValueError("models.dev payload is not a JSON object")
    tables: dict[str, dict[str, CatalogModel]] = {}
    for entry_id in ("opencode", "opencode-go"):
        entry = data.get(entry_id)
        if not isinstance(entry, dict):
            continue
        models = entry.get("models")
        if not isinstance(models, dict):
            continue
        table: dict[str, CatalogModel] = {}
        for model_id, raw in models.items():
            parsed = _parse_model(model_id, raw)
            if parsed is not None:
                table[model_id] = parsed
        if table:
            tables[entry_id] = table
    if not tables:
        raise ValueError("models.dev payload carries no OpenCode entries")
    return tables


def _cached_tables() -> dict[str, dict[str, CatalogModel]] | None:
    """Return the tables when the cache is warm, else None.

    A failed fetch caches an empty success-shaped result for the negative
    window (the fetch must not be retried per request), signalled as ``{}``.
    """
    if _cache is None:
        return None
    fetched_at, tables = _cache
    age = time.monotonic() - fetched_at
    if tables is None:
        return {} if age <= _NEGATIVE_TTL_SECONDS else None
    return tables if age <= _POSITIVE_TTL_SECONDS else None


async def load_catalog(
    client: Any,
    *,
    url: str = DEFAULT_CATALOG_URL,
) -> dict[str, dict[str, CatalogModel]]:
    """Return the parsed catalog tables, fetching through ``client`` once.

    ``client`` is the pooled ``AsyncSession`` the caller already holds (the
    adapter's transport client in production, a mock in tests). Returns an
    empty dict while the catalog is unavailable — callers degrade to the
    prefix heuristics, never fail the request because of the catalog.
    """
    global _cache
    cached = _cached_tables()
    if cached is not None:
        return cached

    async with _fetch_lock:
        cached = _cached_tables()
        if cached is not None:
            return cached
        from llm_proxy.http.client import fetch_json

        try:
            data = await asyncio.wait_for(
                fetch_json(client, url, headers={"Accept": "application/json"}),
                timeout=_FETCH_TIMEOUT_SECONDS,
            )
            tables = parse_catalog(data)
        except Exception as exc:
            # Unknown shape, dead network, provider outage: heuristics carry
            # the requests until the next window.
            logger.warning(f"OpenCode model catalog unavailable, using fallback heuristics: {exc}")
            _cache = (time.monotonic(), None)
            return {}
        _cache = (time.monotonic(), tables)
        return tables


def peek() -> dict[str, dict[str, CatalogModel]]:
    """Return the warm cache without fetching; ``{}`` when cold or failed.

    The synchronous half of the resolution path (pipeline hooks that run
    before the adapter's async entry points) reads only this — the fetch
    itself is the async :func:`load_catalog`'s job.
    """
    return _cached_tables() or {}


def lookup(
    tables: dict[str, dict[str, CatalogModel]],
    entry_id: str,
    model: str | None,
) -> CatalogModel | None:
    """Look a model up in one provider entry of the catalog."""
    if not model:
        return None
    return tables.get(entry_id, {}).get(model)


def reset_cache_for_tests() -> None:
    """Drop the module cache (test isolation; not used in production paths)."""
    global _cache
    _cache = None


__all__ = [
    "DEFAULT_CATALOG_URL",
    "CatalogModel",
    "load_catalog",
    "lookup",
    "parse_catalog",
    "peek",
    "reset_cache_for_tests",
]
