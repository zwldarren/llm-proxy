"""Per-provider-mapping latency statistics for strategy-based selection.

In-memory EWMA store keyed by the same provider mapping key used by the
circuit breaker (``provider:model:index``). Fed by successful requests:
streaming requests record time-to-first-token, non-streaming requests record
total response time.

Designed as an application-level singleton (stored on ``app.state``),
mirroring :class:`llm_proxy.core.circuit_breaker.CircuitBreakerStore`. Stats
are process-local and lost on restart; the ``balanced`` provider strategy
degrades gracefully to cost ordering while the store is cold.
"""

import time
from dataclasses import dataclass, field

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

# Weight of each new sample. Matches the EWMA alpha used by smart routing's
# per-model experience store (llm_proxy.routing.model_experience).
EWMA_ALPHA = 0.3


@dataclass
class _LatencyStats:
    """Internal per-key latency statistics."""

    ewma_ms: float
    samples: int
    last_updated: float = field(default_factory=time.time)


class ProviderStatsStore:
    """In-memory EWMA latency stats per provider mapping key.

    Thread-safety: asyncio is cooperative single-threaded; this store does not
    need explicit locking for typical FastAPI use (same rationale as
    :class:`CircuitBreakerStore`).
    """

    def __init__(self) -> None:
        self._stats: dict[str, _LatencyStats] = {}

    def observe(self, provider_key: str, latency_ms: float) -> None:
        """Record a successful request's latency for *provider_key*.

        Args:
            provider_key: The unique provider mapping key (provider:model:index).
            latency_ms: Observed latency in milliseconds. Negative values are
                ignored defensively.
        """
        if latency_ms < 0:
            return
        stats = self._stats.get(provider_key)
        if stats is None:
            self._stats[provider_key] = _LatencyStats(ewma_ms=latency_ms, samples=1)
            return
        stats.ewma_ms = EWMA_ALPHA * latency_ms + (1 - EWMA_ALPHA) * stats.ewma_ms
        stats.samples += 1
        stats.last_updated = time.time()

    def get(self, provider_key: str) -> float | None:
        """Return the EWMA latency in ms, or ``None`` when no samples exist."""
        stats = self._stats.get(provider_key)
        if stats is None or stats.samples == 0:
            return None
        return stats.ewma_ms

    def sample_count(self, provider_key: str) -> int:
        """Return the number of recorded samples for *provider_key*."""
        stats = self._stats.get(provider_key)
        return stats.samples if stats is not None else 0

    def reset(self, provider_key: str | None = None) -> None:
        """Reset stats for one key, or all keys when ``None``."""
        if provider_key is None:
            self._stats.clear()
        else:
            self._stats.pop(provider_key, None)

    @property
    def key_count(self) -> int:
        """Return the number of tracked provider keys."""
        return len(self._stats)

    def get_all(self) -> dict[str, dict]:
        """Return a snapshot of all stats (observability/debugging)."""
        return {
            key: {
                "ewma_ms": stats.ewma_ms,
                "samples": stats.samples,
                "last_updated": stats.last_updated,
            }
            for key, stats in self._stats.items()
        }


__all__ = ["EWMA_ALPHA", "ProviderStatsStore"]
