"""Per-provider-mapping latency statistics for strategy-based selection.

In-memory EWMA store keyed by the same provider mapping key used by the
circuit breaker. Fed by successful requests, in one of two buckets: a
streaming request reports time-to-first-token (what a user actually waits
for), a non-streaming request reports the whole response time. The two are
kept apart because a provider that is quick to first token but slow overall
(and the reverse) must not average into one meaningless number.

Designed as an application-level singleton, installed on the app's
process-lifetime services (:mod:`llm_proxy.services`), mirroring
:class:`llm_proxy.core.circuit_breaker.CircuitBreakerStore`. Stats are
process-local and lost on restart; the ``balanced`` provider strategy degrades
gracefully to cost ordering while the store is cold.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

# Weight of each new sample. Matches the EWMA alpha used by smart routing's
# per-model experience store (llm_proxy.routing.model_experience).
EWMA_ALPHA = 0.3


class LatencyKind(StrEnum):
    """Latency measurements that balanced selection may compare."""

    TTFT = "ttft"
    TOTAL = "total"


KIND_TTFT = LatencyKind.TTFT
KIND_TOTAL = LatencyKind.TOTAL
#: Fallback bucket for callers that do not say which measurement they recorded.
DEFAULT_LATENCY_KIND = KIND_TOTAL

#: How long a sample stays meaningful. Provider latency drifts over minutes, so
#: a provider that has not served recently is treated as unmeasured instead of
#: keeping a stale rank — ranking it from old data is how "always fastest wins
#: and the runner-up is never sampled again" happens.
SAMPLE_TTL_SECONDS = 300.0


@dataclass
class _LatencyStats:
    """Internal per-key latency statistics for one measurement kind."""

    ewma_ms: float
    samples: int
    last_updated: float = field(default_factory=time.time)


class ProviderStatsStore:
    """In-memory EWMA latency stats per provider mapping key and kind.

    Thread-safety: asyncio is cooperative single-threaded; this store does not
    need explicit locking for typical FastAPI use (same rationale as
    :class:`CircuitBreakerStore`).
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._stats: dict[str, dict[LatencyKind, _LatencyStats]] = {}
        self._clock = clock

    def observe(
        self,
        provider_key: str,
        latency_ms: float,
        kind: LatencyKind = DEFAULT_LATENCY_KIND,
    ) -> None:
        """Record a successful request's latency for *provider_key*.

        Args:
            provider_key: The unique provider mapping key.
            latency_ms: Observed latency in milliseconds. Negative values are
                ignored defensively.
            kind: Which measurement this is (:data:`KIND_TTFT` or
                :data:`KIND_TOTAL`).
        """
        kind = LatencyKind(kind)
        if latency_ms < 0:
            return
        buckets = self._stats.setdefault(provider_key, {})
        stats = buckets.get(kind)
        now = self._clock()
        if stats is None:
            buckets[kind] = _LatencyStats(ewma_ms=latency_ms, samples=1, last_updated=now)
            return
        stats.ewma_ms = EWMA_ALPHA * latency_ms + (1 - EWMA_ALPHA) * stats.ewma_ms
        stats.samples += 1
        stats.last_updated = now

    def get(self, provider_key: str, kind: LatencyKind = DEFAULT_LATENCY_KIND) -> float | None:
        """Return the EWMA latency in ms, or ``None`` when unmeasured/stale."""
        kind = LatencyKind(kind)
        stats = self._stats.get(provider_key, {}).get(kind)
        if stats is None or stats.samples == 0:
            return None
        if self._clock() - stats.last_updated > SAMPLE_TTL_SECONDS:
            return None
        return stats.ewma_ms

    def sample_count(self, provider_key: str, kind: LatencyKind = DEFAULT_LATENCY_KIND) -> int:
        """Return the number of recorded samples for *provider_key* and *kind*."""
        kind = LatencyKind(kind)
        stats = self._stats.get(provider_key, {}).get(kind)
        return stats.samples if stats is not None else 0

    def reset(self, provider_key: str | None = None) -> None:
        """Reset stats for one key (all kinds), or all keys when ``None``."""
        if provider_key is None:
            self._stats.clear()
        else:
            self._stats.pop(provider_key, None)

    @property
    def key_count(self) -> int:
        """Return the number of tracked provider keys."""
        return len(self._stats)

    def get_all(self) -> dict[str, dict[str, dict]]:
        """Return a snapshot of all stats (observability/debugging)."""
        return {
            key: {
                kind.value: {
                    "ewma_ms": stats.ewma_ms,
                    "samples": stats.samples,
                    "last_updated": stats.last_updated,
                }
                for kind, stats in buckets.items()
            }
            for key, buckets in self._stats.items()
        }


__all__ = [
    "DEFAULT_LATENCY_KIND",
    "EWMA_ALPHA",
    "KIND_TOTAL",
    "KIND_TTFT",
    "LatencyKind",
    "SAMPLE_TTL_SECONDS",
    "ProviderStatsStore",
]
