"""Per-model EWMA experience + Thompson sampling support, DB-backed."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from llm_proxy.observability.logger import get_logger
from llm_proxy.routing.types import ModelExperience

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateExperience:
    """Snapshot of a candidate model's experience stats for selection."""

    reliability: float = 0.5
    latency: float = 0.5
    cache_affinity: float = 0.5
    input_cost_multiplier: float = 1.0
    reward_mean: float = 0.5
    samples: int = 0
    # Explicit-feedback channel. Kept even when samples == 0 because feedback
    # is independent of request observations.
    preference_ewma: float = 0.0


_DEFAULT = lambda name: ModelExperience(name=name)  # noqa: E731
_EWMA_ALPHA = 0.3

# Explicit-feedback tuning (ported from UncommonRoute model_experience.py):
# reward blends into reward_mean via EWMA; the preference delta accumulates
# into preference_ewma and is projected to the persisted `feedback` column.
_FEEDBACK_REWARDS = {"ok": 0.9, "weak": 0.15, "strong": 0.35}
_FEEDBACK_PREFERENCE_DELTAS = {"ok": 0.14, "weak": -0.22, "strong": -0.10}
FEEDBACK_SIGNALS = frozenset(_FEEDBACK_REWARDS)


class ModelExperienceStore:
    """In-memory cache with optional async DB persistence.

    When ``session`` is None, behaves as a pure in-memory store (used in tests
    and before the DB table exists). When a session is provided, ``observe``
    also upserts a ``ModelExperienceRecord`` row.
    """

    def __init__(self, session: Any | None) -> None:
        self._session = session
        self._cache: dict[str, ModelExperience] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, name: str) -> asyncio.Lock:
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    def get(self, name: str) -> ModelExperience:
        if name not in self._cache:
            self._cache[name] = _DEFAULT(name)
        return self._cache[name]

    def snapshot(self) -> dict[str, ModelExperience]:
        return dict(self._cache)

    def observe(self, name: str, success: bool, latency_ms: int, cache_hit: bool = False) -> None:
        exp = self.get(name)
        reward = 1.0 if success else 0.0
        exp.samples += 1
        exp.reward_mean = _ewma(exp.reward_mean, reward, _EWMA_ALPHA)
        norm_latency = max(0.0, min(1.0, latency_ms / 5000.0))
        exp.latency = _ewma(exp.latency, norm_latency, _EWMA_ALPHA)
        exp.reliability = _ewma(exp.reliability, reward, _EWMA_ALPHA)
        exp.cache_affinity = _ewma(exp.cache_affinity, 1.0 if cache_hit else 0.0, _EWMA_ALPHA)
        exp.name = name
        self._cache[name] = exp
        # DB persistence is live via _persist (ModelExperienceRecord upsert).

    def record_feedback(self, name: str, signal: str) -> None:
        """Apply explicit user feedback (ok/weak/strong) to a model's experience.

        The reward update flows into Thompson sampling through ``reward_mean``
        (see selector base_quality); the preference delta is persisted via the
        ``feedback`` projection.
        """
        if signal not in _FEEDBACK_REWARDS:
            raise ValueError(f"Unknown feedback signal: {signal!r}")
        exp = self.get(name)
        exp.reward_mean = _ewma(exp.reward_mean, _FEEDBACK_REWARDS[signal], _EWMA_ALPHA)
        preference = max(-1.0, min(1.0, exp.preference_ewma + _FEEDBACK_PREFERENCE_DELTAS[signal]))
        exp.preference_ewma = preference
        exp.feedback = max(0.0, min(1.0, 0.5 + preference * 0.5))
        self._cache[name] = exp

    async def record_feedback_async(self, name: str, signal: str) -> None:
        """DB-backed variant of record_feedback, mirroring observe_async."""
        async with self._lock(name):
            await self.load_from_db(name)
            # Snapshot the current cache entry so we can revert on persist failure
            before = self._snapshot_entry(name)
            self.record_feedback(name, signal)
            try:
                await self._persist(name)
            except Exception:
                # Revert in-memory cache to pre-feedback state
                if before is not None:
                    self._cache[name] = before
                raise

    async def observe_async(
        self, name: str, success: bool, latency_ms: int, cache_hit: bool = False
    ) -> None:
        async with self._lock(name):
            await self.load_from_db(name)
            # Snapshot the current cache entry so we can revert on persist failure
            before = self._snapshot_entry(name)
            self.observe(name, success, latency_ms, cache_hit)
            try:
                await self._persist(name)
            except Exception:
                # Revert in-memory cache to pre-observe state
                if before is not None:
                    self._cache[name] = before
                raise

    def _snapshot_entry(self, name: str) -> ModelExperience | None:
        import copy

        entry = self._cache.get(name)
        if entry is None:
            return None
        return copy.deepcopy(entry)

    async def load_from_db(self, name: str) -> None:
        """Hydrate the in-memory cache from the DB row if one exists."""
        if self._session is None:
            return

        stmt = select_by_name(name)
        existing = await self._session.execute(stmt)
        row = existing.scalar_one_or_none()
        if row is not None:
            self._cache[name] = ModelExperience(
                name=name,
                samples=row.samples,
                reward_mean=row.reward_mean,
                latency=row.latency,
                reliability=row.reliability,
                cache_affinity=row.cache_affinity,
                feedback=row.feedback,
                # Inverse of the persisted projection (see ModelExperience),
                # so preference survives restarts without a dedicated column.
                preference_ewma=(row.feedback - 0.5) * 2.0,
            )

    async def _persist(self, name: str) -> None:
        if self._session is None:
            return
        from llm_proxy.database.tables import ModelExperienceRecord  # lazy

        exp = self.get(name)
        # Upsert by name.
        stmt = select_by_name(name)
        existing = await self._session.execute(stmt)
        row = existing.scalar_one_or_none()
        if row is None:
            row = ModelExperienceRecord(name=name)
            self._session.add(row)
        row.samples = exp.samples
        row.reward_mean = exp.reward_mean
        row.latency = exp.latency
        row.reliability = exp.reliability
        row.feedback = exp.feedback
        row.cache_affinity = exp.cache_affinity
        row.updated_at = time.time()
        await self._session.flush()


def _ewma(prev: float, new: float, alpha: float) -> float:
    return prev + alpha * (new - prev)


def select_by_name(name: str):
    from sqlalchemy import select

    from llm_proxy.database.tables import ModelExperienceRecord

    return select(ModelExperienceRecord).where(ModelExperienceRecord.name == name)


async def observe_model_experience(
    event_context: Any,
    context: Any,
    *,
    success: bool,
) -> None:
    """Update model experience EWMA after a routed request completes.

    Opens a fresh session so telemetry failures never break the response.
    """
    from llm_proxy.database.connection import get_async_session_context

    decision = context.routing_decision
    resolved: str | None = getattr(decision, "model", None)
    if resolved is None:
        return

    latency_ms = event_context.latency_ms
    cache_hit = bool(
        getattr(event_context, "cache_read_input_tokens", 0)
        or getattr(event_context, "cached_prompt_tokens", 0)
    )

    try:
        async with get_async_session_context() as session:
            store = ModelExperienceStore(session=session)
            await store.observe_async(
                resolved,
                success=success,
                latency_ms=int(latency_ms),
                cache_hit=cache_hit,
            )
            await session.commit()
    except Exception:  # noqa: BLE001
        logger.debug("Failed to record model experience", exc_info=True)
