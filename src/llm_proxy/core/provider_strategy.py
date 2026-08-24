"""Pluggable provider-selection strategies.

A strategy *orders* the candidates within a single priority group; the
selector takes the head of the ordering on every pick, so per-request
fallback naturally walks the ordering (used/unavailable keys are filtered out
before each pick). Priority groups still dominate: strategies never reorder
across priority levels.

Strategies:

- ``random``: current default behavior — uniform random pick.
- ``session_sticky``: pin a conversation to one provider for cache affinity.
  The Redis-backed sticky mapping is resolved once per request by
  :func:`resolve_sticky_key` (async, called from the provider-selection
  stage); the per-pick ordering itself is synchronous — sticky provider
  first, then rendezvous rank — so when the pinned provider fails mid-request
  the whole conversation consistently drifts to the same backup provider
  (preserving cache affinity during outages). Without Redis (or without a
  conversation key) the ordering degrades to pure rendezvous hashing / random.
- ``cost_optimized``: cheapest provider first (per-mapping pricing with
  model-level fallback). Deterministic, so also cache-friendly.
- ``balanced``: 0.5 * normalized cost + 0.5 * normalized observed latency
  (in-memory EWMA). Degrades to ``cost_optimized`` while no latency samples
  exist (cold start).
"""

import hashlib
import random
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_proxy.config.types.model import (
    ModelConfig,
    ModelProviderConfig,
    ProviderSelectionStrategy,
)
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.core.provider_stats import ProviderStatsStore

logger = get_logger(__name__)

# TTL for the Redis-backed sticky mapping. Matches smart routing's
# conversation-continuity TTL (routing/orchestrator.py).
STICKY_TTL_SECONDS = 1800  # 30 min

# Weights for the balanced strategy score.
_BALANCED_COST_WEIGHT = 0.5
_BALANCED_LATENCY_WEIGHT = 0.5

_default_rng = random.Random()

Candidate = tuple[ModelProviderConfig, str]


@dataclass
class StrategyContext:
    """Per-request inputs for provider selection strategies.

    Attributes:
        model_config: The model configuration (used for pricing fallback).
        model_name: Proxy-facing model name (namespaces the sticky mapping).
        conversation_key: Stable conversation identifier for sticky routing,
            or None when no session/message signal is available.
        redis: Optional async Redis client for the sticky mapping.
        stats_store: Optional EWMA latency stats for the balanced strategy.
        rng: Optional randomness source (tests inject a seeded instance).
    """

    model_config: ModelConfig
    model_name: str
    conversation_key: str | None = None
    redis: Any | None = None
    stats_store: ProviderStatsStore | None = None
    rng: random.Random | None = None


def _sticky_redis_key(conversation_key: str, model_name: str) -> str:
    """Redis key for the sticky provider mapping of a conversation+model pair."""
    return f"routing:conv:{conversation_key}:provider:{model_name}"


def _rendezvous_score(conversation_key: str, candidate_key: str) -> str:
    """Deterministic rendezvous-hashing score (higher wins)."""
    return hashlib.sha256(f"{conversation_key}:{candidate_key}".encode()).hexdigest()


def _estimated_cost(model_config: ModelConfig, provider: ModelProviderConfig) -> float:
    """Estimated per-1M-token cost for a provider mapping.

    Uses per-mapping pricing with model-level fallback. Returns ``inf`` when
    no pricing is configured at either level (sorted last).
    """
    input_cost = provider.input_cost_per_1m
    if input_cost is None:
        input_cost = model_config.input_cost_per_1m
    output_cost = provider.output_cost_per_1m
    if output_cost is None:
        output_cost = model_config.output_cost_per_1m
    if input_cost is None and output_cost is None:
        return float("inf")
    return (input_cost or 0.0) + (output_cost or 0.0)


def _max_normalize(values: list[float]) -> list[float]:
    """Normalize by the maximum, preserving proportional differences.

    Max-normalization (rather than min-max) matters for the common two-
    candidate case: min-max would always map the two values to {0, 1} on both
    dimensions, making equally-weighted scores collapse into a tie.
    """
    peak = max(values) if values else 0.0
    if peak <= 0:
        return [0.0] * len(values)
    return [v / peak for v in values]


def _cost_ordered(available: list[Candidate], ctx: StrategyContext) -> list[Candidate]:
    """Order candidates by ascending estimated cost; shuffle ties."""
    rng = ctx.rng or _default_rng
    # (cost, random tiebreak) keeps ordering deterministic on price while
    # spreading load across equally-priced providers.
    return sorted(
        available,
        key=lambda pk: (_estimated_cost(ctx.model_config, pk[0]), rng.random()),
    )


def _balanced_ordered(available: list[Candidate], ctx: StrategyContext) -> list[Candidate]:
    """Order candidates by combined cost+latency score (ascending)."""
    rng = ctx.rng or _default_rng
    store = ctx.stats_store
    latencies = [store.get(k) if store is not None else None for _, k in available]
    known = [lat for lat in latencies if lat is not None]
    if not known:
        # Cold start: no latency signal yet — behave like cost_optimized.
        return _cost_ordered(available, ctx)

    median = statistics.median(known)
    filled_latency = [lat if lat is not None else median for lat in latencies]

    costs = [_estimated_cost(ctx.model_config, p) for p, _ in available]
    finite_costs = [c for c in costs if c != float("inf")]
    worst_finite = max(finite_costs) if finite_costs else 0.0
    filled_cost = [c if c != float("inf") else worst_finite for c in costs]

    norm_cost = _max_normalize(filled_cost)
    norm_latency = _max_normalize(filled_latency)
    scored = [
        (_BALANCED_COST_WEIGHT * nc + _BALANCED_LATENCY_WEIGHT * nl, rng.random(), pk)
        for pk, nc, nl in zip(available, norm_cost, norm_latency, strict=True)
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    return [pk for _, _, pk in scored]


def _sticky_ranked(available: list[Candidate], conv_key: str) -> list[Candidate]:
    """Rendezvous-hash ordering: deterministic per conversation key."""
    return sorted(
        available,
        key=lambda pk: _rendezvous_score(conv_key, pk[1]),
        reverse=True,
    )


def _sticky_ordered(
    available: list[Candidate],
    ctx: StrategyContext,
    sticky_key: str | None,
) -> list[Candidate]:
    """Order candidates with the conversation's pinned provider first.

    ``sticky_key`` comes from :func:`resolve_sticky_key` (resolved once per
    request). When it is absent (no Redis, or the pinned provider is
    unavailable/tried), the ordering is pure rendezvous rank — deterministic
    for the conversation, so fallbacks drift consistently.
    """
    conv_key = ctx.conversation_key
    rng = ctx.rng or _default_rng
    if not conv_key:
        shuffled = list(available)
        rng.shuffle(shuffled)
        return shuffled

    ordered = _sticky_ranked(available, conv_key)
    available_keys = {k for _, k in available}
    if sticky_key is None or sticky_key not in available_keys:
        return ordered
    head = [pk for pk in ordered if pk[1] == sticky_key]
    tail = [pk for pk in ordered if pk[1] != sticky_key]
    return head + tail


async def resolve_sticky_key(
    available: list[Candidate],
    ctx: StrategyContext,
) -> str | None:
    """Resolve the pinned provider key for a conversation via Redis.

    Returns the sticky provider mapping key when a stored mapping exists and
    is still among ``available``; otherwise picks the rendezvous head,
    persists it (best-effort), and returns it. Returns ``None`` when sticky
    resolution is impossible (no conversation key or no Redis) — the caller
    then relies on the stateless rendezvous ordering.
    """
    conv_key = ctx.conversation_key
    if not conv_key or ctx.redis is None or not available:
        return None

    redis_key = _sticky_redis_key(conv_key, ctx.model_name)
    try:
        stored = await ctx.redis.get(redis_key)
        if stored:
            stored = stored.decode("utf-8") if isinstance(stored, bytes) else stored
            available_keys = {k for _, k in available}
            if stored in available_keys:
                return stored
    except Exception as exc:
        # Degrade gracefully to rendezvous hashing, but leave a diagnostic
        # trace so Redis outages/wrong-types are visible.
        logger.warning(f"Redis sticky-provider lookup failed for key {redis_key}: {exc}")

    chosen_key = _sticky_ranked(available, conv_key)[0][1]
    try:
        await ctx.redis.setex(redis_key, STICKY_TTL_SECONDS, chosen_key)
    except Exception as exc:
        # Losing stickiness is recoverable; log so it is diagnosable.
        logger.warning(f"Redis sticky-provider persist failed for key {redis_key}: {exc}")
    return chosen_key


def order_providers(
    available: list[Candidate],
    strategy: ProviderSelectionStrategy,
    ctx: StrategyContext,
    sticky_key: str | None = None,
) -> list[Candidate]:
    """Order same-priority candidates according to the selection strategy.

    Args:
        available: Candidates (provider mapping, unique key) that are neither
            tried this request nor circuit-broken.
        strategy: The configured selection strategy.
        ctx: Per-request strategy inputs.
        sticky_key: Pinned provider key for ``session_sticky`` (from
            :func:`resolve_sticky_key`), if resolved.

    Returns:
        The candidates reordered; the caller picks ``result[0]``.
    """
    rng = ctx.rng or _default_rng
    if strategy is ProviderSelectionStrategy.SESSION_STICKY:
        return _sticky_ordered(available, ctx, sticky_key)
    if strategy is ProviderSelectionStrategy.COST_OPTIMIZED:
        return _cost_ordered(available, ctx)
    if strategy is ProviderSelectionStrategy.BALANCED:
        return _balanced_ordered(available, ctx)
    if strategy is not ProviderSelectionStrategy.RANDOM:
        # Defensive fallback for unknown future strategies: preserve current
        # behavior (random) rather than failing the request.
        logger.warning(f"Unknown provider strategy '{strategy}', falling back to random")
    shuffled = list(available)
    rng.shuffle(shuffled)
    return shuffled


def pick_provider(
    available: list[Candidate],
    strategy: ProviderSelectionStrategy,
    ctx: StrategyContext,
    sticky_key: str | None = None,
) -> Candidate | None:
    """Pick one provider from same-priority candidates per the strategy."""
    if not available:
        return None
    ordered = order_providers(available, strategy, ctx, sticky_key=sticky_key)
    return ordered[0]


__all__ = [
    "STICKY_TTL_SECONDS",
    "ProviderSelectionStrategy",
    "StrategyContext",
    "order_providers",
    "pick_provider",
    "resolve_sticky_key",
]
