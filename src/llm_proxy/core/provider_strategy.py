"""Pluggable provider-selection strategies.

A strategy *orders* the candidates within a single priority group; the
selector takes the head of the ordering on every pick, so per-request
fallback naturally walks the ordering (used/unavailable keys are filtered out
before each pick). Priority groups still dominate: strategies never reorder
across priority levels.

Every mapping also carries a ``weight`` (default 1) that splits traffic inside
the group it belongs to: with a 2 and a 1, the 2 is picked twice as often. How
weight enters the ordering depends on what the strategy means, and the neutral
weight 1 makes it a no-op everywhere:

- probabilistic strategies sample with probability proportional to weight
  (``random``: weighted sampling without replacement; ``session_sticky``:
  weighted rendezvous hashing, still deterministic per conversation);
- ranking strategies treat it as a preference multiplier (``cost_optimized``
  and ``balanced`` divide their score by the weight).

A weight of 0 is "never preferred": the mapping stays a last-resort fallback
but is not picked while any positive-weight candidate in its group remains.

Strategies:

- ``random``: uniform random pick — spreads load evenly, at the cost of
  re-warming a conversation's prompt cache on a different provider each turn.
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
  (in-memory EWMA, in the bucket matching the request: time-to-first-token for
  streaming, total duration otherwise). Degrades to ``cost_optimized`` while no
  latency samples exist. Picks the best score for most requests but hands a
  small share to the runners-up (:data:`_BALANCED_EXPLORE_RATE`): a provider
  that never gets picked never produces a fresh sample, and its stale score
  would otherwise freeze the ordering.
"""

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_proxy.config.types.model import (
    ModelConfig,
    ModelProviderConfig,
    ProviderSelectionStrategy,
)
from llm_proxy.core.provider_stats import DEFAULT_LATENCY_KIND, LatencyKind
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

# Share of balanced picks handed to a runner-up instead of the best score. It is
# what keeps every candidate measured: with a strict best-first ordering the
# runners-up receive traffic only after a failure, so their samples age out and
# the ordering never revisits the decision. The cost is that 1 in 10 turns of a
# conversation starts on a provider whose prompt cache is cold — far cheaper
# than a ranking that cannot change its mind.
_BALANCED_EXPLORE_RATE = 0.1

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
        latency_kind: Which latency measurement the balanced strategy compares
            (:data:`llm_proxy.core.provider_stats.KIND_TTFT` for a streaming
            request, ``KIND_TOTAL`` otherwise) so the two are never averaged
            together.
        rng: Optional randomness source (tests inject a seeded instance).
    """

    model_config: ModelConfig
    model_name: str
    conversation_key: str | None = None
    redis: Any | None = None
    stats_store: ProviderStatsStore | None = None
    latency_kind: LatencyKind = DEFAULT_LATENCY_KIND
    rng: random.Random | None = None


def _sticky_redis_key(conversation_key: str, model_name: str) -> str:
    """Redis key for the sticky provider mapping of a conversation+model pair."""
    return f"routing:conv:{conversation_key}:provider:{model_name}"


def _rendezvous_score(conversation_key: str, candidate_key: str) -> str:
    """Deterministic rendezvous-hashing digest for a conversation/mapping pair.

    Returned as hex so the full 256 bits stay in the ordering; the weighted
    ordering in :func:`_sticky_ranked` maps it into ``(0, 1]`` to apply the
    mapping's weight.
    """
    return hashlib.sha256(f"{conversation_key}:{candidate_key}".encode()).hexdigest()


# Size of the space a rendezvous digest is read into. Keeps the derived value
# in (0, 1] so ``-ln(u)`` is finite at the top of the range.
_RENDEZVOUS_HASH_SPACE = 1 << 256


def _effective_weight(provider: ModelProviderConfig) -> float:
    """Weight of a mapping, clamped to the valid range.

    ``ModelProviderConfig`` validates the field, but strategies are also
    reachable with hand-built candidates from callers and tests, and an
    unusable value must never fail a request: it falls back to the neutral 1.
    """
    weight = getattr(provider, "weight", 1.0)
    if weight is None:
        return 1.0
    try:
        return max(0.0, float(weight))
    except TypeError, ValueError:
        return 1.0


def _preference_score(score: float, provider: ModelProviderConfig) -> float:
    """Scale a lower-is-better score by the mapping's weight.

    ``cost_optimized`` and ``balanced`` rank candidates deterministically, so
    weight enters as a preference multiplier rather than a probability: the
    score is divided by the weight, which is what lets a weight of 2 tolerate
    twice the cost (or combined score) of a weight of 1. The default weight 1
    leaves every score untouched; a zero weight never wins a comparison.
    """
    weight = _effective_weight(provider)
    if weight <= 0:
        return float("inf")
    return score / weight


def _weighted_sample_ordered(available: list[Candidate], rng: random.Random) -> list[Candidate]:
    """Weighted random permutation (Efraimidis-Spirakis sampling keys).

    Each candidate is keyed by ``u ** (1 / weight)`` and the keys are sorted
    descending, so the head is picked with probability proportional to its
    weight (2 vs 1 gives 2/3 vs 1/3) while the tail stays a valid fallback
    order. With every weight at 1 the keys are uniform and this is the same
    uniform shuffle the strategies used before. Zero-weight candidates key to 0,
    so they follow every positive-weight candidate in random order.

    Sampling *without replacement* is what keeps the two properties the
    selector relies on: the head is weighted, and the ordering it walks on
    retry never repeats a mapping.
    """
    keyed: list[tuple[float, float, Candidate]] = []
    for candidate in available:
        weight = _effective_weight(candidate[0])
        sample = rng.random() ** (1.0 / weight) if weight > 0 else 0.0
        keyed.append((sample, rng.random(), candidate))
    keyed.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [candidate for _, _, candidate in keyed]


def _weighted_pick(available: list[Candidate], rng: random.Random) -> Candidate:
    """Pick one candidate with probability proportional to its weight."""
    weights = [_effective_weight(candidate[0]) for candidate in available]
    total = sum(weights)
    if total <= 0:
        # Nothing in the set carries a preference; fall back to a uniform draw
        # rather than failing the request.
        return rng.choice(available)
    target = rng.random() * total
    cumulative = 0.0
    for candidate, weight in zip(available, weights, strict=True):
        if weight <= 0:
            continue
        cumulative += weight
        if cumulative > target:
            return candidate
    return available[-1]


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
    """Order candidates by ascending weighted cost; shuffle ties.

    Weight divides the estimated cost (:func:`_preference_score`), so a heavier
    mapping wins more often without losing the cost ranking among equals.
    """
    rng = ctx.rng or _default_rng
    # (cost, random tiebreak) keeps ordering deterministic on price while
    # spreading load across equally-priced providers.
    return sorted(
        available,
        key=lambda pk: (
            _preference_score(_estimated_cost(ctx.model_config, pk[0]), pk[0]),
            rng.random(),
        ),
    )


def _balanced_ordered(available: list[Candidate], ctx: StrategyContext) -> list[Candidate]:
    """Order candidates by combined cost+latency score (ascending).

    Requests are served by the best score, except for the exploration share
    (:data:`_BALANCED_EXPLORE_RATE`) that starts on a runner-up so the runner-up
    keeps producing fresh samples.
    """
    rng = ctx.rng or _default_rng
    store = ctx.stats_store
    latencies = [
        store.get(k, ctx.latency_kind) if store is not None else None for _, k in available
    ]
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
        (
            _preference_score(_BALANCED_COST_WEIGHT * nc + _BALANCED_LATENCY_WEIGHT * nl, pk[0]),
            rng.random(),
            pk,
        )
        for pk, nc, nl in zip(available, norm_cost, norm_latency, strict=True)
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    ordered = [pk for _, _, pk in scored]
    # Zero-weight mappings are excluded from exploration: their weight says the
    # group's traffic should not land there while anything else is available.
    runner_ups = [pk for pk in ordered[1:] if _effective_weight(pk[0]) > 0]
    if runner_ups and rng.random() < _BALANCED_EXPLORE_RATE:
        # Exploration turn: start on a runner-up (weighted, so heavier mappings
        # are explored more), keep the remaining ordering so a fallback inside
        # this request still drifts consistently.
        head = _weighted_pick(runner_ups, rng)
        return [head, *[pk for pk in ordered if pk[1] != head[1]]]
    return ordered


def _sticky_ranked(available: list[Candidate], conv_key: str) -> list[Candidate]:
    """Weighted rendezvous-hash ordering: deterministic per conversation key.

    Plain rendezvous hashing gives every mapping the same chance of being a
    conversation's head. Weighting it with ``-ln(u) / weight`` keeps the head
    deterministic per conversation while making a weight-2 mapping head twice
    as often as a weight-1 one across conversations. Weight 1 is monotone with
    the bare hash, so existing conversations keep the provider they had.

    A zero-weight mapping scores ``inf`` and is therefore ranked last instead
    of being removed: it stays the group's fallback.
    """

    def score(candidate: Candidate) -> float:
        weight = _effective_weight(candidate[0])
        if weight <= 0:
            return float("inf")
        unit = (int(_rendezvous_score(conv_key, candidate[1]), 16) + 1) / _RENDEZVOUS_HASH_SPACE
        return -math.log(unit) / weight

    return sorted(available, key=score)


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
        # No conversation signal to hash on: pick proportionally to weight.
        return _weighted_sample_ordered(available, rng)

    ordered = _sticky_ranked(available, conv_key)
    available_keys = {k for _, k in available}
    if sticky_key is None or sticky_key not in available_keys:
        return ordered
    head = [pk for pk in ordered if pk[1] == sticky_key]
    tail = [pk for pk in ordered if pk[1] != sticky_key]
    return head + tail


async def _refresh_sticky_ttl(
    redis: Any,
    redis_key: str,
    stored: str,
    conv_key: str,
    available: list[Candidate],
) -> None:
    """Keep a pin that hashing would not reproduce from expiring.

    A conversation whose pin is also the rendezvous head of the currently
    available set carries no information: if the mapping expires, the next
    request derives the same head and persists it again. Skipping the write
    keeps the steady state at one Redis command per request.

    A pin that differs from the head is the opposite: the conversation drifted
    to a backup provider while its head was down and has since warmed that
    provider's prompt cache. Expiring it would drop the conversation back onto
    the head mid-conversation, so the TTL is extended on every hit.
    """
    if available and _sticky_ranked(available, conv_key)[0][1] == stored:
        return
    try:
        await redis.setex(redis_key, STICKY_TTL_SECONDS, stored)
    except Exception as exc:
        # Losing the refresh only shortens the pin's life; log so it is
        # diagnosable rather than silent.
        logger.warning(f"Redis sticky-provider refresh failed for key {redis_key}: {exc}")


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
                await _refresh_sticky_ttl(ctx.redis, redis_key, stored, conv_key, available)
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
    return _weighted_sample_ordered(available, rng)


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
