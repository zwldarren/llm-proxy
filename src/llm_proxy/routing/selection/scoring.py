"""Candidate scoring helpers: experience snapshots and dynamic weight helpers.

Extracted from ``llm_proxy.routing.selector``. Owns the experience-store
snapshots, quality priors and evidence strength, candidate-quality sampling
gates, and the dynamic quality-alignment / quality-cost weight adjustments
driven by mode, tier, complexity, and agent pressure.
"""

import logging
import math
from typing import Any

from llm_proxy.routing.model_experience import CandidateExperience
from llm_proxy.routing.types import (
    BanditConfig,
    ModelExperience,
    RoutingFeatures,
    RoutingMode,
    ServedQuality,
    Tier,
)

logger = logging.getLogger("llm-proxy.routing.scoring")


def _candidate_experience_from_model(exp: ModelExperience) -> CandidateExperience:
    """Convert a flat per-model ``ModelExperience`` record to a ``CandidateExperience``.

    Adaptation: llm-proxy's ``ModelExperienceStore`` tracks flat per-model EWMAs
    (no per-mode/tier breakdown and no ``input_cost_multiplier``), so we map the
    overlapping fields and default ``input_cost_multiplier`` to 1.0.
    """
    if exp.samples == 0:
        # preference_ewma is feedback-driven and independent of request
        # observations, so it is kept even with zero samples.
        return CandidateExperience(preference_ewma=exp.preference_ewma)
    return CandidateExperience(
        reliability=exp.reliability,
        latency=exp.latency,
        cache_affinity=exp.cache_affinity,
        input_cost_multiplier=1.0,
        reward_mean=exp.reward_mean,
        samples=exp.samples,
        preference_ewma=exp.preference_ewma,
    )


def _experience_snapshot(
    store: Any,
    model: str,
    mode: RoutingMode,
    tier: Tier,
) -> CandidateExperience:
    if store is None:
        return CandidateExperience()
    snapshot: object | None = None
    if hasattr(store, "snapshot"):
        try:
            snapshot = store.snapshot(model, mode, tier)
        except TypeError:
            try:
                snapshot = store.snapshot(model, mode)
            except TypeError:
                try:
                    # llm-proxy's ModelExperienceStore.snapshot() takes no args
                    # and returns dict[str, ModelExperience].
                    snapshot = store.snapshot()
                except TypeError:
                    snapshot = None
        if isinstance(snapshot, CandidateExperience):
            return snapshot
        if isinstance(snapshot, dict) and model in snapshot:
            return _candidate_experience_from_model(snapshot[model])
    # Stores without a compatible snapshot() but exposing get(name) -> ModelExperience.
    if hasattr(store, "get"):
        try:
            return _candidate_experience_from_model(store.get(model))
        except Exception as exc:
            # Real reliability/latency/cost signals are discarded here; a
            # degraded provider could win if this fires silently, so log it.
            logger.warning(
                "Experience store get(%r) failed; using neutral default: %s",
                model,
                exc,
            )
            return CandidateExperience()
    return CandidateExperience()


def _quality_prior_scores(
    models: list[str],
    benchmark_quality: dict[str, float] | None = None,
) -> dict[str, float]:
    """Quality prior from benchmark data.  Returns 0.5 (neutral) for
    unknown models — the experience system learns actual quality over time.
    """
    if benchmark_quality:
        return {m: benchmark_quality.get(m, 0.5) for m in models}
    return {m: 0.5 for m in models}


def _quality_prior_evidence_strength(
    quality_estimate: object | None,
    default_prior_n: float,
) -> float:
    """Return pseudo-count strength for external quality priors.

    Benchmark priors are not all equal. An exact, fresh, multi-sample estimate
    should reduce exploration variance more than a fuzzy or unknown prior,
    while still leaving room for local outcome feedback to override it.
    """
    base = max(0.0, float(default_prior_n))
    if quality_estimate is None:
        return base

    try:
        confidence = max(0.0, min(1.0, float(getattr(quality_estimate, "confidence", 0.0))))
    except TypeError, ValueError:
        confidence = 0.0
    try:
        samples = max(0, int(getattr(quality_estimate, "sample_count", 0)))
    except TypeError, ValueError:
        samples = 0

    if confidence <= 0.0 or samples <= 0:
        return base

    # Diminishing returns: 40 external runs should stabilize exploration, but
    # not dominate real local feedback forever.
    external_strength = min(30.0, 5.0 * math.sqrt(float(samples))) * confidence
    return max(base, base + external_strength)


def _dynamic_quality_alignment_weight(
    base_weight: float,
    *,
    mode: RoutingMode,
    tier: Tier,
    complexity: float,
    confidence: float,
    target: ServedQuality,
    step_risk: str,
    agent_pressure: float,
) -> float:
    """Adjust fit pressure only when target fit matters.

    This is intentionally a scoring signal, not a model cap. A model outside
    the target quality can still win when its measured quality/cost is better.
    """
    if mode is not RoutingMode.AUTO:
        return base_weight

    normalized_step_risk = str(step_risk or "normal").strip().lower()
    if target is ServedQuality.ECONOMY and (tier is Tier.SIMPLE or normalized_step_risk == "low"):
        return max(base_weight, 0.22)

    if target is ServedQuality.BALANCED and agent_pressure >= 0.55:
        pressure = max(0.0, min(1.0, (agent_pressure - 0.55) / 0.45))
        low_risk_pressure = 0.5 if normalized_step_risk == "low" else 1.0
        return base_weight + (0.08 * pressure * low_risk_pressure)

    if tier is not Tier.COMPLEX or target is not ServedQuality.PREMIUM:
        return base_weight

    complexity_pressure = max(0.0, min(1.0, (complexity - 0.84) / 0.16))
    confidence_pressure = max(0.0, min(1.0, (confidence - 0.60) / 0.30))
    rescue_pressure = max(0.0, min(1.0, (agent_pressure - 0.55) / 0.45))
    pressure = max(min(complexity_pressure, confidence_pressure), rescue_pressure)
    return base_weight + (0.30 * pressure)


def _dynamic_quality_cost_weight(
    default_weight: float,
    *,
    mode: RoutingMode,
    tier: Tier,
    complexity: float,
    target: ServedQuality,
    step_risk: str,
    agent_pressure: float,
) -> float:
    """Lower marginal quality weight once an economy model is the right fit."""
    if mode is not RoutingMode.AUTO:
        return default_weight
    normalized_step_risk = str(step_risk or "normal").strip().lower()
    if target is not ServedQuality.ECONOMY:
        return default_weight
    if agent_pressure >= 0.55 and normalized_step_risk != "low":
        return default_weight
    if tier is not Tier.SIMPLE and normalized_step_risk != "low":
        return default_weight

    economy_weight = 0.42 + (0.20 * max(0.0, min(1.0, complexity)))
    return min(default_weight, economy_weight)


def _should_sample_candidate_quality(
    *,
    bandit_active: bool,
    mode: RoutingMode,
    tier: Tier,
    candidate_quality: ServedQuality,
    candidate_cost: float,
    cheapest_cost: float,
    bandit_config: BanditConfig,
    features: RoutingFeatures,
) -> bool:
    if not bandit_active:
        return False
    if mode is not RoutingMode.AUTO:
        return True
    if tier is Tier.COMPLEX:
        return True
    if candidate_quality is not ServedQuality.PREMIUM:
        return True

    if cheapest_cost <= 0:
        materially_expensive = candidate_cost > 0
    else:
        materially_expensive = candidate_cost > cheapest_cost * max(
            2.0, float(bandit_config.max_cost_ratio)
        )
    if not materially_expensive:
        return True

    return bool(
        str(features.step_risk or "").strip().lower() == "high"
        or features.continuity_quality_floor is ServedQuality.PREMIUM
    )


def _should_disable_routine_auto_exploration(
    *,
    mode: RoutingMode,
    tier: Tier,
    features: RoutingFeatures,
) -> bool:
    if mode is not RoutingMode.AUTO:
        return False
    if tier is Tier.COMPLEX:
        return False
    return not bool(
        str(features.step_risk or "").strip().lower() == "high"
        or features.continuity_quality_floor is ServedQuality.PREMIUM
    )
