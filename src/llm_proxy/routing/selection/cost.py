"""Pricing, cost estimation, and cost-aware selection guards.

Extracted from ``llm_proxy.routing.selector``. Provides conservative
pricing fallbacks for unknown models, per-candidate dollar-cost estimation,
log-scale cost normalization, and the three post-ranking cost guards
(cost-sanity, premium-cost-benefit, routine-premium) that prevent exploration
from over-buying dominated or premium models on routine steps.
"""

import math

from llm_proxy.routing.quality import quality_rank
from llm_proxy.routing.types import (
    CandidateScore,
    ModelPricing,
    RoutingFeatures,
    RoutingMode,
    ServedQuality,
    Tier,
    pressure_rescue_premium_allowed,
)

_UNKNOWN_MODEL_PRICING = ModelPricing(5.0, 25.0)


def _pricing_for_model(model: str, pricing: dict[str, ModelPricing]) -> ModelPricing:
    """Use conservative pricing for unknown models instead of treating them as free."""
    candidate = pricing.get(model) or _UNKNOWN_MODEL_PRICING
    if (
        candidate.input_price < 0
        or candidate.output_price < 0
        or not math.isfinite(candidate.input_price)
        or not math.isfinite(candidate.output_price)
    ):
        return _UNKNOWN_MODEL_PRICING
    return candidate


def _calc_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, ModelPricing],
    *,
    input_cost_multiplier: float = 1.0,
) -> float:
    mp = _pricing_for_model(model, pricing)
    effective_multiplier = max(0.1, min(2.0, input_cost_multiplier))
    return (input_tokens / 1_000_000) * mp.input_price * effective_multiplier + (
        output_tokens / 1_000_000
    ) * mp.output_price


def _apply_cost_sanity_guard(
    ranked: list[CandidateScore],
    *,
    mode: RoutingMode,
    tier: Tier,
    confidence: float,
    features: RoutingFeatures,
) -> tuple[list[CandidateScore], str]:
    """Prevent AUTO/FAST exploration from preferring dominated expensive peers.

    Thompson sampling intentionally explores, but it should not let a same-
    quality, lower-prior, materially more expensive model beat a cheaper peer
    with comparable or stronger benchmark quality. BEST is left quality-first.
    """
    if mode is RoutingMode.BEST or len(ranked) < 2:
        return ranked, ""

    selected = ranked[0]
    selected_cost = max(0.0, selected.predicted_cost)
    if selected_cost <= 0:
        return ranked, ""

    selected_quality_rank = quality_rank(selected.served_quality)
    quality_tolerance = 0.03
    material_savings = 0.25

    alternatives = [
        score
        for score in ranked[1:]
        if quality_rank(score.served_quality) >= selected_quality_rank
        and score.editorial >= selected.editorial - quality_tolerance
        and score.predicted_cost > 0
        and score.predicted_cost <= selected_cost * (1.0 - material_savings)
    ]
    if not alternatives:
        return ranked, ""

    replacement = max(
        alternatives,
        key=lambda score: (
            quality_rank(score.served_quality),
            score.editorial,
            -score.predicted_cost,
            score.total,
        ),
    )
    if replacement.model == selected.model:
        return ranked, ""

    reordered = [replacement]
    reordered.extend(score for score in ranked if score.model != replacement.model)
    note = (
        "cost-sanity="
        f"{selected.model}->{replacement.model}"
        f"({selected_cost:.6f}->{replacement.predicted_cost:.6f})"
    )
    return reordered, note


def _apply_premium_cost_benefit_guard(
    ranked: list[CandidateScore],
    *,
    mode: RoutingMode,
    tier: Tier,
    complexity: float,
    confidence: float,
    features: RoutingFeatures,
) -> tuple[list[CandidateScore], str]:
    """Prefer measured near-peer balanced models over marginal premium wins.

    This is deliberately model-agnostic. Premium still wins for BEST mode,
    initial complex planning, and high-confidence hard recovery. In routine or
    uncertain agent steps, a premium model needs a meaningful measured quality
    advantage to justify an order-of-magnitude higher predicted cost.
    """
    if mode is not RoutingMode.AUTO or tier is not Tier.COMPLEX or len(ranked) < 2:
        return ranked, ""

    selected = ranked[0]
    if quality_rank(selected.served_quality) < quality_rank(ServedQuality.PREMIUM):
        return ranked, ""

    selected_cost = max(0.0, selected.predicted_cost)
    if selected_cost <= 0:
        return ranked, ""

    initial_complex_planning = (
        not features.has_tool_results
        and not features.session_present
        and features.agent_step_count == 0
    )
    if initial_complex_planning:
        return ranked, ""

    high_confidence_hard_step = (
        str(features.step_risk or "").strip().lower() == "high" and confidence >= 0.85
    )
    if high_confidence_hard_step:
        return ranked, ""
    high_pressure_review_step = pressure_rescue_premium_allowed(
        tier=tier,
        complexity=complexity,
        confidence=confidence,
        step_risk=features.step_risk,
        agent_pressure=features.agent_pressure,
        agent_step_count=features.agent_step_count,
        has_tool_results=features.has_tool_results,
        is_agentic=features.is_agentic,
        is_coding=features.is_coding,
        verification_failed=features.verification_failed,
    )
    if high_pressure_review_step:
        return ranked, ""

    max_quality_gap = 0.16
    economical_quality_gap = 0.14
    min_cost_ratio = 8.0
    decisive_cost_ratio = 12.0
    max_total_margin = max(0.025, abs(selected.total) * 0.035)
    alternatives: list[CandidateScore] = []
    for score in ranked[1:]:
        if quality_rank(score.served_quality) < quality_rank(ServedQuality.BALANCED):
            continue
        if score.predicted_cost <= 0:
            continue
        if selected_cost < score.predicted_cost * min_cost_ratio:
            continue
        quality_gap = selected.predicted_quality - score.predicted_quality
        if quality_gap > max_quality_gap:
            continue
        has_quality_basis = (
            score.editorial >= 0.60 or score.quality_prior_confidence >= 0.30 or score.samples > 0
        )
        if not has_quality_basis:
            continue
        near_total = score.total >= selected.total - max_total_margin
        decisive_cost_savings = (
            selected_cost >= score.predicted_cost * decisive_cost_ratio
            and quality_gap <= economical_quality_gap
        )
        if near_total or decisive_cost_savings:
            alternatives.append(score)
    if not alternatives:
        return ranked, ""

    replacement = max(
        alternatives,
        key=lambda score: (
            score.predicted_quality,
            score.editorial,
            -score.predicted_cost,
            score.total,
        ),
    )
    if replacement.model == selected.model:
        return ranked, ""

    reordered = [replacement]
    reordered.extend(score for score in ranked if score.model != replacement.model)
    note = (
        "premium-cost-benefit="
        f"{selected.model}->{replacement.model}"
        f"(q={selected.predicted_quality:.3f}->{replacement.predicted_quality:.3f},"
        f" cost={selected_cost:.6f}->{replacement.predicted_cost:.6f})"
    )
    return reordered, note


def _apply_routine_premium_guard(
    ranked: list[CandidateScore],
    *,
    mode: RoutingMode,
    tier: Tier,
    target: ServedQuality,
    features: RoutingFeatures,
) -> tuple[list[CandidateScore], str]:
    """Keep routine AUTO SIMPLE/MEDIUM routing from over-buying premium models."""
    if mode is not RoutingMode.AUTO or tier is Tier.COMPLEX or len(ranked) < 2:
        return ranked, ""
    if features.continuity_quality_floor is ServedQuality.PREMIUM:
        return ranked, ""

    selected = ranked[0]
    if quality_rank(selected.served_quality) < quality_rank(ServedQuality.PREMIUM):
        return ranked, ""
    selected_cost = max(0.0, selected.predicted_cost)
    if selected_cost <= 0:
        return ranked, ""

    target_rank = quality_rank(target)
    max_quality_gap = 0.18 if target is ServedQuality.BALANCED else 0.12
    min_cost_ratio = 3.0 if target is ServedQuality.BALANCED else 2.0
    alternatives: list[CandidateScore] = []
    for score in ranked[1:]:
        if quality_rank(score.served_quality) > max(
            target_rank, quality_rank(ServedQuality.BALANCED)
        ):
            continue
        if score.predicted_cost <= 0 or selected_cost < score.predicted_cost * min_cost_ratio:
            continue
        quality_gap = selected.predicted_quality - score.predicted_quality
        if quality_gap <= max_quality_gap:
            alternatives.append(score)
    if not alternatives:
        return ranked, ""

    replacement = max(
        alternatives,
        key=lambda score: (
            quality_rank(score.served_quality),
            score.predicted_quality,
            score.editorial,
            -score.predicted_cost,
            score.total,
        ),
    )
    if replacement.model == selected.model:
        return ranked, ""

    reordered = [replacement]
    reordered.extend(score for score in ranked if score.model != replacement.model)
    note = (
        "routine-premium-guard="
        f"{selected.model}->{replacement.model}"
        f"(q={selected.predicted_quality:.3f}->{replacement.predicted_quality:.3f},"
        f" cost={selected_cost:.6f}->{replacement.predicted_cost:.6f})"
    )
    return reordered, note
