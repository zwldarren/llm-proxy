"""Model selection with cost estimation and fallback chain.

Ported source-faithfully from ``uncommon_route/router/selector.py``.

Supports two selection modes:
  1. **Tier-based** (legacy): picks from a pre-assigned model list per tier.
  2. **Pool-based** (v2): all discovered models compete, complexity score
     adjusts cost-vs-quality weights dynamically.

Adaptations from the source (documented inline):
  - Imports renamed from ``uncommon_route.*`` to ``llm_proxy.routing.*``.
  - The source's ``uncommon_route.benchmark`` quality-estimate cache (a
    dynamic discovery dependency) is not ported; ``select_from_pool`` keeps
    the source's "benchmark unavailable" defaults (quality priors = 0.5,
    quality_estimate = None), which is exactly the source's except-branch.
  - ``_experience_snapshot`` additionally handles llm-proxy's
    ``ModelExperienceStore`` API (``get(name)`` / no-arg ``snapshot()``), which
    tracks flat per-model EWMAs rather than per-(model, mode, tier) snapshots.
  - The return ``RoutingDecision`` is mapped to llm-proxy's simplified shape:
    ``reasoning`` is a ``dict`` (``{"text", "parts"}`` or ``{"text","method"}``)
    rather than the source's ``str``; ``candidate_scores`` is
    ``dict[str, float]`` (model -> total) rather than ``list[CandidateScore]``.
    The rich per-candidate ``CandidateScore`` records are still computed and
    used internally by the cost guards and ranking.
"""

import contextlib
import logging
import math
import random
from dataclasses import dataclass, replace
from typing import Any

from llm_proxy.routing.model_experience import CandidateExperience
from llm_proxy.routing.quality import (
    QualityGuardResult,
    apply_quality_guards,
    continuity_alignment_score,
    quality_alignment_score,
    scoring_served_quality_target,
)
from llm_proxy.routing.selection.constraints import (
    _apply_constraints,
    _raise_budget_infeasible,
    _raise_constraint_infeasible,
    _raise_no_available_models,
)
from llm_proxy.routing.selection.cost import (
    _apply_cost_sanity_guard,
    _apply_premium_cost_benefit_guard,
    _apply_routine_premium_guard,
    _calc_cost,
)
from llm_proxy.routing.selection.scoring import (
    _dynamic_quality_alignment_weight,
    _dynamic_quality_cost_weight,
    _experience_snapshot,
    _quality_prior_evidence_strength,
    _quality_prior_scores,
    _should_disable_routine_auto_exploration,
    _should_sample_candidate_quality,
)
from llm_proxy.routing.structural import estimate_output_budget
from llm_proxy.routing.types import (
    BanditConfig,
    CandidateScore,
    CandidateScorecard,
    ModelPricing,
    RoutingConstraints,
    RoutingDecision,
    RoutingFailureCode,
    RoutingFeatures,
    RoutingInfeasibility,
    RoutingInfeasibleError,
    RoutingMode,
    SelectionWeights,
    ServedQuality,
    Tier,
    pressure_rescue_active,
    pressure_rescue_premium_allowed,
    pressure_rescue_premium_window,
)

logger = logging.getLogger("llm-proxy.routing.selector")

_rng = random.Random()

__all__ = [
    "select_from_pool",
]


def _candidate_scorecard(score: CandidateScore) -> CandidateScorecard:
    """Build the canonical serializable scorecard from a CandidateScore.

    ``CandidateScorecard.as_dict()`` is the single source of truth for the
    persisted scorecard shape, so the verbose log payload and
    ``RoutingDecision.candidate_scorecards`` can never drift apart.
    """
    return CandidateScorecard(
        model=score.model,
        total=score.total,
        predicted_cost=score.predicted_cost,
        predicted_quality=score.predicted_quality,
        cost=score.cost,
        latency=score.latency,
        reliability=score.reliability,
        cache_affinity=score.cache_affinity,
        quality_alignment=score.quality_alignment,
        continuity_bias=score.continuity_bias,
        editorial=score.editorial,
        bandit_mean=score.bandit_mean,
        exploration_bonus=score.exploration_bonus,
        samples=score.samples,
        served_quality=score.served_quality,
        feedback_preference=score.feedback_preference,
    )


def _weights_dict(weights: SelectionWeights) -> dict[str, float]:
    """Serialize the selection weights used for a routing decision."""
    return {
        "editorial": weights.editorial,
        "cost": weights.cost,
        "latency": weights.latency,
        "reliability": weights.reliability,
        "cache_affinity": weights.cache_affinity,
        "quality_alignment": weights.quality_alignment,
        "continuity": weights.continuity,
        "feedback_preference": weights.feedback_preference,
    }


def _stabilize_agent_step_selection(features: RoutingFeatures) -> bool:
    """Disable stochastic exploration for the current agent/tool step.

    This is intentionally not a hard model lock and does not make routing
    session-level. It only removes Thompson-sampling randomness on steps where
    the current request carries tool/protocol state.
    """
    return bool(
        features.is_agentic
        or features.has_tool_results
        or features.step_type in {"tool-selection", "tool-result-followup"}
    )


# ---------------------------------------------------------------------------
# Pool-based selection (v2) — all models compete
# ---------------------------------------------------------------------------


def _derive_tier(complexity: float) -> Tier:
    """Map continuous complexity back to the public 3-band tier model."""
    if complexity < 0.33:
        return Tier.SIMPLE
    if complexity < 0.67:
        return Tier.MEDIUM
    return Tier.COMPLEX


def _pressure_rescue_note(
    features: RoutingFeatures,
    *,
    tier: Tier,
    complexity: float,
    confidence: float,
) -> str:
    if not pressure_rescue_active(
        agent_pressure=features.agent_pressure,
        agent_step_count=features.agent_step_count,
        has_tool_results=features.has_tool_results,
        is_agentic=features.is_agentic,
        is_coding=features.is_coding,
    ):
        return ""
    if pressure_rescue_premium_allowed(
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
    ):
        if features.verification_failed:
            return "pressure-rescue=verification-review"
        return "pressure-rescue=premium-window"
    if pressure_rescue_premium_window(
        agent_pressure=features.agent_pressure,
        agent_step_count=features.agent_step_count,
        has_tool_results=features.has_tool_results,
        is_agentic=features.is_agentic,
        is_coding=features.is_coding,
    ):
        return "pressure-rescue=step-up"
    return "pressure-rescue=rolling-rebid"


def _filter_available_models(
    available_models: list[str],
    *,
    mode: RoutingMode,
    routing_assignments: dict[str, list[str]] | None,
    require_images: bool,
    supports_images: dict[str, bool] | None,
    context_lengths: dict[str, int | None] | None,
    estimated_input_tokens: int,
    max_output_tokens: int,
) -> list[str]:
    """Apply routing-assignment, image-support, and context-length filters.

    Raises the same infeasibility errors as the inline code it replaces, in
    the same order: empty pool, no mode-compatible model, no image-capable
    model, no model with sufficient context length.
    """
    if not available_models:
        _raise_no_available_models()

    # Filter by routing_assignments: if a model has explicit routing_assignments,
    # it is only considered for modes listed in its assignments. Models without
    # routing_assignments (None) are available for all modes.
    if routing_assignments:
        mode_value = mode.value if hasattr(mode, "value") else str(mode)
        available_models = [
            m
            for m in available_models
            if m not in routing_assignments
            or not routing_assignments[m]
            or mode_value in routing_assignments[m]
        ]
        if not available_models:
            _raise_no_available_models()

    # Filter by image support: when the request contains images, exclude
    # models that don't support image input.
    if require_images and supports_images is not None:
        available_models = [m for m in available_models if supports_images.get(m, False)]
        if not available_models:
            raise RoutingInfeasibleError(
                RoutingInfeasibility(
                    code=RoutingFailureCode.NO_AVAILABLE_MODELS,
                    message=(
                        "No models support image input. "
                        "Mark at least one model with 'Supports Images' enabled."
                    ),
                )
            )

    if context_lengths:
        required_capacity = estimated_input_tokens + max_output_tokens
        context_feasible: list[str] = []
        for m in available_models:
            cl = context_lengths.get(m)
            if cl is None or cl >= required_capacity:
                context_feasible.append(m)
        if context_feasible:
            available_models = context_feasible
        else:
            max_ctx = max(
                (cl for cl in context_lengths.values() if cl is not None),
                default=None,
            )
            raise RoutingInfeasibleError(
                RoutingInfeasibility(
                    code=RoutingFailureCode.NO_AVAILABLE_MODELS,
                    message=(
                        f"No routed model has sufficient context length for "
                        f"~{required_capacity} tokens (input + output). "
                        f"Largest configured context_length: {max_ctx}. "
                        "Increase a model's context_length or add a model "
                        "with a larger context window."
                    ),
                )
            )
    return available_models


def _apply_constraints_or_raise(
    available_models: list[str],
    hard_constraints: RoutingConstraints,
) -> list[str]:
    """Apply hard constraints, raising the canonical infeasibility error."""
    candidates, failed_constraint, applied_constraints = _apply_constraints(
        available_models,
        hard_constraints,
    )
    if failed_constraint is not None:
        _raise_constraint_infeasible(
            available_models=available_models,
            candidate_count=len(available_models),
            constraints=hard_constraints,
            failed_constraint=failed_constraint,
            applied_constraints=applied_constraints,
        )
    return candidates


def _estimate_effective_output(prompt: str, tier: Tier, max_output_tokens: int) -> int:
    """Clamp the prompt-derived output budget to the request's max tokens."""
    difficulty_tier_label = tier.value
    budget = estimate_output_budget(prompt, difficulty_tier_label)
    return min(max_output_tokens, max(1, budget))


@dataclass(frozen=True, slots=True)
class _QualityState:
    """Served-quality guard result plus the scoring alignment target."""

    guard: QualityGuardResult
    alignment_target: ServedQuality


def _apply_quality_guards_and_target(
    candidates: list[str],
    *,
    mode: RoutingMode,
    tier: Tier,
    served_qualities: dict[str, ServedQuality] | None,
    features: RoutingFeatures,
    complexity: float,
    confidence: float,
) -> tuple[list[str], _QualityState]:
    """Apply served-quality guards and derive the scoring alignment target."""
    quality_guard = apply_quality_guards(
        candidates,
        mode=mode,
        tier=tier,
        served_qualities=served_qualities or {},
        continuity_floor=features.continuity_quality_floor,
        step_risk=features.step_risk,
        agent_pressure=features.agent_pressure,
        is_agentic=features.is_agentic,
        has_tool_results=features.has_tool_results,
    )
    alignment_target = scoring_served_quality_target(
        mode,
        tier,
        quality_guard.target,
        quality_guard.floor,
        complexity=complexity,
        confidence=confidence,
        step_risk=features.step_risk,
        is_agentic=features.is_agentic,
        is_coding=features.is_coding,
        has_tool_results=features.has_tool_results,
        session_present=features.session_present,
        agent_step_count=features.agent_step_count,
        agent_pressure=features.agent_pressure,
        verification_failed=features.verification_failed,
    )
    return quality_guard.allowed_models, _QualityState(
        guard=quality_guard,
        alignment_target=alignment_target,
    )


def _collect_experience(
    candidates: list[str],
    model_experience: object | None,
    mode: RoutingMode,
    tier: Tier,
) -> dict[str, CandidateExperience]:
    """Snapshot per-model experience, falling back to the best-known tier."""
    experience: dict[str, CandidateExperience] = {}
    for m in candidates:
        exp_snapshot = _experience_snapshot(model_experience, m, mode, tier)
        if exp_snapshot.samples == 0:
            best_alt = exp_snapshot
            for alt_tier in (Tier.SIMPLE, Tier.MEDIUM, Tier.COMPLEX):
                alt = _experience_snapshot(model_experience, m, mode, alt_tier)
                if alt.samples > best_alt.samples:
                    best_alt = alt
            exp_snapshot = best_alt
        experience[m] = exp_snapshot
    return experience


@dataclass(frozen=True, slots=True)
class _CostModel:
    """Per-candidate dollar costs plus the max-cost-filtered candidate set."""

    candidates: list[str]
    quality_priors: dict[str, float]
    experience: dict[str, CandidateExperience]
    dollar_costs: dict[str, float]
    cheapest_cost: float
    actual_cost_norm: dict[str, float]


def _build_cost_model(
    candidates: list[str],
    model_experience: object | None,
    mode: RoutingMode,
    tier: Tier,
    *,
    estimated_input_tokens: int,
    effective_output: int,
    pricing: dict[str, ModelPricing],
    hard_constraints: RoutingConstraints,
    available_models: list[str],
) -> _CostModel:
    """Compute per-candidate dollar costs and apply the max_cost constraint.

    Raises ``RoutingInfeasibleError`` (budget) when no candidate fits
    ``max_cost``.
    """
    quality_priors = _quality_prior_scores(candidates)
    experience = _collect_experience(candidates, model_experience, mode, tier)
    dollar_costs = {
        m: _calc_cost(
            m,
            estimated_input_tokens,
            effective_output,
            pricing,
            input_cost_multiplier=experience[m].input_cost_multiplier,
        )
        for m in candidates
    }
    cheapest_cost = min(dollar_costs.values()) if dollar_costs else 0.0
    log_dollar_costs = {m: math.log1p(c * 1000) for m, c in dollar_costs.items()}
    max_log_dc = max(log_dollar_costs.values()) if log_dollar_costs else 1.0
    min_log_dc = min(log_dollar_costs.values()) if log_dollar_costs else 0.0
    span_dc = max_log_dc - min_log_dc
    actual_cost_norm = {
        m: (log_dollar_costs[m] - min_log_dc) / span_dc if span_dc > 0 else 0.5
        for m in dollar_costs
    }
    if hard_constraints.max_cost is not None:
        affordable = [
            model for model in candidates if dollar_costs[model] <= hard_constraints.max_cost
        ]
        if affordable:
            candidates = affordable
            quality_priors = _quality_prior_scores(candidates)
            experience = {m: experience[m] for m in candidates}
            dollar_costs = {m: dollar_costs[m] for m in candidates}
            cheapest_cost = min(dollar_costs.values()) if dollar_costs else 0.0
        else:
            _raise_budget_infeasible(
                available_models=available_models,
                candidate_count=len(candidates),
                constraints=hard_constraints,
                max_cost=hard_constraints.max_cost,
                cheapest_cost=cheapest_cost if dollar_costs else None,
            )
    return _CostModel(
        candidates=candidates,
        quality_priors=quality_priors,
        experience=experience,
        dollar_costs=dollar_costs,
        cheapest_cost=cheapest_cost,
        actual_cost_norm=actual_cost_norm,
    )


@dataclass(frozen=True, slots=True)
class _BanditState:
    """Thompson-sampling state resolved for this request."""

    active: bool
    routine_exploration_disabled: bool
    prior_n: float
    config: BanditConfig


def _resolve_bandit_state(
    bc: BanditConfig,
    tier: Tier,
    mode: RoutingMode,
    features: RoutingFeatures,
) -> _BanditState:
    """Decide whether Thompson sampling is active for this request."""
    bandit_active = bc.enabled and tier in bc.enabled_tiers
    routine_exploration_disabled = bandit_active and _should_disable_routine_auto_exploration(
        mode=mode,
        tier=tier,
        features=features,
    )
    if routine_exploration_disabled:
        bandit_active = False
    return _BanditState(
        active=bandit_active,
        routine_exploration_disabled=routine_exploration_disabled,
        prior_n=max(0.0, float(bc.prior_n)),
        config=bc,
    )


@dataclass(frozen=True, slots=True)
class _ResolvedWeights:
    """Computed quality-vs-cost weights for this request."""

    q_weight: float
    default_q_weight: float
    quality_alignment_weight: float
    gate_fraction: float


def _compute_weights(
    mode: RoutingMode,
    mode_weights: dict[str, float] | None,
    *,
    complexity: float,
    confidence: float,
    alignment_target: ServedQuality,
    weights: SelectionWeights,
    tier: Tier,
    features: RoutingFeatures,
) -> _ResolvedWeights:
    """Resolve quality-vs-cost weights and the relative quality gate.

    Mode controls quality-vs-cost preference:
      FAST  → strongly prefer cheap (low cost_sensitivity = quality matters less)
      AUTO  → balanced
      BEST  → strongly prefer quality (high cost_sensitivity = cost matters less)
    Quality weight: how much does quality matter vs cost.
      FAST: cost-dominant — pick the cheapest decent model
      AUTO: balanced — best quality-per-dollar
      BEST: quality-only — pick the highest quality, ignore cost

    mode_weights can be configured via the admin UI (SmartRoutingConfig).
    When not set, fall back to hardcoded defaults.
    """
    mu = complexity
    _default_mode_weights = {
        RoutingMode.FAST: 0.35,
        RoutingMode.AUTO: 0.65,
        RoutingMode.BEST: 1.0,
    }
    if mode_weights is not None:
        merged = dict(_default_mode_weights)
        for k, v in mode_weights.items():
            with contextlib.suppress(ValueError, TypeError):
                merged[RoutingMode(k)] = v
        mode_quality_weight = merged
    else:
        mode_quality_weight = _default_mode_weights
    base_q_weight = mode_quality_weight.get(mode, 0.65)
    default_q_weight = base_q_weight + mu * (1.0 - base_q_weight) * 0.8
    q_weight = _dynamic_quality_cost_weight(
        default_q_weight,
        mode=mode,
        tier=tier,
        complexity=complexity,
        target=alignment_target,
        step_risk=features.step_risk,
        agent_pressure=features.agent_pressure,
    )
    q_weight = max(0.0, min(1.0, q_weight))
    quality_alignment_weight = _dynamic_quality_alignment_weight(
        weights.quality_alignment,
        mode=mode,
        tier=tier,
        complexity=complexity,
        confidence=confidence,
        target=alignment_target,
        step_risk=features.step_risk,
        agent_pressure=features.agent_pressure,
    )

    # Relative quality gate: exclude models below X% of the best available.
    mode_gate_fraction = {
        RoutingMode.FAST: 0.50,
        RoutingMode.AUTO: 0.60,
        RoutingMode.BEST: 0.85,
    }
    gate_fraction = mode_gate_fraction.get(mode, 0.60)
    return _ResolvedWeights(
        q_weight=q_weight,
        default_q_weight=default_q_weight,
        quality_alignment_weight=quality_alignment_weight,
        gate_fraction=gate_fraction,
    )


def _score_candidates(
    cost_model: _CostModel,
    *,
    quality_state: _QualityState,
    features: RoutingFeatures,
    bandit_state: _BanditState,
    mode: RoutingMode,
    tier: Tier,
    complexity: float,
    weights: SelectionWeights,
    resolved_weights: _ResolvedWeights,
    rng: random.Random,
) -> tuple[list[CandidateScore], dict[str, float], int]:
    """Score every candidate; return (ranked, qualities, premium blocked)."""
    mu = complexity
    ranked: list[CandidateScore] = []
    all_predicted_qualities: dict[str, float] = {}
    premium_exploration_blocked = 0

    for model in cost_model.candidates:
        exp = cost_model.experience[model]
        benchmark_q = cost_model.quality_priors.get(model, 0.5)
        # Benchmark quality-estimate cache intentionally not ported (see module
        # docstring): "benchmark unavailable" means a None estimate here.
        quality_estimate = None
        cost_norm = cost_model.actual_cost_norm.get(model, 0.5)
        candidate_quality = quality_state.guard.quality_by_model.get(
            model, quality_state.guard.floor
        )
        quality_alignment = quality_alignment_score(
            candidate_quality, quality_state.alignment_target
        )
        continuity_bias = continuity_alignment_score(
            candidate_quality, features.previous_served_quality
        )
        model_stickiness = (
            0.06 if features.previous_model and model == features.previous_model else 0.0
        )

        evidence_prior_n = _quality_prior_evidence_strength(quality_estimate, bandit_state.prior_n)
        base_quality = (evidence_prior_n * benchmark_q + exp.samples * exp.reward_mean) / (
            evidence_prior_n + exp.samples
        )

        predicted_quality = base_quality

        # Thompson Sampling: use external benchmark evidence as prior
        # concentration. Complex tasks should not turn reliable priors into
        # high-variance random draws just because they are hard.
        exploration_scale = max(3.0, 4.0 + mu * 6.0)
        ts_concentration = max(exploration_scale, evidence_prior_n + exp.samples)
        ts_alpha = max(0.5, ts_concentration * base_quality)
        ts_beta = max(0.5, ts_concentration * (1.0 - base_quality))
        should_sample = _should_sample_candidate_quality(
            bandit_active=bandit_state.active,
            mode=mode,
            tier=tier,
            candidate_quality=candidate_quality,
            candidate_cost=cost_model.dollar_costs[model],
            cheapest_cost=cost_model.cheapest_cost,
            bandit_config=bandit_state.config,
            features=features,
        )
        if should_sample:
            predicted_quality = rng.betavariate(ts_alpha, ts_beta)
        elif bandit_state.active and candidate_quality is ServedQuality.PREMIUM:
            premium_exploration_blocked += 1
        exploration_bonus = 0.0
        all_predicted_qualities[model] = predicted_quality

        auxiliary = (
            weights.latency * exp.latency
            + weights.reliability * exp.reliability
            + weights.cache_affinity * exp.cache_affinity
            + resolved_weights.quality_alignment_weight * quality_alignment
            + weights.continuity * continuity_bias
            + model_stickiness
            + weights.feedback_preference * exp.preference_ewma
        )
        total = (
            resolved_weights.q_weight * predicted_quality
            - (1.0 - resolved_weights.q_weight) * cost_model.actual_cost_norm[model]
            + auxiliary
        )

        ranked.append(
            CandidateScore(
                model=model,
                total=total,
                predicted_cost=cost_model.dollar_costs[model],
                predicted_quality=predicted_quality,
                editorial=benchmark_q,
                quality_prior_confidence=float(getattr(quality_estimate, "confidence", 0.0)),
                cost=cost_norm,
                latency=exp.latency,
                reliability=exp.reliability,
                cache_affinity=exp.cache_affinity,
                quality_alignment=quality_alignment,
                continuity_bias=continuity_bias,
                served_quality=candidate_quality.value,
                bandit_mean=exp.reward_mean,
                exploration_bonus=exploration_bonus,
                samples=exp.samples,
                feedback_preference=exp.preference_ewma,
            )
        )
    return ranked, all_predicted_qualities, premium_exploration_blocked


@dataclass(frozen=True, slots=True)
class _GuardNotes:
    """Guardrail notes emitted by the ranking guards."""

    cost: str
    premium_cost: str
    routine_premium: str


def _rank_and_apply_guards(
    ranked: list[CandidateScore],
    all_predicted_qualities: dict[str, float],
    gate_fraction: float,
    *,
    mode: RoutingMode,
    tier: Tier,
    complexity: float,
    confidence: float,
    alignment_target: ServedQuality,
    features: RoutingFeatures,
) -> tuple[list[CandidateScore], _GuardNotes]:
    """Apply the relative quality gate and the cost/premium guardrails."""
    # Relative quality gate: exclude models below gate_fraction of best
    best_quality = max(all_predicted_qualities.values()) if all_predicted_qualities else 0.5
    quality_gate = best_quality * gate_fraction

    gated = [s for s in ranked if s.predicted_quality >= quality_gate]
    if gated:
        gated.sort(key=lambda s: s.total, reverse=True)
        below = [s for s in ranked if s.predicted_quality < quality_gate]
        below.sort(key=lambda s: s.total, reverse=True)
        ranked = gated + below
    else:
        ranked.sort(key=lambda s: s.predicted_quality, reverse=True)

    ranked, cost_guard_note = _apply_cost_sanity_guard(
        ranked,
        mode=mode,
        tier=tier,
        confidence=confidence,
        features=features,
    )
    ranked, premium_cost_note = _apply_premium_cost_benefit_guard(
        ranked,
        mode=mode,
        tier=tier,
        complexity=complexity,
        confidence=confidence,
        features=features,
    )
    ranked, routine_premium_note = _apply_routine_premium_guard(
        ranked,
        mode=mode,
        tier=tier,
        target=alignment_target,
        features=features,
    )
    return ranked, _GuardNotes(
        cost=cost_guard_note,
        premium_cost=premium_cost_note,
        routine_premium=routine_premium_note,
    )


def _compute_savings(estimated_input_tokens: int, effective_output: int, cost: float) -> float:
    """Savings vs a fixed 5/25 per-MToken baseline, floored at zero."""
    bp = ModelPricing(5.0, 25.0)
    baseline_cost = (estimated_input_tokens / 1_000_000) * bp.input_price + (
        effective_output / 1_000_000
    ) * bp.output_price
    return max(0.0, (baseline_cost - cost) / baseline_cost) if baseline_cost > 0 else 0.0


def _build_reasoning_parts(
    reasoning_text: str,
    *,
    mode: RoutingMode,
    complexity: float,
    constraint_tags: tuple[str, ...],
    step_stable: bool,
    features: RoutingFeatures,
    selected: CandidateScore,
    pressure_rescue_note: str,
    guard_notes: _GuardNotes,
    quality_state: _QualityState,
    resolved_weights: _ResolvedWeights,
    weights: SelectionWeights,
    bandit_state: _BanditState,
    premium_exploration_blocked: int,
) -> list[str]:
    """Assemble the human-readable reasoning trail for the decision."""
    reasoning_parts = [
        reasoning_text,
        f"chooser=pool(complexity={complexity:.2f})",
        f"mode={mode.value}",
    ]

    if constraint_tags:
        reasoning_parts.append(f"constraints={','.join(constraint_tags)}")
    if step_stable:
        reasoning_parts.append("step-stable=no-bandit")
    if features.previous_model and selected.model == features.previous_model:
        reasoning_parts.append(f"model-stickiness={selected.model}")
    elif features.previous_model:
        reasoning_parts.append(f"model-switch={features.previous_model}->{selected.model}")
    if abs(selected.feedback_preference) >= 0.05:
        reasoning_parts.append(f"feedback-preference={selected.feedback_preference:+.2f}")
    if pressure_rescue_note:
        reasoning_parts.append(pressure_rescue_note)
    if guard_notes.cost:
        reasoning_parts.append(guard_notes.cost)
    if guard_notes.premium_cost:
        reasoning_parts.append(guard_notes.premium_cost)
    if guard_notes.routine_premium:
        reasoning_parts.append(guard_notes.routine_premium)
    if bandit_state.routine_exploration_disabled:
        reasoning_parts.append("routine-exploration=base-prior")
    if premium_exploration_blocked:
        reasoning_parts.append(f"premium-exploration=base-prior({premium_exploration_blocked})")
    reasoning_parts.extend(quality_state.guard.notes)
    if quality_state.alignment_target is not quality_state.guard.target:
        reasoning_parts.append(
            f"served-quality-score-target={quality_state.alignment_target.value}"
        )
    if resolved_weights.q_weight < resolved_weights.default_q_weight - 0.001:
        reasoning_parts.append(f"economy-fit=cost-aware(q={resolved_weights.q_weight:.2f})")
    if resolved_weights.quality_alignment_weight > weights.quality_alignment:
        reasoning_parts.append(
            f"served-quality-fit-weight={resolved_weights.quality_alignment_weight:.2f}"
        )

    return reasoning_parts


def _build_decision(
    *,
    selected: CandidateScore,
    tier: Tier,
    complexity: float,
    confidence: float,
    reasoning_parts: list[str],
    savings: float,
    ranked: list[CandidateScore],
    weights: SelectionWeights,
    signal_votes: dict[str, Any] | None,
) -> RoutingDecision:
    """Assemble the final RoutingDecision from the ranked candidates."""
    scorecards = [_candidate_scorecard(s).as_dict() for s in ranked]
    return RoutingDecision(
        model=selected.model,
        tier=tier,
        complexity=complexity,
        confidence=confidence,
        reasoning={
            "text": " | ".join(reasoning_parts),
            "parts": tuple(reasoning_parts),
            "method": "pool",
        },
        cost_estimate=selected.predicted_cost,
        savings=savings,
        candidate_scores={s.model: s.total for s in ranked},
        candidate_scorecards=scorecards,
        weights_used=_weights_dict(weights),
        guardrail_notes=list(reasoning_parts),
        signal_votes=signal_votes or {},
    )


def select_from_pool(
    complexity: float,
    mode: RoutingMode,
    confidence: float,
    reasoning_text: str,
    available_models: list[str],
    estimated_input_tokens: int,
    max_output_tokens: int,
    prompt: str,
    pricing: dict[str, ModelPricing],
    constraints: RoutingConstraints | None = None,
    routing_features: RoutingFeatures | None = None,
    selection_weights: SelectionWeights | None = None,
    bandit_config: BanditConfig | None = None,
    model_experience: object | None = None,
    raw_confidence: float | None = None,
    rng: random.Random | None = None,
    routing_assignments: dict[str, list[str]] | None = None,
    mode_weights: dict[str, float] | None = None,
    served_qualities: dict[str, ServedQuality] | None = None,
    supports_images: dict[str, bool] | None = None,
    require_images: bool = False,
    context_lengths: dict[str, int | None] | None = None,
    signal_votes: dict[str, Any] | None = None,
) -> RoutingDecision:
    """Select the best model from the full discovered pool.

    Unlike ``select_model`` which picks from a per-tier list, this evaluates ALL
    available models and lets ``complexity`` drive the cost-vs-quality trade-off
    via weight interpolation.
    """
    effective_rng = rng if rng is not None else _rng
    weights = selection_weights or SelectionWeights()
    bc = bandit_config or BanditConfig()
    hard_constraints = constraints or RoutingConstraints()
    tier = _derive_tier(complexity)
    effective_features = routing_features or RoutingFeatures()
    step_stable = _stabilize_agent_step_selection(effective_features)
    if step_stable and bc.enabled:
        bc = replace(bc, enabled=False)

    available_models = _filter_available_models(
        available_models,
        mode=mode,
        routing_assignments=routing_assignments,
        require_images=require_images,
        supports_images=supports_images,
        context_lengths=context_lengths,
        estimated_input_tokens=estimated_input_tokens,
        max_output_tokens=max_output_tokens,
    )

    candidates = _apply_constraints_or_raise(available_models, hard_constraints)
    effective_output = _estimate_effective_output(prompt, tier, max_output_tokens)

    candidates, quality_state = _apply_quality_guards_and_target(
        candidates,
        mode=mode,
        tier=tier,
        served_qualities=served_qualities,
        features=effective_features,
        complexity=complexity,
        confidence=confidence,
    )

    cost_model = _build_cost_model(
        candidates,
        model_experience,
        mode,
        tier,
        estimated_input_tokens=estimated_input_tokens,
        effective_output=effective_output,
        pricing=pricing,
        hard_constraints=hard_constraints,
        available_models=available_models,
    )

    bandit_state = _resolve_bandit_state(bc, tier, mode, effective_features)
    resolved_weights = _compute_weights(
        mode,
        mode_weights,
        complexity=complexity,
        confidence=confidence,
        alignment_target=quality_state.alignment_target,
        weights=weights,
        tier=tier,
        features=effective_features,
    )

    ranked, all_predicted_qualities, premium_exploration_blocked = _score_candidates(
        cost_model,
        quality_state=quality_state,
        features=effective_features,
        bandit_state=bandit_state,
        mode=mode,
        tier=tier,
        complexity=complexity,
        weights=weights,
        resolved_weights=resolved_weights,
        rng=effective_rng,
    )

    ranked, guard_notes = _rank_and_apply_guards(
        ranked,
        all_predicted_qualities,
        resolved_weights.gate_fraction,
        mode=mode,
        tier=tier,
        complexity=complexity,
        confidence=confidence,
        alignment_target=quality_state.alignment_target,
        features=effective_features,
    )

    selected = ranked[0]
    savings = _compute_savings(estimated_input_tokens, effective_output, selected.predicted_cost)

    reasoning_parts = _build_reasoning_parts(
        reasoning_text,
        mode=mode,
        complexity=complexity,
        constraint_tags=hard_constraints.tags(),
        step_stable=step_stable,
        features=effective_features,
        selected=selected,
        pressure_rescue_note=_pressure_rescue_note(
            effective_features,
            tier=tier,
            complexity=complexity,
            confidence=confidence,
        ),
        guard_notes=guard_notes,
        quality_state=quality_state,
        resolved_weights=resolved_weights,
        weights=weights,
        bandit_state=bandit_state,
        premium_exploration_blocked=premium_exploration_blocked,
    )

    return _build_decision(
        selected=selected,
        tier=tier,
        complexity=complexity,
        confidence=confidence,
        reasoning_parts=reasoning_parts,
        savings=savings,
        ranked=ranked,
        weights=weights,
        signal_votes=signal_votes,
    )
