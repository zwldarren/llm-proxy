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
from dataclasses import replace
from typing import Any

from llm_proxy.routing.quality import (
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
    difficulty_tier_label = tier.value
    budget = estimate_output_budget(prompt, difficulty_tier_label)
    effective_output = min(max_output_tokens, max(1, int(budget * max(0.1, 1.0))))

    quality_guard = apply_quality_guards(
        candidates,
        mode=mode,
        tier=tier,
        served_qualities=served_qualities or {},
        continuity_floor=effective_features.continuity_quality_floor,
        step_risk=effective_features.step_risk,
        agent_pressure=effective_features.agent_pressure,
        is_agentic=effective_features.is_agentic,
        has_tool_results=effective_features.has_tool_results,
    )
    candidates = quality_guard.allowed_models
    alignment_target = scoring_served_quality_target(
        mode,
        tier,
        quality_guard.target,
        quality_guard.floor,
        complexity=complexity,
        confidence=confidence,
        step_risk=effective_features.step_risk,
        is_agentic=effective_features.is_agentic,
        is_coding=effective_features.is_coding,
        has_tool_results=effective_features.has_tool_results,
        session_present=effective_features.session_present,
        agent_step_count=effective_features.agent_step_count,
        agent_pressure=effective_features.agent_pressure,
        verification_failed=effective_features.verification_failed,
    )

    # Adaptation: the source consults an ``uncommon_route.benchmark`` quality
    # cache (dynamic discovery of benchmark quality estimates), which llm-proxy
    # does not port. The selector already supports "benchmark unavailable" —
    # priors default to 0.5 and quality_estimate is None, so CandidateScore
    # prior fields fall back to their defaults. This matches the source's
    # except-branch behavior exactly.
    benchmark_quality: dict[str, float] | None = None
    benchmark_quality_estimates: dict[str, object] = {}

    quality_priors = _quality_prior_scores(candidates, benchmark_quality=benchmark_quality)
    experience = {}
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
            quality_priors = _quality_prior_scores(candidates, benchmark_quality=benchmark_quality)
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

    mu = complexity
    bandit_active = bc.enabled and tier in bc.enabled_tiers
    routine_exploration_disabled = bandit_active and _should_disable_routine_auto_exploration(
        mode=mode,
        tier=tier,
        features=effective_features,
    )
    if routine_exploration_disabled:
        bandit_active = False
    prior_n = max(0.0, float(bc.prior_n))

    # Mode controls quality-vs-cost preference:
    #   FAST  → strongly prefer cheap (low cost_sensitivity = quality matters less)
    #   AUTO  → balanced
    #   BEST  → strongly prefer quality (high cost_sensitivity = cost matters less)
    # Quality weight: how much does quality matter vs cost.
    #   FAST: cost-dominant — pick the cheapest decent model
    #   AUTO: balanced — best quality-per-dollar
    #   BEST: quality-only — pick the highest quality, ignore cost
    #
    # mode_weights can be configured via the admin UI (SmartRoutingConfig).
    # When not set, fall back to hardcoded defaults.
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
        step_risk=effective_features.step_risk,
        agent_pressure=effective_features.agent_pressure,
    )
    q_weight = max(0.0, min(1.0, q_weight))
    quality_alignment_weight = _dynamic_quality_alignment_weight(
        weights.quality_alignment,
        mode=mode,
        tier=tier,
        complexity=complexity,
        confidence=confidence,
        target=alignment_target,
        step_risk=effective_features.step_risk,
        agent_pressure=effective_features.agent_pressure,
    )

    # Relative quality gate: exclude models below X% of the best available.
    mode_gate_fraction = {
        RoutingMode.FAST: 0.50,
        RoutingMode.AUTO: 0.60,
        RoutingMode.BEST: 0.85,
    }
    gate_fraction = mode_gate_fraction.get(mode, 0.60)

    ranked: list[CandidateScore] = []
    all_predicted_qualities: dict[str, float] = {}
    premium_exploration_blocked = 0

    for model in candidates:
        exp = experience[model]
        benchmark_q = quality_priors.get(model, 0.5)
        quality_estimate = benchmark_quality_estimates.get(model)
        cost_norm = actual_cost_norm.get(model, 0.5)
        candidate_quality = quality_guard.quality_by_model.get(model, quality_guard.floor)
        quality_alignment = quality_alignment_score(candidate_quality, alignment_target)
        continuity_bias = continuity_alignment_score(
            candidate_quality, effective_features.previous_served_quality
        )
        model_stickiness = (
            0.06
            if effective_features.previous_model and model == effective_features.previous_model
            else 0.0
        )

        evidence_prior_n = _quality_prior_evidence_strength(quality_estimate, prior_n)
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
            bandit_active=bandit_active,
            mode=mode,
            tier=tier,
            candidate_quality=candidate_quality,
            candidate_cost=dollar_costs[model],
            cheapest_cost=cheapest_cost,
            bandit_config=bc,
            features=effective_features,
        )
        if should_sample:
            predicted_quality = effective_rng.betavariate(ts_alpha, ts_beta)
        elif bandit_active and candidate_quality is ServedQuality.PREMIUM:
            premium_exploration_blocked += 1
        exploration_bonus = 0.0
        all_predicted_qualities[model] = predicted_quality

        auxiliary = (
            weights.latency * exp.latency
            + weights.reliability * exp.reliability
            + weights.cache_affinity * exp.cache_affinity
            + quality_alignment_weight * quality_alignment
            + weights.continuity * continuity_bias
            + model_stickiness
            + weights.feedback_preference * exp.preference_ewma
        )
        total = (
            q_weight * predicted_quality - (1.0 - q_weight) * actual_cost_norm[model] + auxiliary
        )

        ranked.append(
            CandidateScore(
                model=model,
                total=total,
                predicted_cost=dollar_costs[model],
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
        features=effective_features,
    )
    ranked, premium_cost_note = _apply_premium_cost_benefit_guard(
        ranked,
        mode=mode,
        tier=tier,
        complexity=complexity,
        confidence=confidence,
        features=effective_features,
    )
    ranked, routine_premium_note = _apply_routine_premium_guard(
        ranked,
        mode=mode,
        tier=tier,
        target=alignment_target,
        features=effective_features,
    )

    selected = ranked[0]
    model = selected.model
    cost = selected.predicted_cost

    bp = ModelPricing(5.0, 25.0)
    baseline_cost = (estimated_input_tokens / 1_000_000) * bp.input_price + (
        effective_output / 1_000_000
    ) * bp.output_price
    savings = max(0.0, (baseline_cost - cost) / baseline_cost) if baseline_cost > 0 else 0.0

    method_note = "pool"
    constraint_tags = hard_constraints.tags()
    reasoning_parts = [
        reasoning_text,
        f"chooser=pool(complexity={complexity:.2f})",
        f"mode={mode.value}",
    ]

    if constraint_tags:
        reasoning_parts.append(f"constraints={','.join(constraint_tags)}")
    if step_stable:
        reasoning_parts.append("step-stable=no-bandit")
    if effective_features.previous_model and model == effective_features.previous_model:
        reasoning_parts.append(f"model-stickiness={model}")
    elif effective_features.previous_model:
        reasoning_parts.append(f"model-switch={effective_features.previous_model}->{model}")
    if abs(selected.feedback_preference) >= 0.05:
        reasoning_parts.append(f"feedback-preference={selected.feedback_preference:+.2f}")
    pressure_rescue_note = _pressure_rescue_note(
        effective_features,
        tier=tier,
        complexity=complexity,
        confidence=confidence,
    )
    if pressure_rescue_note:
        reasoning_parts.append(pressure_rescue_note)
    if cost_guard_note:
        reasoning_parts.append(cost_guard_note)
    if premium_cost_note:
        reasoning_parts.append(premium_cost_note)
    if routine_premium_note:
        reasoning_parts.append(routine_premium_note)
    if routine_exploration_disabled:
        reasoning_parts.append("routine-exploration=base-prior")
    if premium_exploration_blocked:
        reasoning_parts.append(f"premium-exploration=base-prior({premium_exploration_blocked})")
    reasoning_parts.extend(quality_guard.notes)
    if alignment_target is not quality_guard.target:
        reasoning_parts.append(f"served-quality-score-target={alignment_target.value}")
    if q_weight < default_q_weight - 0.001:
        reasoning_parts.append(f"economy-fit=cost-aware(q={q_weight:.2f})")
    if quality_alignment_weight > weights.quality_alignment:
        reasoning_parts.append(f"served-quality-fit-weight={quality_alignment_weight:.2f}")

    scorecards = [_candidate_scorecard(s).as_dict() for s in ranked]
    weights_used = _weights_dict(weights)
    guardrail_notes = list(reasoning_parts)

    return RoutingDecision(
        model=model,
        tier=tier,
        complexity=complexity,
        confidence=confidence,
        reasoning={
            "text": " | ".join(reasoning_parts),
            "parts": tuple(reasoning_parts),
            "method": method_note,
        },
        cost_estimate=cost,
        savings=savings,
        candidate_scores={s.model: s.total for s in ranked},
        candidate_scorecards=scorecards,
        weights_used=weights_used,
        guardrail_notes=guardrail_notes,
        signal_votes=signal_votes or {},
    )
