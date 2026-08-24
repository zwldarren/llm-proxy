"""Capability-lane and served-quality helpers for routing."""

from dataclasses import dataclass

from llm_proxy.routing.types import (
    RoutingMode,
    ServedQuality,
    Tier,
)

_QUALITY_RANK = {
    ServedQuality.ECONOMY: 0,
    ServedQuality.BALANCED: 1,
    ServedQuality.PREMIUM: 2,
}


def quality_rank(value: ServedQuality | str | None) -> int:
    if isinstance(value, ServedQuality):
        return _QUALITY_RANK[value]
    normalized = str(value or "").strip().lower()
    for quality, rank in _QUALITY_RANK.items():
        if quality.value == normalized:
            return rank
    return _QUALITY_RANK[ServedQuality.ECONOMY]


def target_served_quality(mode: RoutingMode, tier: Tier) -> ServedQuality:
    """Return the desired served quality for a (mode, tier) pair.

    Mode shifts the quality target relative to the tier baseline:
      FAST  → target one step below the tier baseline (prefer cheaper models)
      AUTO  → use the tier baseline
      BEST  → target one step above the tier baseline (prefer stronger models)

    The shift is clamped so it never goes below ECONOMY or above PREMIUM.
    """
    # Tier baseline: SIMPLE→economy, MEDIUM→balanced, COMPLEX→premium
    baseline: ServedQuality
    if tier is Tier.SIMPLE:
        baseline = ServedQuality.ECONOMY
    elif tier is Tier.MEDIUM:
        baseline = ServedQuality.BALANCED
    else:
        baseline = ServedQuality.PREMIUM

    rank = quality_rank(baseline)
    if mode is RoutingMode.FAST:
        rank = max(0, rank - 1)
    elif mode is RoutingMode.BEST:
        rank = min(2, rank + 1)
    # AUTO: rank unchanged

    return (ServedQuality.ECONOMY, ServedQuality.BALANCED, ServedQuality.PREMIUM)[rank]


def minimum_served_quality(mode: RoutingMode, tier: Tier) -> ServedQuality:
    """Return the minimum acceptable served quality for a (mode, tier) pair.

    For FAST mode the floor matches the (lowered) target so that economy
    models are not excluded from MEDIUM/COMPLEX tiers. For BEST mode the floor
    is also elevated to match its target.
    """
    return target_served_quality(mode, tier)


def stronger_quality(
    left: ServedQuality | None,
    right: ServedQuality | None,
) -> ServedQuality | None:
    if left is None:
        return right
    if right is None:
        return left
    return left if quality_rank(left) >= quality_rank(right) else right


def quality_alignment_score(
    candidate_quality: ServedQuality,
    target_quality: ServedQuality,
) -> float:
    delta = quality_rank(candidate_quality) - quality_rank(target_quality)
    if delta == 0:
        return 1.0
    if delta > 0:
        return 0.82
    if delta == -1:
        return 0.18
    return 0.0


def scoring_served_quality_target(
    mode: RoutingMode,
    tier: Tier,
    target: ServedQuality,
    floor: ServedQuality,
    *,
    complexity: float | None = None,
    confidence: float | None = None,
    step_risk: str = "normal",
    is_agentic: bool = False,
    is_coding: bool = False,
    has_tool_results: bool = False,
    session_present: bool = False,
    agent_step_count: int = 0,
    agent_pressure: float = 0.0,
    verification_failed: bool = False,
) -> ServedQuality:
    """Return the quality level used for score alignment.

    Public tiers are intentionally distinct served-quality bands:
    SIMPLE=economy, MEDIUM=balanced, COMPLEX=premium. Floors can only raise
    the selected quality when a feature requires more capability.
    """
    if quality_rank(floor) > quality_rank(target):
        return floor
    return target


def continuity_alignment_score(
    candidate_quality: ServedQuality,
    previous_quality: ServedQuality | None,
) -> float:
    if previous_quality is None:
        return 0.0
    delta = quality_rank(candidate_quality) - quality_rank(previous_quality)
    if delta >= 0:
        return 0.18 if delta == 0 else 0.08
    return -0.22 * abs(delta)


@dataclass(frozen=True, slots=True)
class QualityGuardResult:
    allowed_models: list[str]
    quality_by_model: dict[str, ServedQuality]
    target: ServedQuality
    floor: ServedQuality
    continuity_floor: ServedQuality | None
    notes: tuple[str, ...]


def apply_quality_guards(
    candidates: list[str],
    *,
    mode: RoutingMode,
    tier: Tier,
    served_qualities: dict[str, ServedQuality],
    continuity_floor: ServedQuality | None = None,
    step_risk: str = "normal",
    agent_pressure: float = 0.0,
    is_agentic: bool = False,
    has_tool_results: bool = False,
) -> QualityGuardResult:
    quality_by_model = dict(served_qualities)
    target = target_served_quality(mode, tier)
    normalized_step_risk = str(step_risk or "normal").strip().lower()
    floor = minimum_served_quality(mode, tier)
    risk_floor = ServedQuality.BALANCED if normalized_step_risk == "high" else None
    hard_continuity_floor = continuity_floor if mode is RoutingMode.BEST else None
    effective_floor = (
        stronger_quality(
            stronger_quality(floor, risk_floor),
            hard_continuity_floor,
        )
        or floor
    )
    preferred_threshold = (
        stronger_quality(stronger_quality(target, effective_floor), hard_continuity_floor) or target
    )
    notes: list[str] = [
        f"served-quality-target={target.value}",
        f"served-quality-floor={effective_floor.value}",
    ]
    if normalized_step_risk != "normal":
        notes.append(f"step-risk={normalized_step_risk}")
    if agent_pressure >= 0.35:
        notes.append(f"agent-pressure={agent_pressure:.2f}")
    if risk_floor is not None and normalized_step_risk == "high":
        notes.append(f"step-risk-floor={risk_floor.value}")
    elif risk_floor is not None:
        notes.append(f"agent-pressure-floor={risk_floor.value}")
    if continuity_floor is not None and hard_continuity_floor is None:
        notes.append(f"continuity-soft={continuity_floor.value}")
    exact_quality = stronger_quality(preferred_threshold, effective_floor) or target
    exact = [model for model in candidates if quality_by_model[model] is exact_quality]
    if exact:
        notes.append(f"served-quality=tier({exact_quality.value},{len(exact)}/{len(candidates)})")
        return QualityGuardResult(
            allowed_models=exact,
            quality_by_model=quality_by_model,
            target=target,
            floor=exact_quality,
            continuity_floor=hard_continuity_floor,
            notes=tuple(notes),
        )

    preferred = [
        model
        for model in candidates
        if quality_rank(quality_by_model[model]) >= quality_rank(preferred_threshold)
    ]
    if preferred:
        if hard_continuity_floor is not None and quality_rank(preferred_threshold) > quality_rank(
            target
        ):
            notes.append(f"continuity-floor={hard_continuity_floor.value}")
        notes.append(f"served-quality>=target({len(preferred)}/{len(candidates)})")
        return QualityGuardResult(
            allowed_models=preferred,
            quality_by_model=quality_by_model,
            target=target,
            floor=effective_floor,
            continuity_floor=hard_continuity_floor,
            notes=tuple(notes),
        )

    floor_candidates = [
        model
        for model in candidates
        if quality_rank(quality_by_model[model]) >= quality_rank(effective_floor)
    ]
    if floor_candidates:
        if hard_continuity_floor is not None:
            notes.append(f"continuity-floor-unavailable={hard_continuity_floor.value}")
        notes.append(f"served-quality>=floor({len(floor_candidates)}/{len(candidates)})")
        return QualityGuardResult(
            allowed_models=floor_candidates,
            quality_by_model=quality_by_model,
            target=target,
            floor=effective_floor,
            continuity_floor=hard_continuity_floor,
            notes=tuple(notes),
        )

    if hard_continuity_floor is not None:
        notes.append(f"continuity-floor-unavailable={hard_continuity_floor.value}")
    notes.append("served-quality-floor-unavailable")
    return QualityGuardResult(
        allowed_models=list(candidates),
        quality_by_model=quality_by_model,
        target=target,
        floor=effective_floor,
        continuity_floor=hard_continuity_floor,
        notes=tuple(notes),
    )
