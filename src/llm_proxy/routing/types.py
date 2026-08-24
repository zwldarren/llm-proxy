"""Core type definitions for the smart routing engine.

Ported from UncommonRoute with adaptations for llm-proxy.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Tier(StrEnum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


class ServedQuality(StrEnum):
    ECONOMY = "economy"
    BALANCED = "balanced"
    PREMIUM = "premium"


class RoutingMode(StrEnum):
    AUTO = "auto"
    FAST = "fast"
    BEST = "best"


PRESSURE_RESCUE_MIN_STEPS = 24
PRESSURE_RESCUE_PREMIUM_WINDOW_STEPS = 10
PRESSURE_RESCUE_PREMIUM_COMPLEXITY = 0.86
PRESSURE_RESCUE_PREMIUM_CONFIDENCE = 0.30
VERIFICATION_RESCUE_PREMIUM_COMPLEXITY = 0.68


class RoutingFailureCode(StrEnum):
    NO_AVAILABLE_MODELS = "no_available_models"
    ALLOWLIST_EXHAUSTED = "allowlist_exhausted"
    ROUTING_CONSTRAINTS_UNMET = "routing_constraints_unmet"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass(frozen=True, slots=True)
class TierVote:
    """A signal's prediction: tier_id (0-3) or None to abstain."""

    tier_id: int | None
    confidence: float

    def __post_init__(self):
        if self.tier_id is not None:
            if not isinstance(self.tier_id, int):
                raise TypeError(f"tier_id must be int or None, got {type(self.tier_id).__name__}")
            if not (0 <= self.tier_id <= 3):
                raise ValueError(f"tier_id must be 0-3 or None, got {self.tier_id}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")

    @property
    def abstained(self) -> bool:
        return self.tier_id is None


@dataclass(frozen=True, slots=True)
class DimensionScore:
    """One structural dimension with its score and an optional signal tag."""

    name: str
    score: float  # [-1, 1]
    signal: str | None = None


@dataclass(frozen=True, slots=True)
class ScoringResult:
    """Classification result from the structural classifier."""

    tier: Tier | None
    confidence: float
    signals: tuple[str, ...]
    dimensions: tuple[DimensionScore, ...] = ()
    complexity: float = 0.33


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """Result from the weighted ensemble over signal predictions."""

    tier_id: int | None
    confidence: float
    raw_confidence: float
    method: str  # "direct" | "escalated" | "conservative" | "abstain"
    tier_scores: list[float]


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-model pricing in dollars per 1M tokens."""

    input_price: float
    output_price: float


@dataclass(frozen=True, slots=True)
class RoutingConstraints:
    """Hard constraints on which models/providers are eligible."""

    allowed_models: tuple[str, ...] = ()
    allowed_providers: tuple[str, ...] = ()
    max_cost: float | None = None

    def tags(self) -> tuple[str, ...]:
        labels: list[str] = []
        if self.allowed_models:
            labels.append("model-subset")
        if self.allowed_providers:
            labels.append("provider-subset")
        if self.max_cost is not None:
            labels.append("budget-cap")
        return tuple(labels)


@dataclass(frozen=True, slots=True)
class RoutingFeatures:
    """Feature vector extracted from a request for routing decisions."""

    step_type: str = "general"
    has_tool_results: bool = False
    step_risk: str = "normal"
    requested_max_output_tokens: int | None = None
    tier_floor: Tier | None = None
    tier_cap: Tier | None = None
    tier_cap_reason: str = ""
    session_present: bool = False
    agent_step_count: int = 0
    agent_pressure: float = 0.0
    previous_model: str | None = None
    previous_served_quality: ServedQuality | None = None
    continuity_quality_floor: ServedQuality | None = None
    verification_failed: bool = False
    failure_kind: str = ""

    @property
    def has_tools(self) -> bool:
        return self.has_tool_results

    @property
    def is_agentic(self) -> bool:
        """True if request involves tool execution (agentic behavior)."""
        return self.has_tool_results

    @property
    def is_coding(self) -> bool:
        """True if request involves tool execution (alias for is_agentic)."""
        return self.has_tool_results

    def tags(self) -> tuple[str, ...]:
        labels: list[str] = []
        if self.step_type != "general":
            labels.append(f"step:{self.step_type}")
        if self.has_tool_results:
            labels.append("tool-results")
        if self.step_risk != "normal":
            labels.append(f"risk:{self.step_risk}")
        if self.session_present:
            labels.append("session")
        if self.tier_cap is not None and self.tier_cap_reason:
            labels.append(f"cap:{self.tier_cap_reason}")
        if self.agent_step_count > 0:
            labels.append(f"agent-steps:{min(99, self.agent_step_count)}")
        if self.agent_pressure >= 0.35:
            labels.append(f"agent-pressure:{min(1.0, max(0.0, self.agent_pressure)):.2f}")
        if self.previous_model is not None:
            labels.append(f"prev-model:{self.previous_model}")
        if self.previous_served_quality is not None:
            labels.append(f"prev-quality:{self.previous_served_quality.value}")
        if self.continuity_quality_floor is not None:
            labels.append(f"continuity-floor:{self.continuity_quality_floor.value}")
        if self.verification_failed:
            labels.append("verification-failed")
        if self.failure_kind:
            labels.append(f"failure:{self.failure_kind}")
        return tuple(labels)


def pressure_rescue_active(
    *,
    agent_pressure: float,
    agent_step_count: int,
    has_tool_results: bool,
    is_agentic: bool,
    is_coding: bool,
) -> bool:
    """Whether an agent loop is stuck enough to raise the tier floor.

    This is a per-step rescue signal, not session stickiness.
    """
    return bool(
        agent_pressure >= 0.70
        and agent_step_count >= PRESSURE_RESCUE_MIN_STEPS
        and has_tool_results
        and (is_agentic or is_coding)
    )


def pressure_rescue_premium_window(
    *,
    agent_pressure: float,
    agent_step_count: int,
    has_tool_results: bool,
    is_agentic: bool,
    is_coding: bool,
) -> bool:
    """Allow a bounded premium burst before cost-aware rebidding resumes."""
    return bool(
        pressure_rescue_active(
            agent_pressure=agent_pressure,
            agent_step_count=agent_step_count,
            has_tool_results=has_tool_results,
            is_agentic=is_agentic,
            is_coding=is_coding,
        )
        and agent_step_count < PRESSURE_RESCUE_MIN_STEPS + PRESSURE_RESCUE_PREMIUM_WINDOW_STEPS
    )


def pressure_rescue_premium_allowed(
    *,
    tier: Tier,
    complexity: float | None,
    confidence: float | None,
    step_risk: str,
    agent_pressure: float,
    agent_step_count: int,
    has_tool_results: bool,
    is_agentic: bool,
    is_coding: bool,
    verification_failed: bool = False,
) -> bool:
    """Whether pressure rescue may temporarily bypass premium cost rebidding."""
    active = pressure_rescue_active(
        agent_pressure=agent_pressure,
        agent_step_count=agent_step_count,
        has_tool_results=has_tool_results,
        is_agentic=is_agentic,
        is_coding=is_coding,
    )
    if (
        verification_failed
        and tier is Tier.COMPLEX
        and complexity is not None
        and confidence is not None
        and complexity >= VERIFICATION_RESCUE_PREMIUM_COMPLEXITY
        and confidence >= PRESSURE_RESCUE_PREMIUM_CONFIDENCE
        and str(step_risk or "normal").strip().lower() == "high"
        and active
    ):
        return True

    return bool(
        tier is Tier.COMPLEX
        and complexity is not None
        and confidence is not None
        and complexity >= PRESSURE_RESCUE_PREMIUM_COMPLEXITY
        and confidence >= PRESSURE_RESCUE_PREMIUM_CONFIDENCE
        and str(step_risk or "normal").strip().lower() != "low"
        and pressure_rescue_premium_window(
            agent_pressure=agent_pressure,
            agent_step_count=agent_step_count,
            has_tool_results=has_tool_results,
            is_agentic=is_agentic,
            is_coding=is_coding,
        )
    )


@dataclass(frozen=True, slots=True)
class CandidateScorecard:
    """Serializable, human-readable breakdown of one candidate's routing score."""

    model: str
    total: float
    predicted_cost: float
    predicted_quality: float
    cost: float
    latency: float
    reliability: float
    cache_affinity: float
    quality_alignment: float
    continuity_bias: float
    editorial: float
    bandit_mean: float
    exploration_bonus: float
    samples: int
    served_quality: str
    feedback_preference: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "model": self.model,
            "total": self.total,
            "predicted_cost": self.predicted_cost,
            "predicted_quality": self.predicted_quality,
            "cost": self.cost,
            "latency": self.latency,
            "reliability": self.reliability,
            "cache_affinity": self.cache_affinity,
            "quality_alignment": self.quality_alignment,
            "continuity_bias": self.continuity_bias,
            "editorial": self.editorial,
            "bandit_mean": self.bandit_mean,
            "exploration_bonus": self.exploration_bonus,
            "samples": self.samples,
            "served_quality": self.served_quality,
            "feedback_preference": self.feedback_preference,
        }


@dataclass
class RoutingDecision:
    """Result of a routing decision: which model to use and why."""

    model: str
    tier: Tier
    complexity: float = 0.33
    confidence: float = 0.0
    reasoning: dict[str, object] = field(default_factory=dict)
    cost_estimate: float = 0.0
    savings: float = 0.0
    candidate_scores: dict[str, float] = field(default_factory=dict)
    candidate_scorecards: list[dict[str, Any]] = field(default_factory=list)
    weights_used: dict[str, float] = field(default_factory=dict)
    guardrail_notes: list[str] = field(default_factory=list)
    signal_votes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePool:
    """Bundle of candidate models + their pricing/quality/routing assignments."""

    available_models: list[str]
    pricing: dict[str, ModelPricing]
    served_qualities: dict[str, ServedQuality]
    routing_assignments: dict[str, list[str]]
    supports_images: dict[str, bool]
    context_lengths: dict[str, int | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    model: str
    total: float
    predicted_cost: float
    predicted_quality: float = 0.5
    editorial: float = 0.0
    quality_prior_confidence: float = 0.0
    cost: float = 0.0
    latency: float = 0.0
    reliability: float = 0.0
    cache_affinity: float = 0.0
    quality_alignment: float = 0.0
    continuity_bias: float = 0.0
    served_quality: str = ""
    bandit_mean: float = 0.5
    exploration_bonus: float = 0.0
    samples: int = 0
    feedback_preference: float = 0.0


@dataclass(frozen=True, slots=True)
class RoutingInfeasibility:
    code: RoutingFailureCode
    message: str
    available_model_count: int = 0
    candidate_count: int = 0
    constraint_tags: tuple[str, ...] = ()
    failed_constraints: tuple[str, ...] = ()
    max_cost: float | None = None
    cheapest_cost: float | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code.value,
            "message": self.message,
            "available_model_count": self.available_model_count,
            "candidate_count": self.candidate_count,
        }
        if self.constraint_tags:
            payload["constraint_tags"] = list(self.constraint_tags)
        if self.failed_constraints:
            payload["failed_constraints"] = list(self.failed_constraints)
        if self.max_cost is not None:
            payload["max_cost"] = self.max_cost
        if self.cheapest_cost is not None:
            payload["cheapest_cost"] = self.cheapest_cost
        return payload


class RoutingInfeasibleError(RuntimeError):
    def __init__(self, infeasibility: RoutingInfeasibility) -> None:
        super().__init__(infeasibility.message)
        self.infeasibility = infeasibility


@dataclass
class ModelExperience:
    """Per-model EWMA stats read by Thompson sampling.

    ``preference_ewma`` (in [-1, 1]) accumulates explicit user feedback
    deltas; ``feedback`` (in [0, 1]) is its DB-persisted projection. The two
    are linked by ``feedback = 0.5 + preference_ewma * 0.5`` so the DB row
    only needs the single ``feedback`` column.
    """

    name: str
    samples: int = 0
    reward_mean: float = 0.5
    latency: float = 0.0
    reliability: float = 1.0
    cache_affinity: float = 0.0
    preference_ewma: float = 0.0
    feedback: float = 0.5


# ---------------------------------------------------------------------------
# Routing configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectionWeights:
    editorial: float = 0.4
    cost: float = 0.2
    latency: float = 0.1
    reliability: float = 0.1
    cache_affinity: float = 0.05
    quality_alignment: float = 0.08
    continuity: float = 0.06
    # Weight for the explicit-feedback preference channel (preference_ewma in
    # [-1, 1]). Unlike reward_mean it only moves on user feedback, so negative
    # feedback sticks instead of being diluted by successful observations.
    feedback_preference: float = 0.15


@dataclass(frozen=True, slots=True)
class BanditConfig:
    enabled: bool = True
    reward_weight: float = 0.12
    exploration_weight: float = 0.18
    prior_n: float = 5.0
    warmup_pulls: int = 2
    min_samples_for_guardrail: int = 3
    min_reliability: float = 0.25
    max_cost_ratio: float = 3.0
    enabled_tiers: tuple[Tier, ...] = (Tier.SIMPLE, Tier.MEDIUM)


@dataclass(frozen=True, slots=True)
class ModeConfig:
    selection: SelectionWeights = field(default_factory=SelectionWeights)
    bandit: BanditConfig = field(default_factory=BanditConfig)


@dataclass
class StructuralWeights:
    """Weights for language-agnostic structural features."""

    normalized_length: float = 0.05
    enumeration_density: float = 0.10
    sentence_count: float = 0.08
    code_markers: float = 0.07
    math_symbols: float = 0.06
    nesting_depth: float = 0.03
    vocabulary_diversity: float = 0.03
    avg_word_length: float = 0.03
    alphabetic_ratio: float = 0.03
    functional_intent: float = 0.06
    unique_concept_density: float = 0.07
    requirement_phrases: float = 0.06


@dataclass
class TierBoundaries:
    simple_medium: float = -0.02
    medium_complex: float = 0.15


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    structural_weights: StructuralWeights = field(default_factory=StructuralWeights)
    tier_boundaries: TierBoundaries = field(default_factory=TierBoundaries)
    confidence_steepness: float = 18.0
    confidence_threshold: float = 0.55


@dataclass
class RoutingConfig:
    version: str = "5.0"
    modes: dict[RoutingMode, ModeConfig] = field(default_factory=dict)
