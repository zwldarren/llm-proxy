"""When the routing judge is consulted, and what a consultation leaves behind.

Pure policy, no I/O (ADR-0018). It lives here rather than in the request path so
the evaluation rig can measure the *same* rule the proxy ships — a gate that is
only approximated offline is a gate nobody has evidence for.

Three questions, in order:

1. **May the judge speak at all?** Enabled, configured with a model, and the
   request's virtual model is listed in ``modes``.
2. **Is this turn one the judge can actually judge?** Only the first turn of a
   conversation qualifies, and never a background call: the client is not waiting
   on the answer, so there is no client latency budget to spend (ADR-0018 names
   tool-loop steps, subagents and background calls as non-judged callers). The
   judge is shown user turns and capability flags
   but never assistant or tool content, so from turn two onward it cannot see the
   agent state that decides difficulty — the ensemble can, and it keeps the call.
3. **Does the gate fire?** Predicates are measured, not assumed: the rig reports
   candidate triggers side by side (``GateSpec`` in
   :mod:`llm_proxy.routing.eval.harness`). Every recorded corpus prefers that the
   gate stay *closed* once the ensemble's over-tier distortion is fixed
   (ADR-0029): no predicate set leaves the judge in shadow-only sampling, and
   ``always=True`` is the operator's deliberate open-gate override.
   ``not_low_anchor`` is configurable — the measured default of ADR-0028, kept
   for deployments whose rig run shows a judge-only band.

Shadow mode is expressed here too, because "consult but do not obey" is a
property of the consultation, not of the caller: a shadow consultation records a
verdict and its cost, and :attr:`JudgeConsultation.owns_tier` stays false.
"""

from dataclasses import dataclass
from typing import Any

from llm_proxy.config.types.smart_routing import RoutingJudgeConfig
from llm_proxy.routing.judge.rubric import RUBRIC_VERSION, JudgeVerdict
from llm_proxy.routing.types import RoutingMode

#: Roles whose presence means the conversation is no longer on its first turn.
_FOLLOWUP_ROLES = frozenset({"assistant", "tool"})


def complexity_band_fires(complexity: float, band: tuple[float, float]) -> bool:
    """Whether ``complexity`` lands inside the half-open band ``[low, high)``.

    Half-open to match the public tier mapping the band is read from
    (:func:`llm_proxy.routing.selector._derive_tier` maps ``[0.33, 0.67)`` to
    MEDIUM) and ADR-0018, which writes the band ``[0.33, 0.67)``. Configuration
    rejects a degenerate band, because a band this predicate can never fire was
    a trigger nobody measured.
    """
    low, high = band
    return low <= complexity < high


def not_low_anchor_fires(complexity: float) -> bool:
    """Whether the ensemble placed the turn anywhere but the trivial low anchor.

    The ensemble's complexity snaps to the tier anchors ``{0.00, 0.40, 0.68,
    0.90}``, so "not the low anchor" is the same predicate the rig reports as
    ``not_low_anchor``: every turn the ensemble did not already call trivial.
    It consults 80–96% of eligible turns — the ship-measured default of
    ADR-0028 that was amended off by default once the recorded corpora showed
    the judge below the fixed ensemble on every high-call gate (ADR-0029).
    """
    return complexity > 0.0


def verdict_consumable(
    verdict: JudgeVerdict,
    *,
    min_confidence: float | None,
    min_escalation: float | None,
) -> bool:
    """Whether a non-abstaining verdict may own the tier (the consumption bar).

    A consulted verdict is not automatically right: on the recorded corpora the
    judge's confident verdicts are far more accurate than its hedged ones, and
    its escalation probability carries independent signal (ADR-0029). The bar is
    the judge's *own* meta-confidence, never the ensemble's: the tier
    probability mass on the option it chose (``verdict.confidence``) or the
    probability it puts on "a frontier model would measurably beat a cheap one"
    (``verdict.escalation_probability``). Configuring either threshold applies
    the verdict when **either** bar is met; both unset apply every
    non-abstaining verdict — the behavior the bars amend.

    The thresholds live on the consultation (carried from configuration at
    consult time), so the same pure rule decides ownership on the request path
    and in the rig — a consumption policy approximated only offline is a policy
    nobody has evidence for, exactly as for the gate.
    """
    if verdict.tier is None:
        return False
    if min_confidence is None and min_escalation is None:
        return True
    if min_confidence is not None and verdict.confidence >= min_confidence:
        return True
    return (
        min_escalation is not None
        and verdict.escalation_probability is not None
        and verdict.escalation_probability >= min_escalation
    )


def is_first_turn(messages: list[dict[str, Any]] | None) -> bool:
    """True when this request opens a conversation rather than continuing one.

    One user turn and no assistant or tool output. A system prompt does not
    disqualify the turn: it is part of the ask, not history.
    """
    rows = [m for m in (messages or []) if isinstance(m, dict)]
    if not rows:
        return False
    user_turns = 0
    for message in rows:
        role = message.get("role")
        if role == "user":
            user_turns += 1
            if user_turns > 1:
                return False
        elif role in _FOLLOWUP_ROLES or message.get("tool_calls"):
            return False
    return user_turns == 1


@dataclass(frozen=True, slots=True)
class JudgeConsultPlan:
    """Whether to consult the judge for one turn, and why.

    ``reason`` is written into the decision's telemetry, so it reads as a short
    machine-checkable label rather than prose.
    """

    consult: bool
    reason: str
    shadow: bool = False


def plan_judge_consult(
    judge: RoutingJudgeConfig,
    *,
    mode: RoutingMode,
    messages: list[dict[str, Any]] | None,
    confidence: float,
    complexity: float,
    sample: float,
    is_background: bool = False,
) -> JudgeConsultPlan:
    """Decide whether this turn gets a judge call.

    ``sample`` is a pre-drawn uniform value in ``[0, 1)``: drawing it here would
    make the policy untestable and the rig's shadow sampling unreproducible.

    ``is_background`` marks a request the client polls for instead of waiting on
    (the OpenResponses protocol's ``background`` mode). It is excluded whatever
    the gate says: a background call has no client latency budget for the judge
    to spend, so it is never a judged turn (ADR-0018).
    """
    if not judge.is_configured:
        return JudgeConsultPlan(consult=False, reason="judge-disabled")
    if mode.value not in judge.modes:
        return JudgeConsultPlan(consult=False, reason=f"mode-not-judged({mode.value})")
    if is_background:
        return JudgeConsultPlan(consult=False, reason="background-call")
    if not is_first_turn(messages):
        return JudgeConsultPlan(consult=False, reason="not-first-turn")

    if judge.confidence_below is not None and confidence < judge.confidence_below:
        return JudgeConsultPlan(
            consult=True,
            reason=f"gate:confidence({confidence:.2f}<{judge.confidence_below:g})",
            shadow=judge.shadow,
        )
    band = judge.complexity_between
    if band is not None and complexity_band_fires(complexity, band):
        return JudgeConsultPlan(
            consult=True,
            reason=f"gate:complexity({complexity:.2f} in [{band[0]:g},{band[1]:g}))",
            shadow=judge.shadow,
        )
    if judge.not_low_anchor and not_low_anchor_fires(complexity):
        return JudgeConsultPlan(
            consult=True,
            reason=f"gate:not-low-anchor({complexity:.2f})",
            shadow=judge.shadow,
        )
    if judge.gate_is_open:
        return JudgeConsultPlan(consult=True, reason="gate:open", shadow=judge.shadow)

    if judge.shadow and sample < judge.shadow_sample_rate:
        # The gate did not fire, but the shadow window needs turns *outside* it:
        # agreement measured only inside the gate cannot tell whether the gate is
        # letting ambiguous turns through.
        return JudgeConsultPlan(consult=True, reason="shadow-sample", shadow=True)

    return JudgeConsultPlan(consult=False, reason="gate-not-fired")


@dataclass(frozen=True, slots=True)
class JudgeConsultation:
    """One judge call: why it happened, what came back, what it cost.

    Deliberately not a signal vote. The judge does not contribute a fourth
    prediction to the ensemble; it either owns the tier for an eligible turn or
    abstains and leaves the decision untouched (ADR-0018).
    """

    plan: JudgeConsultPlan
    model: str = ""
    verdict: JudgeVerdict | None = None
    latency_ms: float = 0.0
    provider_name: str | None = None
    #: The concrete model the judge's provider reported serving the call, when it
    #: reported one. ``model`` is the configured name; this is the deployment's.
    provider_model_name: str | None = None
    #: Wall-clock bounds of the call, epoch seconds. The call's own log row is
    #: timestamped from these, so it sits in the log store when it happened —
    #: during routing, before the client request that triggered it finished.
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    #: Input tokens the judge call billed, when the upstream reported usage.
    input_tokens: int | None = None
    #: Output tokens the judge call billed. System One charges input only, so
    #: this is normally 0 — recorded for completeness, not for accounting.
    output_tokens: int | None = None
    #: What the call cost, in USD, and where that number came from. Judge spend
    #: is real even on the fail-open path (an HTTP 200 with an answer the proxy
    #: rejects has still been billed), which is why it rides the decision's
    #: telemetry rather than a classifier row of its own (ADR-0018).
    cost: float | None = None
    #: ``"provider"`` for an upstream-reported charge, ``"estimated"`` for one
    #: priced from the model's configured per-1M rates, ``None`` when neither
    #: was available. Never both: a reported charge is never re-estimated.
    cost_source: str | None = None
    #: Consumption bar carried from configuration at consult time (ADR-0029):
    #: the verdict's own tier probability mass and escalation probability must
    #: clear these for the verdict to own the tier. ``None`` bars are unset.
    apply_min_confidence: float | None = None
    apply_min_escalation: float | None = None

    @property
    def owns_tier(self) -> bool:
        """True when this verdict replaces the ensemble's tier for this turn.

        Three ways not to own it: the consultation is shadow (never obeyed),
        the verdict abstained, or the verdict cleared neither consumption bar
        (ADR-0029). A held verdict is still recorded in full — it is evidence
        for the next consumption-policy measurement, not waste.
        """
        if self.plan.shadow or self.verdict is None:
            return False
        return verdict_consumable(
            self.verdict,
            min_confidence=self.apply_min_confidence,
            min_escalation=self.apply_min_escalation,
        )

    def as_meta(self) -> dict[str, Any]:
        """Flatten the consultation into the decision's telemetry record.

        ``confidence`` here is the judge's own mass on the option it chose, kept
        for measurement only: it is a distribution-peakiness statistic, not
        P(correct), and it must never reach a :class:`TierVote`.

        ``rubric_version`` pins the verdict to the rubric that produced it: the
        rubric module bumps the constant on any incompatible edit precisely so
        logged verdicts stay comparable within one version.
        """
        verdict = self.verdict
        return {
            "gate": self.plan.reason,
            "shadow": self.plan.shadow,
            "owns_tier": self.owns_tier,
            "model": self.model,
            "rubric_version": RUBRIC_VERSION,
            "answered": self.verdict is not None and bool(self.verdict.answers),
            "tier": verdict.tier.value
            if verdict is not None and verdict.tier is not None
            else None,
            "probabilities": dict(verdict.probabilities) if verdict is not None else None,
            "confidence": round(verdict.confidence, 4) if verdict is not None else None,
            "abstain_probability": (
                round(verdict.abstain_probability, 4) if verdict is not None else None
            ),
            "escalation_probability": (
                round(verdict.escalation_probability, 4)
                if verdict is not None and verdict.escalation_probability is not None
                else None
            ),
            "upstream_confidence": (
                round(verdict.upstream_confidence, 4)
                if verdict is not None and verdict.upstream_confidence is not None
                else None
            ),
            "latency_ms": round(self.latency_ms, 1),
            "provider": self.provider_name,
            "provider_model_name": self.provider_model_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "cost_source": self.cost_source,
            "error": self.error,
        }


__all__ = [
    "JudgeConsultPlan",
    "JudgeConsultation",
    "complexity_band_fires",
    "not_low_anchor_fires",
    "is_first_turn",
    "plan_judge_consult",
    "verdict_consumable",
]
