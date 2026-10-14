"""Public API — the route() orchestration entry point.

Ported from UncommonRoute ``router/api.py`` with the following adaptations:

* ``uncommon_route.*`` imports rewritten to ``llm_proxy.routing.*``.
* Duplicated inline feature helpers (``tool_result_is_error``,
  ``tool_result_failure_kind``, ``agent_state_pressure``,
  ``messages_contextual_followup_floor``, ``_user_context_text``) are removed
  in favor of the shared implementations in ``features.py``.
* Phase-2 online-learning hooks removed: no ``SignalWeightTracker``/
  ``v2_lifecycle`` weight learning (static ensemble weights), no Signal B
  shadow mode (``use_signal_b`` is always True). The ensemble's confidence
  is Platt-calibrated via the vendored temperature in
  ``assets/calibration_params.json`` (see ``decision/calibration.py``);
  ``calibration_*`` args to ``select_from_pool`` stay at their defaults.
* Circuit-breaker / dynamic discovery removed: candidates come from the
  ``CandidatePool`` passed by the caller. No ``ModelMapper``.
* Agent gating (design §7.2): the agent-specific floor/cap branches
  (pressure-rescue floor, contextual follow-up floor, final-verification
  floor, early-semantic-failure cap, routine-success cap) execute only when
  the request is part of a tool loop (``features.has_tools`` or
  ``features.agent_step_count > 0``). Non-agent chat skips those branches;
  structured-output floor, vision floor, substance floors, and
  ``_apply_tier_bounds``/``_derive_tier`` always run.
"""

import random
from dataclasses import dataclass, replace
from typing import Any

from llm_proxy.routing.config import (
    DEFAULT_CONFIG,
    get_bandit_config,
    get_selection_weights,
)
from llm_proxy.routing.decision.calibration import PlattCalibrator, load_calibration_temperature
from llm_proxy.routing.decision.ensemble import Ensemble
from llm_proxy.routing.features import (
    _latest_tool_result_message,
    _message_text,
    _tool_result_is_verification_failure,
    extract_routing_features,
    messages_contextual_followup_floor,
)
from llm_proxy.routing.judge.policy import JudgeConsultation
from llm_proxy.routing.selector import _derive_tier, select_from_pool
from llm_proxy.routing.signal_tuning import (
    DEFAULT_SIGNAL_TUNING,
    extract_closed_form_ask,
    is_closed_form_ask,
    system_prompt_has_structured_output_constraint,
    text_high_substance_score,
    text_substance_score,
    vision_prompt_needs_medium_floor,
)
from llm_proxy.routing.signals.metadata import MetadataSignal
from llm_proxy.routing.signals.structural import StructuralSignal
from llm_proxy.routing.structural import estimate_tokens
from llm_proxy.routing.types import (
    CandidatePool,
    RoutingConfig,
    RoutingConstraints,
    RoutingDecision,
    RoutingFeatures,
    RoutingMode,
    Tier,
    TierVote,
    pressure_rescue_active,
    pressure_rescue_premium_window,
)

# ─── Tier <-> complexity maps (inverse of _derive_tier boundaries) ───

_TIER_ID_TO_COMPLEXITY: frozendict[int, float] = frozendict({0: 0.0, 1: 0.40, 2: 0.68, 3: 0.90})
_TIER_ORDER: frozendict[Tier, int] = frozendict({Tier.SIMPLE: 0, Tier.MEDIUM: 1, Tier.COMPLEX: 2})
_PRESSURE_RESCUE_NEXT_TIER: frozendict[Tier, Tier] = frozendict(
    {
        Tier.SIMPLE: Tier.MEDIUM,
        Tier.MEDIUM: Tier.COMPLEX,
        Tier.COMPLEX: Tier.COMPLEX,
    }
)
_PUBLIC_TIER_COMPLEXITY: frozendict[Tier, float] = frozendict(
    {
        Tier.SIMPLE: 0.0,
        Tier.MEDIUM: 0.40,
        Tier.COMPLEX: 0.68,
    }
)

# The source's route() accepts max_output_tokens explicitly (default 4096).
# Our route() signature derives it from features or falls back to this default.
_DEFAULT_MAX_OUTPUT_TOKENS = 4096

# ─── Signal A/B singletons ───
# Signals A (metadata) and B (structural) are cheap, stateless, and eagerly
# initialized at module load time to avoid thread-safety concerns with
# check-then-create lazy init under concurrent requests.
_v2_sig_a: MetadataSignal = MetadataSignal()
_v2_sig_b: StructuralSignal = StructuralSignal()


# fmt: off
def _get_sig_a() -> MetadataSignal: return _v2_sig_a
def _get_sig_b() -> StructuralSignal: return _v2_sig_b
# fmt: on


# Calibrator singleton. Lazily created on first use: the vendored temperature
# asset deploys to the platform data dir on first read, which should not
# happen at module import time.
_calibrator: PlattCalibrator | None = None


def _get_calibrator() -> PlattCalibrator:
    global _calibrator
    if _calibrator is None:
        _calibrator = PlattCalibrator(temperature=load_calibration_temperature())
    return _calibrator


def _derive_prompt(messages: list[dict[str, Any]] | None) -> str:
    """Best-effort current-turn prompt text for token estimation / selection.

    The source's route() receives ``prompt`` from the proxy caller. This port
    only sees the message list, so we derive the latest user turn (falling back
    to the latest message) as the representative prompt.
    """
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return _message_text(message.get("content"))
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            return _message_text(last.get("content"))
    return ""


def _estimate_conversation_tokens(messages: list[dict[str, Any]] | None) -> int:
    """Estimate tokens for the **full** conversation, not just the last user message."""
    if not messages:
        return 0
    msg_token_overhead = 4
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        text = _message_text(content)
        total += estimate_tokens(text)
        total += msg_token_overhead
    return total


def _messages_contain_images(messages: list[dict[str, Any]] | None) -> bool:
    """Check if any message contains image content.

    Handles multiple protocol formats:
    - OpenAI /v1/chat/completions: "image_url"
    - Anthropic /v1/messages: "image"
    - OpenAI /v1/responses: "input_image"
    - Legacy/inline: "inline_data"
    """
    if not messages:
        return False
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in {
                    "image_url",
                    "image",
                    "input_image",
                    "inline_data",
                }:
                    return True
    return False


def _enrich_features_from_messages(
    features: RoutingFeatures,
    messages: list[dict[str, Any]] | None,
    *,
    max_output_tokens: int,
) -> RoutingFeatures:
    """Preserve caller-provided features while honoring explicit latest failures."""
    if not messages:
        return features

    inferred = extract_routing_features(messages, max_output_tokens=max_output_tokens)
    if not inferred.verification_failed:
        return features

    tier_floor = features.tier_floor
    if tier_floor is None or _TIER_ORDER[tier_floor] < _TIER_ORDER[Tier.MEDIUM]:
        tier_floor = Tier.MEDIUM

    return replace(
        features,
        has_tool_results=features.has_tool_results or inferred.has_tool_results,
        step_risk="high",
        is_agentic=features.is_agentic or inferred.is_agentic,
        is_coding=features.is_coding or inferred.is_coding,
        tier_floor=tier_floor,
        tier_cap=None,
        tier_cap_reason="",
        agent_step_count=max(features.agent_step_count, inferred.agent_step_count),
        agent_pressure=max(features.agent_pressure, inferred.agent_pressure),
        verification_failed=True,
        failure_kind=inferred.failure_kind or features.failure_kind,
    )


def _apply_tier_bounds(
    complexity: float,
    *,
    tier_floor: Tier | None,
    tier_cap: Tier | None,
) -> tuple[float, list[str]]:
    bounded = complexity
    notes: list[str] = []
    effective_tier = _derive_tier(bounded)

    if tier_floor is not None and _TIER_ORDER[effective_tier] < _TIER_ORDER[tier_floor]:
        bounded = _PUBLIC_TIER_COMPLEXITY[tier_floor]
        effective_tier = tier_floor
        notes.append(f"tier-floor={tier_floor.value}")

    if tier_cap is not None and _TIER_ORDER[effective_tier] > _TIER_ORDER[tier_cap]:
        bounded = _PUBLIC_TIER_COMPLEXITY[tier_cap]
        notes.append(f"tier-cap={tier_cap.value}")

    return bounded, notes


def _agent_pressure_supports_rescue(features: RoutingFeatures) -> bool:
    return pressure_rescue_active(
        agent_pressure=features.agent_pressure,
        agent_step_count=features.agent_step_count,
        has_tool_results=features.has_tool_results,
        is_agentic=features.is_agentic,
        is_coding=features.is_coding,
    )


def _pressure_rescue_tier_floor(
    v2: V2ClassifyResult,
    features: RoutingFeatures,
) -> tuple[Tier | None, str | None]:
    if not _agent_pressure_supports_rescue(features):
        return None, None

    predicted_tier = _derive_tier(v2.complexity)

    if features.verification_failed:
        return Tier.COMPLEX, "agent-pressure-floor=COMPLEX(verification-failed)"

    if not pressure_rescue_premium_window(
        agent_pressure=features.agent_pressure,
        agent_step_count=features.agent_step_count,
        has_tool_results=features.has_tool_results,
        is_agentic=features.is_agentic,
        is_coding=features.is_coding,
    ):
        return None, None

    rescue_floor = _PRESSURE_RESCUE_NEXT_TIER[predicted_tier]
    cap_reason = str(features.tier_cap_reason or "").strip().lower()
    if (
        cap_reason in {"routine-success", "short-observation", "recoverable-tool-error"}
        and predicted_tier is not Tier.COMPLEX
    ):
        return None, None
    return rescue_floor, f"agent-pressure-floor={rescue_floor.value}(from={predicted_tier.value})"


def _soften_tier_cap_for_agent_state(
    v2: V2ClassifyResult,
    features: RoutingFeatures,
    tier_cap: Tier | None,
    pressure_rescue_floor: Tier | None,
) -> tuple[Tier | None, str | None]:
    if tier_cap is None:
        return tier_cap, None

    predicted_tier = _derive_tier(v2.complexity)
    pressure_support = features.agent_pressure >= 0.55 and v2.tier_id >= 2
    high_risk_support = features.step_risk == "high" and v2.tier_id >= 2
    cap_reason = str(features.tier_cap_reason or "").strip().lower()
    rescue_exceeds_cap = (
        pressure_rescue_floor is not None
        and _TIER_ORDER[pressure_rescue_floor] > _TIER_ORDER[tier_cap]
    )

    if cap_reason == "environment-or-routine":
        if str(features.step_risk or "").strip().lower() == "low":
            return tier_cap, f"tier-cap-preserved({cap_reason}:low-risk-step)"
        routine_pressure_support = features.agent_pressure >= 0.65 and v2.tier_id >= 3
        if not routine_pressure_support:
            return tier_cap, f"tier-cap-preserved({cap_reason})"
    elif (
        cap_reason == "environment-recovery"
        or cap_reason == "invocation-recovery"
        or cap_reason in {"suggestion-mode", "title-generation"}
    ):
        return tier_cap, f"tier-cap-preserved({cap_reason})"
    elif cap_reason == "low-risk":
        standalone_complex_support = (
            not features.has_tool_results
            and not features.is_agentic
            and v2.tier_id >= 2
            and predicted_tier is Tier.COMPLEX
            and v2.confidence >= 0.25
        )
        if standalone_complex_support:
            return None, "tier-cap-softened(current-complex-evidence)"
        pressure_cap_support = rescue_exceeds_cap and (v2.tier_id >= 1 or v2.confidence >= 0.30)
        if pressure_cap_support:
            return None, f"tier-cap-softened(agent-pressure={features.agent_pressure:.2f})"
        return tier_cap, f"tier-cap-preserved({cap_reason})"
    elif cap_reason in {
        "routine-success",
        "short-observation",
        "recoverable-tool-error",
    }:
        pressure_cap_support = rescue_exceeds_cap and (v2.tier_id >= 1 or v2.confidence >= 0.30)
        if pressure_cap_support:
            return None, f"tier-cap-softened(agent-pressure={features.agent_pressure:.2f})"
        return tier_cap, f"tier-cap-preserved({cap_reason})"

    if _TIER_ORDER[predicted_tier] <= _TIER_ORDER[tier_cap]:
        return tier_cap, None

    if pressure_support or high_risk_support:
        reasons: list[str] = []
        if pressure_support:
            reasons.append(f"agent-pressure={features.agent_pressure:.2f}")
        if high_risk_support:
            reasons.append("risk=high")
        return None, "tier-cap-softened(" + ",".join(reasons) + ")"

    return tier_cap, None


def _strongest_non_structural_tier(vote_a: TierVote) -> int | None:
    """Return the strongest sufficiently confident non-structural support."""
    supported: list[int] = []
    if not vote_a.abstained and vote_a.tier_id is not None and vote_a.confidence >= 0.65:
        supported.append(vote_a.tier_id)
    return max(supported) if supported else None


def _latest_user_has_high_substance(row: dict[str, Any]) -> bool:
    ask = _row_closed_form_ask_text(row)
    if ask is not None:
        return False
    return (
        text_high_substance_score(_row_latest_user_text(row))
        >= DEFAULT_SIGNAL_TUNING.high_substance_score
    )


def _row_latest_user_text(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages", [])):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            return _message_text(message.get("content"))
    return ""


def _row_closed_form_ask_text(row: dict[str, Any]) -> str | None:
    """The bounded selection ask of the latest user turn, when one exists.

    Two shapes qualify: a long message whose terminal block is a closed-form
    ask (yes/no directive, lettered options) — the ask is extracted and the
    material behind it is reference volume — and a latest user turn that is
    itself a bounded selection question (the corpus convention of a document
    turn followed by a question turn). When one exists, the difficulty-
    relevant text is the ask, and the substance/structural floors stand down:
    they exist for implementation/design asks, not for selection over
    supplied material.
    """
    text = _row_latest_user_text(row)
    ask = extract_closed_form_ask(text)
    if ask is not None:
        return ask
    # The "turn is itself a bounded selection question" shape only holds for a
    # short question turn. Running the marker test over a long message would
    # stand the substance/structural floors down on any pasted volume that
    # merely mentions a marker, so long text qualifies only through the
    # guarded ``extract_closed_form_ask`` above.
    if len(text) <= DEFAULT_SIGNAL_TUNING.closed_form_ask_min_chars and is_closed_form_ask(text):
        return text
    return None


def _row_has_structured_system_prompt(row: dict[str, Any]) -> bool:
    for message in row.get("messages", []):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "system":
            continue
        if system_prompt_has_structured_output_constraint(_message_text(message.get("content"))):
            return True
    return False


def _allow_short_structural_medium_floor(row: dict[str, Any]) -> bool:
    """Use Signal B's soft medium floor only for short asks with real substance.

    Short prompts like "write one subject line" and "give me a shell one-liner"
    often trip the structural classifier's medium band; substance checks keep
    the floor from inflating trivial asks. (The embedding signal this floor was
    once gated behind was removed with Signal C.) A closed-form ask — yes/no
    directive, lettered options — is a selection task, not an implementation
    ask, so the floor never applies to it regardless of substance.
    """
    if _row_closed_form_ask_text(row) is not None:
        return False
    if _row_has_structured_system_prompt(row):
        return True
    text = _row_latest_user_text(row)
    if not text.strip():
        return True
    if text_substance_score(text) >= DEFAULT_SIGNAL_TUNING.short_medium_substance_score:
        return True
    return len(text.split()) > 15 and text_substance_score(text) > 0.0


def _latest_tool_text_and_command(
    messages: list[dict[str, Any]] | None,
) -> tuple[str, str]:
    _message, text, command = _latest_tool_result_message(messages)
    return text, command


def _messages_have_explicit_final_verification_failure(
    messages: list[dict[str, Any]] | None,
) -> bool:
    text, command = _latest_tool_text_and_command(messages)
    lowered = f"{command}\n{text}".lower()
    return bool("final verification" in lowered and _tool_result_is_verification_failure(text))


def _messages_need_vision_analysis_floor(messages: list[dict[str, Any]] | None) -> bool:
    if not messages:
        return False
    has_vision = False
    text_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"image_url", "input_image"}:
                    has_vision = True
                if item.get("type") in {"text", "input_text"}:
                    text_parts.append(str(item.get("text") or ""))
        elif isinstance(content, str):
            text_parts.append(content)
    if not has_vision:
        return False
    return vision_prompt_needs_medium_floor(
        has_vision=True,
        prompt="\n".join(text_parts),
    )


def _cap_uncorroborated_structural_high(
    row: dict[str, Any],
    vote_a: TierVote,
    vote_b: TierVote,
) -> tuple[TierVote, bool, int | None]:
    """Prevent short-prompt structure alone from forcing the highest v2 tier.

    Signal B is a useful safety floor, but on standalone agent prompts it can
    over-read task scaffolding ("plan, use tools, edit files") as highest-tier
    complexity. Require Signal A (plus text substance) to corroborate before
    allowing B to vote at tier 3 on short no-tool context.
    """
    if vote_b.abstained or vote_b.tier_id is None:
        return vote_b, False, None
    if vote_b.tier_id < 3 or vote_b.confidence < 0.95:
        return vote_b, False, None

    messages = row.get("messages", [])
    tool_msg_count = sum(1 for m in messages if m.get("role") == "tool" or m.get("tool_calls"))
    if tool_msg_count > 0 or len(messages) > 3:
        return vote_b, False, None

    support_tier = _strongest_non_structural_tier(vote_a)
    if support_tier is not None and support_tier >= 3:
        return vote_b, False, None
    # Cap target follows Signal A's vote, not the amount of text: "the ask
    # has substance" and "the ask is an open plan" are different claims, and
    # the second is precisely what an uncorroborated structural high vote
    # was not allowed to make. Signal A voting SIMPLE holds the cap at one
    # vote above the low anchor — MEDIUM — because the disagreement is real
    # (Signal B saw something); it is just not a COMPLEX vote.
    capped_tier = min(2, max(1, support_tier or 1))
    if vote_b.tier_id <= capped_tier:
        return vote_b, False, None
    return TierVote(capped_tier, vote_b.confidence), True, capped_tier


@dataclass(frozen=True)
class V2ClassifyResult:
    """Full result from v2 classification, including all signal votes."""

    complexity: float
    confidence: float
    tier_id: int
    method: str
    signals_text: tuple[str, ...]
    vote_a: TierVote
    vote_b: TierVote


@dataclass(frozen=True, slots=True)
class RoutingClassification:
    """One classification pass: the features extracted, and the ensemble on them.

    Callers that need the ensemble's verdict *before* routing — the judge gate
    does, since it decides whether to ask a second model — classify once and pass
    the result to :func:`route`, so the confidence the gate acts on is the
    confidence the decision was made from.
    """

    features: RoutingFeatures
    v2: V2ClassifyResult


def _build_signal_row(
    messages: list[dict[str, Any]] | None,
    routing_features: RoutingFeatures | None,
) -> dict[str, Any]:
    """Build a row dict for v2 signals from the message list.

    The source also accepts ``prompt``/``system_prompt``/``context_features``
    and synthesizes messages from them when ``messages`` is absent. This port
    always has ``messages``, so those inputs are dropped (YAGNI); when
    ``messages`` is empty the signals abstain or fall back to tier-0 metadata.
    """
    msgs: list[dict[str, Any]] = []
    if messages:
        for m in messages:
            normalized = dict(m)  # preserve tool_calls, tool_call_id, name, etc.
            content = normalized.get("content", "")
            if not isinstance(content, str):
                if isinstance(content, list):
                    parts = []
                    has_multimodal = False
                    for p in content:
                        if isinstance(p, dict):
                            p_type = p.get("type")
                            if p_type == "text" or p_type == "input_text":
                                parts.append(p.get("text", ""))
                            elif p_type in {
                                "input_image",
                                "image",
                                "image_url",
                                "input_file",
                                "input_audio",
                                "input_video",
                            }:
                                has_multimodal = True
                            elif p_type == "tool_result":
                                # Anthropic tool_result: {'type': 'tool_result', 'content': ...}
                                result_content = p.get("content", "")
                                if isinstance(result_content, str):
                                    parts.append(result_content)
                                elif isinstance(result_content, list):
                                    for item in result_content:
                                        if isinstance(item, dict) and item.get("type") == "text":
                                            parts.append(item.get("text", ""))
                        elif not isinstance(p, (dict, str)) and hasattr(p, "text"):
                            parts.append(getattr(p, "text", ""))
                        elif isinstance(p, str):
                            parts.append(p)
                    content = " ".join(parts)
                    # Preserve multimodal marker for signals to not abstain
                    if not content and has_multimodal:
                        content = "[multimodal_content]"
                else:
                    content = str(content)
                normalized["content"] = content
            msgs.append(normalized)

    msg_count = len(msgs)
    step_index = max(1, msg_count // 2)
    total_steps = max(step_index + 3, 10)

    scenario = "general"
    if routing_features:
        if routing_features.is_coding:
            scenario = "code_swe"
        elif routing_features.step_type == "tool-result-followup":
            scenario = "general_agent"

    return {
        "messages": msgs,
        "benchmark": "",
        "scenario": scenario,
        "step_index": step_index,
        "total_steps": total_steps,
        "routing_features": routing_features,
    }


def _v2_classify(
    messages: list[dict[str, Any]] | None,
    routing_features: RoutingFeatures | None,
    risk_tolerance: float = 0.5,
) -> V2ClassifyResult:
    """Run the v2 signal ensemble. Returns full result with all signal votes.

    Signal A (metadata) and Signal B (structural) always run; Signal C
    (embedding) and its heavy ML stack were removed, so the ensemble decides
    on A+B alone. Phase-2 online weight learning and shadow-mode gating are
    removed: Signal B is always active and the ensemble keeps the source's
    static weights (A=0.50, B=0.10).
    """
    row = _build_signal_row(messages, routing_features)
    closed_form_ask = _row_closed_form_ask_text(row)
    tool_msg_count = sum(
        1 for m in row.get("messages", []) if m.get("role") == "tool" or m.get("tool_calls")
    )
    has_system_prompt = any(m.get("role") == "system" for m in row.get("messages", []))

    vote_a = _get_sig_a().predict(row)
    vote_b = _get_sig_b().predict(row)

    effective_vote_b, structural_high_capped, structural_cap_tier = (
        _cap_uncorroborated_structural_high(
            row,
            vote_a,
            vote_b,
        )
    )

    # Signal B is always active (Phase-2 shadow mode removed).
    active_votes = [vote_a]
    active_weights = [0.50]
    if not effective_vote_b.abstained:
        active_votes.append(effective_vote_b)
        active_weights.append(0.10)

    ensemble = Ensemble(
        weights=active_weights, risk_tolerance=risk_tolerance, calibrator=_get_calibrator()
    )
    result = ensemble.decide(active_votes)

    tier_id = result.tier_id if result.tier_id is not None else 1

    structural_floor_applied = False
    if (
        tool_msg_count == 0
        and len(row.get("messages", [])) <= 3
        and not effective_vote_b.abstained
        and closed_form_ask is None
        and effective_vote_b.confidence >= (0.70 if has_system_prompt else 0.95)
        and (effective_vote_b.tier_id or 0) >= 1
        and tier_id < (effective_vote_b.tier_id or 0)
    ):
        tier_id = effective_vote_b.tier_id or 0
        structural_floor_applied = True
    structural_medium_floor_applied = False
    substance_structural_high_floor_applied = False
    substance_complexity_floor_applied = False
    if (
        not structural_floor_applied
        and tool_msg_count == 0
        and len(row.get("messages", [])) <= 3
        and not effective_vote_b.abstained
        and effective_vote_b.confidence >= 0.70
        and (effective_vote_b.tier_id or 0) >= 1
        and tier_id < 1
        and _allow_short_structural_medium_floor(row)
    ):
        # Short standalone implementation/design prompts are often capped to
        # 0.70 confidence by Signal B's short-text dampener. Do not let
        # metadata-only priors route those steps as economy, but avoid
        # escalating them all the way to premium unless Signal B is stronger.
        tier_id = 1
        structural_medium_floor_applied = True
    if (
        not structural_floor_applied
        and tool_msg_count == 0
        and len(row.get("messages", [])) <= 3
        and not effective_vote_b.abstained
        and not structural_high_capped
        and (vote_b.tier_id or 0) >= 3
        and vote_b.confidence >= 0.70
        and tier_id < 2
        and _latest_user_has_high_substance(row)
    ):
        # Keyed on Signal B's *uncapped* vote so the floor still lifts from a
        # genuinely corroborated high vote, and barred through a capped one:
        # once the cap has decided that B's high reading failed corroboration,
        # a held MEDIUM vote must not be re-promoted to COMPLEX by text
        # substance — the two signals disagreed and that disagreement was,
        # measured, mostly Signal B over-reading volume (ADR-0029).
        tier_id = 2
        substance_structural_high_floor_applied = True
    if (
        tier_id < 2
        and tool_msg_count == 0
        and len(row.get("messages", [])) <= 3
        and not effective_vote_b.abstained
        and not structural_high_capped
        and (effective_vote_b.tier_id or 0) >= 1
        and effective_vote_b.confidence >= 0.70
        and _latest_user_has_high_substance(row)
    ):
        # The uncapped condition: a capped vote is B's high reading that
        # failed corroboration, held at MEDIUM — text substance about the
        # same volume must not turn that MEDIUM into COMPLEX here.
        tier_id = 2
        substance_complexity_floor_applied = True
    complexity = _TIER_ID_TO_COMPLEXITY.get(tier_id, 0.40)

    signals_parts = [
        f"v2:metadata={vote_a.tier_id}({vote_a.confidence:.2f})",
        f"v2:structural={vote_b.tier_id}({vote_b.confidence:.2f})[active]",
        f"v2:tier={tier_id} complexity={complexity:.2f} method={result.method}",
    ]
    if structural_floor_applied:
        signals_parts.append("v2:structural-floor")
    if structural_medium_floor_applied:
        signals_parts.append("v2:structural-medium-floor")
    if substance_structural_high_floor_applied:
        signals_parts.append("v2:substance-structural-high-floor")
    if substance_complexity_floor_applied:
        signals_parts.append("v2:substance-complexity-floor")
    if structural_high_capped:
        signals_parts.append(f"v2:structural-high-cap={structural_cap_tier}")
    signals_text = tuple(signals_parts)

    return V2ClassifyResult(
        complexity=complexity,
        confidence=result.confidence,
        tier_id=tier_id,
        method=result.method,
        signals_text=signals_text,
        vote_a=vote_a,
        vote_b=vote_b,
    )


def classify(
    messages: list[dict[str, Any]] | None,
    features: RoutingFeatures | None = None,
    *,
    max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
    risk_tolerance: float = 0.5,
) -> RoutingClassification:
    """Extract routing features and run the signal ensemble, in one pass.

    ``route()`` performs exactly this work when it is not handed the result, so
    callers that need the ensemble's confidence or complexity *before* deciding
    anything — the judge gate is the only one today — can classify once and pass
    it along instead of classifying twice.
    """
    if features is None:
        features = extract_routing_features(messages, max_output_tokens=max_output_tokens)
    else:
        features = _enrich_features_from_messages(
            features,
            messages,
            max_output_tokens=max_output_tokens,
        )
    v2 = _v2_classify(
        messages,
        features,
        risk_tolerance=risk_tolerance,
    )
    return RoutingClassification(features=features, v2=v2)


def route(
    messages: list[dict[str, Any]],
    features: RoutingFeatures | None,
    pool: CandidatePool,
    mode: RoutingMode,
    config: RoutingConfig | None = None,
    experience_store: object | None = None,
    previous_model: str | None = None,
    *,
    rng: random.Random | None = None,
    mode_weights: dict[str, float] | None = None,
    require_images: bool = False,
    classification: RoutingClassification | None = None,
    judge: JudgeConsultation | None = None,
    degrade: bool = False,
) -> RoutingDecision:
    """Route a request to the best model using the v2 multi-signal ensemble.

    Orchestrates: feature extraction → Signal A/B → Ensemble → tier floor/cap
    heuristics → ``_apply_tier_bounds`` → ``select_from_pool``. Agent-specific
    floor/cap branches run only when a tool loop is detected (design §7.2).

    Parameters
    ----------
    previous_model: If set, the model used for the previous turn in this
        conversation. The router gives it a stickiness bonus to preserve
        provider-side prompt caching across turns.
    classification: A prior :func:`classify` result to reuse instead of
        classifying again, so a caller that already looked at the ensemble's
        confidence is deciding from the numbers it saw.
    degrade: Enable graceful degradation (the fallback ladder): two
        relaxations — mode assignment and the output half of the context
        window — apply instead of refusing the request. Recorded as
        ``degraded=...`` reasoning notes.
    judge: A judge consultation for this turn. When its verdict owns the tier,
        that tier replaces the ensemble's complexity before floors and caps are
        applied — so a judge can never lift a request above a cap or drop it
        below a floor, and it never contributes a fourth signal vote
        (ADR-0018). Consultations in shadow mode are recorded and ignored.
    """
    cfg = config or DEFAULT_CONFIG
    constraints = RoutingConstraints()
    classification = classification or classify(messages, features)
    features = classification.features
    mode = mode if isinstance(mode, RoutingMode) else RoutingMode(mode)
    effective_max_output_tokens = features.requested_max_output_tokens or _DEFAULT_MAX_OUTPUT_TOKENS

    prompt = _derive_prompt(messages)
    estimated_tokens = _estimate_conversation_tokens(messages)

    # ─── v2: multi-signal ensemble ───
    v2 = classification.v2

    signal_votes = {
        "metadata": {"tier_id": v2.vote_a.tier_id, "confidence": v2.vote_a.confidence},
        "structural": {"tier_id": v2.vote_b.tier_id, "confidence": v2.vote_b.confidence},
    }

    sel_weights = get_selection_weights(cfg, mode)
    bc = get_bandit_config(cfg, mode)
    candidates = pool.available_models
    effective_pricing = pool.pricing
    routing_assignments = getattr(pool, "routing_assignments", None)

    # Detect images in messages: if present, require image-capable models
    if not require_images and _messages_contain_images(messages):
        require_images = True

    # ─── Conversation continuity: previous model stickiness + quality ───
    if previous_model:
        features = replace(features, previous_model=previous_model)
        prev_quality = pool.served_qualities.get(previous_model)
        if prev_quality is not None and features.previous_served_quality is None:
            features = replace(features, previous_served_quality=prev_quality)

    effective_tier_floor = features.tier_floor
    effective_tier_cap = features.tier_cap
    feature_bound_notes: list[str] = []
    early_semantic_failure_cap_applied = False
    pressure_rescue_floor: Tier | None = None
    pressure_floor_note: str | None = None

    # Agent-only floor/cap branches (design §7.2). For non-agent chat these are
    # skipped entirely; the non-agent floors below still run.
    agent_active = features.has_tools or features.agent_step_count > 0

    if agent_active:
        contextual_followup_floor = messages_contextual_followup_floor(messages)
        if contextual_followup_floor is not None and (
            effective_tier_floor is None
            or _TIER_ORDER[effective_tier_floor] < _TIER_ORDER[contextual_followup_floor]
        ):
            effective_tier_floor = contextual_followup_floor
            feature_bound_notes.append(f"context-followup-floor={contextual_followup_floor.value}")

    # ─── Non-agent floors (always run) ───
    if _row_has_structured_system_prompt({"messages": messages or []}) and (
        effective_tier_floor is None or _TIER_ORDER[effective_tier_floor] < _TIER_ORDER[Tier.MEDIUM]
    ):
        effective_tier_floor = Tier.MEDIUM
        feature_bound_notes.append("structured-output-floor=MEDIUM")
    if _messages_need_vision_analysis_floor(messages) and (
        effective_tier_floor is None or _TIER_ORDER[effective_tier_floor] < _TIER_ORDER[Tier.MEDIUM]
    ):
        effective_tier_floor = Tier.MEDIUM
        feature_bound_notes.append("vision-floor=MEDIUM")

    if agent_active:
        if (
            features.verification_failed
            and _messages_have_explicit_final_verification_failure(messages)
            and (
                effective_tier_floor is None
                or _TIER_ORDER[effective_tier_floor] < _TIER_ORDER[Tier.COMPLEX]
            )
        ):
            effective_tier_floor = Tier.COMPLEX
            effective_tier_cap = None
            feature_bound_notes.append("final-verification-floor=COMPLEX")
        elif (
            features.verification_failed
            and features.agent_pressure < 0.55
            and effective_tier_cap is None
        ):
            effective_tier_cap = Tier.MEDIUM
            early_semantic_failure_cap_applied = True
            feature_bound_notes.append("early-semantic-failure-cap=MEDIUM")
        if (
            str(features.step_risk or "").strip().lower() == "low"
            and features.tier_cap_reason == "routine-success"
            and not features.verification_failed
            and features.agent_pressure < 0.35
        ):
            effective_tier_cap = Tier.SIMPLE
            feature_bound_notes.append("routine-success-cap=SIMPLE")
        pressure_rescue_floor, pressure_floor_note_candidate = _pressure_rescue_tier_floor(
            v2, features
        )
        if pressure_rescue_floor is not None and (
            effective_tier_floor is None
            or _TIER_ORDER[effective_tier_floor] < _TIER_ORDER[pressure_rescue_floor]
        ):
            effective_tier_floor = pressure_rescue_floor
            pressure_floor_note = pressure_floor_note_candidate

    # _soften_tier_cap_for_agent_state self-no-ops when the cap is None (the
    # non-agent case), so it is safe to run unconditionally.
    effective_tier_cap, cap_softened_note = _soften_tier_cap_for_agent_state(
        v2,
        features,
        effective_tier_cap,
        pressure_rescue_floor,
    )
    if early_semantic_failure_cap_applied and cap_softened_note:
        effective_tier_cap = Tier.MEDIUM
        cap_softened_note = "tier-cap-preserved(early-semantic-failure)"
    # ─── Judge: an eligible verdict owns the tier, bounds still apply after ───
    # The substitution happens here — below the floor/cap machinery, above
    # ``_apply_tier_bounds`` — which is what keeps the judge from overruling a
    # floor or a cap. The ensemble's own vote stays visible in ``signals_text``
    # and ``signal_votes``, so disagreement stays measurable instead of being
    # overwritten by the verdict.
    judge_meta: dict[str, Any] | None = None
    judge_note: str | None = None
    base_complexity = v2.complexity
    verdict = judge.verdict if judge is not None else None
    if judge is not None:
        judge_meta = judge.as_meta()
        if judge.owns_tier and verdict is not None and verdict.tier is not None:
            base_complexity = _PUBLIC_TIER_COMPLEXITY[verdict.tier]
            judge_note = f"judge={verdict.tier.value}({judge.plan.reason})"
        elif judge.plan.shadow:
            shadow_tier = verdict.tier.value if verdict is not None and verdict.tier else "abstain"
            judge_note = f"judge-shadow={shadow_tier}({judge.plan.reason})"
        elif verdict is not None and verdict.tier is not None:
            # A consulted verdict that cleared no consumption bar: recorded in
            # full, but the ensemble's tier stands (ADR-0029). Distinct from an
            # abstention so the log tells "no answer" from "an answer not
            # confident enough to obey".
            judge_note = f"judge-held={verdict.tier.value}({judge.plan.reason})"
        else:
            judge_note = f"judge=abstained({judge.error or 'ambiguous'})"

    bounded_complexity, bound_notes = _apply_tier_bounds(
        base_complexity,
        tier_floor=effective_tier_floor,
        tier_cap=effective_tier_cap,
    )

    # Phase-2 RouteConfidenceCalibrator removed: ensemble confidence is used
    # directly and the calibration_* args to select_from_pool stay default.
    confidence = v2.confidence
    reasoning_parts = list(v2.signals_text)
    if judge_note:
        reasoning_parts.append(judge_note)
    if pressure_floor_note:
        reasoning_parts.append(pressure_floor_note)
    if cap_softened_note:
        reasoning_parts.append(cap_softened_note)
    reasoning_parts.extend(feature_bound_notes)
    reasoning_parts.extend(bound_notes)
    reasoning = ", ".join(reasoning_parts)

    decision = select_from_pool(
        complexity=bounded_complexity,
        mode=mode,
        confidence=confidence,
        reasoning_text=reasoning,
        available_models=candidates,
        estimated_input_tokens=estimated_tokens,
        max_output_tokens=effective_max_output_tokens,
        prompt=prompt,
        pricing=effective_pricing,
        constraints=constraints,
        routing_features=features,
        selection_weights=sel_weights,
        bandit_config=bc,
        model_experience=experience_store,
        rng=rng,
        routing_assignments=routing_assignments,
        mode_weights=mode_weights,
        served_qualities=pool.served_qualities,
        supports_images=pool.supports_images,
        require_images=require_images,
        context_lengths=getattr(pool, "context_lengths", None),
        signal_votes=signal_votes,
        degrade=degrade,
    )

    if judge_meta is None:
        return decision
    return replace(decision, judge=judge_meta)
