"""Tests for the judge gate policy and the consultation record it produces.

The gate is the part of the judge design that had to be measured rather than
assumed (ADR-0018): these tests pin the *rule*, while the evaluation rig
measures which configuration of it is worth shipping.
"""

import pytest

from llm_proxy.config.types.smart_routing import RoutingJudgeConfig
from llm_proxy.routing.judge.policy import (
    JudgeConsultation,
    JudgeConsultPlan,
    is_first_turn,
    plan_judge_consult,
)
from llm_proxy.routing.judge.rubric import RUBRIC_VERSION, JudgeVerdict
from llm_proxy.routing.types import RoutingMode, Tier

FIRST_TURN = [{"role": "user", "content": "Summarise this PR."}]
VERDICT = JudgeVerdict(
    tier=Tier.MEDIUM,
    probabilities={"simple": 0.2, "medium": 0.7, "complex": 0.05, "ambiguous": 0.05},
    escalation_probability=0.1234567,
    upstream_confidence=0.8123456,
    answers={"tier": {"choice": "medium"}, "escalate": {"noul": 0.12}},
)


def _judge(**overrides) -> RoutingJudgeConfig:
    kwargs = {"enabled": True, "model": "judge"}
    kwargs.update(overrides)
    return RoutingJudgeConfig(**kwargs)


def _plan(judge: RoutingJudgeConfig, **overrides) -> JudgeConsultPlan:
    kwargs = {
        "mode": RoutingMode.AUTO,
        "messages": FIRST_TURN,
        "confidence": 0.9,
        "complexity": 0.5,
        "sample": 0.999,
    }
    kwargs.update(overrides)
    return plan_judge_consult(judge, **kwargs)


def _consultation(**overrides) -> JudgeConsultation:
    kwargs = {
        "plan": JudgeConsultPlan(consult=True, reason="gate:open"),
        "model": "judge",
        "verdict": VERDICT,
        "latency_ms": 245.64,
        "provider_name": "local-ollama",
        "provider_model_name": "judge:0.6b",
        "started_at": 1000.0,
        "finished_at": 1000.2456,
    }
    kwargs.update(overrides)
    return JudgeConsultation(**kwargs)


# ─── Eligibility: first turn only ───


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([{"role": "user", "content": "hi"}], True),
        ([{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}], True),
        ([], False),
        ([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], False),
        ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], False),
        ([{"role": "user", "content": "a"}, {"role": "tool", "content": "b"}], False),
        (
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            ],
            False,
        ),
        (["not-a-dict"], False),
    ],
)
def test_first_turn_is_one_user_turn_and_no_model_output(messages, expected):
    """The judge sees user turns only, so only turn one is judgeable at all."""
    assert is_first_turn(messages) is expected


# ─── The gate ───


def test_a_disabled_judge_is_never_consulted():
    plan = _plan(RoutingJudgeConfig(enabled=False, model="judge"))
    assert plan.consult is False
    assert plan.reason == "judge-disabled"


def test_an_enabled_judge_without_a_model_is_not_configured():
    plan = _plan(RoutingJudgeConfig(enabled=True, model="   "))
    assert plan.consult is False
    assert plan.reason == "judge-disabled"


def test_fast_is_not_judged_by_default():
    """Cheap model choice is fast's purpose; a judge call works against it."""
    plan = _plan(_judge(), mode=RoutingMode.FAST)
    assert plan.consult is False
    assert plan.reason == "mode-not-judged(fast)"


def test_a_follow_up_turn_is_not_judged():
    plan = _plan(
        _judge(),
        messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
    )
    assert plan.consult is False
    assert plan.reason == "not-first-turn"


def test_no_predicate_means_every_eligible_turn_is_judged():
    plan = _plan(_judge())
    assert plan.consult is True
    assert plan.reason == "gate:open"
    assert plan.shadow is True  # shadow is the default for an unconfigured rollout


def test_the_confidence_gate_fires_below_its_threshold():
    judge = _judge(confidence_below=0.55)
    assert _plan(judge, confidence=0.54).reason == "gate:confidence(0.54<0.55)"
    assert _plan(judge, confidence=0.54).consult is True
    below = _plan(judge, confidence=0.56)
    assert below.consult is False
    assert below.reason == "gate-not-fired"


def test_the_complexity_band_is_half_open_at_the_high_end():
    """Half-open [low, high), matching the public tier mapping and the rig."""
    judge = _judge(complexity_between=(0.33, 0.67))
    assert _plan(judge, complexity=0.33).consult is True
    assert _plan(judge, complexity=0.33).reason == "gate:complexity(0.33 in [0.33,0.67))"
    borderline = _plan(judge, complexity=0.67)
    assert borderline.consult is False
    assert borderline.reason == "gate-not-fired"
    outside = _plan(judge, complexity=0.0)
    assert outside.consult is False
    assert outside.reason == "gate-not-fired"


def test_a_background_call_is_never_judged_whatever_the_gate_says():
    """A polled request has no client latency budget; the ensemble decides it."""
    plan = _plan(_judge(), is_background=True)
    assert plan.consult is False
    assert plan.reason == "background-call"


def test_the_shadow_window_samples_turns_the_gate_did_not_fire():
    """Agreement measured only inside the gate cannot tell if the gate leaks."""
    judge = _judge(confidence_below=0.10, shadow=True, shadow_sample_rate=0.05)
    sampled = _plan(judge, confidence=0.9, sample=0.01)
    assert sampled.consult is True
    assert sampled.reason == "shadow-sample"
    assert sampled.shadow is True

    unsampled = _plan(judge, confidence=0.9, sample=0.5)
    assert unsampled.consult is False


def test_sampling_is_off_when_shadow_is_off():
    judge = _judge(confidence_below=0.10, shadow=False, shadow_sample_rate=0.5)
    assert _plan(judge, confidence=0.9, sample=0.0).consult is False


# ─── What a consultation is ───


def test_a_shadow_verdict_does_not_own_the_tier():
    consultation = _consultation(
        plan=JudgeConsultPlan(consult=True, reason="gate:open", shadow=True)
    )
    assert consultation.owns_tier is False


@pytest.mark.parametrize("verdict", [None, JudgeVerdict()])
def test_an_abstention_does_not_own_the_tier(verdict):
    assert _consultation(verdict=verdict).owns_tier is False


def test_a_live_verdict_owns_the_tier():
    assert _consultation().owns_tier is True


def test_telemetry_carries_the_verdict_without_inventing_a_signal_vote():
    meta = _consultation().as_meta()
    assert meta["gate"] == "gate:open"
    assert meta["shadow"] is False
    assert meta["tier"] == "MEDIUM"
    # The verdict is pinned to the rubric that produced it, so logged verdicts
    # stay comparable across rubric edits.
    assert meta["rubric_version"] == RUBRIC_VERSION
    assert meta["answered"] is True
    assert meta["confidence"] == 0.7  # the judge's own mass, telemetry only
    assert meta["abstain_probability"] == 0.05
    assert meta["escalation_probability"] == 0.1235
    assert meta["upstream_confidence"] == 0.8123
    assert meta["latency_ms"] == 245.6
    assert meta["provider"] == "local-ollama"
    # The call's own row needs the concrete model and the wall-clock bounds: the
    # configured name lives on ``model``, the deployment's on ``provider_model_name``.
    assert meta["provider_model_name"] == "judge:0.6b"
    assert meta["started_at"] == 1000.0
    assert meta["finished_at"] == 1000.2456
    assert meta["error"] is None


def test_telemetry_records_the_failure_when_there_was_no_answer():
    meta = _consultation(verdict=None, error="TimeoutError: deadline expired").as_meta()
    assert meta["tier"] is None
    assert meta["answered"] is False
    assert meta["confidence"] is None
    assert meta["error"] == "TimeoutError: deadline expired"
