"""Tests for the ported UncommonRoute router/selector model selector.

Covers the public ``select_from_pool`` (pool-based v2 path), plus infeasibility errors.
"""

import random

import pytest

from llm_proxy.routing.model_experience import ModelExperienceStore
from llm_proxy.routing.selector import select_from_pool
from llm_proxy.routing.types import (
    BanditConfig,
    ModelExperience,
    ModelPricing,
    RoutingConstraints,
    RoutingInfeasibleError,
    RoutingMode,
    ServedQuality,
    Tier,
)


def _biased_experience_store() -> ModelExperienceStore:
    """Return a store with enough samples to make quality ranks visible."""
    store = ModelExperienceStore(session=None)
    store._cache = {
        "econ/cheap": ModelExperience(
            name="econ/cheap", samples=50, reward_mean=0.45, reliability=0.7
        ),
        "bal/mid": ModelExperience(name="bal/mid", samples=50, reward_mean=0.75, reliability=0.9),
        "prem/pro": ModelExperience(name="prem/pro", samples=50, reward_mean=0.95, reliability=1.0),
    }
    return store


PRICING = {
    "econ/cheap": ModelPricing(0.1, 0.2),
    "bal/mid": ModelPricing(1.0, 2.0),
    "prem/pro": ModelPricing(5.0, 15.0),
}
MODELS = list(PRICING)
SERVED = {
    "econ/cheap": ServedQuality.ECONOMY,
    "bal/mid": ServedQuality.BALANCED,
    "prem/pro": ServedQuality.PREMIUM,
}

# Our Tier enum uses SIMPLE/MEDIUM/COMPLEX (lowercase StrEnum values), not the
# ADVANCED/EXPERT/STANDARD/BASIC names in the task brief. Assertions match the
# actual ported behavior.
VALID_TIERS = (Tier.SIMPLE, Tier.MEDIUM, Tier.COMPLEX)


def test_select_best_complex_returns_valid_decision():
    store = ModelExperienceStore(session=None)
    dec = select_from_pool(
        complexity=0.85,
        mode=RoutingMode.BEST,
        confidence=0.9,
        reasoning_text="complex reasoning task",
        available_models=MODELS,
        estimated_input_tokens=500,
        max_output_tokens=2000,
        prompt="Explain quantum entanglement and derive Bell's inequality step by step.",
        pricing=PRICING,
        model_experience=store,
        served_qualities=SERVED,
    )
    assert dec.model in MODELS
    assert dec.cost_estimate >= 0
    assert dec.tier in VALID_TIERS


def test_select_fast_low_complexity_returns_valid_decision():
    store = ModelExperienceStore(session=None)
    dec = select_from_pool(
        complexity=0.15,
        mode=RoutingMode.FAST,
        confidence=0.95,
        reasoning_text="simple greeting",
        available_models=MODELS,
        estimated_input_tokens=100,
        max_output_tokens=500,
        prompt="hi there",
        pricing=PRICING,
        model_experience=store,
        served_qualities=SERVED,
    )
    assert dec.model in MODELS
    assert dec.cost_estimate >= 0
    assert dec.tier in VALID_TIERS


def test_select_from_pool_reasoning_is_structured_dict():
    store = ModelExperienceStore(session=None)
    dec = select_from_pool(
        complexity=0.85,
        mode=RoutingMode.BEST,
        confidence=0.9,
        reasoning_text="complex reasoning task",
        available_models=MODELS,
        estimated_input_tokens=500,
        max_output_tokens=2000,
        prompt="Explain quantum entanglement and derive Bell's inequality step by step.",
        pricing=PRICING,
        model_experience=store,
        served_qualities=SERVED,
    )
    # Our RoutingDecision.reasoning is a dict (adapted from source's str).
    assert isinstance(dec.reasoning, dict)
    assert "text" in dec.reasoning
    assert isinstance(dec.reasoning["text"], str)
    assert dec.reasoning["text"]


def test_select_from_pool_raises_when_no_models():
    store = ModelExperienceStore(session=None)
    with pytest.raises(RoutingInfeasibleError):
        select_from_pool(
            complexity=0.4,
            mode=RoutingMode.AUTO,
            confidence=0.6,
            reasoning_text="",
            available_models=[],
            estimated_input_tokens=100,
            max_output_tokens=500,
            prompt="hello",
            pricing=PRICING,
            model_experience=store,
            served_qualities={},
        )


def test_select_from_pool_respects_max_cost_constraint():
    store = ModelExperienceStore(session=None)
    # Use SIMPLE-tier complexity so econ/cheap passes the quality guard.
    dec = select_from_pool(
        complexity=0.15,
        mode=RoutingMode.AUTO,
        confidence=0.6,
        reasoning_text="",
        available_models=MODELS,
        estimated_input_tokens=100,
        max_output_tokens=500,
        prompt="hello",
        pricing=PRICING,
        constraints=RoutingConstraints(max_cost=0.0001),
        model_experience=store,
        served_qualities=SERVED,
    )
    assert dec.model == "econ/cheap"
    assert dec.cost_estimate <= 0.0001


def test_fast_and_best_diverge_with_seeded_rng():
    """FAST should stay cheap; BEST should pick the higher-quality premium model.

    With default equal priors and an empty experience store, both modes fall back
    to the first candidate (econ/cheap). We seed a small quality gradient so that
    the mode-driven cost-vs-quality trade-off is visible and deterministic.
    """
    fast_picks: list[str] = []
    best_picks: list[str] = []
    for seed in range(1, 6):
        rng = random.Random(seed)
        fast = select_from_pool(
            complexity=0.15,
            mode=RoutingMode.FAST,
            confidence=0.95,
            reasoning_text="simple greeting",
            available_models=MODELS,
            estimated_input_tokens=100,
            max_output_tokens=500,
            prompt="hi there",
            pricing=PRICING,
            model_experience=_biased_experience_store(),
            rng=rng,
            served_qualities=SERVED,
        )
        rng = random.Random(seed)
        best = select_from_pool(
            complexity=0.85,
            mode=RoutingMode.BEST,
            confidence=0.9,
            reasoning_text="complex reasoning task",
            available_models=MODELS,
            estimated_input_tokens=500,
            max_output_tokens=2000,
            prompt="Explain quantum entanglement and derive Bell's inequality step by step.",
            pricing=PRICING,
            model_experience=_biased_experience_store(),
            rng=rng,
            served_qualities=SERVED,
        )
        assert fast.cost_estimate <= best.cost_estimate
        fast_picks.append(fast.model)
        best_picks.append(best.model)
    assert any(f != b for f, b in zip(fast_picks, best_picks, strict=True))


def test_select_from_pool_populates_scorecards_and_weights():
    store = ModelExperienceStore(session=None)
    # Use flat served qualities so the quality guard does not filter any
    # candidates — this test focuses on scorecard/weight population.
    flat_served = {m: ServedQuality.BALANCED for m in MODELS}
    test_signal_votes = {
        "metadata": {"tier_id": 1, "confidence": 0.8},
        "structural": {"tier_id": 1, "confidence": 0.7},
        "embedding": {"tier_id": 0, "confidence": 0.6},
    }
    dec = select_from_pool(
        complexity=0.5,
        mode=RoutingMode.AUTO,
        confidence=0.8,
        reasoning_text="debug routing",
        available_models=MODELS,
        estimated_input_tokens=200,
        max_output_tokens=1000,
        prompt="Summarize this article",
        pricing=PRICING,
        model_experience=store,
        served_qualities=flat_served,
        signal_votes=test_signal_votes,
    )
    assert len(dec.candidate_scorecards) == len(MODELS)
    assert all("model" in card and "total" in card for card in dec.candidate_scorecards)
    # Winner is first in the final ranking.
    assert dec.candidate_scorecards[0]["model"] == dec.model
    assert set(dec.weights_used.keys()) == {
        "editorial",
        "cost",
        "latency",
        "reliability",
        "cache_affinity",
        "quality_alignment",
        "continuity",
        "feedback_preference",
    }
    assert dec.guardrail_notes
    # signal_votes must reflect what was passed in
    assert set(dec.signal_votes.keys()) == {"metadata", "structural", "embedding"}
    assert dec.signal_votes["metadata"]["tier_id"] == 1
    assert dec.signal_votes["metadata"]["confidence"] == 0.8
    assert dec.signal_votes["structural"]["tier_id"] == 1
    assert dec.signal_votes["embedding"]["tier_id"] == 0


def _two_identical_models_store(penalized_pref: float = 0.0, other_pref: float = 0.0):
    """Two candidates identical in every way except feedback preference."""
    store = ModelExperienceStore(session=None)
    store._cache = {
        "m/a": ModelExperience(
            name="m/a", samples=50, reward_mean=0.8, preference_ewma=penalized_pref
        ),
        "m/b": ModelExperience(name="m/b", samples=50, reward_mean=0.8, preference_ewma=other_pref),
    }
    pricing = {"m/a": ModelPricing(1.0, 2.0), "m/b": ModelPricing(1.0, 2.0)}
    return store, pricing


def _decide_with(store, pricing):
    return select_from_pool(
        complexity=0.5,
        mode=RoutingMode.AUTO,
        confidence=0.9,
        reasoning_text="feedback preference test",
        available_models=["m/a", "m/b"],
        estimated_input_tokens=100,
        max_output_tokens=100,
        prompt="write a small helper function",
        pricing=pricing,
        model_experience=store,
        bandit_config=BanditConfig(enabled=False),
        served_qualities={"m/a": ServedQuality.BALANCED, "m/b": ServedQuality.BALANCED},
    )


def test_negative_feedback_preference_demotes_model():
    # Two weak feedbacks on m/a (-0.44) must flip the otherwise-tied ranking.
    store, pricing = _two_identical_models_store(penalized_pref=-0.44)
    dec = _decide_with(store, pricing)
    assert dec.model == "m/b"


def test_positive_feedback_preference_promotes_model():
    store, pricing = _two_identical_models_store(other_pref=0.28)
    dec = _decide_with(store, pricing)
    assert dec.model == "m/b"
    parts = dec.reasoning.get("parts", ())
    assert any("feedback-preference=+0.28" in p for p in parts)


def test_neutral_feedback_preference_adds_no_reasoning_note():
    store, pricing = _two_identical_models_store()
    dec = _decide_with(store, pricing)
    parts = dec.reasoning.get("parts", ())
    assert not any("feedback-preference" in p for p in parts)


def test_feedback_preference_visible_in_scorecards():
    store, pricing = _two_identical_models_store(penalized_pref=-0.22)
    dec = _decide_with(store, pricing)
    cards = {c["model"]: c for c in dec.candidate_scorecards}
    assert cards["m/a"]["feedback_preference"] == pytest.approx(-0.22)
    assert cards["m/b"]["feedback_preference"] == pytest.approx(0.0)
