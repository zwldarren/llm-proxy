from llm_proxy.routing.types import (
    BanditConfig,
    CandidatePool,
    CandidateScorecard,
    ModeConfig,
    ModelExperience,
    RoutingConfig,
    RoutingDecision,
    RoutingMode,
    ScoringConfig,
    SelectionWeights,
    ServedQuality,
    StructuralWeights,
    Tier,
    TierBoundaries,
    TierVote,
)


def test_enums():
    assert Tier.SIMPLE.value == "SIMPLE"
    assert ServedQuality.PREMIUM.value == "premium"
    assert RoutingMode.AUTO.value == "auto"


def test_dataclasses_construct():
    vote = TierVote(tier_id=2, confidence=0.8)
    assert vote.tier_id == 2 and vote.confidence == 0.8
    pool = CandidatePool(
        available_models=["m1"],
        pricing={},
        served_qualities={},
        routing_assignments={},
        supports_images={},
    )
    assert pool.available_models == ["m1"]
    exp = ModelExperience(
        name="m1",
        samples=0,
        reward_mean=0.5,
        latency=0.0,
        reliability=1.0,
        cache_affinity=0.0,
    )
    assert exp.samples == 0
    dec = RoutingDecision(
        model="m1",
        tier=Tier.MEDIUM,
        complexity=0.5,
        confidence=0.7,
        reasoning={},
        cost_estimate=0.0,
        savings=0.0,
        candidate_scores={},
    )
    assert dec.model == "m1"


def test_config_dataclasses_defaults():
    assert SelectionWeights().editorial == 0.4
    assert BanditConfig().enabled is True
    assert ModeConfig().selection == SelectionWeights()
    assert ScoringConfig().confidence_steepness == 18.0
    assert RoutingConfig().version == "5.0"


def test_candidate_scorecard_as_dict_returns_serializable_fields():
    card = CandidateScorecard(
        model="econ/cheap",
        total=0.75,
        predicted_cost=0.0001,
        predicted_quality=0.6,
        cost=0.8,
        latency=0.7,
        reliability=0.9,
        cache_affinity=0.5,
        quality_alignment=0.6,
        continuity_bias=0.1,
        editorial=0.33,
        bandit_mean=0.55,
        exploration_bonus=0.02,
        samples=10,
        served_quality="economy",
    )
    d = card.as_dict()
    assert d["model"] == "econ/cheap"
    assert d["total"] == 0.75
    assert isinstance(d["samples"], int)


def test_routing_decision_has_extended_fields():
    dec = RoutingDecision(
        model="m1",
        tier=Tier.MEDIUM,
        candidate_scorecards=[{"model": "m1", "total": 0.9}],
        weights_used={"editorial": 0.4, "cost": 0.2},
        guardrail_notes=["budget-cap exceeded"],
        signal_votes={"structural": {"tier_id": 2}},
    )
    assert dec.candidate_scorecards == [{"model": "m1", "total": 0.9}]
    assert dec.weights_used == {"editorial": 0.4, "cost": 0.2}
    assert dec.guardrail_notes == ["budget-cap exceeded"]
    assert dec.signal_votes == {"structural": {"tier_id": 2}}


def test_structural_weights_and_boundaries_are_mutable():
    """Match source mutability: these two dataclasses are not frozen."""
    weights = StructuralWeights()
    weights.normalized_length = 0.99
    assert weights.normalized_length == 0.99

    boundaries = TierBoundaries()
    boundaries.simple_medium = 0.5
    assert boundaries.simple_medium == 0.5
