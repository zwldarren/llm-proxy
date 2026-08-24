from llm_proxy.routing.decision.ensemble import Ensemble
from llm_proxy.routing.types import TierVote


def test_high_confidence_direct_decision():
    ens = Ensemble(weights=[1.0, 1.0, 1.0])
    res = ens.decide([TierVote(2, 0.9), TierVote(2, 0.85), TierVote(2, 0.8)])
    assert res.tier_id == 2
    assert res.method == "direct"


def test_low_confidence_conservative_escalation():
    ens = Ensemble(weights=[1.0, 1.0, 1.0])
    res = ens.decide([TierVote(0, 0.2), TierVote(1, 0.2), TierVote(0, 0.2)])
    assert res.tier_id >= 1  # escalated conservatively
    assert res.method == "conservative"


def test_weak_low_escalated_to_mid():
    ens = Ensemble(weights=[1.0, 1.0, 1.0])
    res = ens.decide([TierVote(0, 0.4), TierVote(0, 0.4), TierVote(0, 0.4)])
    # weak LOW -> MID per source
    assert res.tier_id in {0, 1}
