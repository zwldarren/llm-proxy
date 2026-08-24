"""Tests for the gap-fill selector dependencies ported from UncommonRoute."""

from llm_proxy.routing.model_experience import CandidateExperience
from llm_proxy.routing.types import RoutingConstraints


def test_routing_constraints_tags_max_cost():
    c = RoutingConstraints(max_cost=0.5)
    assert c.tags() == ("budget-cap",)


def test_candidate_experience_defaults():
    ce = CandidateExperience()
    assert ce.reliability == 0.5
    assert ce.latency == 0.5
    assert ce.cache_affinity == 0.5
    assert ce.input_cost_multiplier == 1.0
    assert ce.reward_mean == 0.5
    assert ce.samples == 0
