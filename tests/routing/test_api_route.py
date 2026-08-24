"""Tests for the route() orchestration entry point (ported from UncommonRoute)."""

import orjson

from llm_proxy.routing.api import route
from llm_proxy.routing.config import DEFAULT_CONFIG
from llm_proxy.routing.model_experience import ModelExperienceStore
from llm_proxy.routing.types import (
    CandidatePool,
    ModelPricing,
    RoutingMode,
    ServedQuality,
    Tier,
)


def _pool() -> CandidatePool:
    models = ["econ", "bal", "prem"]
    return CandidatePool(
        available_models=models,
        pricing={
            "econ": ModelPricing(0.15, 0.6),
            "bal": ModelPricing(1.0, 4.0),
            "prem": ModelPricing(5.0, 25.0),
        },
        served_qualities={
            "econ": ServedQuality.ECONOMY,
            "bal": ServedQuality.BALANCED,
            "prem": ServedQuality.PREMIUM,
        },
        routing_assignments={},
        supports_images={m: True for m in models},
    )


def test_route_returns_decision_with_model():
    dec = route(
        messages=[{"role": "user", "content": "hi"}],
        features=None,
        pool=_pool(),
        mode=RoutingMode.AUTO,
        config=DEFAULT_CONFIG,
        experience_store=ModelExperienceStore(session=None),
        embedding_signal=None,  # A+B only
    )
    assert dec.model in {"econ", "bal", "prem"}
    assert 0.0 <= dec.complexity <= 1.0


def _simple_decision():
    """Route a simple non-agent prompt and return the decision."""
    return route(
        messages=[{"role": "user", "content": "Write a sonnet about the sea."}],
        features=None,
        pool=_pool(),
        mode=RoutingMode.AUTO,
        config=DEFAULT_CONFIG,
        experience_store=ModelExperienceStore(session=None),
        embedding_signal=None,
    )


def _agent_decision():
    """Route a tool-loop (agent) prompt and return the decision."""
    msgs = [{"role": "user", "content": "fix the failing test"}]
    msgs += [{"role": "assistant", "tool_calls": [{"id": "t1", "function": {"name": "bash"}}]}]
    msgs += [{"role": "tool", "tool_call_id": "t1", "content": "Error: assertion failed"}] * 6
    return route(
        messages=msgs,
        features=None,
        pool=_pool(),
        mode=RoutingMode.AUTO,
        config=DEFAULT_CONFIG,
        experience_store=ModelExperienceStore(session=None),
        embedding_signal=None,
    )


# Agent-floor substrings that must NOT appear in non-agent reasoning.
_AGENT_FLOOR_MARKERS = ("agent-pressure", "hints=agentic", "step-risk")


def test_route_without_tools_skips_agent_floor_caps():
    # Non-agent request: agent pressure/followup floors must be no-ops.
    dec = _simple_decision()
    assert dec.model in {"econ", "bal", "prem"}

    # Simple prompt should resolve at low complexity (observed: 0.0).
    assert dec.complexity <= 0.40, f"non-agent complexity too high: {dec.complexity}"

    # Tier must be SIMPLE for a trivial prompt (not inflated by agent floors).
    assert dec.tier == Tier.SIMPLE, f"non-agent tier should be SIMPLE, got {dec.tier}"

    # Reasoning must contain no agent-floor markers — if it did, the gate leaked.
    reasoning_text = orjson.dumps(dec.reasoning).decode()
    for marker in _AGENT_FLOOR_MARKERS:
        assert marker not in reasoning_text, f"non-agent reasoning contains agent marker '{marker}'"


def test_route_with_tool_loop_activates_agent_heuristics():
    # Tool loop present -> agent heuristics must fire.
    dec = _agent_decision()
    assert dec.model in {"econ", "bal", "prem"}

    # Agent path must NOT be forced down to SIMPLE tier (observed: COMPLEX, 0.9).
    assert dec.complexity >= 0.40, f"agent complexity suspiciously low: {dec.complexity}"
    assert dec.tier != Tier.SIMPLE, f"agent path tier should not be SIMPLE, got {dec.tier}"

    # Reasoning must reflect agent awareness (at least one agent marker present).
    reasoning_text = orjson.dumps(dec.reasoning).decode()
    agent_markers_found = [m for m in _AGENT_FLOOR_MARKERS if m in reasoning_text]
    assert agent_markers_found, (
        f"agent reasoning contains no agent markers; reasoning: {reasoning_text[:500]}"
    )


def test_route_includes_signal_vote_summary():
    dec = _simple_decision()
    assert "metadata" in dec.signal_votes
    assert "structural" in dec.signal_votes
    assert "embedding" in dec.signal_votes
    assert "tier_id" in dec.signal_votes["metadata"]
    assert "confidence" in dec.signal_votes["metadata"]


def test_agent_path_routes_higher_than_simple():
    # The key regression test: if the agent gate breaks, agent complexity
    # collapses to the same low value as the simple path and this fails.
    simple_dec = _simple_decision()
    agent_dec = _agent_decision()

    assert agent_dec.complexity > simple_dec.complexity, (
        f"agent complexity ({agent_dec.complexity}) must exceed simple ({simple_dec.complexity})"
    )
    # Tier ordering: SIMPLE < MEDIUM < COMPLEX (string comparison works for StrEnum
    # since 'COMPLEX' > 'SIMPLE' lexicographically, but this is fragile — use ordinal).
    _tier_order = {Tier.SIMPLE: 0, Tier.MEDIUM: 1, Tier.COMPLEX: 2}
    assert _tier_order[agent_dec.tier] > _tier_order[simple_dec.tier], (
        f"agent tier ({agent_dec.tier}) must exceed simple tier ({simple_dec.tier})"
    )
