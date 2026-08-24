from llm_proxy.routing.classifier import classify
from llm_proxy.routing.signals.structural import StructuralSignal
from llm_proxy.routing.types import Tier


def test_classify_simple_text():
    result = classify("Say hello in one word.")
    assert result.tier in {Tier.SIMPLE, Tier.MEDIUM}
    assert 0.0 <= result.complexity <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_classify_complex_code():
    result = classify(
        "Refactor this async Python module to add retry logic with exponential "
        "backoff, circuit breaking, and structured logging across all callers.",
        system_prompt="You are a senior engineer.",
    )
    assert result.tier in {Tier.MEDIUM, Tier.COMPLEX}
    assert 0.0 <= result.complexity <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_structural_signal_returns_tier_vote():
    vote = StructuralSignal().predict(
        {
            "messages": [
                {"role": "user", "content": "Refactor this async module with retry/backoff."}
            ],
        }
    )
    assert vote.tier_id in {0, 1, 2, 3}
    assert 0.0 <= vote.confidence <= 1.0
