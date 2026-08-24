"""Tests for context-length-based candidate filtering in select_from_pool().

Verifies that when models declare a ``context_length``, the routing engine
excludes candidates whose context window is too small for the estimated
input + output tokens, while preserving models with unknown (``None``)
context length.
"""

import pytest

from llm_proxy.routing.api import _estimate_conversation_tokens
from llm_proxy.routing.selector import select_from_pool
from llm_proxy.routing.types import (
    ModelPricing,
    RoutingInfeasibleError,
    RoutingMode,
    ServedQuality,
)


def _pricing(models: list[str]) -> dict[str, ModelPricing]:
    return {m: ModelPricing(input_price=1.0, output_price=4.0) for m in models}


def _served_qualities(models: list[str]) -> dict[str, ServedQuality]:
    return {m: ServedQuality.BALANCED for m in models}


def test_context_length_filters_out_small_models():
    """Models with known insufficient context_length are excluded."""
    models = ["small-200k", "large-1m"]
    context_lengths = {"small-200k": 200_000, "large-1m": 1_000_000}

    decision = select_from_pool(
        complexity=0.5,
        mode=RoutingMode.AUTO,
        confidence=0.8,
        reasoning_text="test",
        available_models=models,
        estimated_input_tokens=250_000,  # exceeds 200k
        max_output_tokens=4096,
        prompt="test prompt",
        pricing=_pricing(models),
        served_qualities=_served_qualities(models),
        context_lengths=context_lengths,
    )

    assert decision.model == "large-1m"
    assert "small-200k" not in decision.candidate_scores


def test_context_length_keeps_sufficient_models():
    """Models with enough context_length are all kept."""
    models = ["model-a", "model-b"]
    context_lengths = {"model-a": 200_000, "model-b": 200_000}

    decision = select_from_pool(
        complexity=0.5,
        mode=RoutingMode.AUTO,
        confidence=0.8,
        reasoning_text="test",
        available_models=models,
        estimated_input_tokens=50_000,  # well within 200k
        max_output_tokens=4096,
        prompt="test prompt",
        pricing=_pricing(models),
        served_qualities=_served_qualities(models),
        context_lengths=context_lengths,
    )

    # Both models should appear in candidate scores
    assert "model-a" in decision.candidate_scores
    assert "model-b" in decision.candidate_scores


def test_context_length_none_models_are_kept():
    """Models with context_length=None (unknown) are never filtered out."""
    models = ["unknown-ctx", "small-200k"]
    context_lengths = {"unknown-ctx": None, "small-200k": 200_000}

    decision = select_from_pool(
        complexity=0.5,
        mode=RoutingMode.AUTO,
        confidence=0.8,
        reasoning_text="test",
        available_models=models,
        estimated_input_tokens=250_000,  # exceeds small-200k
        max_output_tokens=4096,
        prompt="test prompt",
        pricing=_pricing(models),
        served_qualities=_served_qualities(models),
        context_lengths=context_lengths,
    )

    assert decision.model == "unknown-ctx"
    assert "small-200k" not in decision.candidate_scores


def test_context_length_all_filtered_raises_infeasible():
    """When all models have insufficient context, raise RoutingInfeasibleError."""
    models = ["small-a", "small-b"]
    context_lengths = {"small-a": 200_000, "small-b": 128_000}

    with pytest.raises(RoutingInfeasibleError) as exc_info:
        select_from_pool(
            complexity=0.5,
            mode=RoutingMode.AUTO,
            confidence=0.8,
            reasoning_text="test",
            available_models=models,
            estimated_input_tokens=500_000,  # exceeds both
            max_output_tokens=4096,
            prompt="test prompt",
            pricing=_pricing(models),
            served_qualities=_served_qualities(models),
            context_lengths=context_lengths,
        )

    msg = str(exc_info.value)
    assert "context length" in msg.lower()
    assert "200000" in msg  # largest configured context_length in message


def test_context_length_not_provided_keeps_all():
    """When context_lengths is None, no filtering is applied."""
    models = ["model-a", "model-b"]

    decision = select_from_pool(
        complexity=0.5,
        mode=RoutingMode.AUTO,
        confidence=0.8,
        reasoning_text="test",
        available_models=models,
        estimated_input_tokens=999_999,  # huge, but no filtering
        max_output_tokens=4096,
        prompt="test prompt",
        pricing=_pricing(models),
        served_qualities=_served_qualities(models),
        context_lengths=None,
    )

    assert "model-a" in decision.candidate_scores
    assert "model-b" in decision.candidate_scores


def test_context_length_boundary_exact_fit():
    """A model whose context_length exactly equals required capacity is kept."""
    models = ["exact-fit"]
    context_lengths = {"exact-fit": 204_096}

    decision = select_from_pool(
        complexity=0.5,
        mode=RoutingMode.AUTO,
        confidence=0.8,
        reasoning_text="test",
        available_models=models,
        estimated_input_tokens=200_000,
        max_output_tokens=4_096,  # 200000 + 4096 = 204096 == context_length
        prompt="test prompt",
        pricing=_pricing(models),
        served_qualities=_served_qualities(models),
        context_lengths=context_lengths,
    )

    assert decision.model == "exact-fit"


# ─── Full-conversation token estimation tests ───


def test_estimate_conversation_tokens_empty():
    """Empty or None messages should return 0 tokens."""
    assert _estimate_conversation_tokens(None) == 0
    assert _estimate_conversation_tokens([]) == 0


def test_estimate_conversation_tokens_single_message():
    """A single short message should produce a small, non-zero token estimate."""
    tokens = _estimate_conversation_tokens([{"role": "user", "content": "hello world"}])
    assert tokens > 0
    # "hello world" is ~2-3 tokens + 4 overhead ≈ 6-7
    assert tokens < 20


def test_estimate_conversation_tokens_multi_message_exceeds_single():
    """Full-conversation estimate must exceed single-message estimate.

    This is the core property that enables context-length filtering: the
    router must see the *entire* conversation, not just the last user message.
    """
    single = _estimate_conversation_tokens([{"role": "user", "content": "fix the failing test"}])
    multi = _estimate_conversation_tokens(
        [
            {"role": "user", "content": "fix the failing test"},
            {"role": "assistant", "content": "I will run the tests first."},
            {"role": "user", "content": "go ahead"},
        ]
    )
    assert multi > single, f"multi-message estimate ({multi}) should exceed single ({single})"


def test_estimate_conversation_tokens_handles_list_content():
    """Multimodal content (list of parts) should be flattened and estimated."""
    tokens = _estimate_conversation_tokens(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
    )
    # Should estimate the text part (non-zero) plus message overhead
    assert tokens > 0
