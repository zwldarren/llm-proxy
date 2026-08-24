# tests/unit/serialization/test_gemini_interactions_usage.py
"""Tests for the Interactions usage -> billable token mapping."""

from llm_proxy.serialization.gemini_interactions.usage import (
    interactions_billable_token_counts,
    interactions_input_tokens_by_modality,
    interactions_normalize_usage,
    interactions_web_search_requests,
)


def test_plain_chat_usage():
    usage = {
        "total_input_tokens": 62,
        "total_output_tokens": 171,
        "total_thought_tokens": 297,
        "total_cached_tokens": 0,
        "total_tool_use_tokens": 0,
        "total_tokens": 530,
    }
    input_tokens, output_tokens = interactions_billable_token_counts(
        usage, has_search_grounding=False
    )
    assert input_tokens == 62
    # thinking folds into output
    assert output_tokens == 468


def test_tool_use_folds_into_input():
    usage = {
        "total_input_tokens": 100,
        "total_output_tokens": 25,
        "total_tool_use_tokens": 50,
        "total_thought_tokens": 0,
        "total_tokens": 125,
    }
    input_tokens, output_tokens = interactions_billable_token_counts(
        usage, has_search_grounding=False
    )
    assert input_tokens == 150
    assert output_tokens == 25


def test_search_grounding_excludes_tool_use():
    usage = {
        "total_input_tokens": 100,
        "total_output_tokens": 25,
        "total_tool_use_tokens": 50,
        "total_thought_tokens": 0,
        "total_tokens": 125,
    }
    input_tokens, _ = interactions_billable_token_counts(usage, has_search_grounding=True)
    assert input_tokens == 100


def test_missing_fields_default_to_zero():
    input_tokens, output_tokens = interactions_billable_token_counts({}, has_search_grounding=False)
    assert input_tokens == 0
    assert output_tokens == 0


def test_web_search_requests_from_grounding_tool_count():
    usage = {
        "grounding_tool_count": [
            {"type": "google_search", "count": 2},
            {"type": "google_maps", "count": 1},
        ]
    }
    assert interactions_web_search_requests(usage) == 2


def test_web_search_requests_empty():
    assert interactions_web_search_requests({}) == 0
    assert interactions_web_search_requests(None) == 0


def test_openai_style_aliases_normalized():
    """The migration guide streams OpenAI-style usage on completed events."""
    usage = {"prompt_tokens": 256, "completion_tokens": 128, "total_tokens": 384}
    normalized = interactions_normalize_usage(usage)
    assert normalized["total_input_tokens"] == 256
    assert normalized["total_output_tokens"] == 128
    input_tokens, output_tokens = interactions_billable_token_counts(
        usage, has_search_grounding=False
    )
    assert input_tokens == 256
    assert output_tokens == 128


def test_new_vocabulary_wins_over_aliases():
    usage = {
        "total_input_tokens": 10,
        "total_output_tokens": 20,
        "prompt_tokens": 999,
        "completion_tokens": 999,
    }
    normalized = interactions_normalize_usage(usage)
    assert normalized["total_input_tokens"] == 10
    assert normalized["total_output_tokens"] == 20


def test_input_tokens_by_modality():
    usage = {
        "input_tokens_by_modality": [
            {"modality": "text", "tokens": 10},
            {"modality": "image", "tokens": 258},
        ]
    }
    by_modality = interactions_input_tokens_by_modality(usage)
    assert by_modality == {"text": 10, "image": 258}


def test_input_tokens_by_modality_missing():
    assert interactions_input_tokens_by_modality({}) == {}
    assert interactions_input_tokens_by_modality({"input_tokens_by_modality": None}) == {}
