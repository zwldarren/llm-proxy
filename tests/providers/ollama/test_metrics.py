"""Unit tests for Ollama metric extraction helpers."""

from llm_proxy.serialization.ollama.metrics import (
    extract_ollama_cached_tokens,
    extract_ollama_token_counts,
)


class TestExtractOllamaTokenCounts:
    def test_neither_counter_present_returns_none(self):
        assert extract_ollama_token_counts({}) is None

    def test_reported_zero_is_a_real_count(self):
        assert extract_ollama_token_counts({"prompt_eval_count": 0, "eval_count": 50}) == (0, 50)

    def test_missing_counter_defaults_to_zero(self):
        assert extract_ollama_token_counts({"eval_count": 50}) == (0, 50)
        assert extract_ollama_token_counts({"prompt_eval_count": 11}) == (11, 0)


class TestExtractOllamaCachedTokens:
    def test_absent_field_returns_none(self):
        assert extract_ollama_cached_tokens({}, 11) is None

    def test_reported_zero_is_zero_not_none(self):
        assert extract_ollama_cached_tokens({"prompt_eval_cached_count": 0}, 11) == 0

    def test_clamped_into_input_range(self):
        assert extract_ollama_cached_tokens({"prompt_eval_cached_count": 99}, 5) == 5
        assert extract_ollama_cached_tokens({"prompt_eval_cached_count": -3}, 5) == 0
