"""Tests for the NanoGPT provider adapter."""

import pytest

from llm_proxy.providers.nanogpt.adapter import NanoGPTAdapter
from llm_proxy.providers.nanogpt.pricing import extract_nanogpt_pricing


class TestExtractNanoGptPricing:
    """Pricing extraction from the x_nanogpt_pricing envelope."""

    @pytest.mark.parametrize(
        "response",
        [
            {},  # missing envelope
            {"x_nanogpt_pricing": []},  # wrong type
        ],
    )
    def test_none_when_envelope_invalid(self, response):
        assert extract_nanogpt_pricing(response) is None

    def test_full_pricing_extracted(self):
        response = {"x_nanogpt_pricing": {"inputTokens": 12, "outputTokens": 8, "cost": 0.02}}
        assert extract_nanogpt_pricing(response) == {
            "input_tokens": 12,
            "output_tokens": 8,
            "nanogpt_cost": 0.02,
            "nanogpt_pricing": response["x_nanogpt_pricing"],
            "inputTokens": 12,
            "outputTokens": 8,
            "cost": 0.02,
        }

    def test_negative_tokens_excluded(self):
        response = {"x_nanogpt_pricing": {"inputTokens": -5, "outputTokens": 3}}
        result = extract_nanogpt_pricing(response)
        assert "input_tokens" not in result
        assert result == {
            "output_tokens": 3,
            "nanogpt_pricing": response["x_nanogpt_pricing"],
            "inputTokens": -5,
            "outputTokens": 3,
        }


class TestStreamTransformChunk:
    """Streaming chunks get NanoGPT pricing normalized into the usage block."""

    def setup_method(self):
        self.adapter = NanoGPTAdapter(api_key="test-key")

    def test_pricing_injected_into_usage(self):
        chunk = {
            "id": "c1",
            "choices": [{"delta": {"content": "hi"}}],
            "x_nanogpt_pricing": {"inputTokens": 20, "outputTokens": 10, "cost": 0.05},
        }

        result = self.adapter._stream_transform_chunk(chunk, context={})

        assert result is chunk
        assert result["usage"]["prompt_tokens"] == 20
        assert result["usage"]["completion_tokens"] == 10
        assert result["usage"]["total_tokens"] == 30
        assert result["usage"]["nanogpt_cost"] == 0.05

    def test_usage_created_when_absent(self):
        chunk = {
            "id": "c2",
            "choices": [{"delta": {"content": "hi"}}],
            "x_nanogpt_pricing": {"inputTokens": 1, "outputTokens": 2},
        }

        result = self.adapter._stream_transform_chunk(chunk, context={})

        assert result["usage"]["prompt_tokens"] == 1
        assert result["usage"]["completion_tokens"] == 2

    def test_no_pricing_leaves_chunk_untouched(self):
        chunk = {"id": "c3", "choices": [{"delta": {"content": "hi"}}]}

        result = self.adapter._stream_transform_chunk(chunk, context={})

        assert "usage" not in result

    def test_chunk_without_choices_is_returned(self):
        chunk = {"x_nanogpt_pricing": {"inputTokens": 1, "outputTokens": 2}, "choices": []}

        result = self.adapter._stream_transform_chunk(chunk, context={})

        assert result is chunk
