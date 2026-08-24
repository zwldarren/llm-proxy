"""Tests for the NanoGPT provider serializer."""

import pytest

from llm_proxy.models import InternalResponse
from llm_proxy.providers.nanogpt.pricing import extract_nanogpt_pricing
from llm_proxy.providers.nanogpt.serializer import NanoGPTProviderSerializer
from llm_proxy.serialization.providers import get_provider_serializer


def _minimal_openai_response() -> dict:
    return {
        "id": "test-id",
        "model": "gpt-4",
        "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class TestExtractNanoGptPricing:
    """x_nanogpt_pricing extraction from the response envelope."""

    @pytest.mark.parametrize(
        "response",
        [
            {"id": "x"},  # missing envelope
            {"x_nanogpt_pricing": "nope"},  # wrong type
        ],
    )
    def test_none_when_envelope_invalid(self, response):
        assert extract_nanogpt_pricing(response) is None

    def test_full_pricing_extracted(self):
        response = {
            "x_nanogpt_pricing": {
                "inputTokens": 10,
                "outputTokens": 5,
                "cost": 0.012,
            }
        }
        assert extract_nanogpt_pricing(response) == {
            "input_tokens": 10,
            "output_tokens": 5,
            "nanogpt_cost": 0.012,
            "nanogpt_pricing": response["x_nanogpt_pricing"],
            "inputTokens": 10,
            "outputTokens": 5,
            "cost": 0.012,
        }

    def test_partial_pricing_only_input(self):
        response = {"x_nanogpt_pricing": {"inputTokens": 10}}
        assert extract_nanogpt_pricing(response) == {
            "input_tokens": 10,
            "nanogpt_pricing": response["x_nanogpt_pricing"],
            "inputTokens": 10,
        }

    def test_negative_tokens_excluded(self):
        response = {"x_nanogpt_pricing": {"inputTokens": -1, "outputTokens": 5}}
        result = extract_nanogpt_pricing(response)
        assert "input_tokens" not in result
        assert result == {
            "output_tokens": 5,
            "nanogpt_pricing": response["x_nanogpt_pricing"],
            "inputTokens": -1,
            "outputTokens": 5,
        }

    def test_zero_cost_excluded(self):
        response = {"x_nanogpt_pricing": {"inputTokens": 10, "outputTokens": 5, "cost": 0}}
        result = extract_nanogpt_pricing(response)
        assert "nanogpt_cost" not in result
        assert result["cost"] == 0
        assert result["nanogpt_pricing"] == response["x_nanogpt_pricing"]


class TestParseProviderResponse:
    """NanoGPT pricing metadata is merged into the parsed InternalResponse."""

    def test_registered_serializer_is_nanogpt(self):
        serializer = get_provider_serializer("nanogpt")
        assert isinstance(serializer, NanoGPTProviderSerializer)

    def test_pricing_merged_into_usage(self):
        serializer = NanoGPTProviderSerializer()
        response = _minimal_openai_response()
        response["x_nanogpt_pricing"] = {
            "inputTokens": 100,
            "outputTokens": 50,
            "cost": 0.3,
        }

        result = serializer.parse_provider_response(response, model="gpt-4")

        assert isinstance(result, InternalResponse)
        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150
        assert result.provider_info.get("nanogpt_cost") == 0.3

    def test_pricing_subfields_preserved_in_provider_info(self):
        serializer = NanoGPTProviderSerializer()
        response = _minimal_openai_response()
        response["x_nanogpt_pricing"] = {
            "inputTokens": 10,
            "outputTokens": 5,
            "cost": 0.001,
            "amount": 1.5,
            "currency": "USD",
            "error": None,
            "paymentSource": "balance",
            "cacheReadTokens": 2,
            "cacheWriteTokens": 3,
            "webSearchRequests": 1,
            "youtubeTranscripts": 2,
            "scrapedUrls": 4,
        }

        result = serializer.parse_provider_response(response, model="gpt-4")

        assert result.provider_info.get("nanogpt_cost") == 0.001
        assert result.provider_info.get("amount") == 1.5
        assert result.provider_info.get("currency") == "USD"
        assert result.provider_info.get("paymentSource") == "balance"
        assert result.provider_info.get("cacheReadTokens") == 2
        assert result.provider_info.get("cacheWriteTokens") == 3
        assert result.provider_info.get("webSearchRequests") == 1
        assert result.provider_info.get("youtubeTranscripts") == 2
        assert result.provider_info.get("scrapedUrls") == 4
        # Full envelope is also kept.
        assert result.provider_info.get("nanogpt_pricing") == response["x_nanogpt_pricing"]

    def test_unknown_fields_preserved_in_provider_info(self):
        serializer = NanoGPTProviderSerializer()
        response = _minimal_openai_response()
        response["x_custom_field"] = "keep-me"

        result = serializer.parse_provider_response(response, model="gpt-4")

        assert result.provider_info.get("x_custom_field") == "keep-me"


class TestChunkConverter:
    """Streaming chunks pass through unchanged (already canonical)."""

    def test_returns_identity_converter(self):
        from llm_proxy.serialization.providers.base import IdentityChunkConverter

        serializer = NanoGPTProviderSerializer()
        converter = serializer.get_chunk_converter(model="gpt-4", request_id="r1")
        assert isinstance(converter, IdentityChunkConverter)
