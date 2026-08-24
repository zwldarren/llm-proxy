# tests/unit/models/test_params.py
"""Tests for GenerationParams and related parameter classes."""

from llm_proxy.models.params import (
    AnthropicSpecificParams,
    GenerationParams,
    OpenAISpecificParams,
)
from llm_proxy.models.types import ResponseFormat


class TestAnthropicSpecificParams:
    def test_create_with_minimal_params(self):
        params = AnthropicSpecificParams()
        assert params.top_k is None
        assert params.metadata is None

    def test_create_with_all_params(self):
        metadata = {"user_id": "user-123", "session_id": "sess-456"}
        params = AnthropicSpecificParams(top_k=40, metadata=metadata)

        assert params.top_k == 40
        assert params.metadata == metadata
        assert params.metadata is not None
        assert params.metadata["user_id"] == "user-123"

    def test_create_with_only_top_k(self):
        params = AnthropicSpecificParams(top_k=50)
        assert params.top_k == 50
        assert params.metadata is None


class TestOpenAISpecificParams:
    def test_create_with_minimal_params(self):
        params = OpenAISpecificParams()
        assert params.logprobs is None
        assert params.top_logprobs is None
        assert params.service_tier is None

    def test_create_with_all_params(self):
        params = OpenAISpecificParams(logprobs=True, top_logprobs=5, service_tier="auto")

        assert params.logprobs is True
        assert params.top_logprobs == 5
        assert params.service_tier == "auto"

    def test_create_with_only_logprobs(self):
        params = OpenAISpecificParams(logprobs=True)
        assert params.logprobs is True
        assert params.top_logprobs is None
        assert params.service_tier is None


class TestGenerationParam:
    def test_create_with_minimal_params(self):
        params = GenerationParams()
        assert params.temperature is None
        assert params.top_p is None
        assert params.max_tokens is None
        assert params.stop is None
        assert params.frequency_penalty is None
        assert params.presence_penalty is None
        assert params.response_format is None
        assert params.seed is None
        assert params.anthropic is None
        assert params.openai is None

    def test_create_with_all_common_params(self):
        params = GenerationParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            stop=["END", "STOP"],
            frequency_penalty=0.5,
            presence_penalty=0.3,
            seed=42,
        )

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 1000
        assert params.stop == ["END", "STOP"]
        assert params.frequency_penalty == 0.5
        assert params.presence_penalty == 0.3
        assert params.seed == 42

    def test_create_with_response_format(self):
        response_format = ResponseFormat(type="json_object")
        params = GenerationParams(response_format=response_format)

        assert params.response_format == response_format
        assert params.response_format is not None
        assert params.response_format.type == "json_object"

    def test_create_with_anthropic_params(self):
        anthropic = AnthropicSpecificParams(top_k=40)
        params = GenerationParams(anthropic=anthropic)

        assert params.anthropic is not None
        assert params.anthropic.top_k == 40

    def test_create_with_openai_params(self):
        openai = OpenAISpecificParams(logprobs=True, top_logprobs=3)
        params = GenerationParams(openai=openai)

        assert params.openai is not None
        assert params.openai.logprobs is True
        assert params.openai.top_logprobs == 3

    def test_create_with_all_params(self):
        response_format = ResponseFormat(type="json_schema", json_schema={"type": "object"})
        anthropic = AnthropicSpecificParams(top_k=40)
        openai = OpenAISpecificParams(logprobs=True, service_tier="auto")

        params = GenerationParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=2000,
            stop=["DONE"],
            frequency_penalty=0.4,
            presence_penalty=0.2,
            response_format=response_format,
            seed=123,
            anthropic=anthropic,
            openai=openai,
        )

        assert params.temperature == 0.8
        assert params.top_p == 0.95
        assert params.max_tokens == 2000
        assert params.stop == ["DONE"]
        assert params.frequency_penalty == 0.4
        assert params.presence_penalty == 0.2
        assert params.response_format == response_format
        assert params.seed == 123
        assert params.anthropic is not None
        assert params.anthropic.top_k == 40
        assert params.openai is not None
        assert params.openai.logprobs is True
