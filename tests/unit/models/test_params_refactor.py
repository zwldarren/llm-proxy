# tests/unit/models/test_params_refactor.py

from llm_proxy.models.params import (
    AnthropicSpecificParams,
    CommonParams,
    GenerationParams,
    OpenAISpecificParams,
)


class TestCommonParams:
    def test_common_params_defaults(self):
        """CommonParams should have sensible defaults."""
        params = CommonParams()
        assert params.temperature is None
        assert params.top_p is None
        assert params.max_tokens is None
        assert params.stop is None
        assert params.frequency_penalty is None
        assert params.presence_penalty is None
        assert params.seed is None
        assert params.response_format is None
        assert params.n is None

    def test_common_params_values(self):
        """CommonParams should accept values."""
        params = CommonParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            stop=["STOP"],
        )
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 1000
        assert params.stop == ["STOP"]


class TestOpenAISpecificParams:
    def test_openai_params_defaults(self):
        """OpenAISpecificParams should have sensible defaults."""
        params = OpenAISpecificParams()
        assert params.logprobs is None
        assert params.top_logprobs is None
        assert params.service_tier is None
        assert params.audio is None
        assert params.modalities is None
        assert params.reasoning_effort is None

    def test_openai_params_values(self):
        """OpenAISpecificParams should accept values."""
        params = OpenAISpecificParams(
            logprobs=True,
            top_logprobs=5,
            reasoning_effort="medium",
        )
        assert params.logprobs is True
        assert params.top_logprobs == 5
        assert params.reasoning_effort == "medium"


class TestAnthropicSpecificParams:
    def test_anthropic_params_defaults(self):
        """AnthropicSpecificParams should have sensible defaults."""
        params = AnthropicSpecificParams()
        assert params.top_k is None
        assert params.stop_sequences is None
        assert params.disable_parallel_tool_use is None

    def test_thinking_via_generation_params(self):
        """Thinking config should be set via GenerationParams, not AnthropicSpecificParams."""
        from llm_proxy.models.types import ThinkingConfig

        params = GenerationParams(
            max_tokens=1000,
            anthropic=AnthropicSpecificParams(top_k=50),
            thinking=ThinkingConfig(type="enabled", budget_tokens=10000),
        )
        assert params.anthropic is not None
        assert params.anthropic.top_k == 50
        assert params.thinking is not None
        assert params.thinking.type == "enabled"
        assert params.thinking.budget_tokens == 10000


class TestGenerationParamsRefactored:
    def test_generation_params_composition(self):
        """GenerationParams should use composition."""

        params = GenerationParams(
            common=CommonParams(temperature=0.7),
            openai=OpenAISpecificParams(logprobs=True),
            anthropic=AnthropicSpecificParams(top_k=50),
        )
        assert params.common.temperature == 0.7
        assert params.openai is not None
        assert params.openai.logprobs is True
        assert params.anthropic is not None
        assert params.anthropic.top_k == 50

    def test_generation_params_common_defaults(self):
        """GenerationParams should have CommonParams by default."""
        params = GenerationParams()
        assert params.common is not None
        assert params.openai is None
        assert params.anthropic is None

    def test_generation_params_property_access(self):
        """GenerationParams should expose common params via properties."""
        params = GenerationParams()
        params.common.temperature = 0.7
        assert params.temperature == 0.7

        params.temperature = 0.9
        assert params.common.temperature == 0.9

    def test_generation_params_all_properties(self):
        """All common params should be accessible via properties."""
        params = GenerationParams()

        params.temperature = 0.7
        assert params.temperature == 0.7

        params.top_p = 0.9
        assert params.top_p == 0.9

        params.max_tokens = 1000
        assert params.max_tokens == 1000

        params.stop = ["STOP"]
        assert params.stop == ["STOP"]

        params.frequency_penalty = 1.5
        assert params.frequency_penalty == 1.5

        params.presence_penalty = 0.5
        assert params.presence_penalty == 0.5

        from llm_proxy.models.types import ResponseFormat

        params.response_format = ResponseFormat(type="json_object")
        assert params.response_format is not None
        assert params.response_format.type == "json_object"

        params.seed = 42
        assert params.seed == 42

        params.n = 3
        assert params.n == 3
