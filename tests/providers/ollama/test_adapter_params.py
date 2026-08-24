"""Tests for Ollama adapter parameter mapping."""

import pytest

from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
    OpenAISpecificParams,
    ResponseFormat,
    TextBlock,
    ThinkingConfig,
)
from llm_proxy.providers.ollama.adapter import OllamaAdapter


class TestBuildRequestBody:
    """Tests for _build_request_body parameter mapping."""

    @pytest.fixture
    def provider(self):
        return OllamaAdapter()

    def test_format_json_object(self, provider):
        """Test response_format json_object maps to format: json."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(response_format=ResponseFormat(type="json_object")),
        )
        body = provider._build_request_body(request)
        assert body["format"] == "json"

    def test_format_json_schema(self, provider):
        """Test response_format json_schema maps to format schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(
                response_format=ResponseFormat(type="json_schema", json_schema=schema)
            ),
        )
        body = provider._build_request_body(request)
        assert body["format"] == schema

    def test_thinking_enabled(self, provider):
        """Test Anthropic thinking enabled maps to think: true."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=None)),
        )
        body = provider._build_request_body(request)
        assert body["think"] is True

    def test_thinking_disabled(self, provider):
        """Test Anthropic thinking disabled maps to think: false."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="disabled", budget_tokens=None)),
        )
        body = provider._build_request_body(request)
        assert body["think"] is False

    def test_thinking_with_budget_low(self, provider):
        """Test thinking with low budget maps to think: low."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=2000)),
        )
        body = provider._build_request_body(request)
        assert body["think"] == "low"

    def test_thinking_with_budget_high(self, provider):
        """Test thinking with high budget maps to think: high."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=30000)),
        )
        body = provider._build_request_body(request)
        assert body["think"] == "high"

    def test_reasoning_effort_levels(self, provider):
        """Test OpenAI reasoning_effort maps to think levels."""
        cases = [
            ("none", False),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "max"),
        ]
        for effort, expected in cases:
            request = InternalRequest(
                model="test",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="hi")])]
                ),
                params=GenerationParams(openai=OpenAISpecificParams(reasoning_effort=effort)),
            )
            body = provider._build_request_body(request)
            assert body["think"] == expected, f"Failed for effort={effort}"

    def test_reasoning_effort_takes_precedence(self, provider):
        """Test reasoning_effort when unified thinking is not set."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(
                openai=OpenAISpecificParams(reasoning_effort="low"),
            ),
        )
        body = provider._build_request_body(request)
        assert body["think"] == "low"

    def test_logprobs_enabled(self, provider):
        """Test logprobs parameter mapping."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(logprobs=True, top_logprobs=5)),
        )
        body = provider._build_request_body(request)
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 5

    def test_ollama_options_from_extra_body(self, provider):
        """Test Ollama options extracted from extra_body."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={
                "num_ctx": 4096,
                "mirostat": 2,
                "seed": 42,
                "unknown_param": "value",
            },
        )
        body = provider._build_request_body(request)
        assert body["options"]["num_ctx"] == 4096
        assert body["options"]["mirostat"] == 2
        assert body["options"]["seed"] == 42
        # With default 'ignore' policy, unknown params are stripped
        assert "unknown_param" not in body

    def test_ollama_options_from_extra_body_passthrough(self, provider):
        """Test Ollama options with passthrough policy."""
        adapter = OllamaAdapter(unknown_fields_policy="passthrough")
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={
                "num_ctx": 4096,
                "mirostat": 2,
                "seed": 42,
                "unknown_param": "value",
            },
        )
        body = adapter._build_request_body(request)
        assert body["options"]["num_ctx"] == 4096
        assert body["options"]["mirostat"] == 2
        assert body["options"]["seed"] == 42
        # With passthrough, unknown params go to top level
        assert body["unknown_param"] == "value"

    def test_keep_alive_at_top_level(self, provider):
        """Test keep_alive goes to top level, not options."""
        adapter = OllamaAdapter(unknown_fields_policy="passthrough")
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"keep_alive": "5m"},
        )
        body = adapter._build_request_body(request)
        assert body["keep_alive"] == "5m"
        assert "keep_alive" not in body.get("options", {})

    def test_keep_alive_at_top_level_with_ignore_policy(self, provider):
        """Test keep_alive is preserved even with default ignore policy."""
        request = InternalRequest(
            model="test",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"keep_alive": "5m"},
        )
        body = provider._build_request_body(request)
        assert body["keep_alive"] == "5m"
        assert "keep_alive" not in body.get("options", {})


class TestConvertLogprobs:
    """Tests for convert_logprobs method in OllamaProviderSerializer."""

    @pytest.fixture
    def serializer(self):
        from llm_proxy.serialization.ollama.serializer import OllamaProviderSerializer

        return OllamaProviderSerializer()

    def test_empty_logprobs(self, serializer):
        """Test empty input returns None."""
        assert serializer.convert_logprobs(None) is None
        assert serializer.convert_logprobs([]) is None

    def test_simple_logprobs(self, serializer):
        """Test simple logprobs conversion."""
        ollama_logprobs = [
            {"token": "Hello", "logprob": -0.5, "bytes": [72, 101, 108, 108, 111]},
        ]
        result = serializer.convert_logprobs(ollama_logprobs)
        assert result is not None
        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["token"] == "Hello"
        assert result["content"][0]["logprob"] == -0.5
        assert result["content"][0]["bytes"] == [72, 101, 108, 108, 111]

    def test_logprobs_with_top_logprobs(self, serializer):
        """Test logprobs with top_logprobs."""
        ollama_logprobs = [
            {
                "token": "Hello",
                "logprob": -0.5,
                "top_logprobs": [
                    {"token": "Hello", "logprob": -0.5},
                    {"token": "Hi", "logprob": -1.2},
                ],
            },
        ]
        result = serializer.convert_logprobs(ollama_logprobs)
        assert result is not None
        assert len(result["content"][0]["top_logprobs"]) == 2
        assert result["content"][0]["top_logprobs"][0]["token"] == "Hello"
        assert result["content"][0]["top_logprobs"][1]["token"] == "Hi"

    def test_logprobs_missing_optional_fields(self, serializer):
        """Test logprobs handles missing optional fields."""
        ollama_logprobs = [
            {"token": "test", "logprob": -0.1},
        ]
        result = serializer.convert_logprobs(ollama_logprobs)
        assert result is not None
        assert "bytes" not in result["content"][0]
        assert "top_logprobs" not in result["content"][0]
