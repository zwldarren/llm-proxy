# tests/unit/models/test_request.py
"""Tests for InternalRequest."""

from llm_proxy.models import (
    ConversationContext,
    FunctionTool,
    GenerationParams,
    InternalRequest,
    Message,
    RequestMetadata,
    SystemMessage,
    TextBlock,
    ToolChoice,
)


class TestRequestMetadata:
    """Test suite for RequestMetadata."""

    def test_metadata_defaults(self):
        """RequestMetadata should have sensible defaults."""
        meta = RequestMetadata()
        assert meta.request_id is None
        assert meta.user is None

    def test_metadata_values(self):
        """RequestMetadata should accept values."""
        meta = RequestMetadata(
            request_id="req_123",
            user="user_abc",
        )
        assert meta.request_id == "req_123"
        assert meta.user == "user_abc"


class TestInternalRequest:
    """Test suite for InternalRequest."""

    def test_minimal_request(self):
        """Create InternalRequest with only required fields."""
        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(),
        )
        assert request.model == "gpt-4"
        assert len(request.conversation.messages) == 0
        assert request.tools is None
        assert request.tool_choice is None
        assert request.stream is False
        assert request.stream_options is None
        assert request.request_id is None
        assert request.user is None
        assert isinstance(request.metadata, RequestMetadata)

    def test_request_with_messages(self):
        """Create InternalRequest with messages."""
        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="Hello")]),
                    Message(role="assistant", content=[TextBlock(text="Hi there!")]),
                ]
            ),
        )
        assert len(request.conversation.messages) == 2
        assert request.conversation.messages[0].role == "user"
        assert request.conversation.messages[1].role == "assistant"

    def test_request_with_system_prompt(self):
        """Create InternalRequest with system prompt."""
        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                system_messages=[
                    SystemMessage.from_text(role="system", text="You are a helpful assistant.")
                ]
            ),
        )
        assert len(request.conversation.system_messages) == 1
        assert (
            request.conversation.system_messages[0].text_content == "You are a helpful assistant."
        )

    def test_request_with_tools(self):
        """Create InternalRequest with tools."""
        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(),
            tools=[
                FunctionTool(name="get_weather", parameters={"type": "object"}),
                FunctionTool(name="search", parameters={"type": "object"}),
            ],
            tool_choice=ToolChoice(mode="auto"),
        )
        assert request.tools is not None
        assert len(request.tools) == 2
        assert isinstance(request.tools[0], FunctionTool)
        assert request.tools[0].name == "get_weather"
        assert isinstance(request.tool_choice, ToolChoice)
        assert request.tool_choice.mode == "auto"

    def test_request_with_params(self):
        """Create InternalRequest with generation params."""
        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(),
            params=GenerationParams(temperature=0.7, max_tokens=100),
        )
        assert request.params.temperature == 0.7
        assert request.params.max_tokens == 100

    def test_request_with_streaming(self):
        """Create InternalRequest with streaming enabled."""
        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(),
            stream=True,
        )
        assert request.stream is True

    def test_request_with_metadata(self):
        """Create InternalRequest with metadata."""
        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(),
        )
        request.request_id = "req_123"
        request.user = "user_abc"
        assert request.request_id == "req_123"
        assert request.user == "user_abc"
        assert isinstance(request.metadata, RequestMetadata)

    def test_request_default_params(self):
        """Verify default params are created."""
        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(),
        )
        # params should default to a new GenerationParams instance
        assert request.params is not None
        assert request.params.temperature is None
        assert request.params.max_tokens is None

    def test_unified_request_with_audio_params(self):
        """Test InternalRequest with audio parameters."""
        from llm_proxy.models.params import OpenAISpecificParams

        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hello")])]
            ),
            params=GenerationParams(
                openai=OpenAISpecificParams(
                    audio={"voice": "alloy", "format": "mp3"},
                    modalities=["text", "audio"],
                )
            ),
        )
        assert request.params.openai is not None
        assert request.params.openai.audio == {"voice": "alloy", "format": "mp3"}
        assert request.params.openai.modalities == ["text", "audio"]

    def test_unified_request_with_reasoning_effort(self):
        """Test InternalRequest with reasoning_effort."""
        from llm_proxy.models.params import OpenAISpecificParams

        request = InternalRequest(
            model="o1",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="solve this")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(reasoning_effort="high")),
        )
        assert request.params.openai is not None
        assert request.params.openai.reasoning_effort == "high"

    def test_unified_request_with_web_search(self):
        """Test InternalRequest with web_search_options."""
        from llm_proxy.models.params import OpenAISpecificParams

        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="search")])]
            ),
            params=GenerationParams(
                openai=OpenAISpecificParams(web_search_options={"search_context_size": "high"})
            ),
        )
        assert request.params.openai is not None
        assert request.params.openai.web_search_options == {"search_context_size": "high"}

    def test_unified_request_with_logit_bias(self):
        """Test InternalRequest with logit_bias."""
        from llm_proxy.models.params import OpenAISpecificParams

        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="test")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(logit_bias={"1234": -100})),
        )
        assert request.params.openai is not None
        assert request.params.openai.logit_bias == {"1234": -100}

    def test_unified_request_with_prediction(self):
        """Test InternalRequest with prediction."""
        from llm_proxy.models.params import OpenAISpecificParams

        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="test")])]
            ),
            params=GenerationParams(
                openai=OpenAISpecificParams(
                    prediction={"type": "content", "content": "expected output"}
                )
            ),
        )
        assert request.params.openai is not None
        assert request.params.openai.prediction == {"type": "content", "content": "expected output"}

    def test_unified_request_with_caching(self):
        """Test InternalRequest with caching parameters."""
        from llm_proxy.models.params import OpenAISpecificParams

        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="test")])]
            ),
            params=GenerationParams(
                openai=OpenAISpecificParams(
                    prompt_cache_key="cache-123",
                    prompt_cache_retention="24h",
                )
            ),
        )
        assert request.params.openai is not None
        assert request.params.openai.prompt_cache_key == "cache-123"
        assert request.params.openai.prompt_cache_retention == "24h"

    def test_unified_request_with_safety(self):
        """Test InternalRequest with safety_identifier."""
        from llm_proxy.models.params import OpenAISpecificParams

        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="test")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(safety_identifier="user-123")),
        )
        assert request.params.openai is not None
        assert request.params.openai.safety_identifier == "user-123"
