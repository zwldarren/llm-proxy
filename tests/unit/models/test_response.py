# tests/unit/models/test_response.py
"""Tests for InternalResponse."""

from llm_proxy.models import (
    InternalResponse,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)


class TestInternalResponse:
    """Test suite for InternalResponse."""

    def test_minimal_response(self):
        """Create InternalResponse with only required fields."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[TextBlock(text="Hello")],
        )
        assert response.id == "resp_123"
        assert response.model == "gpt-4"
        assert len(response.output) == 1
        assert response.status == "completed"
        assert response.usage is None
        assert response.finish_reason is None

    def test_response_with_usage(self):
        """Create InternalResponse with usage information."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[TextBlock(text="Hello")],
            usage=Usage(input_tokens=100, output_tokens=50),
        )
        assert response.usage is not None
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150

    def test_response_with_tool_calls(self):
        """Create InternalResponse with tool calls."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[
                TextBlock(text="Let me check."),
                ToolUseBlock(id="call_1", name="search", input={"q": "test"}),
            ],
        )
        assert len(response.output) == 2
        assert isinstance(response.output[0], TextBlock)
        assert isinstance(response.output[1], ToolUseBlock)
        assert response.output[1].name == "search"
        assert response.output[1].input == {"q": "test"}

    def test_response_with_thinking(self):
        """Create InternalResponse with thinking blocks."""
        response = InternalResponse(
            id="resp_123",
            model="claude-3-opus",
            output=[
                ThinkingBlock(thinking="Let me think about this..."),
                TextBlock(text="The answer is 42."),
            ],
        )
        assert len(response.output) == 2
        assert isinstance(response.output[0], ThinkingBlock)
        assert response.output[0].thinking == "Let me think about this..."

    def test_response_with_status(self):
        """Create InternalResponse with different status values."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[TextBlock(text="Hello")],
            status="incomplete",
        )
        assert response.status == "incomplete"

    def test_response_with_finish_reason(self):
        """Create InternalResponse with finish reason."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[TextBlock(text="Hello")],
            finish_reason="stop",
        )
        assert response.finish_reason == "stop"

    def test_response_with_metadata(self):
        """Create InternalResponse with metadata."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[TextBlock(text="Hello")],
            response_time_ms=1234.5,
            request_id="req_abc",
            provider_info={"provider": "openai", "model_version": "gpt-4-0613"},
        )
        assert response.response_time_ms == 1234.5
        assert response.request_id == "req_abc"
        assert response.provider_info["provider"] == "openai"

    def test_response_multiple_tool_calls(self):
        """Create InternalResponse with multiple tool calls."""
        tool1 = ToolUseBlock(id="call_1", name="get_weather", input={"city": "SF"})
        tool2 = ToolUseBlock(id="call_2", name="get_time", input={"tz": "PST"})
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[tool1, tool2],
        )
        assert len(response.output) == 2
        assert isinstance(response.output[0], ToolUseBlock)
        assert isinstance(response.output[1], ToolUseBlock)
        assert tool1.name == "get_weather"
        assert tool2.name == "get_time"
