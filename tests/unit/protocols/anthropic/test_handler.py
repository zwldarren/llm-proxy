# tests/unit/protocols/anthropic/test_handler.py
"""Tests for AnthropicProtocolEndpoint."""

from llm_proxy.models import (
    FunctionTool,
    InternalResponse,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from llm_proxy.protocols.anthropic.handler import (
    AnthropicStreamingTransformer,
    anthropic_protocol,
)
from llm_proxy.protocols.registry import get_protocol_serializer

_serializer = get_protocol_serializer("anthropic")


class TestAnthropicProtocolEndpoint:
    """Test suite for AnthropicProtocolEndpoint."""

    def test_name(self):
        """Test protocol name."""
        assert anthropic_protocol.name == "anthropic"

    def test_paths(self):
        """Test supported paths."""
        paths = anthropic_protocol.paths
        assert "/v1/messages" in paths

    def test_request_model(self):
        """Test request model is returned."""
        model = anthropic_protocol.request_model
        assert model is not None
        # Verify it's a Pydantic model
        assert hasattr(model, "model_fields")

    def test_parse_simple_request(self):
        """Test parsing a simple request."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
        data = request.model_dump(exclude_none=True)

        unified = _serializer.parse_request(data)
        assert unified.model == "claude-3-sonnet"
        assert len(unified.conversation.messages) == 1
        assert unified.conversation.messages[0].role == "user"

    def test_parse_request_with_system(self):
        """Test parsing request with system prompt."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hi"}],
        )
        data = request.model_dump(exclude_none=True)

        unified = _serializer.parse_request(data)
        assert len(unified.conversation.system_messages) == 1
        assert unified.conversation.system_messages[0].text_content == "You are helpful."
        assert len(unified.conversation.messages) == 1

    def test_parse_request_with_system_blocks(self):
        """Test parsing request with system as content blocks."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            system=[
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ],
            messages=[{"role": "user", "content": "Hi"}],
        )
        data = request.model_dump(exclude_none=True)

        unified = _serializer.parse_request(data)
        assert len(unified.conversation.system_messages) == 1
        assert (
            unified.conversation.system_messages[0].text_content
            == "You are helpful.\n\nBe concise."
        )

    def test_parse_request_with_tools(self):
        """Test parsing request with tools."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "input_schema": {"type": "object"},
                }
            ],
        )
        data = request.model_dump(exclude_none=True)

        unified = _serializer.parse_request(data)
        assert unified.tools is not None
        assert len(unified.tools) == 1
        tool = unified.tools[0]
        assert isinstance(tool, FunctionTool)
        assert tool.name == "get_weather"

    def test_parse_request_with_params(self):
        """Test parsing request with generation params."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            top_k=50,
        )
        data = request.model_dump(exclude_none=True)

        unified = _serializer.parse_request(data)
        assert unified.params.max_tokens == 100
        assert unified.params.temperature == 0.7
        assert unified.params.anthropic is not None
        assert unified.params.anthropic.top_k == 50

    def test_parse_request_with_multimodal_content(self):
        """Test parsing request with image content."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "base64data",
                            },
                        },
                    ],
                }
            ],
        )
        data = request.model_dump(exclude_none=True)

        unified = _serializer.parse_request(data)
        assert len(unified.conversation.messages) == 1
        content = unified.conversation.messages[0].content
        assert len(content) == 2
        assert isinstance(content[0], TextBlock)
        assert content[0].text == "What's in this image?"

    def test_format_simple_response(self):
        """Test formatting a simple response."""
        response = InternalResponse(
            id="msg_123",
            model="claude-3-sonnet",
            output=[TextBlock(text="Hello, world!")],
            usage=Usage(input_tokens=10, output_tokens=5),
        )

        result = _serializer.format_response(response)
        assert result["id"] == "msg_123"
        assert result["model"] == "claude-3-sonnet"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello, world!"
        assert result["role"] == "assistant"
        assert result["type"] == "message"

    def test_format_response_with_tool_use(self):
        """Test formatting response with tool use."""
        response = InternalResponse(
            id="msg_123",
            model="claude-3-sonnet",
            output=[ToolUseBlock(id="toolu_1", name="get_weather", input={"city": "SF"})],
        )

        result = _serializer.format_response(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["id"] == "toolu_1"
        assert result["content"][0]["name"] == "get_weather"
        assert result["content"][0]["input"] == {"city": "SF"}

    def test_format_response_with_thinking(self):
        """Test formatting response with thinking."""
        response = InternalResponse(
            id="msg_123",
            model="claude-3-sonnet",
            output=[
                ThinkingBlock(thinking="Let me think..."),
                TextBlock(text="The answer is 42."),
            ],
        )

        result = _serializer.format_response(response)
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "thinking"
        assert result["content"][0]["thinking"] == "Let me think..."
        assert result["content"][1]["type"] == "text"

    def test_format_response_with_usage(self):
        """Test formatting response with usage."""
        response = InternalResponse(
            id="msg_123",
            model="claude-3-sonnet",
            output=[TextBlock(text="Hello")],
            usage=Usage(input_tokens=100, output_tokens=50),
        )

        result = _serializer.format_response(response)
        assert "usage" in result
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50

    def test_streaming_transformer(self):
        """Test that streaming transformer is returned."""
        transformer_cls = anthropic_protocol.get_streaming_transformer()
        assert transformer_cls is not None

    def test_streaming_transformer_converts_openai_format(self):
        """Test that streaming transformer converts OpenAI chunks to Anthropic format."""
        transformer = AnthropicStreamingTransformer(model="claude-3-opus", request_id="msg_123")

        # Simulate OpenAI-style chunks
        chunk1 = (
            'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
            '"created":1234567890,"model":"claude-3-opus",'
            '"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}'
        )
        result1 = transformer.transform(chunk1)
        assert result1 is not None
        # Should send message_start
        assert "event: message_start" in result1
        assert "msg_123" in result1

        chunk2 = (
            'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
            '"created":1234567890,"model":"claude-3-opus",'
            '"choices":[{"index":0,"delta":{"content":"Hello"}}]}'
        )
        result2 = transformer.transform(chunk2)
        assert result2 is not None
        # Should send content_block_start and content_block_delta
        assert "event: content_block_start" in result2
        assert "event: content_block_delta" in result2
        assert '"type":"text_delta"' in result2
        assert '"text":"Hello"' in result2

        chunk3 = (
            'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
            '"created":1234567890,"model":"claude-3-opus",'
            '"choices":[{"index":0,"delta":{"content":" world"}}]}'
        )
        result3 = transformer.transform(chunk3)
        assert result3 is not None
        # Should only send delta
        assert "event: content_block_delta" in result3
        assert '"text":" world"' in result3

    def test_streaming_transformer_handles_reasoning_content(self):
        """Test that streaming transformer handles reasoning_content (thinking blocks)."""
        transformer = AnthropicStreamingTransformer(model="claude-3-opus", request_id="msg_123")

        # Start with message
        chunk1 = (
            'data: {"id":"chatcmpl-abc","choices":'
            '[{"index":0,"delta":{"role":"assistant","content":""}}]}'
        )
        transformer.transform(chunk1)

        # Reasoning content first - should start thinking block
        chunk2 = (
            'data: {"id":"chatcmpl-abc","choices":'
            '[{"index":0,"delta":{"reasoning_content":"Analyzing"}}]}'
        )
        result2 = transformer.transform(chunk2)
        assert result2 is not None
        assert "event: content_block_start" in result2
        assert '"type":"thinking"' in result2
        assert "event: content_block_delta" in result2
        assert '"type":"thinking_delta"' in result2

    def test_streaming_transformer_switches_from_thinking_to_text(self):
        """Test that text is emitted after reasoning_content starts a thinking block."""
        transformer = AnthropicStreamingTransformer(model="claude-3-opus", request_id="msg_123")

        transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"role":"assistant"}}]}'
        )

        thinking_chunk = (
            'data: {"id":"chatcmpl-abc","choices":'
            '[{"index":0,"delta":{"reasoning_content":"Let me think"}}]}'
        )
        thinking_result = transformer.transform(thinking_chunk)
        assert thinking_result is not None
        assert '"type":"thinking_delta"' in thinking_result

        text_chunk = (
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"content":"Final answer"}}]}'
        )
        text_result = transformer.transform(text_chunk)
        assert text_result is not None
        assert "event: content_block_stop" in text_result
        assert "event: content_block_start" in text_result
        assert '"type":"text"' in text_result
        assert "event: content_block_delta" in text_result
        assert '"type":"text_delta"' in text_result
        assert '"text":"Final answer"' in text_result

    def test_streaming_transformer_ignores_late_reasoning_after_text_started(self):
        """Test delayed reasoning fragments do not split text into multiple blocks."""
        transformer = AnthropicStreamingTransformer(model="claude-3-opus", request_id="msg_123")

        transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"role":"assistant"}}]}'
        )

        # Provider emits initial reasoning first.
        transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":'
            '[{"index":0,"delta":{"reasoning_content":"Think step 1"}}]}'
        )

        # Then starts final text output.
        text_part_1 = transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"content":"Hello"}}]}'
        )
        assert text_part_1 is not None
        assert '"type":"text_delta"' in text_part_1

        # A delayed reasoning tail arrives after text has started.
        late_reasoning = transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":'
            '[{"index":0,"delta":{"reasoning_content":" late tail"}}]}'
        )
        # Must be ignored to avoid creating another block transition.
        assert late_reasoning is None

        # Text should continue in the same block.
        text_part_2 = transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"content":" world"}}]}'
        )
        assert text_part_2 is not None
        assert "event: content_block_start" not in text_part_2
        assert "event: content_block_stop" not in text_part_2
        assert '"type":"text_delta"' in text_part_2
        assert '"text":" world"' in text_part_2

    def test_streaming_transformer_handles_tool_calls(self):
        """Test that streaming transformer handles tool calls."""
        transformer = AnthropicStreamingTransformer(model="claude-3-opus", request_id="msg_123")

        # Start with message
        chunk1 = (
            'data: {"id":"chatcmpl-abc","choices":'
            '[{"index":0,"delta":{"role":"assistant","content":""}}]}'
        )
        transformer.transform(chunk1)

        # Tool call start
        chunk2 = (
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":'
            '{"tool_calls":[{"id":"call_123","type":"function","index":0,'
            '"function":{"name":"get_weather","arguments":""}}]}}]}'
        )
        result2 = transformer.transform(chunk2)
        assert result2 is not None
        assert "event: content_block_start" in result2
        assert '"type":"tool_use"' in result2
        assert '"id":"call_123"' in result2
        assert '"name":"get_weather"' in result2

        # Tool call arguments
        chunk3 = (
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":'
            '{"tool_calls":[{"index":0,"function":'
            '{"arguments":"{\\"city\\": \\"NYC\\"}"}}]}}]}'
        )
        result3 = transformer.transform(chunk3)
        assert result3 is not None
        assert "event: content_block_delta" in result3
        assert '"type":"input_json_delta"' in result3
        assert "NYC" in result3

    def test_streaming_transformer_message_start_uses_usage_from_first_chunk(self):
        """Test that message_start usage reflects usage included in the first chunk."""
        transformer = AnthropicStreamingTransformer(model="glm-5", request_id="msg_123")

        chunk = (
            'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
            '"created":1234567890,"model":"glm-5",'
            '"choices":[{"index":0,"delta":{"role":"assistant","content":""}}],'
            '"usage":{"prompt_tokens":123,"completion_tokens":0,"total_tokens":123}}'
        )

        result = transformer.transform(chunk)
        assert result is not None
        assert "event: message_start" in result
        assert '"input_tokens":123' in result
        assert '"output_tokens":0' in result

    def test_streaming_transformer_message_start_accumulates_usage_from_later_chunk(self):
        """Test that message_start uses accumulated input_tokens when usage comes later.

        This is the common case for OpenAI streaming where usage is sent in the final chunk,
        not the first chunk. The transformer should accumulate usage from any chunk and
        use it for message_start when content arrives first.
        """
        transformer = AnthropicStreamingTransformer(model="gpt-4", request_id="msg_123")

        # First chunk: content only, no usage (common OpenAI pattern)
        chunk1 = 'data: {"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}'
        result1 = transformer.transform(chunk1)

        # message_start should be sent with input_tokens=0 (no usage yet)
        assert result1 is not None
        assert "event: message_start" in result1
        assert '"input_tokens":0' in result1

        # Second chunk: more content
        chunk2 = 'data: {"choices":[{"delta":{"content":"lo"}}]}'
        result2 = transformer.transform(chunk2)
        assert result2 is not None
        assert "event: content_block_delta" in result2

        # Third chunk: finish reason
        chunk3 = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        result3 = transformer.transform(chunk3)
        assert result3 is not None
        assert "event: content_block_stop" in result3

        # Fourth chunk: usage (OpenAI sends this last with stream_options.include_usage)
        chunk4 = 'data: {"choices":[],"usage":{"prompt_tokens":45179,"completion_tokens":5}}'
        result4 = transformer.transform(chunk4)

        # Finalize
        final = transformer.finalize()

        # Combined output should have correct final usage
        combined = (result1 or "") + (result2 or "") + (result3 or "") + (result4 or "") + final
        assert "event: message_delta" in combined
        assert '"input_tokens":45179' in combined
        assert '"output_tokens":5' in combined

    def test_streaming_transformer_handles_finish_reason(self):
        """Test that streaming transformer handles finish_reason correctly."""
        transformer = AnthropicStreamingTransformer(model="claude-3-opus", request_id="msg_123")

        # Start with message
        chunk1 = (
            'data: {"id":"chatcmpl-abc","choices":'
            '[{"index":0,"delta":{"role":"assistant","content":""}}]}'
        )
        transformer.transform(chunk1)

        # Content
        chunk2 = 'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{"content":"Done"}}]}'
        transformer.transform(chunk2)

        # Finish with stop - stop_reason is cached, message_delta sent in finalize()
        chunk3 = (
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        )
        result3 = transformer.transform(chunk3)
        assert result3 is not None
        assert "event: content_block_stop" in result3

        # Finalize sends the message_delta with cached stop_reason
        final = transformer.finalize()
        assert "event: message_delta" in final
        assert '"stop_reason":"end_turn"' in final

    def test_streaming_transformer_filters_non_standard_fields(self):
        """Test that non-standard fields like chutes_verification are filtered out."""
        transformer = AnthropicStreamingTransformer(model="claude-3-opus", request_id="msg_123")

        chunk = (
            'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
            '"created":1234567890,"model":"claude-3-opus",'
            '"choices":[{"index":0,"delta":{"content":"test"}}],'
            '"chutes_verification":"abc123","prompt_sha256":"def456"}'
        )
        result = transformer.transform(chunk)
        assert result is not None

        # Should not contain non-standard fields
        assert "chutes_verification" not in result
        assert "prompt_sha256" not in result
        assert "logprobs" not in result

        # Should contain Anthropic events
        assert "event:" in result
        assert "message_start" in result or "content_block" in result

    def test_parse_request_with_stop_sequences(self):
        """Test parsing request with stop_sequences."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            stop_sequences=["END", "STOP"],
        )
        unified = _serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.anthropic is not None
        assert unified.params.anthropic.stop_sequences == ["END", "STOP"]

    def test_parse_request_with_thinking(self):
        """Test parsing request with thinking config."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=2000,
            messages=[{"role": "user", "content": "think about it"}],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
        unified = _serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.thinking is not None
        assert unified.params.thinking.type == "enabled"
        assert unified.params.thinking.budget_tokens == 1024

    def test_parse_request_with_metadata(self):
        """Test parsing request with metadata."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            metadata={"user_id": "123"},
        )
        unified = _serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.anthropic is not None
        assert unified.params.anthropic.metadata == {"user_id": "123"}

    def test_parse_request_with_container(self):
        """Test parsing request with container."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            container="container_123",
        )
        unified = _serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.anthropic is not None
        assert unified.params.anthropic.container == "container_123"

    def test_parse_request_with_inference_geo(self):
        """Test parsing request with inference_geo."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            inference_geo="us-west-2",
        )
        unified = _serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.anthropic is not None
        assert unified.params.anthropic.inference_geo == "us-west-2"

    def test_parse_request_with_cache_control(self):
        """Test parsing request with cache_control."""
        MessagesRequest = anthropic_protocol.request_model
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            cache_control={"type": "ephemeral"},
        )
        unified = _serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.anthropic is not None
        assert unified.params.anthropic.cache_control == {"type": "ephemeral"}

    def test_parse_request_thinking_budget_validation(self):
        """Test that thinking.budget_tokens bounds are enforced provider-side
        (Claude models only; see TestThinkingBudgetValidation in
        test_anthropic_wire_compatibility.py for the full matrix)."""

        MessagesRequest = anthropic_protocol.request_model

        # The client-facing schema no longer validates the budget: the check
        # moved to the Anthropic provider serializer so third-party
        # Anthropic-compatible upstreams keep accepting what they did before.
        parsed = MessagesRequest(
            model="claude-3-opus",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            thinking={"type": "enabled", "budget_tokens": 200},
        )
        assert parsed.model == "claude-3-opus"

    def test_additional_routes_count_tokens(self):
        """Test that additional_routes returns count_tokens endpoint."""
        routes = anthropic_protocol.additional_routes
        assert len(routes) == 1
        path, request_model, response_model, handler = routes[0]
        assert path == "/v1/messages/count_tokens"
        assert request_model.__name__ == "CountTokensRequest"
        assert response_model is not None
        if response_model is not None:
            assert response_model.__name__ == "CountTokensResponse"
        assert handler.__name__ == "handle_count_tokens"

    def test_count_tokens_simple(self):
        """Test count_tokens with simple message."""
        import asyncio
        from unittest.mock import MagicMock

        from llm_proxy.protocols.anthropic.handler import handle_count_tokens
        from llm_proxy.protocols.anthropic.schemas import CountTokensRequest

        request = CountTokensRequest(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Hello, world!"}],
        )
        mock_req = MagicMock()
        result = asyncio.run(handle_count_tokens(request, mock_req))
        assert result.input_tokens > 0
        assert "input_tokens" in result.model_dump()

    def test_count_tokens_with_system(self):
        """Test count_tokens with system prompt."""
        import asyncio
        from unittest.mock import MagicMock

        from llm_proxy.protocols.anthropic.handler import handle_count_tokens
        from llm_proxy.protocols.anthropic.schemas import CountTokensRequest

        request = CountTokensRequest(
            model="claude-3-opus",
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hi"}],
        )
        mock_req = MagicMock()
        result = asyncio.run(handle_count_tokens(request, mock_req))
        assert result.input_tokens > 0

    def test_count_tokens_with_tools(self):
        """Test count_tokens with tools."""
        import asyncio
        from unittest.mock import MagicMock

        from llm_proxy.protocols.anthropic.handler import handle_count_tokens
        from llm_proxy.protocols.anthropic.schemas import CountTokensRequest

        request = CountTokensRequest(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "input_schema": {"type": "object"},
                }
            ],
        )
        mock_req = MagicMock()
        result = asyncio.run(handle_count_tokens(request, mock_req))
        assert result.input_tokens > 0

    def test_streaming_transformer_single_message_delta_with_usage(self):
        """Test that stop_reason and usage are sent in a single message_delta event.

        This test captures the bug where OpenAI sends:
        1. A chunk with finish_reason="stop"
        2. A separate usage chunk

        Thetransformer should merge these into ONE message_delta event,
        not send two separate message_delta events.
        """
        transformer = AnthropicStreamingTransformer(model="gpt-4", request_id="msg_123")

        # Chunk 1: Initial role
        chunk1 = 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
        transformer.transform(chunk1)

        # Chunk 2: Content
        chunk2 = 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
        transformer.transform(chunk2)

        # Chunk 3: Finish reason (OpenAI sends this first)
        chunk3 = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        result3 = transformer.transform(chunk3)

        # Chunk 4: Usage chunk (OpenAI sends this separately)
        chunk4 = (
            'data: {"choices":[],"usage":'
            '{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}'
        )
        result4 = transformer.transform(chunk4)

        # Finalize should emit exactly one message_delta with latest usage.
        final = transformer.finalize()

        # Combine results
        combined = (result3 or "") + (result4 or "") + final

        # Count message_delta events - should be exactly 1
        message_delta_count = combined.count("event: message_delta")
        assert message_delta_count == 1, (
            f"Expected exactly 1 message_delta event, got {message_delta_count}. "
            f"stop_reason and usage should be merged."
        )

        # Verify the single message_delta contains both stop_reason and usage
        assert '"stop_reason":"end_turn"' in combined
        assert '"input_tokens":10' in combined
        assert '"output_tokens":5' in combined

    def test_streaming_transformer_merges_stop_reason_with_zero_usage(self):
        """Test that stop_reason still merges when usage chunk reports zero output tokens."""
        transformer = AnthropicStreamingTransformer(model="gpt-4", request_id="msg_123")

        transformer.transform('data: {"choices":[{"delta":{"role":"assistant"}}]}')
        transformer.transform('data: {"choices":[{"delta":{"content":"Hello"}}]}')

        chunk_finish = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        chunk_usage_zero = (
            'data: {"choices":[],"usage":'
            '{"prompt_tokens":10,"completion_tokens":0,"total_tokens":10}}'
        )

        result = (transformer.transform(chunk_finish) or "") + (
            transformer.transform(chunk_usage_zero) or ""
        )
        final = transformer.finalize()
        combined = result + final

        assert combined.count("event: message_delta") == 1
        assert '"stop_reason":"end_turn"' in combined
        assert '"input_tokens":10' in combined
        assert '"output_tokens":0' in combined

    def test_streaming_transformer_uses_latest_usage_when_usage_updates_after_finish(self):
        """Test that only final cumulative usage is emitted when usage appears multiple times."""
        transformer = AnthropicStreamingTransformer(model="gpt-4", request_id="msg_123")

        transformer.transform('data: {"choices":[{"delta":{"role":"assistant"}}]}')
        transformer.transform('data: {"choices":[{"delta":{"content":"Hello"}}]}')

        # Finish first, then provider sends updated cumulative usage.
        transformer.transform('data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}')
        usage_76 = 'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":76}}'
        usage_77 = 'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":77}}'
        transformer.transform(usage_76)
        transformer.transform(usage_77)

        final = transformer.finalize()

        assert final.count("event: message_delta") == 1
        assert '"stop_reason":"tool_use"' in final
        assert '"input_tokens":10' in final
        assert '"output_tokens":77' in final

    def test_streaming_transformer_closes_previous_tool_block_before_new_tool_block(self):
        """Test new tool_use block starts only after previous tool_use block is closed."""
        transformer = AnthropicStreamingTransformer(model="gpt-4", request_id="msg_123")

        transformer.transform('data: {"choices":[{"delta":{"role":"assistant"}}]}')

        chunk = (
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"id":"call_1","type":"function","index":0,'
            '"function":{"name":"Glob","arguments":"{\\"pattern\\":\\"a\\"}"}},'
            '{"id":"call_2","type":"function","index":1,'
            '"function":{"name":"Glob","arguments":"{\\"pattern\\":\\"b\\"}"}}]}}]}'
        )

        result = transformer.transform(chunk)
        assert result is not None
        assert result.count("event: content_block_start") == 2
        assert result.count("event: content_block_stop") == 1
        assert '"index":0,"content_block":{"type":"tool_use","id":"call_1"' in result
        assert '"index":1,"content_block":{"type":"tool_use","id":"call_2"' in result

    def test_streaming_transformer_emits_usage_without_stop_reason_on_finalize(self):
        """Test finalize emits usage-only message_delta when no finish_reason was received."""
        transformer = AnthropicStreamingTransformer(model="gpt-4", request_id="msg_123")

        transformer.transform('data: {"choices":[{"delta":{"role":"assistant"}}]}')
        transformer.transform('data: {"choices":[{"delta":{"content":"Hello"}}]}')
        usage_chunk = 'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}'
        transformer.transform(usage_chunk)

        final = transformer.finalize()

        assert "event: message_delta" in final
        assert '"input_tokens":12' in final
        assert '"output_tokens":3' in final
        assert "event: message_stop" in final

    def test_streaming_transformer_finalize_closes_open_block(self):
        """Test finalize emits content_block_stop when a block is still open."""
        transformer = AnthropicStreamingTransformer(model="gpt-4", request_id="msg_123")

        transformer.transform('data: {"choices":[{"delta":{"role":"assistant"}}]}')
        partial = transformer.transform('data: {"choices":[{"delta":{"reasoning_content":"I"}}]}')

        assert partial is not None
        assert "event: content_block_start" in partial
        assert "event: content_block_delta" in partial

        final = transformer.finalize()

        assert "event: content_block_stop" in final
        assert "event: message_stop" in final

    def test_streaming_transformer_no_null_stop_sequence(self):
        """Test that stop_sequence field is omitted when None, not sent as null.

        Anthropic API spec: stop_sequence should be omitted if not applicable,
        not sent as null.
        """
        transformer = AnthropicStreamingTransformer(model="gpt-4", request_id="msg_123")

        # Chunk 1: Initial
        chunk1 = 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
        transformer.transform(chunk1)

        # Chunk 2: Content
        chunk2 = 'data: {"choices":[{"delta":{"content":"test"}}]}'
        transformer.transform(chunk2)

        # Chunk 3: Finish
        chunk3 = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        result = transformer.transform(chunk3)

        # stop_sequence should NOT be in the output at all (not even as null)
        assert result is not None
        assert '"stop_sequence": null' not in result
        assert '"stop_sequence":' not in result

    def test_streaming_transformer_finish_reason_mapping(self):
        """Test all finish_reason mappings toAnthropic stop_reason."""
        # Test various finish_reasons
        test_cases = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("content_filter", "refusal"),
        ]

        for openai_reason, expected_anthropic in test_cases:
            # Reset transformer for each test
            transformer = AnthropicStreamingTransformer(model="gpt-4", request_id="msg_123")

            chunk1 = 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
            transformer.transform(chunk1)

            chunk2 = f'data: {{"choices":[{{"delta":{{}},"finish_reason":"{openai_reason}"}}]}}'
            result = transformer.transform(chunk2)

            # finish_reason is cached, so result is None or just content_block_stop
            # We need to call finalize() to get the message_delta
            final = transformer.finalize()

            combined = (result or "") + final
            assert f'"stop_reason":"{expected_anthropic}"' in combined, (
                f"OpenAI '{openai_reason}' should map toAnthropic '{expected_anthropic}'"
            )

    def test_format_response_with_tool_result_no_is_error_when_false(self):
        """Test that tool_result does not include is_error when False.

        Per Anthropic API spec, is_error should only be present when True,
        not when False.
        """
        response = InternalResponse(
            id="msg_123",
            model="claude-3-sonnet",
            output=[
                ToolResultBlock(tool_use_id="toolu_1", content="Success result", is_error=False)
            ],
        )

        result = _serializer.format_response(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "toolu_1"
        assert result["content"][0]["content"] == "Success result"
        assert "is_error" not in result["content"][0]

    def test_format_response_with_tool_result_has_is_error_when_true(self):
        """Test that tool_result includes is_error when True."""
        response = InternalResponse(
            id="msg_123",
            model="claude-3-sonnet",
            output=[ToolResultBlock(tool_use_id="toolu_1", content="Error result", is_error=True)],
        )

        result = _serializer.format_response(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "toolu_1"
        assert result["content"][0]["content"] == "Error result"
        assert result["content"][0]["is_error"] is True

    def test_format_response_with_nested_tool_result(self):
        """Test formatting response with nested content in tool_result."""
        response = InternalResponse(
            id="msg_123",
            model="claude-3-sonnet",
            output=[
                ToolResultBlock(
                    tool_use_id="toolu_1",
                    content=[TextBlock(text="Line 1"), TextBlock(text="Line 2")],
                    is_error=False,
                )
            ],
        )

        result = _serializer.format_response(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "toolu_1"
        assert isinstance(result["content"][0]["content"], list)
        assert len(result["content"][0]["content"]) == 2
        assert result["content"][0]["content"][0]["type"] == "text"
        assert "is_error" not in result["content"][0]

    def test_streaming_transformer_uses_estimated_input_tokens_as_fallback(self):
        """Test that message_start uses estimated_input_tokens when provider doesn't send usage.

        This handles the case where providers like Ollama don't return usage information.
        The transformer should use the pre-estimated input tokens as a fallback.
        """
        transformer = AnthropicStreamingTransformer(
            model="llama3.2",
            request_id="msg_123",
            estimated_input_tokens=500,  # Pre-estimated from request
        )

        # First chunk: content only, no usage (Ollama doesn't send usage)
        chunk1 = 'data: {"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}'
        result1 = transformer.transform(chunk1)

        # message_start should use estimated_input_tokens
        assert result1 is not None
        assert "event: message_start" in result1
        assert '"input_tokens":500' in result1
        assert '"output_tokens":0' in result1

        # Continue with content
        chunk2 = 'data: {"choices":[{"delta":{"content":"lo"}}]}'
        result2 = transformer.transform(chunk2)
        assert result2 is not None

        # Finish without any usage information
        chunk3 = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        result3 = transformer.transform(chunk3)

        # Finalize - should still have the estimated input_tokens
        final = transformer.finalize()
        combined = (result1 or "") + (result2 or "") + (result3 or "") + final

        # Final message_delta should still show estimated input_tokens
        assert "event: message_delta" in combined
        # input_tokens should still be 500 (no provider usage to override)
        assert '"input_tokens":500' in combined

    def test_streaming_transformer_provider_usage_overrides_estimated(self):
        """Test that provider-provided usage overrides estimated input tokens.

        When provider sends actual usage information, it should replace the estimate.
        """
        transformer = AnthropicStreamingTransformer(
            model="gpt-4",
            request_id="msg_123",
            estimated_input_tokens=500,  # Pre-estimated
        )

        # First chunk: content only
        chunk1 = 'data: {"choices":[{"delta":{"role":"assistant","content":"Hi"}}]}'
        result1 = transformer.transform(chunk1)

        # message_start should initially use estimated_input_tokens
        assert result1 is not None
        assert "event: message_start" in result1
        assert '"input_tokens":500' in result1

        # Later chunk: provider sends actual usage (different from estimate)
        chunk2 = 'data: {"choices":[],"usage":{"prompt_tokens":1234,"completion_tokens":10}}'
        result2 = transformer.transform(chunk2)

        # Finalize
        final = transformer.finalize()

        # Combined output should have provider's actual usage, not the estimate
        combined = (result1 or "") + (result2 or "") + final
        assert "event: message_delta" in combined
        assert '"input_tokens":1234' in combined
        assert '"output_tokens":10' in combined

    def test_streaming_transformer_accumulates_tool_use_block(self):
        """Test that streaming transformer accumulates ToolUseBlock in get_accumulated_output()."""
        transformer = AnthropicStreamingTransformer(model="claude-3-opus", request_id="msg_123")

        # Start with message
        chunk1 = (
            'data: {"id":"chatcmpl-abc","choices":'
            '[{"index":0,"delta":{"role":"assistant","content":""}}]}'
        )
        transformer.transform(chunk1)

        # Tool call start
        chunk2 = (
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":'
            '{"tool_calls":[{"id":"call_123","type":"function","index":0,'
            '"function":{"name":"web_search","arguments":""}}]}}]}'
        )
        transformer.transform(chunk2)

        # Tool call arguments
        chunk3 = (
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":'
            '{"tool_calls":[{"index":0,"function":'
            '{"arguments":"{\\"query\\": \\"test search\\"}"}}]}}]}'
        )
        transformer.transform(chunk3)

        # Finish
        chunk4 = 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}'
        transformer.transform(chunk4)

        # Check accumulated output
        output = transformer.get_accumulated_output()
        assert len(output) == 1
        assert isinstance(output[0], ToolUseBlock)
        assert output[0].id == "call_123"
        assert output[0].name == "web_search"
        assert output[0].input == {"query": "test search"}

    def test_streaming_transformer_accumulates_multiple_tool_use_blocks(self):
        """Test that streaming transformer accumulates multiple ToolUseBlocks."""
        transformer = AnthropicStreamingTransformer(model="claude-3-opus", request_id="msg_123")

        # Start with message
        transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":'
            '{"role":"assistant","content":""}}]}'
        )

        # First tool call
        transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":'
            '{"tool_calls":[{"id":"call_1","type":"function","index":0,'
            '"function":{"name":"web_search","arguments":""}}]}}]}'
        )
        transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":'
            '{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\": \\"1\\"}"}}]}}]}'
        )

        # Second tool call (closes first tool and opens second)
        transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":'
            '{"tool_calls":[{"id":"call_2","type":"function","index":1,'
            '"function":{"name":"get_weather","arguments":""}}]}}]}'
        )
        transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":'
            '{"tool_calls":[{"index":1,"function":{"arguments":"{\\"city\\": \\"NYC\\"}"}}]}}]}'
        )

        # Finish
        transformer.transform('data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}')

        # Check accumulated output
        output = transformer.get_accumulated_output()
        assert len(output) == 2
        tool1 = output[0]
        assert isinstance(tool1, ServerToolUseBlock)
        assert tool1.id == "call_1"
        assert tool1.name == "web_search"
        assert tool1.input == {"q": "1"}
        tool2 = output[1]
        assert isinstance(tool2, ToolUseBlock)
        assert tool2.id == "call_2"
        assert tool2.name == "get_weather"
        assert tool2.input == {"city": "NYC"}

    def test_web_search_tool_call_emitted_as_tool_use_when_not_intercepting(self):
        """When intercept_web_search=False, a WebSearch tool call is emitted as a
        regular ``tool_use`` block, not ``server_tool_use``.

        Otherwise clients that did not request proxy-side web search receive a bare
        ``server_tool_use`` (a server-side tool the proxy never fulfills) and fail to
        parse the tool call. Regression test for the kimi-k3 WebSearch case.
        """
        transformer = AnthropicStreamingTransformer(
            model="claude-3-5-sonnet",
            request_id="msg_123",
            intercept_web_search=False,
        )

        # Tool call start (provider-native WebSearch, CamelCase name)
        start = transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":'
            '{"tool_calls":[{"id":"call_1","type":"function","index":0,'
            '"function":{"name":"WebSearch","arguments":""}}]}}]}'
        )
        assert start is not None
        assert '"type":"tool_use"' in start
        assert '"type":"server_tool_use"' not in start

    def test_web_search_tool_call_emitted_as_server_tool_use_when_intercepting(self):
        """When intercept_web_search=True (proxy intercepts), a web_search tool call
        is still emitted as ``server_tool_use`` so the proxy can inject results inline.
        """
        transformer = AnthropicStreamingTransformer(
            model="claude-3-5-sonnet",
            request_id="msg_123",
            intercept_web_search=True,
        )

        start = transformer.transform(
            'data: {"id":"chatcmpl-abc","choices":[{"index":0,"delta":'
            '{"tool_calls":[{"id":"call_1","type":"function","index":0,'
            '"function":{"name":"web_search","arguments":""}}]}}]}'
        )
        assert start is not None
        assert '"type":"server_tool_use"' in start
