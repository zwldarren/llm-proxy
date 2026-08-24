"""Tests for Anthropic adapter streaming."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
    RequestMetadata,
    TextBlock,
)
from llm_proxy.providers.anthropic import AnthropicAdapter  # noqa: F401 - triggers registration


@pytest.fixture
def anthropic_adapter():
    """Create an Anthropic adapter for testing."""
    return AnthropicAdapter(
        api_key="test-key",
        base_url="https://api.anthropic.com",
    )


@pytest.fixture
def basic_request():
    """Create a basic chat request."""
    return InternalRequest(
        model="claude-sonnet-4-20250514",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(max_tokens=100),
        metadata=RequestMetadata(request_id="test_req_123"),
    )


class MockStreamResponse:
    """Mock HTTP response for streaming."""

    def __init__(self, lines: list[bytes], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def iter_lines(self):
        for line in self._lines:
            yield line


# Long JSON SSE data strings used across test methods
_MSG_START_DATA = (
    '{"type":"message_start","message":{"id":"msg_123","type":"message",'
    '"role":"assistant","content":[],"model":"claude-sonnet-4-20250514",'
    '"stop_reason":null,"stop_sequence":null,'
    '"usage":{"input_tokens":10,"output_tokens":1}}}'
)
_MSG_START_TOOL_DATA = (
    '{"type":"message_start","message":{"id":"msg_tool","type":"message",'
    '"role":"assistant","content":[],"model":"claude-sonnet-4-20250514",'
    '"stop_reason":null,"stop_sequence":null,'
    '"usage":{"input_tokens":50,"output_tokens":5}}}'
)
_MSG_DELTA_DATA = (
    '{"type":"message_delta","delta":{"stop_reason":"end_turn",'
    '"stop_sequence":null},"usage":{"output_tokens":12}}'
)
_MSG_DELTA_TOOL_DATA = (
    '{"type":"message_delta","delta":{"stop_reason":"tool_use",'
    '"stop_sequence":null},"usage":{"output_tokens":15}}'
)


def _make_sse_events(events: list[tuple[str, str]]) -> list[bytes]:
    lines = []
    for event_type, data in events:
        lines.append(f"event: {event_type}\n".encode())
        lines.append(f"data: {data}\n\n".encode())
    return lines


class TestAnthropicStreamingText:
    """Tests for text-only streaming."""

    @pytest.mark.asyncio
    async def test_simple_text_stream(self, anthropic_adapter, basic_request):
        """Anthropic text SSE events are converted to OpenAI chat.completion.chunk format."""
        sse_events = _make_sse_events(
            [
                ("message_start", _MSG_START_DATA),
                (
                    "content_block_start",
                    '{"type":"content_block_start","index":0,'
                    '"content_block":{"type":"text","text":""}}',
                ),
                (
                    "content_block_delta",
                    '{"type":"content_block_delta","index":0,'
                    '"delta":{"type":"text_delta","text":"Hello"}}',
                ),
                (
                    "content_block_delta",
                    '{"type":"content_block_delta","index":0,'
                    '"delta":{"type":"text_delta","text":" World"}}',
                ),
                ("content_block_stop", '{"type":"content_block_stop","index":0}'),
                ("message_delta", _MSG_DELTA_DATA),
                ("message_stop", '{"type":"message_stop"}'),
            ]
        )

        mock_response = MockStreamResponse(sse_events)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(anthropic_adapter, "_get_client", return_value=mock_client):
            stream_gen = await anthropic_adapter.stream_chat_completion(basic_request)
            chunks = []
            async for chunk in stream_gen:
                chunks.append(chunk)

        # First chunk should be message_start -> role chunk
        assert len(chunks) > 0, "Should yield at least one chunk"
        first = chunks[0]
        assert isinstance(first, dict), "Chunks should be dicts"
        assert "choices" in first, "Chunks should have choices"
        assert first["choices"][0]["delta"].get("role") == "assistant"
        assert first["choices"][0]["delta"].get("content") == ""

        # Content chunks
        text_chunks = [
            c
            for c in chunks
            if isinstance(c, dict) and c.get("choices", [{}])[0].get("delta", {}).get("content")
        ]
        texts = [c["choices"][0]["delta"]["content"] for c in text_chunks]
        assert "".join(texts) == "Hello World"

        # Last non-DONE chunk should have finish_reason
        non_done = [c for c in chunks if c != "[DONE]"]
        last = non_done[-1]
        assert last["choices"][0].get("finish_reason") == "stop"
        assert "usage" in last

        # Should end with [DONE]
        assert chunks[-1] == "[DONE]"

    @pytest.mark.asyncio
    async def test_streaming_request_body(self, anthropic_adapter, basic_request):
        """Streaming sends correct headers and body to Anthropic API."""
        sse_events = _make_sse_events(
            [
                ("message_start", _MSG_START_DATA),
                (
                    "content_block_start",
                    '{"type":"content_block_start","index":0,'
                    '"content_block":{"type":"text","text":""}}',
                ),
                (
                    "content_block_delta",
                    '{"type":"content_block_delta","index":0,'
                    '"delta":{"type":"text_delta","text":"Hi"}}',
                ),
                ("content_block_stop", '{"type":"content_block_stop","index":0}'),
                (
                    "message_delta",
                    '{"type":"message_delta","delta":{"stop_reason":'
                    '"end_turn"},"usage":{"output_tokens":5}}',
                ),
                ("message_stop", '{"type":"message_stop"}'),
            ]
        )

        mock_response = MockStreamResponse(sse_events)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(anthropic_adapter, "_get_client", return_value=mock_client):
            stream_gen = await anthropic_adapter.stream_chat_completion(basic_request)
            async for _ in stream_gen:
                pass

        # Verify the request
        call_args = mock_client.post.call_args
        assert call_args is not None, "Should have made HTTP request"
        url = call_args.args[0]
        headers = call_args.kwargs["headers"]
        body = call_args.kwargs["json"]

        assert anthropic_adapter.CHAT_ENDPOINT in url
        assert headers["x-api-key"] == "test-key"
        assert headers["anthropic-version"] == anthropic_adapter.EXTRA_HEADERS["anthropic-version"]
        assert body["model"] == "claude-sonnet-4-20250514"
        assert body["stream"] is True


class TestAnthropicStreamingToolUse:
    """Tests for tool use in streaming."""

    @pytest.mark.asyncio
    async def test_streaming_tool_use(self, anthropic_adapter, basic_request):
        """Tool use in streaming is converted to OpenAI tool_calls format."""
        sse_events = _make_sse_events(
            [
                ("message_start", _MSG_START_TOOL_DATA),
                (
                    "content_block_start",
                    '{"type":"content_block_start","index":0,'
                    '"content_block":{"type":"tool_use","id":"toolu_123",'
                    '"name":"get_weather","input":{}}}',
                ),
                (
                    "content_block_delta",
                    '{"type":"content_block_delta","index":0,'
                    '"delta":{"type":"input_json_delta",'
                    '"partial_json":"{\\"location\\": \\"San"}}',
                ),
                (
                    "content_block_delta",
                    '{"type":"content_block_delta","index":0,'
                    '"delta":{"type":"input_json_delta",'
                    '"partial_json":" Francisco\\"}"}}',
                ),
                ("content_block_stop", '{"type":"content_block_stop","index":0}'),
                ("message_delta", _MSG_DELTA_TOOL_DATA),
                ("message_stop", '{"type":"message_stop"}'),
            ]
        )

        mock_response = MockStreamResponse(sse_events)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(anthropic_adapter, "_get_client", return_value=mock_client):
            stream_gen = await anthropic_adapter.stream_chat_completion(basic_request)
            chunks = []
            async for chunk in stream_gen:
                chunks.append(chunk)

        # Find tool call chunks
        tool_call_chunks = []
        for c in chunks:
            if isinstance(c, dict):
                choices = c.get("choices", [])
                if choices and choices[0].get("delta", {}).get("tool_calls"):
                    tool_call_chunks.append(c)

        assert len(tool_call_chunks) > 0, "Should have tool call chunks"

        # First tool call chunk should have id and name
        first_tc = tool_call_chunks[0]
        tc_delta = first_tc["choices"][0]["delta"]["tool_calls"][0]
        assert tc_delta["id"] == "toolu_123"
        assert tc_delta["function"]["name"] == "get_weather"

        # Arguments should be accumulated across deltas
        all_args = ""
        for tc in tool_call_chunks:
            args = tc["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
            if args:
                all_args += args
        assert "San Francisco" in all_args

        # Final chunk should have finish_reason = tool_calls
        non_done = [c for c in chunks if c != "[DONE]"]
        last = non_done[-1]
        assert last["choices"][0].get("finish_reason") == "tool_calls"


class TestAnthropicStreamingErrors:
    """Tests for error handling in streaming."""

    @pytest.mark.asyncio
    async def test_streaming_http_error(self, anthropic_adapter, basic_request):
        """HTTP errors during streaming raise appropriate ProviderError."""
        from llm_proxy.core.exceptions import ProviderError

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = AsyncMock(return_value='{"error": {"message": "Rate limit exceeded"}}')
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(anthropic_adapter, "_get_client", return_value=mock_client):
            stream_gen = await anthropic_adapter.stream_chat_completion(basic_request)
            with pytest.raises(ProviderError):
                async for _ in stream_gen:
                    pass

    @pytest.mark.asyncio
    async def test_cancel_token_stops_stream(self, anthropic_adapter, basic_request):
        """Cancel token stops stream iteration."""
        import asyncio

        cancel_token = asyncio.Event()

        _msg_start = (
            b'data: {"type":"message_start","message":{"id":"msg_1",'
            b'"type":"message","role":"assistant","content":[],'
            b'"model":"claude",'
            b'"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":1}}}'
            b"\n\n"
        )
        _cb_start = (
            b'data: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}'
            b"\n\n"
        )
        _cb_delta = (
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"text_delta","text":"Hello"}}'
            b"\n\n"
        )

        mock_response = MockStreamResponse(
            [
                b"event: message_start\n",
                _msg_start,
                b"event: content_block_start\n",
                _cb_start,
                b"event: content_block_delta\n",
                _cb_delta,
            ]
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(anthropic_adapter, "_get_client", return_value=mock_client):
            stream_gen = await anthropic_adapter.stream_chat_completion(
                basic_request, cancel_token=cancel_token
            )
            chunks = []
            async for chunk in stream_gen:
                chunks.append(chunk)
                # Signal cancellation after first chunk
                cancel_token.set()

        # Should have yielded at least one chunk before cancellation
        assert len(chunks) > 0, "Should yield some chunks before cancellation"
        # Should NOT end with [DONE] (stream was cancelled)
        assert chunks[-1] != "[DONE]"


class TestAnthropicNativeStreaming:
    """Tests for protocol-native streaming passthrough."""

    def test_supports_native_streaming(self, anthropic_adapter):
        """Native streaming is enabled only for the anthropic protocol."""
        assert anthropic_adapter.supports_native_streaming("anthropic") is True
        assert anthropic_adapter.supports_native_streaming("openai") is False
        assert anthropic_adapter.supports_native_streaming("") is False

    @pytest.mark.asyncio
    async def test_stream_chat_completion_native_yields_raw_sse_frames(
        self, anthropic_adapter, basic_request
    ):
        """Native streaming yields complete raw SSE event frames."""
        sse_events = _make_sse_events(
            [
                ("message_start", _MSG_START_DATA),
                (
                    "content_block_delta",
                    '{"type":"content_block_delta","index":0,'
                    '"delta":{"type":"text_delta","text":"Hello"}}',
                ),
                ("message_stop", '{"type":"message_stop"}'),
            ]
        )

        mock_response = MockStreamResponse(sse_events)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(anthropic_adapter, "_get_client", return_value=mock_client):
            stream_gen = await anthropic_adapter.stream_chat_completion_native(basic_request)
            frames = []
            async for frame in stream_gen:
                frames.append(frame)

        assert len(frames) > 0, "Should yield at least one frame"
        # Each frame should be a complete SSE event block ending with blank line
        assert all(frame.endswith("\n\n") for frame in frames)
        assert any("event: message_start" in f for f in frames)
        assert any("data:" in f for f in frames)
        # No OpenAI dict conversion should happen in native mode
        assert not any(isinstance(f, dict) for f in frames)
