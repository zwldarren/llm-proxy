"""Streaming tests for /v1/chat/completions proxy endpoint.

Tests streaming functionality including tool calling in streaming mode,
which is a complex scenario where tool calls are built incrementally
through multiple SSE chunks.
"""

from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from llm_proxy.models import (
    ConversationContext,
    FunctionTool,
    InternalRequest,
    Message,
    TextBlock,
)


class MockResponse:
    """Mock HTTP response for AsyncSession.

    For non-streaming responses, json() is synchronous in httpx2.
    For streaming responses (error handling), json() would be async.
    """

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict | None = None,
        stream_chunks: Sequence[str | bytes] | None = None,
    ):
        self.status_code = status_code
        self._json_data = json_data
        self._stream_chunks = stream_chunks

    def json(self):
        """Return JSON data (synchronous for non-streaming responses)."""
        return self._json_data or {}

    def iter_lines(self):
        """httpx2 uses iter_lines() for async iteration."""
        return MockAsyncIterator(self._stream_chunks)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class MockAsyncIterator:
    """Async iterator for streaming response content."""

    def __init__(self, chunks: Sequence[str | bytes] | None = None):
        self._chunks = chunks or []
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        if isinstance(chunk, str):
            return chunk.encode()
        return chunk


class TestStreamingChatCompletionsOpenAI:
    """Streaming tests for OpenAI provider chat completions."""

    @pytest.fixture
    def openai_adapter(self):
        """Create an OpenAI adapter for testing."""
        from llm_proxy.providers.openai_compatible import OpenAICompatibleBase

        return OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

    @pytest.mark.asyncio
    async def test_streaming_simple_response(self, openai_adapter):
        """Test simple streaming text response.

        The adapter yields dict chunks for single-serialization path.
        "[DONE]" marker is yielded as a string.
        """
        streaming_chunks = [
            'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","model":"gpt-4",'
            '"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","model":"gpt-4",'
            '"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","model":"gpt-4",'
            '"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","model":"gpt-4",'
            '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, stream_chunks=streaming_chunks)
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hi")])]
                ),
                stream=True,
            )

            chunks = []
            async for chunk in await openai_adapter.stream_chat_completion(request):
                chunks.append(chunk)

            # Verify chunks are received: 4 dict chunks + 1 "[DONE]" string = 5
            assert len(chunks) == 5
            # Verify content is accumulated from dict chunks
            full_content = ""
            for chunk in chunks:
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            full_content += delta["content"]

            assert full_content == "Hello there!"

    @pytest.mark.asyncio
    async def test_streaming_tool_call_single(self, openai_adapter):
        """Test streaming response with single tool call.

        Tool calls in streaming mode are built incrementally:
        1. First chunk: tool call id, type, and function name
        2. Subsequent chunks: function arguments streamed character by character
        3. Final chunk: finish_reason='tool_calls'

        See: https://platform.openai.com/docs/api-reference/streaming#streaming/mode
        """
        streaming_chunks = [
            # Tool call start - includes id, type, and function name
            (
                'data: {"id":"chatcmpl-tool-stream","object":"chat.completion.chunk",'
                '"model":"gpt-4-turbo","choices":[{"index":0,"delta":{"tool_calls":'
                '[{"index":0,"id":"call_stream_1","type":"function","function":'
                '{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n'
            ),
            # Tool call argument streaming - arguments are JSON string chunks
            (
                'data: {"id":"chatcmpl-tool-stream","object":"chat.completion.chunk",'
                '"model":"gpt-4-turbo","choices":[{"index":0,"delta":{"tool_calls":'
                '[{"index":0,"function":{"arguments":"{\\"location\\": \\"San Francisco\\"}"}}'
                ']},"finish_reason":null}]}\n\n'
            ),
            # Tool call complete
            (
                'data: {"id":"chatcmpl-tool-stream","object":"chat.completion.chunk",'
                '"model":"gpt-4-turbo","choices":[{"index":0,"delta":{},'
                '"finish_reason":"tool_calls"}]}\n\n'
            ),
            "data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, stream_chunks=streaming_chunks)
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4-turbo",
                conversation=ConversationContext(
                    messages=[
                        Message(role="user", content=[TextBlock(text="What's the weather in SF?")])
                    ]
                ),
                tools=[FunctionTool(name="get_weather", parameters={"type": "object"})],
                stream=True,
            )

            chunks = []
            async for chunk in await openai_adapter.stream_chat_completion(request):
                chunks.append(chunk)

            # Verify tool call chunks (now dicts instead of SSE strings)
            tool_call_started = False
            accumulated_args = ""

            for chunk in chunks:
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        tool_calls = delta.get("tool_calls", [])
                        for tc in tool_calls:
                            if tc.get("id"):
                                tool_call_started = True
                                assert tc["id"] == "call_stream_1"
                                assert tc["type"] == "function"
                            if tc.get("function", {}).get("name"):
                                assert tc["function"]["name"] == "get_weather"
                            if tc.get("function", {}).get("arguments"):
                                accumulated_args += tc["function"]["arguments"]

            assert tool_call_started
            # Verify the accumulated arguments form valid JSON
            assert accumulated_args == '{"location": "San Francisco"}'
            parsed_args = orjson.loads(accumulated_args)
            assert parsed_args == {"location": "San Francisco"}

    @pytest.mark.asyncio
    async def test_streaming_tool_call_multiple(self, openai_adapter):
        """Test streaming response with multiple tool calls (parallel function calls).

        When multiple tool calls are returned, they are interleaved in the stream
        with each chunk potentially containing delta for different tool calls.
        """
        streaming_chunks = [
            # First tool call start
            'data: {"id":"chatcmpl-multi","object":"chat.completion.chunk","model":"gpt-4-turbo",'
            '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            '"type":"function","function":{"name":"get_weather","arguments":""}}]},'
            '"finish_reason":null}]}\n\n',
            # Second tool call start (interleaved)
            'data: {"id":"chatcmpl-multi","object":"chat.completion.chunk","model":"gpt-4-turbo",'
            '"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2",'
            '"type":"function","function":{"name":"get_weather","arguments":""}}]},'
            '"finish_reason":null}]}\n\n',
            # First tool call arguments
            'data: {"id":"chatcmpl-multi","object":"chat.completion.chunk","model":"gpt-4-turbo",'
            '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{'
            '"arguments":"{\\"location\\": \\"SF\\"}"}}]},"finish_reason":null}]}\n\n',
            # Second tool call arguments
            'data: {"id":"chatcmpl-multi","object":"chat.completion.chunk","model":"gpt-4-turbo",'
            '"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{'
            '"arguments":"{\\"location\\": \\"NYC\\"}"}}]},"finish_reason":null}]}\n\n',
            # Complete
            'data: {"id":"chatcmpl-multi","object":"chat.completion.chunk","model":"gpt-4-turbo",'
            '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, stream_chunks=streaming_chunks)
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4-turbo",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user", content=[TextBlock(text="Compare weather in SF and NYC")]
                        )
                    ]
                ),
                tools=[FunctionTool(name="get_weather", parameters={"type": "object"})],
                stream=True,
            )

            tool_call_args = {0: "", 1: ""}

            async for chunk in await openai_adapter.stream_chat_completion(request):
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        tool_calls = delta.get("tool_calls", [])
                        for tc in tool_calls:
                            idx = tc.get("index")
                            if idx is not None and tc.get("function", {}).get("arguments"):
                                tool_call_args[idx] += tc["function"]["arguments"]

            # Verify both tool calls were received
            assert tool_call_args[0] == '{"location": "SF"}'
            assert tool_call_args[1] == '{"location": "NYC"}'

    @pytest.mark.asyncio
    async def test_streaming_with_usage_chunk(self, openai_adapter):
        """Test streaming response with usage information.

        OpenAI includes usage in the final chunk when stream_options.include_usage=true.
        """
        streaming_chunks = [
            'data: {"id":"chatcmpl-usage","object":"chat.completion.chunk","model":"gpt-4",'
            '"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-usage","object":"chat.completion.chunk","model":"gpt-4",'
            '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: {"id":"chatcmpl-usage","object":"chat.completion.chunk","model":"gpt-4",'
            '"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, stream_chunks=streaming_chunks)
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hi")])]
                ),
                stream=True,
            )

            usage_found = False
            async for chunk in await openai_adapter.stream_chat_completion(request):
                if isinstance(chunk, dict) and "usage" in chunk:
                    assert chunk["usage"]["prompt_tokens"] == 10
                    assert chunk["usage"]["completion_tokens"] == 5
                    usage_found = True

            assert usage_found, "Usage chunk not found in stream"


class TestStreamingChatCompletionsGemini:
    """Streaming tests for Gemini provider chat completions."""

    @pytest.fixture
    def gemini_adapter(self):
        """Create a Gemini adapter for testing."""
        from llm_proxy.providers.gemini import GeminiAdapter

        return GeminiAdapter(
            api_key="test-key",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )

    @pytest.mark.asyncio
    async def test_streaming_simple_response(self, gemini_adapter):
        """Test Gemini streaming response conversion to OpenAI format."""
        # Gemini SSE format (with alt=sse)
        streaming_chunks = [
            'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},'
            '"finishReason":"STOP"}]}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, stream_chunks=streaming_chunks)
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.gemini.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            gemini_adapter._http_client = mock_client

            request = InternalRequest(
                model="gemini-pro",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hi")])]
                ),
                stream=True,
            )

            chunks = []
            async for chunk in await gemini_adapter.stream_chat_completion(request):
                chunks.append(chunk)

            # Adapter now yields OpenAI-format dicts, with "[DONE]" as stream terminator
            assert len(chunks) >= 1
            for chunk in chunks:
                if isinstance(chunk, dict):
                    assert "choices" in chunk
                    assert "delta" in chunk["choices"][0]


class TestStreamingChatCompletionsOllama:
    """Streaming tests for Ollama provider chat completions."""

    @pytest.fixture
    def ollama_adapter(self):
        """Create an Ollama adapter for testing."""
        from llm_proxy.providers.ollama import OllamaAdapter

        return OllamaAdapter(
            api_key="test-key",
            base_url="http://localhost:11434",
        )

    @pytest.mark.asyncio
    async def test_streaming_simple_response(self, ollama_adapter):
        """Test Ollama streaming response (native format converted to OpenAI)."""
        # Ollama native streaming format (JSON lines)
        native_chunks = [
            orjson.dumps(
                {
                    "model": "llama3",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "Hello"},
                    "done": False,
                }
            )
            + b"\n",
            orjson.dumps(
                {
                    "model": "llama3",
                    "created_at": "2024-01-01T00:00:01Z",
                    "message": {"role": "assistant", "content": " from Ollama"},
                    "done": False,
                }
            )
            + b"\n",
            orjson.dumps(
                {
                    "model": "llama3",
                    "created_at": "2024-01-01T00:00:02Z",
                    "message": {"role": "assistant", "content": "!"},
                    "done": True,
                    "done_reason": "stop",
                }
            )
            + b"\n",
        ]

        mock_response = MockResponse(200, stream_chunks=native_chunks)
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.ollama.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            ollama_adapter._http_client = mock_client

            request = InternalRequest(
                model="llama3",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hi")])]
                ),
                stream=True,
            )

            chunks = []
            async for chunk in await ollama_adapter.stream_chat_completion(request):
                chunks.append(chunk)

            full_content = ""
            for chunk in chunks:
                if isinstance(chunk, dict) and chunk.get("choices"):
                    content = chunk["choices"][0].get("delta", {}).get("content")
                    if content:
                        full_content += content

            assert full_content == "Hello from Ollama!"

    @pytest.mark.asyncio
    async def test_streaming_native_format(self, ollama_adapter):
        """Test Ollama native streaming format conversion via serializer."""
        # This test validates the OllamaProviderSerializer's convert_native_chunk method
        # which is used when processing streaming chunks from Ollama's native API
        from llm_proxy.serialization.ollama.serializer import OllamaProviderSerializer

        serializer = OllamaProviderSerializer()

        # Test native format chunk conversion
        native_chunk = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Hello"},
            "done": False,
        }

        converted = serializer.convert_native_chunk(native_chunk)
        assert "choices" in converted  # Should be in OpenAI format
        assert converted["choices"][0]["delta"]["content"] == "Hello"

        # Test final chunk with done=True
        final_chunk = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:02Z",
            "message": {"role": "assistant", "content": "!"},
            "done": True,
            "done_reason": "stop",
        }

        converted_final = serializer.convert_native_chunk(final_chunk)
        assert "choices" in converted_final
        assert converted_final["choices"][0]["finish_reason"] == "stop"

        # Test chunk with tool calls
        tool_chunk = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "test", "arguments": {"arg": "value"}},
                    }
                ],
            },
            "done": False,
        }

        converted_tool = serializer.convert_native_chunk(tool_chunk)
        assert "choices" in converted_tool
        delta = converted_tool["choices"][0]["delta"]
        assert "tool_calls" in delta
        assert delta["tool_calls"][0]["function"]["name"] == "test"


class TestStreamingEdgeCases:
    """Tests for streaming edge cases and error handling."""

    @pytest.fixture
    def openai_adapter(self):
        """Create an OpenAI adapter for testing."""
        from llm_proxy.providers.openai_compatible import OpenAICompatibleBase

        return OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

    @pytest.mark.asyncio
    async def test_streaming_with_empty_content_delta(self, openai_adapter):
        """Test handling of empty content deltas in streaming.

        Some providers send empty content deltas which should be filtered out.
        """
        streaming_chunks = [
            'data: {"id":"chatcmpl-empty","object":"chat.completion.chunk","model":"gpt-4",'
            '"choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-empty","object":"chat.completion.chunk","model":"gpt-4",'
            '"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-empty","object":"chat.completion.chunk","model":"gpt-4",'
            '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, stream_chunks=streaming_chunks)
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hi")])]
                ),
                stream=True,
            )

            content_chunks = 0
            async for chunk in await openai_adapter.stream_chat_completion(request):
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if (
                        choices
                        and choices[0].get("delta", {}).get("content")
                        and choices[0]["delta"]["content"]  # Non-empty
                    ):
                        content_chunks += 1

            assert content_chunks == 1  # Only "Hello" should be counted

    @pytest.mark.asyncio
    async def test_streaming_tool_call_incomplete(self, openai_adapter):
        """Test handling of incomplete tool call chunks.

        Some providers may send incomplete tool call data that should be filtered.
        """
        streaming_chunks = [
            # Incomplete tool call (missing function field) - should be filtered
            (
                'data: {"id":"chatcmpl-incomplete","object":"chat.completion.chunk",'
                '"model":"gpt-4-turbo","choices":[{"index":0,"delta":{"tool_calls":'
                '[{"index":0,"type":"function"}]},"finish_reason":null}]}\n\n'
            ),
            # Complete tool call
            (
                'data: {"id":"chatcmpl-incomplete","object":"chat.completion.chunk",'
                '"model":"gpt-4-turbo","choices":[{"index":0,"delta":{"tool_calls":'
                '[{"index":0,"id":"call_complete","type":"function","function":'
                '{"name":"test","arguments":""}}]},"finish_reason":null}]}\n\n'
            ),
            (
                'data: {"id":"chatcmpl-incomplete","object":"chat.completion.chunk",'
                '"model":"gpt-4-turbo","choices":[{"index":0,"delta":{},'
                '"finish_reason":"tool_calls"}]}\n\n'
            ),
            "data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, stream_chunks=streaming_chunks)
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4-turbo",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Test")])]
                ),
                tools=[FunctionTool(name="test", parameters={"type": "object"})],
                stream=True,
            )

            valid_tool_calls = 0
            async for chunk in await openai_adapter.stream_chat_completion(request):
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        tool_calls = choices[0].get("delta", {}).get("tool_calls", [])
                        for tc in tool_calls:
                            # Only count tool calls with valid function field
                            if tc.get("function"):
                                valid_tool_calls += 1
                                assert tc["id"] == "call_complete"

            assert valid_tool_calls == 1

    @pytest.mark.asyncio
    async def test_streaming_with_reasoning_content(self, openai_adapter):
        """Test streaming with reasoning_content (o1-style models).

        Some models return reasoning_content for thinking models.
        """
        streaming_chunks = [
            (
                'data: {"id":"chatcmpl-reasoning","object":"chat.completion.chunk",'
                '"model":"o1","choices":[{"index":0,"delta":'
                '{"reasoning_content":"Let me think..."},"finish_reason":null}]}\n\n'
            ),
            (
                'data: {"id":"chatcmpl-reasoning","object":"chat.completion.chunk",'
                '"model":"o1","choices":[{"index":0,"delta":{"content":"The answer is 42."},'
                '"finish_reason":null}]}\n\n'
            ),
            (
                'data: {"id":"chatcmpl-reasoning","object":"chat.completion.chunk",'
                '"model":"o1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            ),
            "data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, stream_chunks=streaming_chunks)
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="o1",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="What's the answer?")])]
                ),
                stream=True,
            )

            reasoning_found = False
            content_found = False

            async for chunk in await openai_adapter.stream_chat_completion(request):
                if isinstance(chunk, dict):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("reasoning_content"):
                            reasoning_found = True
                            assert delta["reasoning_content"] == "Let me think..."
                        if delta.get("content"):
                            content_found = True
                            assert delta["content"] == "The answer is 42."

            assert reasoning_found, "Reasoning content not found"
            assert content_found, "Content not found"
