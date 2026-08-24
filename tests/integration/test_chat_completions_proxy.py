"""Integration tests for /v1/chat/completions proxy endpoint.

Tests the proxy functionality across OpenAI, Gemini, and Ollama providers,
including complex scenarios like tool calling, multi-turn conversations,
and streaming responses.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from llm_proxy.models import (
    ConversationContext,
    FunctionTool,
    InternalRequest,
    InternalResponse,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


class MockResponse:
    """Mock HTTP response for AsyncSession.

    For non-streaming responses, json() is synchronous in httpx2.
    """

    def __init__(self, status_code: int = 200, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        """Return JSON data (synchronous for non-streaming responses)."""
        return self._json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def setup_mock_adapter(adapter, _module_path: str):
    """Helper to set up mock HTTP client for an adapter."""
    mock_client = MagicMock()
    adapter._http_client = mock_client
    return mock_client


class TestOpenAIChatCompletionsProxy:
    """Tests for OpenAI provider chat completions through the proxy."""

    @pytest.fixture
    def openai_adapter(self):
        """Create an OpenAI adapter for testing."""
        from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase

        return OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

    @pytest.mark.asyncio
    async def test_simple_chat_completion(self, openai_adapter):
        """Test simple chat completion request/response."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help you?",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hello")])]
                ),
            )
            response = await openai_adapter.chat_completion(request)

            assert isinstance(response, InternalResponse)
            assert len(response.output) == 1
            assert isinstance(response.output[0], TextBlock)
            assert response.output[0].text == "Hello! How can I help you?"
            assert response.model == "gpt-4"
            assert response.finish_reason == "stop"
            assert response.usage is not None
            assert response.usage.input_tokens == 10
            assert response.usage.output_tokens == 8

    @pytest.mark.asyncio
    async def test_chat_completion_with_tool_call(self, openai_adapter):
        """Test chat completion with function calling."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "San Francisco, CA"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            },
        )

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
            )
            response = await openai_adapter.chat_completion(request)

            assert isinstance(response, InternalResponse)
            assert len(response.output) == 1
            assert isinstance(response.output[0], ToolUseBlock)
            assert response.output[0].id == "call_abc123"
            assert response.output[0].name == "get_weather"
            assert response.output[0].input == {"location": "San Francisco, CA"}
            assert response.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_chat_completion_parallel_tool_calls(self, openai_adapter):
        """Test chat completion with parallel function calls."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-456",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "San Francisco"}',
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "New York"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 60,
                    "completion_tokens": 40,
                    "total_tokens": 100,
                },
            },
        )

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
                            role="user",
                            content=[TextBlock(text="Compare weather in SF and NYC")],
                        )
                    ]
                ),
                tools=[
                    FunctionTool(name="get_weather", parameters={"type": "object"}),
                ],
            )
            response = await openai_adapter.chat_completion(request)

            assert len(response.output) == 2
            assert all(isinstance(block, ToolUseBlock) for block in response.output)
            assert response.output[0].name == "get_weather"
            assert response.output[0].input == {"location": "San Francisco"}
            assert response.output[1].name == "get_weather"
            assert response.output[1].input == {"location": "New York"}

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_with_tool_results(self, openai_adapter):
        """Test multi-turn conversation with tool call results."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-789",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "The weather in San Francisco is currently "
                                "72°F and sunny. Perfect day to be outside!"
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "total_tokens": 125,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4-turbo",
                conversation=ConversationContext(
                    messages=[
                        Message(role="user", content=[TextBlock(text="What's the weather in SF?")]),
                        Message(
                            role="assistant",
                            content=[
                                ToolUseBlock(
                                    id="call_abc123",
                                    name="get_weather",
                                    input={"location": "San Francisco"},
                                )
                            ],
                        ),
                        Message(
                            role="tool",
                            content=[
                                ToolResultBlock(
                                    tool_use_id="call_abc123",
                                    content="72°F, sunny",
                                    is_error=False,
                                )
                            ],
                        ),
                    ]
                ),
            )
            response = await openai_adapter.chat_completion(request)

            assert isinstance(response.output[0], TextBlock)
            assert "72°F" in response.output[0].text
            assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_choice_auto(self, openai_adapter):
        """Test tool_choice='auto' parameter."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-tool-auto",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "I'd be happy to help with the weather. "
                                "What location are you interested in?"
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4-turbo",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="What's the weather?")])]
                ),
                tools=[FunctionTool(name="get_weather", parameters={"type": "object"})],
            )
            response = await openai_adapter.chat_completion(request)

            assert response.finish_reason == "stop"
            assert isinstance(response.output[0], TextBlock)

    @pytest.mark.asyncio
    async def test_tool_choice_required(self, openai_adapter):
        """Test tool_choice='required' parameter."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-tool-required",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_required",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "latest news"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4-turbo",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Find latest news")])]
                ),
                tools=[
                    FunctionTool(name="search", parameters={"type": "object"}),
                ],
            )
            response = await openai_adapter.chat_completion(request)

            assert response.finish_reason == "tool_calls"
            assert isinstance(response.output[0], ToolUseBlock)

    @pytest.mark.asyncio
    async def test_complex_nested_parameters(self, openai_adapter):
        """Test function calling with complex nested parameters."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-complex",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_complex",
                                    "type": "function",
                                    "function": {
                                        "name": "create_order",
                                        "arguments": orjson.dumps(
                                            {
                                                "customer": {
                                                    "name": "John Doe",
                                                    "email": "john@example.com",
                                                    "address": {
                                                        "street": "123 Main St",
                                                        "city": "San Francisco",
                                                        "zip": "94102",
                                                    },
                                                },
                                                "items": [
                                                    {
                                                        "product_id": "prod_123",
                                                        "quantity": 2,
                                                        "price": 29.99,
                                                    },
                                                    {
                                                        "product_id": "prod_456",
                                                        "quantity": 1,
                                                        "price": 49.99,
                                                    },
                                                ],
                                                "shipping_method": "express",
                                            }
                                        ).decode(),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 60,
                    "total_tokens": 140,
                },
            },
        )

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
                            role="user",
                            content=[TextBlock(text="Create an order for John Doe with 2 items")],
                        )
                    ]
                ),
                tools=[
                    FunctionTool(
                        name="create_order",
                        parameters={
                            "type": "object",
                            "properties": {
                                "customer": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "email": {"type": "string"},
                                        "address": {
                                            "type": "object",
                                            "properties": {
                                                "street": {"type": "string"},
                                                "city": {"type": "string"},
                                                "zip": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "product_id": {"type": "string"},
                                            "quantity": {"type": "integer"},
                                            "price": {"type": "number"},
                                        },
                                    },
                                },
                            },
                        },
                    )
                ],
            )
            response = await openai_adapter.chat_completion(request)

            tool_block = response.output[0]
            assert isinstance(tool_block, ToolUseBlock)
            assert tool_block.name == "create_order"
            assert "customer" in tool_block.input
            assert tool_block.input["customer"]["name"] == "John Doe"
            assert len(tool_block.input["items"]) == 2


class TestGeminiChatCompletionsProxy:
    """Tests for Gemini provider chat completions through the proxy."""

    @pytest.fixture
    def gemini_adapter(self):
        """Create a Gemini adapter for testing."""
        from llm_proxy.providers.gemini import GeminiAdapter

        return GeminiAdapter(
            api_key="test-key",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )

    @pytest.mark.asyncio
    async def test_simple_chat_completion(self, gemini_adapter):
        """Test simple chat completion with Gemini provider."""
        mock_response = MockResponse(
            200,
            json_data={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Hello from Gemini!"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.gemini.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            gemini_adapter._http_client = mock_client

            request = InternalRequest(
                model="gemini-pro",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hello")])]
                ),
            )
            response = await gemini_adapter.chat_completion(request)

            assert isinstance(response, InternalResponse)
            assert isinstance(response.output[0], TextBlock)
            assert response.output[0].text == "Hello from Gemini!"
            assert response.usage is not None
            assert response.usage.input_tokens == 10
            assert response.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_chat_completion_with_function_call(self, gemini_adapter):
        """Test Gemini function calling."""
        mock_response = MockResponse(
            200,
            json_data={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"location": "San Francisco"},
                                    }
                                }
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 50,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 70,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.gemini.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            gemini_adapter._http_client = mock_client

            request = InternalRequest(
                model="gemini-pro",
                conversation=ConversationContext(
                    messages=[
                        Message(role="user", content=[TextBlock(text="What's the weather in SF?")])
                    ]
                ),
                tools=[
                    FunctionTool(
                        name="get_weather",
                        description="Get weather for a location",
                        parameters={
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                        },
                    )
                ],
            )
            response = await gemini_adapter.chat_completion(request)

            assert isinstance(response.output[0], ToolUseBlock)
            assert response.output[0].name == "get_weather"
            assert response.output[0].input == {"location": "San Francisco"}


class TestOllamaChatCompletionsProxy:
    """Tests for Ollama provider chat completions through the proxy."""

    @pytest.fixture
    def ollama_adapter(self):
        """Create an Ollama adapter for testing."""
        from llm_proxy.providers.ollama import OllamaAdapter

        return OllamaAdapter(
            api_key="test-key",
            base_url="http://localhost:11434",
        )

    @pytest.mark.asyncio
    async def test_simple_chat_completion(self, ollama_adapter):
        """Test simple chat completion with Ollama provider."""
        mock_response = MockResponse(
            200,
            json_data={
                "model": "llama3",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "Hello from Ollama!",
                },
                "done": True,
                "done_reason": "stop",
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.ollama.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            ollama_adapter._http_client = mock_client

            request = InternalRequest(
                model="llama3",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hello")])]
                ),
            )
            response = await ollama_adapter.chat_completion(request)

            assert isinstance(response, InternalResponse)
            assert isinstance(response.output[0], TextBlock)
            assert response.output[0].text == "Hello from Ollama!"

    @pytest.mark.asyncio
    async def test_chat_completion_with_tool_call(self, ollama_adapter):
        """Test Ollama tool calling (native format)."""
        mock_response = MockResponse(
            200,
            json_data={
                "model": "llama3.1",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_ollama_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"location": "San Francisco"},
                            },
                        }
                    ],
                },
                "done": True,
                "done_reason": "tool_calls",
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.ollama.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            ollama_adapter._http_client = mock_client

            request = InternalRequest(
                model="llama3.1",
                conversation=ConversationContext(
                    messages=[
                        Message(role="user", content=[TextBlock(text="What's the weather in SF?")])
                    ]
                ),
                tools=[
                    FunctionTool(
                        name="get_weather",
                        parameters={
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name"},
                            },
                            "required": ["location"],
                        },
                    )
                ],
            )
            response = await ollama_adapter.chat_completion(request)

            assert isinstance(response.output[0], ToolUseBlock)
            assert response.output[0].name == "get_weather"
            assert response.output[0].input == {"location": "San Francisco"}
            assert response.finish_reason == "tool_calls"


class TestToolCallingEdgeCases:
    """Tests for edge cases in tool calling functionality."""

    @pytest.fixture
    def openai_adapter(self):
        """Create an OpenAI adapter for testing."""
        from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase

        return OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

    @pytest.mark.asyncio
    async def test_tool_call_with_empty_arguments(self, openai_adapter):
        """Test tool call with no arguments (empty object)."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-empty-args",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_empty",
                                    "type": "function",
                                    "function": {
                                        "name": "get_current_time",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4-turbo",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="What time is it?")])]
                ),
                tools=[FunctionTool(name="get_current_time", parameters={"type": "object"})],
            )
            response = await openai_adapter.chat_completion(request)

            tool_block = response.output[0]
            assert isinstance(tool_block, ToolUseBlock)
            assert tool_block.name == "get_current_time"
            assert tool_block.input == {}

    @pytest.mark.asyncio
    async def test_tool_call_with_special_characters_in_arguments(self, openai_adapter):
        """Test tool call with special characters and unicode in arguments."""
        special_args = {
            "query": 'Search for: café ☕️ and "quotes"',
            "path": "/home/user/my file.txt",
            "regex": "^\\d+\\s*\\w+$",
        }
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-special",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_special",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_command",
                                        "arguments": orjson.dumps(special_args).decode(),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4-turbo",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Search for café ☕️")])]
                ),
                tools=[FunctionTool(name="execute_command", parameters={"type": "object"})],
            )
            response = await openai_adapter.chat_completion(request)

            tool_block = response.output[0]
            assert tool_block.input["query"] == 'Search for: café ☕️ and "quotes"'
            assert tool_block.input["path"] == "/home/user/my file.txt"
            assert tool_block.input["regex"] == "^\\d+\\s*\\w+$"

    @pytest.mark.asyncio
    async def test_mixed_content_and_tool_calls(self, openai_adapter):
        """Test response with both text content and tool calls."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-mixed",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I'll check the weather for you.",
                            "tool_calls": [
                                {
                                    "id": "call_mixed",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "SF"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
            },
        )

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
            )
            response = await openai_adapter.chat_completion(request)

            assert len(response.output) == 2
            assert isinstance(response.output[0], TextBlock)
            assert response.output[0].text == "I'll check the weather for you."
            assert isinstance(response.output[1], ToolUseBlock)
            assert response.output[1].name == "get_weather"

    @pytest.mark.asyncio
    async def test_tool_result_with_error(self, openai_adapter):
        """Test handling of tool execution error in tool result."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-error",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "I apologize, but I couldn't retrieve the weather data. "
                                "Let me try a different approach."
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 20,
                    "total_tokens": 100,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4-turbo",
                conversation=ConversationContext(
                    messages=[
                        Message(role="user", content=[TextBlock(text="What's the weather?")]),
                        Message(
                            role="assistant",
                            content=[ToolUseBlock(id="call_err", name="get_weather", input={})],
                        ),
                        Message(
                            role="tool",
                            content=[
                                ToolResultBlock(
                                    tool_use_id="call_err",
                                    content="Error: API rate limit exceeded",
                                    is_error=True,
                                )
                            ],
                        ),
                    ]
                ),
            )
            response = await openai_adapter.chat_completion(request)

            assert isinstance(response.output[0], TextBlock)
            assert "couldn't retrieve" in response.output[0].text.lower()
