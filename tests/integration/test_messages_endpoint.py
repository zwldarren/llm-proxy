"""Integration tests for /v1/messages endpoint (Anthropic protocol).

Tests the Anthropic Messages API protocol across OpenAI, Gemini, and Ollama providers,
including complex scenarios like tool calling, extended thinking, multi-turn conversations,
and streaming responses.

See Anthropic Messages API docs: https://docs.anthropic.com/en/api/messages
"""

from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from llm_proxy.models import (
    ConversationContext,
    FunctionTool,
    InternalRequest,
    Message,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.tools import ToolChoice, ToolChoiceFunction


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


class TestAnthropicProtocolWithOpenAI:
    """Tests for /v1/messages endpoint proxied to OpenAI provider.

    Tests the Anthropic Messages API format being translated and sent to OpenAI.
    This validates the protocol translation layer works correctly.
    """

    @pytest.fixture
    def openai_adapter(self):
        """Create an OpenAI adapter for testing."""
        from llm_proxy.providers.openai_compatible import OpenAICompatibleBase

        return OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

    @pytest.fixture
    def anthropic_serializer(self):
        """Create an Anthropic serializer for request/response handling."""
        from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer
        from llm_proxy.serialization.anthropic.serializer import AnthropicProviderSerializer

        class AnthropicSerializer(AnthropicProtocolSerializer, AnthropicProviderSerializer):
            pass

        return AnthropicSerializer()

    @pytest.mark.asyncio
    async def test_simple_message(self, openai_adapter, anthropic_serializer):
        """Test simple message request/response through Anthropic protocol.

        Validates that a basic Anthropic-style message is correctly translated
        to OpenAI format and the response is correctly formatted back.
        """
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
                            "content": "Hello! I'm Claude, how can I help you today?",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 12, "total_tokens": 27},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            # Anthropic-style request
            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hello, who are you?")])]
                ),
            )
            response = await openai_adapter.chat_completion(request)

            # Format as Anthropic response
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["type"] == "message"
            assert anthropic_response["role"] == "assistant"
            assert len(anthropic_response["content"]) == 1
            assert anthropic_response["content"][0]["type"] == "text"
            assert (
                anthropic_response["content"][0]["text"]
                == "Hello! I'm Claude, how can I help you today?"
            )
            assert anthropic_response["stop_reason"] == "end_turn"
            assert anthropic_response["usage"]["input_tokens"] == 15
            assert anthropic_response["usage"]["output_tokens"] == 12

    @pytest.mark.asyncio
    async def test_message_with_system_prompt(self, openai_adapter, anthropic_serializer):
        """Test message with system prompt in Anthropic format.

        Anthropic accepts system prompts as a top-level 'system' field,
        which should be translated to OpenAI's messages format.
        """
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-sys",
                "object": "chat.completion",
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": ("I am a helpful coding assistant specialized in Python."),
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
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    system_messages=[
                        SystemMessage.from_text(role="system", text="You are a coding assistant.")
                    ],
                    messages=[
                        Message(
                            role="user", content=[TextBlock(text="What are you specialized in?")]
                        )
                    ],
                ),
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["content"][0]["text"] == (
                "I am a helpful coding assistant specialized in Python."
            )

    @pytest.mark.asyncio
    async def test_tool_use_single(self, openai_adapter, anthropic_serializer):
        """Test single tool use (function calling) through Anthropic protocol.

        Anthropic uses tool_use content blocks in responses, while OpenAI uses
        tool_calls in the message. This tests the translation between formats.

        See: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
        """
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-tool",
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
                                    "id": "toolu_01A09q90qw90lq91723",
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
                "usage": {"prompt_tokens": 55, "completion_tokens": 18, "total_tokens": 73},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            # Anthropic-style tool definition
            tools = [
                {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                }
            ]

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[TextBlock(text="What is the weather like in San Francisco?")],
                        )
                    ]
                ),
                tools=anthropic_serializer._parse_tools(tools),
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["stop_reason"] == "tool_use"
            assert len(anthropic_response["content"]) == 1
            assert anthropic_response["content"][0]["type"] == "tool_use"
            assert anthropic_response["content"][0]["id"] == "toolu_01A09q90qw90lq91723"
            assert anthropic_response["content"][0]["name"] == "get_weather"
            assert anthropic_response["content"][0]["input"] == {"location": "San Francisco, CA"}

    @pytest.mark.asyncio
    async def test_tool_use_parallel(self, openai_adapter, anthropic_serializer):
        """Test parallel tool use (multiple function calls in one response).

        Anthropic supports parallel tool use where the model can make multiple
        tool calls in a single response for independent operations.

        See: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#parallel-tool-use
        """
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-parallel",
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
                                    "id": "toolu_01A1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "San Francisco"}',
                                    },
                                },
                                {
                                    "id": "toolu_01A2",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "New York"}',
                                    },
                                },
                                {
                                    "id": "toolu_01A3",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "London"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 45,
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
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[TextBlock(text="Compare the weather in SF, NYC, and London")],
                        )
                    ]
                ),
                tools=[
                    FunctionTool(
                        name="get_weather",
                        description="Get weather for a location",
                        parameters={
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    )
                ],
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["stop_reason"] == "tool_use"
            assert len(anthropic_response["content"]) == 3
            for i, location in enumerate(["San Francisco", "New York", "London"]):
                assert anthropic_response["content"][i]["type"] == "tool_use"
                assert anthropic_response["content"][i]["name"] == "get_weather"
                assert anthropic_response["content"][i]["input"]["location"] == location

    @pytest.mark.asyncio
    async def test_tool_result_multi_turn(self, openai_adapter, anthropic_serializer):
        """Test multi-turn conversation with tool results.

        After a tool call, the client sends the tool result back as a user message
        with a tool_result content block. The model then uses this to provide
        a final answer.

        See: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#handling-tool-use-and-tool-result-content
        """
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-final",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "The current weather in San Francisco is sunny with "
                                "a temperature of 72°F (22°C). It's a beautiful day "
                                "with a light breeze coming from the west."
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 95,
                    "completion_tokens": 35,
                    "total_tokens": 130,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            # Request includes tool result from previous turn
            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[TextBlock(text="What's the weather in San Francisco?")],
                        ),
                        Message(
                            role="assistant",
                            content=[
                                ToolUseBlock(
                                    id="toolu_01A",
                                    name="get_weather",
                                    input={"location": "San Francisco"},
                                )
                            ],
                        ),
                        Message(
                            role="user",
                            content=[
                                ToolResultBlock(
                                    tool_use_id="toolu_01A",
                                    content=(
                                        '{"temperature": "72°F", "condition": "sunny", '
                                        '"wind": "light breeze from west"}'
                                    ),
                                    is_error=False,
                                )
                            ],
                        ),
                    ]
                ),
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["stop_reason"] == "end_turn"
            assert anthropic_response["content"][0]["type"] == "text"
            assert "72°F" in anthropic_response["content"][0]["text"]
            assert "sunny" in anthropic_response["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tool_result_with_error(self, openai_adapter, anthropic_serializer):
        """Test tool result with error flag set.

        When a tool execution fails, the client should set is_error=True
        so the model can handle the error appropriately.
        """
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-err",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "I apologize, but I encountered an error while trying "
                                "to fetch the weather data. The weather service is "
                                "currently unavailable. Would you like me to try again "
                                "or check a different location?"
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 85,
                    "completion_tokens": 40,
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
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[TextBlock(text="What's the weather in Tokyo?")],
                        ),
                        Message(
                            role="assistant",
                            content=[
                                ToolUseBlock(
                                    id="toolu_err",
                                    name="get_weather",
                                    input={"location": "Tokyo"},
                                )
                            ],
                        ),
                        Message(
                            role="user",
                            content=[
                                ToolResultBlock(
                                    tool_use_id="toolu_err",
                                    content="Error: Weather service unavailable (503)",
                                    is_error=True,
                                )
                            ],
                        ),
                    ]
                ),
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert "error" in anthropic_response["content"][0]["text"].lower()
            assert "unavailable" in anthropic_response["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_tool_choice_auto(self, openai_adapter, anthropic_serializer):
        """Test tool_choice='auto' parameter.

        With tool_choice='auto', the model decides whether to use tools
        or respond with text based on the context.
        """
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-auto",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "I can help you with weather information! "
                                "Which city would you like to know about?"
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 45, "completion_tokens": 18, "total_tokens": 63},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Can you help me?")])]
                ),
                tools=[
                    FunctionTool(
                        name="get_weather",
                        parameters={"type": "object"},
                    )
                ],
                tool_choice=ToolChoice(mode="auto"),
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["stop_reason"] == "end_turn"
            assert anthropic_response["content"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_tool_choice_any(self, openai_adapter, anthropic_serializer):
        """Test tool_choice='any' (required in Anthropic) parameter.

        With tool_choice='any', the model must use at least one tool.
        Anthropic uses 'any' while OpenAI uses 'required'.
        """
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-any",
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
                                    "id": "toolu_required",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "general search"}',
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
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hello")])]
                ),
                tools=[
                    FunctionTool(
                        name="search",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                    )
                ],
                tool_choice=ToolChoice(mode="any"),
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_tool_choice_specific_tool(self, openai_adapter, anthropic_serializer):
        """Test forcing a specific tool by name.

        With tool_choice={'type': 'tool', 'name': 'get_weather'},
        the model must call that specific tool.
        """
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-specific",
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
                                    "id": "toolu_specific",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "default"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 55, "completion_tokens": 15, "total_tokens": 70},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Tell me something")])]
                ),
                tools=[
                    FunctionTool(
                        name="get_weather",
                        parameters={
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    ),
                    FunctionTool(
                        name="search",
                        parameters={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    ),
                ],
                tool_choice=ToolChoiceFunction(name="get_weather"),
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["content"][0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_complex_nested_tool_arguments(self, openai_adapter, anthropic_serializer):
        """Test tool use with deeply nested JSON arguments.

        Some tools require complex nested structures as arguments.
        This validates that nested objects are preserved correctly.
        """
        complex_args = {
            "user": {
                "profile": {
                    "name": "Alice Johnson",
                    "email": "alice@example.com",
                    "preferences": {
                        "language": "en",
                        "timezone": "America/Los_Angeles",
                        "notifications": {
                            "email": True,
                            "sms": False,
                            "push": True,
                        },
                    },
                },
                "subscription": {
                    "plan": "premium",
                    "billing": {
                        "cycle": "monthly",
                        "amount": 29.99,
                        "currency": "USD",
                    },
                },
            },
            "actions": [
                {"type": "update_settings", "settings": {"theme": "dark"}},
                {"type": "send_notification", "message": "Welcome back!"},
            ],
        }

        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-nested",
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
                                    "id": "toolu_nested",
                                    "type": "function",
                                    "function": {
                                        "name": "update_user",
                                        "arguments": orjson.dumps(complex_args).decode(),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 95,
                    "total_tokens": 215,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[
                                TextBlock(
                                    text=(
                                        "Update Alice's settings to dark theme "
                                        "and send welcome notification"
                                    )
                                )
                            ],
                        )
                    ]
                ),
                tools=[
                    FunctionTool(
                        name="update_user",
                        parameters={"type": "object"},
                    )
                ],
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            tool_input = anthropic_response["content"][0]["input"]
            assert tool_input["user"]["profile"]["name"] == "Alice Johnson"
            assert tool_input["user"]["profile"]["preferences"]["notifications"]["email"] is True
            assert tool_input["user"]["subscription"]["plan"] == "premium"
            assert len(tool_input["actions"]) == 2

    @pytest.mark.asyncio
    async def test_thinking_block(self, openai_adapter, anthropic_serializer):
        """Test extended thinking (reasoning) support.

        Anthropic's extended thinking feature returns thinking blocks
        that show the model's reasoning process. OpenAI uses reasoning_content.
        """
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-think",
                "object": "chat.completion",
                "model": "o1-preview",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The answer is 42.",
                            "reasoning_content": (
                                "Let me think about this step by step. "
                                "First, I need to understand the question. "
                                "The user is asking about the meaning of life. "
                                "According to Douglas Adams..."
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 100,
                    "total_tokens": 150,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[TextBlock(text="What is the meaning of life?")],
                        )
                    ]
                ),
            )
            response = await openai_adapter.chat_completion(request)

            # Check that reasoning_content is converted to ThinkingBlock in output
            # OpenAI's reasoning_content is converted to a ThinkingBlock
            thinking_blocks = [b for b in response.output if isinstance(b, ThinkingBlock)]
            assert len(thinking_blocks) >= 1
            assert "step by step" in thinking_blocks[0].thinking

    @pytest.mark.asyncio
    async def test_thinking_parameter_conversion(self, openai_adapter, anthropic_serializer):
        """Test thinking parameter conversion from Anthropic to OpenAI format.

        When using Anthropic endpoint with OpenAI-compatible provider,
        the thinking parameter should be correctly converted and included
        in the request body sent to the provider.
        """
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-think-param",
                "object": "chat.completion",
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            # Parse Anthropic request with thinking parameter
            from llm_proxy.models.params import GenerationParams
            from llm_proxy.models.types import ThinkingConfig

            request = InternalRequest(
                model="kimi-k2.5",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hello")])]
                ),
                params=GenerationParams(thinking=ThinkingConfig(type="disabled")),
            )
            await openai_adapter.chat_completion(request)

            # Verify the request body sent to OpenAI-compatible provider
            # contains the reasoning_effort parameter (converted from thinking)
            call_args = mock_client.post.call_args
            request_body = call_args.kwargs.get("json", {})

            assert "reasoning_effort" in request_body, "reasoning_effort should be in request body"
            assert request_body["reasoning_effort"] == "none"

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_use(self, openai_adapter, anthropic_serializer):
        """Test response with both text content and tool calls.

        The model may respond with text explaining what it's doing,
        followed by tool calls to execute the actions.
        """
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
                            "content": "I'll help you find the weather for both cities.",
                            "tool_calls": [
                                {
                                    "id": "toolu_mixed_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Paris"}',
                                    },
                                },
                                {
                                    "id": "toolu_mixed_2",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Berlin"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 60,
                    "completion_tokens": 35,
                    "total_tokens": 95,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[TextBlock(text="What's the weather in Paris and Berlin?")],
                        )
                    ]
                ),
                tools=[
                    FunctionTool(
                        name="get_weather",
                        parameters={
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    )
                ],
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            # Should have text block first, then tool use blocks
            assert len(anthropic_response["content"]) == 3
            assert anthropic_response["content"][0]["type"] == "text"
            assert "both cities" in anthropic_response["content"][0]["text"]
            assert anthropic_response["content"][1]["type"] == "tool_use"
            assert anthropic_response["content"][2]["type"] == "tool_use"

    @pytest.mark.asyncio
    async def test_cache_control_usage(self, openai_adapter, anthropic_serializer):
        """Test response with cache-related usage metrics.

        OpenAI returns cached_tokens in prompt_tokens_details for prompt caching.
        Anthropic returns cache_read_input_tokens and cache_creation_input_tokens.
        """
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-cache",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Cached response!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_tokens_details": {
                        "cached_tokens": 500,
                        "audio_tokens": 0,
                    },
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hello")])]
                ),
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["usage"]["input_tokens"] == 100
            assert anthropic_response["usage"]["output_tokens"] == 5
            # OpenAI's cached_tokens is in prompt_tokens_details
            assert response.usage is not None
            assert response.usage.prompt_tokens_details is not None
            assert response.usage.prompt_tokens_details.cached_tokens == 500


class TestAnthropicProtocolWithGemini:
    """Tests for /v1/messages endpoint proxied to Gemini provider.

    Tests the Anthropic Messages API format being translated and sent to
    Google's Gemini API. Validates protocol translation works correctly.
    """

    @pytest.fixture
    def gemini_adapter(self):
        """Create a Gemini adapter for testing."""
        from llm_proxy.providers.gemini import GeminiAdapter

        return GeminiAdapter(
            api_key="test-key",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )

    @pytest.fixture
    def anthropic_serializer(self):
        """Create an Anthropic serializer for request/response handling."""
        from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer
        from llm_proxy.serialization.anthropic.serializer import AnthropicProviderSerializer

        class AnthropicSerializer(AnthropicProtocolSerializer, AnthropicProviderSerializer):
            pass

        return AnthropicSerializer()

    @pytest.mark.asyncio
    async def test_simple_message(self, gemini_adapter, anthropic_serializer):
        """Test simple message through Gemini provider.

        Gemini uses different request/response format with 'contents' and 'parts'.
        """
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
                    "promptTokenCount": 15,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 25,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.gemini.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            gemini_adapter._http_client = mock_client

            request = InternalRequest(
                model="gemini-3.1-pro-preview",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hello")])]
                ),
            )
            response = await gemini_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["type"] == "message"
            assert anthropic_response["content"][0]["type"] == "text"
            assert anthropic_response["content"][0]["text"] == "Hello from Gemini!"
            assert anthropic_response["stop_reason"] == "end_turn"
            assert anthropic_response["usage"]["input_tokens"] == 15
            assert anthropic_response["usage"]["output_tokens"] == 10

    @pytest.mark.asyncio
    async def test_function_call(self, gemini_adapter, anthropic_serializer):
        """Test function calling with Gemini provider.

        Gemini uses functionCall format which needs to be translated to
        Anthropic's tool_use format.

        See: https://ai.google.dev/gemini-api/docs/function-calling
        """
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
                                        "args": {
                                            "location": "San Francisco",
                                            "unit": "celsius",
                                        },
                                    }
                                }
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 60,
                    "candidatesTokenCount": 25,
                    "totalTokenCount": 85,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.gemini.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            gemini_adapter._http_client = mock_client

            request = InternalRequest(
                model="gemini-3.1-pro-preview",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[TextBlock(text="What's the weather in SF in Celsius?")],
                        )
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
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                        },
                    )
                ],
            )
            response = await gemini_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert len(anthropic_response["content"]) == 1
            assert anthropic_response["content"][0]["type"] == "tool_use"
            assert anthropic_response["content"][0]["name"] == "get_weather"
            assert anthropic_response["content"][0]["input"]["location"] == "San Francisco"
            assert anthropic_response["content"][0]["input"]["unit"] == "celsius"

    @pytest.mark.asyncio
    async def test_parallel_function_calls(self, gemini_adapter, anthropic_serializer):
        """Test parallel function calls with Gemini.

        Gemini can return multiple function calls in a single response.
        """
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
                                        "args": {"location": "Tokyo"},
                                    }
                                },
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"location": "Seoul"},
                                    }
                                },
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 70,
                    "candidatesTokenCount": 30,
                    "totalTokenCount": 100,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.gemini.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            gemini_adapter._http_client = mock_client

            request = InternalRequest(
                model="gemini-3.1-pro-preview",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[TextBlock(text="Compare weather in Tokyo and Seoul")],
                        )
                    ]
                ),
                tools=[
                    FunctionTool(
                        name="get_weather",
                        parameters={
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    )
                ],
            )
            response = await gemini_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert len(anthropic_response["content"]) == 2
            assert all(c["type"] == "tool_use" for c in anthropic_response["content"])
            locations = [c["input"]["location"] for c in anthropic_response["content"]]
            assert "Tokyo" in locations
            assert "Seoul" in locations

    @pytest.mark.asyncio
    async def test_system_instruction(self, gemini_adapter, anthropic_serializer):
        """Test system prompt/instruction with Gemini.

        Gemini uses systemInstruction field for system prompts.
        """
        mock_response = MockResponse(
            200,
            json_data={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "I'm a pirate assistant! Arrr, how can I help ye?"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 30,
                    "candidatesTokenCount": 15,
                    "totalTokenCount": 45,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.gemini.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            gemini_adapter._http_client = mock_client

            request = InternalRequest(
                model="gemini-3.1-pro-preview",
                conversation=ConversationContext(
                    system_messages=[
                        SystemMessage(
                            role="system",
                            content="You are a helpful assistant who speaks like a pirate.",
                        )
                    ],
                    messages=[Message(role="user", content=[TextBlock(text="Hello!")])],
                ),
            )
            response = await gemini_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert "pirate" in anthropic_response["content"][0]["text"].lower()
            assert "arrr" in anthropic_response["content"][0]["text"].lower()


class TestAnthropicProtocolWithOllama:
    """Tests for /v1/messages endpoint proxied to Ollama provider.

    Tests the Anthropic Messages API format being translated and sent to
    Ollama's native API. Ollama uses /api/chat endpoint.
    """

    @pytest.fixture
    def ollama_adapter(self):
        """Create an Ollama adapter for testing."""
        from llm_proxy.providers.ollama import OllamaAdapter

        return OllamaAdapter(
            api_key="test-key",
            base_url="http://localhost:11434",
        )

    @pytest.fixture
    def anthropic_serializer(self):
        """Create an Anthropic serializer for request/response handling."""
        from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer
        from llm_proxy.serialization.anthropic.serializer import AnthropicProviderSerializer

        class AnthropicSerializer(AnthropicProtocolSerializer, AnthropicProviderSerializer):
            pass

        return AnthropicSerializer()

    @pytest.mark.asyncio
    async def test_simple_message(self, ollama_adapter, anthropic_serializer):
        """Test simple message through Ollama provider.

        Ollama uses native /api/chat format with model, message, and done fields.
        """
        mock_response = MockResponse(
            200,
            json_data={
                "model": "llama3.2",
                "created_at": "2024-01-15T12:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "Hello from Ollama! How can I assist you today?",
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 15,
                "eval_count": 12,
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.ollama.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            ollama_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hello")])]
                ),
            )
            response = await ollama_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["type"] == "message"
            assert anthropic_response["content"][0]["type"] == "text"
            assert anthropic_response["content"][0]["text"] == (
                "Hello from Ollama! How can I assist you today?"
            )
            assert anthropic_response["stop_reason"] == "end_turn"
            assert anthropic_response["usage"]["input_tokens"] == 15
            assert anthropic_response["usage"]["output_tokens"] == 12

    @pytest.mark.asyncio
    async def test_tool_call(self, ollama_adapter, anthropic_serializer):
        """Test tool calling with Ollama.

        Ollama supports tool calling for models like llama3.1, mistral, etc.
        The format is similar to OpenAI but with some differences.

        See: https://ollama.com/blog/tool-support
        """
        mock_response = MockResponse(
            200,
            json_data={
                "model": "llama3.1",
                "created_at": "2024-01-15T12:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_ollama_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"location": "London"},
                            },
                        }
                    ],
                },
                "done": True,
                "done_reason": "tool_calls",
                "prompt_eval_count": 50,
                "eval_count": 18,
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.ollama.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            ollama_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[TextBlock(text="What's the weather in London?")],
                        )
                    ]
                ),
                tools=[
                    FunctionTool(
                        name="get_weather",
                        parameters={
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    )
                ],
            )
            response = await ollama_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert len(anthropic_response["content"]) == 1
            assert anthropic_response["content"][0]["type"] == "tool_use"
            assert anthropic_response["content"][0]["name"] == "get_weather"
            assert anthropic_response["content"][0]["input"]["location"] == "London"

    @pytest.mark.asyncio
    async def test_thinking_with_ollama(self, ollama_adapter, anthropic_serializer):
        """Test thinking/reasoning content with Ollama.

        Ollama supports 'thinking' field in the message for models with
        reasoning capabilities (e.g., deepseek-r1, qwen3).
        """
        mock_response = MockResponse(
            200,
            json_data={
                "model": "deepseek-r1",
                "created_at": "2024-01-15T12:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "The answer is 17.",
                    "thinking": (
                        "Let me solve this step by step. "
                        "First, I need to understand what 2 + 15 equals..."
                    ),
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 25,
                "eval_count": 50,
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.ollama.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            ollama_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="What is 2 + 15?")])]
                ),
            )
            response = await ollama_adapter.chat_completion(request)

            # Check that thinking is captured as ThinkingBlock in output
            from llm_proxy.models import ThinkingBlock

            thinking_blocks = [b for b in response.output if isinstance(b, ThinkingBlock)]
            assert len(thinking_blocks) == 1
            assert "step by step" in thinking_blocks[0].thinking

    @pytest.mark.asyncio
    async def test_tool_result_roundtrip(self, ollama_adapter, anthropic_serializer):
        """Test multi-turn with tool result through Ollama.

        Validates that tool results are correctly formatted when sent to Ollama.
        """
        mock_response = MockResponse(
            200,
            json_data={
                "model": "llama3.1",
                "created_at": "2024-01-15T12:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": (
                        "The weather in Tokyo is currently sunny with a temperature "
                        "of 25°C. It's a pleasant day!"
                    ),
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 85,
                "eval_count": 25,
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.ollama.adapter.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            ollama_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[TextBlock(text="What's the weather in Tokyo?")],
                        ),
                        Message(
                            role="assistant",
                            content=[
                                ToolUseBlock(
                                    id="toolu_ollama",
                                    name="get_weather",
                                    input={"location": "Tokyo"},
                                )
                            ],
                        ),
                        Message(
                            role="user",
                            content=[
                                ToolResultBlock(
                                    tool_use_id="toolu_ollama",
                                    content='{"temp": "25°C", "condition": "sunny"}',
                                    is_error=False,
                                )
                            ],
                        ),
                    ]
                ),
            )
            response = await ollama_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["content"][0]["type"] == "text"
            assert "25°C" in anthropic_response["content"][0]["text"]
            assert "sunny" in anthropic_response["content"][0]["text"]


class TestAnthropicProtocolEdgeCases:
    """Tests for edge cases and complex scenarios in /v1/messages endpoint."""

    @pytest.fixture
    def openai_adapter(self):
        """Create an OpenAI adapter for testing."""
        from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase

        return OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

    @pytest.fixture
    def anthropic_serializer(self):
        """Create an Anthropic serializer for request/response handling."""
        from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer
        from llm_proxy.serialization.anthropic.serializer import AnthropicProviderSerializer

        class AnthropicSerializer(AnthropicProtocolSerializer, AnthropicProviderSerializer):
            pass

        return AnthropicSerializer()

    @pytest.mark.asyncio
    async def test_empty_tool_arguments(self, openai_adapter, anthropic_serializer):
        """Test tool call with no arguments (empty object)."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-empty",
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
                                    "id": "toolu_empty",
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
                "usage": {"prompt_tokens": 25, "completion_tokens": 8, "total_tokens": 33},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="What time is it?")])]
                ),
                tools=[
                    FunctionTool(
                        name="get_current_time",
                        parameters={"type": "object"},
                    )
                ],
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["content"][0]["type"] == "tool_use"
            assert anthropic_response["content"][0]["name"] == "get_current_time"
            assert anthropic_response["content"][0]["input"] == {}

    @pytest.mark.asyncio
    async def test_unicode_and_special_chars_in_tool_args(
        self, openai_adapter, anthropic_serializer
    ):
        """Test tool call with unicode and special characters in arguments."""
        special_args = {
            "query": "Search for: café ☕️ and emojis 🎉",
            "path": "/home/user/文档/文件.txt",
            "regex": "^\\d+\\s*[\\w\\-]+$",
            "message": 'He said "Hello" and left',
        }

        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-unicode",
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
                                    "id": "toolu_unicode",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": orjson.dumps(special_args).decode(),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 25, "total_tokens": 65},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Search for café ☕️")])]
                ),
                tools=[FunctionTool(name="search", parameters={"type": "object"})],
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            tool_input = anthropic_response["content"][0]["input"]
            assert "café ☕️" in tool_input["query"]
            assert "文档" in tool_input["path"]

    @pytest.mark.asyncio
    async def test_long_tool_name_and_args(self, openai_adapter, anthropic_serializer):
        """Test tool with very long name and arguments."""
        long_name = "very_long_tool_name_that_exceeds_normal_length_for_testing_purposes"
        long_args = {
            "description": "A" * 1000,
            "items": [{"id": f"item_{i}", "value": f"value_{i}"} for i in range(100)],
        }

        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-long",
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
                                    "id": "toolu_long",
                                    "type": "function",
                                    "function": {
                                        "name": long_name,
                                        "arguments": orjson.dumps(long_args).decode(),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 500,
                    "total_tokens": 600,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Process large data")])]
                ),
                tools=[
                    FunctionTool(
                        name=long_name,
                        parameters={"type": "object"},
                    )
                ],
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["content"][0]["name"] == long_name
            assert len(anthropic_response["content"][0]["input"]["items"]) == 100
            assert len(anthropic_response["content"][0]["input"]["description"]) == 1000

    @pytest.mark.asyncio
    async def test_multiple_tools_different_types(self, openai_adapter, anthropic_serializer):
        """Test response with multiple different tool types in sequence."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-multi",
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
                                    "id": "toolu_search",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "latest AI news"}',
                                    },
                                },
                                {
                                    "id": "toolu_fetch",
                                    "type": "function",
                                    "function": {
                                        "name": "fetch_url",
                                        "arguments": '{"url": "https://example.com"}',
                                    },
                                },
                                {
                                    "id": "toolu_save",
                                    "type": "function",
                                    "function": {
                                        "name": "save_file",
                                        "arguments": '{"path": "/tmp/results.json"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 45,
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
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[
                                TextBlock(
                                    text="Search for AI news, fetch the top result, and save it"
                                )
                            ],
                        )
                    ]
                ),
                tools=[
                    FunctionTool(
                        name="web_search",
                        parameters={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    ),
                    FunctionTool(
                        name="fetch_url",
                        parameters={
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                        },
                    ),
                    FunctionTool(
                        name="save_file",
                        parameters={
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    ),
                ],
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert len(anthropic_response["content"]) == 3
            tool_names = [c["name"] for c in anthropic_response["content"]]
            assert tool_names == ["web_search", "fetch_url", "save_file"]

    @pytest.mark.asyncio
    async def test_tool_result_with_nested_content(self, openai_adapter, anthropic_serializer):
        """Test tool result containing structured/nested content."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-nested-result",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I've analyzed the data structure.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 10,
                    "total_tokens": 160,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            # Tool result with nested content blocks
            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="user",
                            content=[TextBlock(text="Analyze this data")],
                        ),
                        Message(
                            role="assistant",
                            content=[
                                ToolUseBlock(
                                    id="toolu_analyze",
                                    name="analyze_data",
                                    input={"data_id": "123"},
                                )
                            ],
                        ),
                        Message(
                            role="user",
                            content=[
                                ToolResultBlock(
                                    tool_use_id="toolu_analyze",
                                    content=orjson.dumps(
                                        {
                                            "summary": "Processed 100 records",
                                            "errors": [],
                                            "warnings": ["2 deprecated fields"],
                                        }
                                    ).decode(),
                                    is_error=False,
                                )
                            ],
                        ),
                    ]
                ),
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["content"][0]["type"] == "text"
            assert "analyzed" in anthropic_response["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_max_tokens_stop_reason(self, openai_adapter, anthropic_serializer):
        """Test that max_tokens limit is reflected in stop_reason."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "chatcmpl-max",
                "object": "chat.completion",
                "model": "gpt-4-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "This is a long response that was cut off...",
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120,
                },
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            openai_adapter._http_client = mock_client

            request = InternalRequest(
                model="claude-sonnet-4-5-20250929",
                conversation=ConversationContext(
                    messages=[
                        Message(role="user", content=[TextBlock(text="Tell me a very long story")])
                    ]
                ),
            )
            response = await openai_adapter.chat_completion(request)
            anthropic_response = anthropic_serializer.format_response(response)

            assert anthropic_response["stop_reason"] == "max_tokens"
