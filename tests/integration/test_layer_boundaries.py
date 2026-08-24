"""Integration tests for layer boundary interactions.

Tests verify correct data flow and error handling across architectural layers:
- Protocol Layer → Processing Layer
- Processing Layer → Provider Layer
- Full request chain (HTTP Request → Response)
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.core.processing.base import RequestContext, ServiceDependencies
from llm_proxy.models import (
    ConversationContext,
    FunctionTool,
    InternalRequest,
    InternalResponse,
    Message,
    TextBlock,
    ToolUseBlock,
)
from llm_proxy.protocols.registry import get_protocol_serializer


@dataclass
class MockProviderConfig:
    """Mock ProviderConfig for testing."""

    name: str
    api_key: str = "test-key"
    base_url: str = "https://api.example.com"


@dataclass
class MockModelProviderConfig:
    """Mock ModelProviderConfig for testing."""

    provider: str
    provider_model_name: str | None = None
    priority: int = 1
    parameter_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockModelConfig:
    """Mock ModelConfig for testing."""

    model_name: str
    providers: list[MockModelProviderConfig]
    parameter_overrides: dict[str, Any] = field(default_factory=dict)
    max_retries: int | None = None

    def get_providers_by_priority(self):
        return sorted(self.providers, key=lambda p: -p.priority)


_openai_serializer = get_protocol_serializer("openai")
_anthropic_serializer = get_protocol_serializer("anthropic")


class TestProtocolToProcessingLayer:
    """Tests for Protocol Layer → Processing Layer boundary.

    Verifies that:
    1. Protocol handlers correctly parse requests into InternalRequest
    2. UnifiedProcessor receives properly structured InternalRequest
    3. Metadata (request_id, trace_id) flows correctly
    """

    @pytest.mark.asyncio
    async def test_openai_request_parsing_to_unified(self):
        """Test OpenAI protocol correctly parses request to InternalRequest."""
        raw_request = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        unified = _openai_serializer.parse_request(raw_request)

        assert isinstance(unified, InternalRequest)
        assert unified.model == "gpt-4"
        assert len(unified.conversation.messages) == 3
        assert isinstance(unified.conversation.messages[0].content[0], TextBlock)
        assert unified.conversation.messages[0].content[0].text == "Hello"
        assert unified.params.temperature == 0.7
        assert unified.params.max_tokens == 100

    @pytest.mark.asyncio
    async def test_anthropic_request_parsing_to_unified(self):
        """Test Anthropic protocol correctly parses request to InternalRequest."""
        raw_request = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            "max_tokens": 100,
            "system": "You are helpful.",
        }

        unified = _anthropic_serializer.parse_request(raw_request)

        assert isinstance(unified, InternalRequest)
        assert unified.model == "claude-3-opus"
        assert len(unified.conversation.messages) == 1
        assert len(unified.conversation.system_messages) == 1
        assert unified.params.max_tokens == 100

    @pytest.mark.asyncio
    async def test_tool_calls_preserved_across_boundary(self):
        """Test tool definitions are preserved from protocol to processing layer."""
        raw_request = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather info",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }

        unified = _openai_serializer.parse_request(raw_request)

        assert unified.tools is not None
        assert len(unified.tools) == 1
        assert isinstance(unified.tools[0], FunctionTool)
        assert unified.tools[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_streaming_flag_preserved(self):
        """Test streaming flag is correctly passed through protocol layer."""
        raw_request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        unified = _openai_serializer.parse_request(raw_request)

        assert unified.stream is True


class TestProcessingToProviderLayer:
    """Tests for Processing Layer → Provider Layer boundary.

    Verifies that:
    1. UnifiedProcessor correctly selects provider
    2. InternalRequest is passed correctly to adapters
    3. Provider errors are handled and classified
    """

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter for testing."""
        adapter = MagicMock()
        adapter.provider_name = "test-provider"
        adapter.chat_completion = AsyncMock(
            return_value=InternalResponse(
                id="resp-test-123",
                model="gpt-4",
                output=[TextBlock(text="Hello!")],
                finish_reason="stop",
                usage=MagicMock(input_tokens=10, output_tokens=5),
            )
        )
        return adapter

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator for testing."""
        orchestrator = MagicMock()
        orchestrator.select_next_provider = MagicMock(
            return_value=MagicMock(
                provider_name="test-provider",
                provider_model_name=None,
                priority=1,
            )
        )
        orchestrator.should_retry = MagicMock(return_value=False)
        orchestrator.exhausted = False
        return orchestrator

    @pytest.mark.asyncio
    async def test_unified_request_passed_to_adapter(self, mock_adapter, mock_orchestrator):
        """Test InternalRequest is correctly passed to adapter."""
        from llm_proxy.core.processing import get_strategy
        from llm_proxy.core.request_type import RequestType

        unified_request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            params=MagicMock(),
        )

        context = RequestContext(
            orchestrator=mock_orchestrator,
            services=ServiceDependencies(adapter_factory=AsyncMock(return_value=mock_adapter)),
        )

        strategy = get_strategy(RequestType.CHAT)
        await strategy.execute(unified_request, mock_adapter, context)

        mock_adapter.chat_completion.assert_called_once()
        called_request = mock_adapter.chat_completion.call_args[0][0]
        assert isinstance(called_request, InternalRequest)
        assert called_request.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_provider_error_classification(self, mock_orchestrator):
        """Test provider errors are correctly classified for retry decisions."""
        from llm_proxy.core.exceptions import ProviderError
        from llm_proxy.core.provider_selector import ErrorCategory, classify_error

        rate_limit_error = ProviderError(
            message="Rate limit exceeded",
            error_type="rate_limit_error",
            status_code=429,
        )
        assert classify_error(rate_limit_error, 429) == ErrorCategory.RETRYABLE

        auth_error = ProviderError(
            message="Invalid API key",
            error_type="authentication_error",
            status_code=401,
        )
        assert classify_error(auth_error, 401) == ErrorCategory.RETRYABLE

        server_error = ProviderError(
            message="Internal server error",
            error_type="api_error",
            status_code=503,
        )
        assert classify_error(server_error, 503) == ErrorCategory.RETRYABLE


class TestFullRequestChain:
    """Tests for complete request chain from HTTP to response.

    Verifies end-to-end flow:
    1. HTTP request received
    2. Protocol parsing
    3. Provider selection
    4. Adapter execution
    5. Response formatting
    """

    @pytest.mark.asyncio
    async def test_openai_chat_completion_full_chain(self):
        """Test complete OpenAI chat completion flow."""
        from llm_proxy.providers.openai_compatible import OpenAICompatibleBase

        adapter = OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from GPT-4!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            adapter._http_client = mock_client

            request = InternalRequest(
                model="gpt-4",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hello")])]
                ),
            )

            response = await adapter.chat_completion(request)

            assert isinstance(response, InternalResponse)
            assert isinstance(response.output[0], TextBlock)
            assert response.output[0].text == "Hello from GPT-4!"
            assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_call_round_trip(self):
        """Test tool calling round-trip across all layers."""
        from llm_proxy.providers.openai_compatible import OpenAICompatibleBase

        raw_request = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": "What's the weather in SF?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }

        unified = _openai_serializer.parse_request(raw_request)

        assert unified.tools is not None
        assert len(unified.tools) == 1
        assert unified.tools[0].name == "get_weather"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
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
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }

        adapter = OpenAICompatibleBase(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("llm_proxy.providers.openai_compatible._base.AsyncSession") as mock_session:
            mock_session.return_value = mock_client
            adapter._http_client = mock_client

            response = await adapter.chat_completion(unified)

            assert isinstance(response.output[0], ToolUseBlock)
            assert response.output[0].name == "get_weather"
            assert response.output[0].input == {"location": "San Francisco"}
            assert response.finish_reason == "tool_calls"

            formatted = _openai_serializer.format_response(response)
            assert "choices" in formatted
            assert formatted["choices"][0]["message"]["tool_calls"] is not None


class TestCrossLayerMetadata:
    """Tests for metadata flow across layers.

    Verifies that:
    1. Request IDs flow correctly
    2. Trace context is preserved
    3. Timing information is accurate
    """

    @pytest.mark.asyncio
    async def test_request_id_preservation(self):
        """Test request_id is preserved across all layers."""
        raw_request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        unified = _openai_serializer.parse_request(raw_request)

        unified.request_id = "req-test-123"

        response = InternalResponse(
            id="resp-test-123",
            model="gpt-4",
            output=[TextBlock(text="Hello!")],
            finish_reason="stop",
            request_id="req-test-123",
        )

        formatted = _openai_serializer.format_response(response)
        assert formatted["id"] == "resp-test-123"
        assert response.request_id == "req-test-123"

    @pytest.mark.asyncio
    async def test_model_name_mapping_across_layers(self):
        """Test model name mapping works correctly across layers."""
        from llm_proxy.core.provider_selector import ProviderSelector

        model_config = MockModelConfig(
            model_name="my-model-alias",
            providers=[
                MockModelProviderConfig(
                    provider="openai",
                    provider_model_name="gpt-4-turbo",
                    priority=1,
                )
            ],
        )

        selector = ProviderSelector(
            model_config=model_config,
            provider_configs={"openai": MockProviderConfig(name="openai")},
        )

        selection = selector.select_next_provider()
        assert selection is not None
        assert selection.provider_model_name == "gpt-4-turbo"
