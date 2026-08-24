"""Additional tests for OpenRouter-specific features beyond the base adapter.

Covers streaming comment filtering, request body construction with
OpenRouter-specific parameters, and provider metadata extraction.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    InternalResponse,
    Message,
    OpenAISpecificParams,
    TextBlock,
    ThinkingBlock,
    ThinkingConfig,
)
from llm_proxy.models.audio import InternalTranscriptionRequest
from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter():
    return OpenRouterAdapter(api_key="sk-or-test")


# ---------------------------------------------------------------------------
# Streaming comment filtering
# ---------------------------------------------------------------------------


class TestStreamFilterLine:
    """OpenRouter emits SSE comments like ``: OPENROUTER PROCESSING``."""

    def test_filters_colon_prefix_lines(self, adapter):
        """Lines starting with ``:`` are filtered out (SSE comment/keepalive)."""
        assert adapter._stream_filter_line(": OPENROUTER PROCESSING") is None
        assert adapter._stream_filter_line(": keepalive") is None
        assert adapter._stream_filter_line(":\n") is None

    def test_passes_data_lines(self, adapter):
        """Lines starting with ``data: `` are extracted."""
        result = adapter._stream_filter_line('data: {"choices":[]}')
        assert result == '{"choices":[]}'

    def test_returns_none_for_empty_lines(self, adapter):
        """Non-SSE lines return None."""
        # The method doesn't strip, so a bare empty string should match nothing
        assert adapter._stream_filter_line("") is None

    def test_returns_none_for_non_data_non_colon(self, adapter):
        """Lines that are neither ``data:`` nor ``:`` prefix return None."""
        assert adapter._stream_filter_line("event: message") is None
        assert adapter._stream_filter_line("id: 1") is None
        assert adapter._stream_filter_line("retry: 3000") is None

    def test_preserves_json_with_internal_colons(self, adapter):
        """JSON payloads containing colons are preserved intact."""
        payload = 'data: {"id":"gen-123","choices":[{"delta":{"content":"a: b"}}]}'
        result = adapter._stream_filter_line(payload)
        assert result is not None
        assert "a: b" in result


# ---------------------------------------------------------------------------
# Request body — OpenRouter-specific parameters
# ---------------------------------------------------------------------------


class TestOpenRouterRequestBody:
    """OpenRouter request body includes provider routing & reasoning params."""

    def test_reasoning_effort_passthrough(self, adapter):
        """``reasoning_effort`` is included when set in OpenAISpecificParams."""
        request = InternalRequest(
            model="openai/gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(reasoning_effort="high")),
        )
        body = adapter._build_request_body(request)
        assert body["reasoning_effort"] == "high"

    def test_provider_routing_via_extra_passthrough(self, adapter):
        """OpenRouter provider routing preferences passed via extra.

        When the adapter is configured with ``unknown_fields_policy="passthrough"``,
        OpenRouter-specific routing fields (``provider``, ``models``, ``plugins``)
        are forwarded to the upstream API.
        """
        adapter_passthrough = OpenRouterAdapter(
            api_key="sk-or-test", unknown_fields_policy="passthrough"
        )
        request = InternalRequest(
            model="auto",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            extra={
                "provider": {
                    "order": ["OpenAI", "Anthropic"],
                    "allow_fallbacks": False,
                },
                "models": ["openai/gpt-4o", "anthropic/claude-3-opus"],
                "plugins": ["web"],
            },
        )
        body = adapter_passthrough._build_request_body(request)
        assert body["provider"]["order"] == ["OpenAI", "Anthropic"]
        assert body["provider"]["allow_fallbacks"] is False
        assert body["models"] == ["openai/gpt-4o", "anthropic/claude-3-opus"]
        assert body["plugins"] == ["web"]

    def test_reasoning_field_always_reasoning(self, adapter):
        """OpenRouter always emits assistant reasoning as ``reasoning`` field."""
        request = InternalRequest(
            model="openai/gpt-4o",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[ThinkingBlock(thinking="Previous reasoning")],
                    )
                ]
            ),
        )
        body = adapter._build_request_body(request)
        assistant_msg = body["messages"][0]
        assert "reasoning" in assistant_msg
        assert assistant_msg["reasoning"] == "Previous reasoning"
        assert "reasoning_content" not in assistant_msg

    def test_thinking_config_emits_reasoning_effort(self, adapter):
        """ThinkingConfig is converted to reasoning_effort for OpenRouter."""
        request = InternalRequest(
            model="openai/gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Deep question")])]
            ),
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", effort="high")),
        )
        body = adapter._build_request_body(request)
        assert body["reasoning_effort"] == "high"

    def test_max_completion_tokens(self, adapter):
        """OpenRouter supports max_completion_tokens."""
        request = InternalRequest(
            model="openai/gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(max_completion_tokens=4096)),
        )
        body = adapter._build_request_body(request)
        assert body["max_completion_tokens"] == 4096

    def test_top_logprobs_included(self, adapter):
        """logprobs + top_logprobs are forwarded for OpenRouter."""
        request = InternalRequest(
            model="openai/gpt-4o",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
            params=GenerationParams(openai=OpenAISpecificParams(logprobs=True, top_logprobs=5)),
        )
        body = adapter._build_request_body(request)
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 5


class TestTranscriptionRequest:
    """OpenRouter STT uses JSON ``input_audio`` rather than multipart."""

    def test_prompt_included_in_transcription_body(self, adapter):
        """Regression: prompt must be forwarded to the OpenRouter STT body."""
        request = InternalTranscriptionRequest(
            model="whisper-large-v3",
            file=b"fake-audio",
            filename="audio.mp3",
            prompt="Use formal English.",
            language="en",
            response_format="verbose_json",
            temperature=0.5,
        )
        url, headers, body = adapter._build_transcription_request(request)
        assert body["prompt"] == "Use formal English."
        assert body["language"] == "en"
        assert body["response_format"] == "verbose_json"
        assert body["temperature"] == 0.5
        assert "input_audio" in body
        assert headers["Content-Type"] == "application/json"

    def test_prompt_none_omitted(self, adapter):
        request = InternalTranscriptionRequest(
            model="whisper-large-v3",
            file=b"fake-audio",
            filename="audio.mp3",
        )
        url, headers, body = adapter._build_transcription_request(request)
        assert "prompt" not in body


# ---------------------------------------------------------------------------
# Response post-processing
# ---------------------------------------------------------------------------


class MockResponse:
    def __init__(self, status_code: int, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.headers = MagicMock()

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class TestPostProcessChatResponse:
    """OpenRouter-specific response post-processing."""

    def test_unknown_fields_extracted_to_provider_info(self, adapter):
        """OpenRouter response metadata (provider, cost, etc.) is captured.

        OpenRouter returns fields like ``provider`` (the actual upstream
        provider), ``cost``, and ``native_finish_reason`` that are not part
        of the standard OpenAI schema. These are extracted and stored in
        ``provider_info`` for downstream consumers (cost tracking, analytics).

        Note: the ``provider`` field from the OpenRouter response overwrites
        the serializer default ``provider: "openai"`` — this is intentional so
        callers see which provider actually served the request.
        """
        serializer = adapter._get_serializer()
        response_data = {
            "id": "gen-abc123",
            "model": "anthropic/claude-3-opus",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "provider": "Anthropic",
            "cost": 0.0025,
            "native_finish_reason": "end_turn",
        }
        result = serializer.parse_provider_response(response_data, model="test")
        processed = adapter._post_process_chat_response(response_data, result)
        # The OpenRouter `provider` field tells which actual provider served
        # the request. It overwrites the default `provider: "openai"`.
        assert processed.provider_info["provider"] == "Anthropic"
        assert processed.provider_info["cost"] == 0.0025

    def test_response_with_extra_fields_added_to_provider_info(self, adapter):
        """Fields not in the known OpenAI schema are added to provider_info.

        The OpenRouter post-processing uses ``extract_unknown_response_fields``
        which adds any field not in the known response schema to provider_info.
        This includes ``system_fingerprint``, which is part of the OpenAI spec
        but treated as provider-specific metadata.
        """
        serializer = adapter._get_serializer()
        response_data = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "system_fingerprint": "fp_123",
            "object": "chat.completion",
            "created": 1234567890,
        }
        result = serializer.parse_provider_response(response_data, model="gpt-4")
        processed = adapter._post_process_chat_response(response_data, result)
        # system_fingerprint is expected in provider_info
        assert "system_fingerprint" in processed.provider_info
        assert processed.provider_info["system_fingerprint"] == "fp_123"


# ---------------------------------------------------------------------------
# Chat completion integration
# ---------------------------------------------------------------------------


class TestChatCompletionIntegration:
    """End-to-end chat completion tests for OpenRouter."""

    @pytest.mark.asyncio
    async def test_chat_completion_basic(self, adapter):
        """Basic chat completion with OpenRouter adapter."""
        mock_response = MockResponse(
            200,
            json_data={
                "id": "gen-abc123",
                "model": "openai/gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from OpenRouter!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "llm_proxy.providers.openai_compatible._base.AsyncSession",
            return_value=mock_client,
        ):
            adapter._http_client = mock_client
            request = InternalRequest(
                model="openai/gpt-4o",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Hi")])]
                ),
            )
            response = await adapter.chat_completion(request)
            assert isinstance(response, InternalResponse)
            assert len(response.output) == 1
            assert response.output[0].text == "Hello from OpenRouter!"
            assert response.model == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_chat_completion_with_tool_calls(self, adapter):
        """OpenRouter chat completion with tool calls."""
        from llm_proxy.models import ToolUseBlock

        mock_response = MockResponse(
            200,
            json_data={
                "id": "gen-tool-123",
                "model": "openai/gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location":"London"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "llm_proxy.providers.openai_compatible._base.AsyncSession",
            return_value=mock_client,
        ):
            adapter._http_client = mock_client
            request = InternalRequest(
                model="openai/gpt-4o",
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="Weather?")])]
                ),
            )
            response = await adapter.chat_completion(request)
            tool_blocks = [b for b in response.output if isinstance(b, ToolUseBlock)]
            assert len(tool_blocks) == 1
            assert tool_blocks[0].name == "get_weather"
            assert tool_blocks[0].input == {"location": "London"}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestOpenRouterErrorHandling:
    """OpenRouter-specific error scenarios."""

    @pytest.fixture
    def adapter(self):
        return OpenRouterAdapter(api_key="sk-or-test")

    def test_stream_filter_line_with_error_response(self, adapter):
        """Error responses in SSE format are passed through."""
        error_line = 'data: {"error":{"message":"Rate limited","code":429}}'
        result = adapter._stream_filter_line(error_line)
        assert result is not None
        assert "Rate limited" in result

    def test_stream_filter_line_with_done(self, adapter):
        """``[DONE]`` marker is passed through."""
        result = adapter._stream_filter_line("data: [DONE]")
        assert result == "[DONE]"
