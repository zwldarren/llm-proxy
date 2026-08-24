# tests/test_serialization/test_openai.py
"""Tests for OpenAI serializers."""

import pytest

from llm_proxy.models import (
    AudioBlock,
    ConversationContext,
    GenerationParams,
    InternalRequest,
    InternalResponse,
    Message,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer
from llm_proxy.serialization.openai.serializer import OpenAIResponsesProviderSerializer


@pytest.fixture
def protocol_serializer():
    return OpenAIProtocolSerializer()


@pytest.fixture
def provider_serializer():
    from llm_proxy.serialization.providers.chat_completions import OpenAIProviderSerializer

    return OpenAIProviderSerializer()


def test_serializer_registered():
    from llm_proxy.protocols.registry import get_protocol_serializer
    from llm_proxy.serialization.providers import get_provider_serializer

    ps = get_protocol_serializer("openai")
    assert isinstance(ps, OpenAIProtocolSerializer)

    pvs = get_provider_serializer("openai")
    assert isinstance(pvs, OpenAIResponsesProviderSerializer)


def test_parse_simple_request(protocol_serializer):
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello, world!"}]}

    request = protocol_serializer.parse_request(data)

    assert request.model == "gpt-4"
    assert len(request.conversation.messages) == 1
    assert request.conversation.messages[0].role == "user"


def test_parse_content_string(protocol_serializer):
    blocks = protocol_serializer.parse_content_blocks("Hello, world!")

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == "Hello, world!"


def test_parse_content_list(protocol_serializer):
    blocks = protocol_serializer.parse_content_blocks(
        [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
    )

    assert len(blocks) == 2
    assert blocks[0].text == "Hello"
    assert blocks[1].text == "World"


def test_format_response(protocol_serializer):
    response = InternalResponse(id="test-id", model="gpt-4", output=[TextBlock(text="Hello!")])

    result = protocol_serializer.format_response(response)

    assert result["id"] == "test-id"
    assert result["model"] == "gpt-4"
    assert result["object"] == "chat.completion"
    assert len(result["choices"]) == 1
    assert result["choices"][0]["message"]["content"] == "Hello!"


def test_build_provider_request(provider_serializer):
    request = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(max_tokens=100, temperature=0.7),
    )

    body = provider_serializer.build_provider_request(request)

    assert body["model"] == "gpt-4"
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.7
    assert len(body["messages"]) == 1


def test_max_completion_tokens_preserved(provider_serializer):
    from llm_proxy.models import OpenAISpecificParams

    request = InternalRequest(
        model="o3-mini",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(
            openai=OpenAISpecificParams(max_completion_tokens=500),
        ),
    )

    body = provider_serializer.build_provider_request(request)

    assert body["max_completion_tokens"] == 500
    assert "max_tokens" not in body


def test_max_tokens_used_when_no_completion_tokens(provider_serializer):
    request = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(max_tokens=100),
    )

    body = provider_serializer.build_provider_request(request)

    assert body["max_tokens"] == 100
    assert "max_completion_tokens" not in body


def test_explicit_max_tokens_wins_over_max_completion_tokens_when_differing(provider_serializer):
    """When both max_tokens and max_completion_tokens are sent with different values,
    an explicit max_tokens wins on the Chat Completions wire (consistent with the
    common-field precedence used by the Responses/Anthropic/Gemini/Ollama builders).
    """
    from llm_proxy.models import OpenAISpecificParams

    request = InternalRequest(
        model="o3-mini",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(
            max_tokens=100,
            openai=OpenAISpecificParams(max_completion_tokens=500),
        ),
    )

    body = provider_serializer.build_provider_request(request)

    assert body["max_tokens"] == 100
    assert "max_completion_tokens" not in body


def test_equal_max_tokens_and_max_completion_tokens_emits_completion_field(provider_serializer):
    """When both fields are present with the same value, emit the o-series-compatible
    ``max_completion_tokens`` field (value is identical, so this is safe and universal).
    """
    from llm_proxy.models import OpenAISpecificParams

    request = InternalRequest(
        model="o3-mini",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(
            max_tokens=500,
            openai=OpenAISpecificParams(max_completion_tokens=500),
        ),
    )

    body = provider_serializer.build_provider_request(request)

    assert body["max_completion_tokens"] == 500
    assert "max_tokens" not in body


def test_parse_max_completion_tokens(protocol_serializer):
    data = {
        "model": "o3-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_completion_tokens": 500,
    }

    request = protocol_serializer.parse_request(data)

    assert request.params.openai is not None
    assert request.params.openai.max_completion_tokens == 500
    # When only max_completion_tokens is provided it is also mirrored to the
    # common max_tokens field so all provider builders can read it.
    assert request.params.max_tokens == 500


def test_parse_max_tokens(protocol_serializer):
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
    }

    request = protocol_serializer.parse_request(data)

    assert request.params.max_tokens == 100
    assert request.params.openai is None or request.params.openai.max_completion_tokens is None


def test_round_trip_max_completion_tokens(protocol_serializer, provider_serializer):
    original_data = {
        "model": "o3-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_completion_tokens": 500,
    }

    request = protocol_serializer.parse_request(original_data)
    provider_body = provider_serializer.build_provider_request(request)

    assert provider_body["max_completion_tokens"] == 500
    assert "max_tokens" not in provider_body


def test_max_completion_tokens_falls_back_to_common_max_tokens_for_responses_api(
    protocol_serializer,
):
    data = {
        "model": "o3-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_completion_tokens": 500,
    }

    request = protocol_serializer.parse_request(data)

    # The common field is populated so provider builders that only read
    # params.max_tokens (e.g. OpenAI Responses API) still get the limit.
    assert request.params.max_tokens == 500


def test_max_tokens_takes_priority_over_max_completion_tokens(protocol_serializer):
    data = {
        "model": "o3-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_completion_tokens": 500,
        "max_tokens": 100,
    }

    request = protocol_serializer.parse_request(data)

    assert request.params.openai is not None
    assert request.params.openai.max_completion_tokens == 500
    # Explicit max_tokens always wins for the common field.
    assert request.params.max_tokens == 100


def test_parse_provider_response(provider_serializer):
    provider_response = {
        "id": "test-id",
        "model": "gpt-4",
        "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    response = provider_serializer.parse_provider_response(provider_response, model="gpt-4")

    assert response.id == "test-id"
    assert response.model == "gpt-4"
    assert len(response.output) == 1
    assert response.output[0].text == "Hello!"
    assert response.finish_reason == "stop"


def test_round_trip(protocol_serializer, provider_serializer):
    original_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
    }

    request = protocol_serializer.parse_request(original_data)
    provider_body = provider_serializer.build_provider_request(request)

    assert provider_body["model"] == "gpt-4"
    assert len(provider_body["messages"]) == 1
    assert provider_body["messages"][0]["role"] == "user"


def test_parse_tool_calls(protocol_serializer):
    data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "Boston"}'},
                    }
                ],
            }
        ],
    }

    request = protocol_serializer.parse_request(data)

    assert len(request.conversation.messages) == 1
    msg = request.conversation.messages[0]
    assert msg.role == "assistant"
    assert len(msg.content) == 2
    assert isinstance(msg.content[1], ToolUseBlock)
    assert msg.content[1].id == "call_123"
    assert msg.content[1].name == "get_weather"
    assert msg.content[1].input == {"location": "Boston"}


def test_parse_tool_calls_with_thought_signature(protocol_serializer):
    """Test that thought_signature in OpenAI tool_calls is parsed into ToolUseBlock."""
    data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "Boston"}'},
                        "thought_signature": "sig_abc_123",
                    }
                ],
            }
        ],
    }

    request = protocol_serializer.parse_request(data)

    msg = request.conversation.messages[0]
    tool_use = msg.content[1]
    assert isinstance(tool_use, ToolUseBlock)
    assert tool_use.extra.get("thought_signature") == "sig_abc_123"


def test_parse_provider_response_with_tool_calls(provider_serializer):
    provider_response = {
        "id": "test-id",
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Boston"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    response = provider_serializer.parse_provider_response(provider_response)

    assert len(response.output) == 1
    assert isinstance(response.output[0], ToolUseBlock)
    assert response.output[0].id == "call_123"
    assert response.output[0].name == "get_weather"
    assert response.output[0].input == {"location": "Boston"}
    assert response.finish_reason == "tool_calls"


def test_parse_provider_response_with_tool_calls_thought_signature(provider_serializer):
    """Test that thought_signature in provider response is parsed into ToolUseBlock."""
    provider_response = {
        "id": "test-id",
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Boston"}',
                            },
                            "thought_signature": "sig_abc_123",
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    response = provider_serializer.parse_provider_response(provider_response)

    assert isinstance(response.output[0], ToolUseBlock)
    assert response.output[0].extra.get("thought_signature") == "sig_abc_123"


def test_parse_provider_response_with_refusal(provider_serializer):
    provider_response = {
        "id": "test-id",
        "model": "gpt-4",
        "choices": [
            {
                "message": {"content": None, "refusal": "I cannot help with that request."},
                "finish_reason": "content_filter",
            }
        ],
    }

    response = provider_serializer.parse_provider_response(provider_response)

    assert len(response.output) == 1
    assert isinstance(response.output[0], RefusalBlock)
    assert response.output[0].refusal == "I cannot help with that request."
    assert response.finish_reason == "content_filter"


def test_parse_provider_response_with_reasoning_content(provider_serializer):
    provider_response = {
        "id": "test-id",
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me think about this step by step...",
                },
                "finish_reason": "stop",
            }
        ],
    }

    response = provider_serializer.parse_provider_response(provider_response)

    assert len(response.output) == 2
    assert isinstance(response.output[0], TextBlock)
    assert response.output[0].text == "The answer is 42."
    assert isinstance(response.output[1], ThinkingBlock)
    assert response.output[1].thinking == "Let me think about this step by step..."


def test_parse_provider_response_with_reasoning_fallback(provider_serializer):
    provider_response = {
        "id": "test-id",
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "content": "The answer is 42.",
                    "reasoning": "Alternative reasoning field.",
                },
                "finish_reason": "stop",
            }
        ],
    }

    response = provider_serializer.parse_provider_response(provider_response)

    assert len(response.output) == 2
    assert isinstance(response.output[1], ThinkingBlock)
    assert response.output[1].thinking == "Alternative reasoning field."


def test_parse_provider_response_with_audio(provider_serializer):
    provider_response = {
        "id": "test-id",
        "model": "gpt-4o-audio-preview",
        "choices": [
            {
                "message": {
                    "content": None,
                    "audio": {
                        "data": "base64encodedaudiodata",
                        "id": "audio-123",
                        "expires_at": 1234567890,
                        "transcript": "Hello, this is the audio transcript.",
                    },
                },
                "finish_reason": "stop",
            }
        ],
    }

    response = provider_serializer.parse_provider_response(provider_response)

    assert len(response.output) == 1
    assert isinstance(response.output[0], AudioBlock)
    assert response.output[0].source.type == "base64"
    assert response.output[0].source.data == "base64encodedaudiodata"
    assert response.output[0].source.id == "audio-123"
    assert response.output[0].source.expires_at == 1234567890
    assert response.output[0].source.transcript == "Hello, this is the audio transcript."


def test_parse_provider_response_with_annotations(provider_serializer):
    provider_response = {
        "id": "test-id",
        "model": "gpt-4",
        "choices": [
            {
                "message": {
                    "content": "Here is some information.",
                    "annotations": [
                        {"type": "url_citation", "url": "https://example.com", "title": "Example"}
                    ],
                },
                "finish_reason": "stop",
            }
        ],
    }

    response = provider_serializer.parse_provider_response(provider_response)

    assert len(response.output) == 1
    annotations = response.provider_info.get("annotations")
    assert annotations is not None
    assert len(annotations) == 1
    assert annotations[0]["type"] == "url_citation"
    assert annotations[0]["url"] == "https://example.com"


def test_parallel_tool_calls_false_converts_to_disable_parallel_tool_use(protocol_serializer):
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "parallel_tool_calls": False,
    }

    request = protocol_serializer.parse_request(data)

    assert request.extra.get("disable_parallel_tool_use") is True
    assert request.params.openai is not None
    assert request.params.openai.parallel_tool_calls is False


def test_parallel_tool_calls_true_not_converted(protocol_serializer):
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "parallel_tool_calls": True,
    }

    request = protocol_serializer.parse_request(data)

    assert request.params.openai is not None
    assert request.params.openai.parallel_tool_calls is True
    assert (
        request.params.anthropic is None
        or request.params.anthropic.disable_parallel_tool_use is None
    )


def test_build_provider_request_with_parallel_tool_calls(provider_serializer):
    from llm_proxy.models import OpenAISpecificParams

    request = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(openai=OpenAISpecificParams(parallel_tool_calls=False)),
    )

    body = provider_serializer.build_provider_request(request)

    assert body["parallel_tool_calls"] is False


class TestCitationPreservation:
    """Citations from Anthropic provider TextBlocks must be preserved in OpenAI protocol format."""

    def test_citations_preserved_in_format_response(self, protocol_serializer):
        from llm_proxy.models.content_blocks import CitationCharLocation
        from llm_proxy.models.types import Usage

        internal = InternalResponse(
            id="resp_1",
            model="claude-sonnet-4-6",
            output=[
                TextBlock(
                    text="Paris",
                    citations=[
                        CitationCharLocation(
                            cited_text="Paris",
                            document_index=0,
                            start_char_index=0,
                            end_char_index=5,
                        )
                    ],
                )
            ],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason="stop",
        )
        result = protocol_serializer.format_response(internal)
        message = result["choices"][0]["message"]
        assert "annotations" in message, "Citations should be preserved as annotations"
        assert len(message["annotations"]) == 1
        assert message["annotations"][0]["type"] == "char_location"

    def test_multiple_citations_across_text_blocks(self, protocol_serializer):
        from llm_proxy.models.content_blocks import CitationCharLocation
        from llm_proxy.models.types import Usage

        internal = InternalResponse(
            id="resp_2",
            model="claude-sonnet-4-6",
            output=[
                TextBlock(
                    text="Paris is the capital",
                    citations=[
                        CitationCharLocation(
                            cited_text="Paris", start_char_index=0, end_char_index=5
                        )
                    ],
                ),
                TextBlock(
                    text="of France.",
                    citations=[
                        CitationCharLocation(
                            cited_text="France", start_char_index=3, end_char_index=9
                        )
                    ],
                ),
            ],
            usage=Usage(input_tokens=10, output_tokens=10, total_tokens=20),
            finish_reason="stop",
        )
        result = protocol_serializer.format_response(internal)
        message = result["choices"][0]["message"]
        assert "annotations" in message
        assert len(message["annotations"]) == 2


class TestOpenAIResponseParserDetails:
    """Regression tests for OpenAI-format response detail preservation."""

    def test_upstream_model_overrides_request_model(self):
        from llm_proxy.serialization.openai.components.response_parser import (
            OpenAIResponseParser,
        )

        parser = OpenAIResponseParser()
        response = {
            "id": "resp-1",
            "model": "openai/gpt-5",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        }
        result = parser.parse(response, model="requested-model")
        assert result.model == "openai/gpt-5"

    def test_usage_details_with_cache_write_video_and_image_tokens(self):
        from llm_proxy.serialization.openai.components.response_parser import (
            OpenAIResponseParser,
        )

        parser = OpenAIResponseParser()
        response = {
            "id": "resp-1",
            "model": "gpt-4",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {
                    "audio_tokens": 1,
                    "cached_tokens": 3,
                    "image_tokens": 4,
                    "cache_write_tokens": 2,
                    "video_tokens": 7,
                },
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 1,
                    "audio_tokens": 1,
                    "reasoning_tokens": 4,
                    "rejected_prediction_tokens": 1,
                    "image_tokens": 2,
                },
            },
        }
        result = parser.parse(response, model="gpt-4")
        assert result.usage is not None
        ptd = result.usage.prompt_tokens_details
        assert ptd is not None
        assert ptd.cache_write_tokens == 2
        assert ptd.video_tokens == 7
        ctd = result.usage.completion_tokens_details
        assert ctd is not None
        assert ctd.image_tokens == 2

    def test_native_finish_reason_from_choices(self):
        from llm_proxy.serialization.openai.components.response_parser import (
            OpenAIResponseParser,
        )

        parser = OpenAIResponseParser()
        response = {
            "id": "resp-1",
            "model": "gpt-4",
            "choices": [
                {
                    "message": {"content": "hi"},
                    "finish_reason": "stop",
                    "native_finish_reason": "end_turn",
                }
            ],
        }
        result = parser.parse(response, model="gpt-4")
        assert result.provider_info.get("native_finish_reasons") == ["end_turn"]


def test_build_provider_request_respects_stream_usage_false(provider_serializer):
    """stream_options.include_usage=false must be forwarded as-is: the proxy
    does not override it. Billing falls back to token estimation when the
    provider returns no usage chunk."""
    from llm_proxy.models.types import StreamOptions

    request = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=GenerationParams(),
        stream=True,
        stream_options=StreamOptions(include_usage=False),
    )

    body = provider_serializer.build_provider_request(request)

    assert body["stream"] is True
    assert body["stream_options"]["include_usage"] is False


def test_build_provider_request_always_full_converts(provider_serializer):
    """build_provider_request no longer gates on compatible_protocols: even
    with a stash present it rebuilds from the parsed request. The
    wire-compatible rebuild shortcut lives in the conversion seam
    (prepare_wire_reuse_body), tested in tests/core/test_conversion_tiers.py."""
    raw = {
        "model": "client-alias",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": False},
    }
    request = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(),
        stream=True,
    )
    request.metadata.protocol_name = "openai"
    request._raw_protocol_data = raw

    body = provider_serializer.build_provider_request(request)

    # Full conversion: the client's alias is replaced by the request's model,
    # and the stash is never touched.
    assert body["model"] == "gpt-4"
    assert raw["model"] == "client-alias"


def test_build_provider_request_stream_without_stream_options_sends_none(provider_serializer):
    """A streaming request without stream_options must not get a fabricated
    stream_options block on the provider body."""
    request = InternalRequest(
        model="gpt-4",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(),
        stream=True,
    )

    body = provider_serializer.build_provider_request(request)

    assert body["stream"] is True
    assert "stream_options" not in body


def test_parse_provider_response_maps_deepseek_cache_fields(provider_serializer):
    """DeepSeek top-level prompt_cache_hit_tokens folds into
    prompt_tokens_details.cached_tokens so cache pricing applies."""
    response = {
        "id": "chatcmpl-x",
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_cache_hit_tokens": 64,
            "prompt_cache_miss_tokens": 36,
        },
    }

    result = provider_serializer.parse_provider_response(response, model="deepseek-chat")

    assert result.usage is not None
    assert result.usage.input_tokens == 100
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 64
