# tests/unit/protocols/openai/test_handler.py
"""Tests for openai_protocol."""

import orjson

from llm_proxy.models import (
    AudioBlock,
    FunctionTool,
    ImageBlock,
    InternalResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from llm_proxy.protocols.openai.handler import openai_protocol
from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.serialization.openai import format_conversation

_openai_serializer = get_protocol_serializer("openai")


class TestOpenAIProtocolEndpoint:
    """Test suite for openai_protocol."""

    def test_name(self):
        """Test protocol name."""
        assert openai_protocol.name == "openai"

    def test_paths(self):
        """Test supported paths."""
        paths = openai_protocol.paths
        assert "/v1/chat/completions" in paths

    def test_request_model(self):
        """Test request model is returned."""
        model = openai_protocol.request_model
        assert model is not None
        # Verify it's a Pydantic model
        assert hasattr(model, "model_fields")

    def test_parse_simple_request(self):
        """Test parsing a simple request."""
        ChatRequest = openai_protocol.request_model
        request = ChatRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        unified = _openai_serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.model == "gpt-4"
        assert len(unified.conversation.messages) == 1
        assert unified.conversation.messages[0].role == "user"

    def test_parse_request_with_system(self):
        """Test parsing request with system prompt."""
        ChatRequest = openai_protocol.request_model
        request = ChatRequest(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        )

        unified = _openai_serializer.parse_request(request.model_dump(exclude_none=True))
        assert len(unified.conversation.system_messages) == 1
        assert unified.conversation.system_messages[0].text_content == "You are helpful."
        assert len(unified.conversation.messages) == 1

    def test_parse_request_with_tools(self):
        """Test parsing request with tools."""
        ChatRequest = openai_protocol.request_model
        request = ChatRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

        unified = _openai_serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.tools is not None
        assert len(unified.tools) == 1
        tool = unified.tools[0]
        assert isinstance(tool, FunctionTool)
        assert tool.name == "get_weather"

    def test_parse_request_with_params(self):
        """Test parsing request with generation params."""
        ChatRequest = openai_protocol.request_model
        request = ChatRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        )

        unified = _openai_serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.temperature == 0.7
        assert unified.params.max_tokens == 100

    def test_format_simple_response(self):
        """Test formatting a simple response."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[TextBlock(text="Hello, world!")],
            usage=Usage(input_tokens=10, output_tokens=5),
        )

        result = _openai_serializer.format_response(response)
        assert result["id"] == "resp_123"
        assert result["model"] == "gpt-4"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Hello, world!"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_format_response_with_tool_calls(self):
        """Test formatting response with tool calls."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[
                TextBlock(text="Let me check that."),
                ToolUseBlock(id="call_1", name="get_weather", input={"city": "SF"}),
            ],
        )

        result = _openai_serializer.format_response(response)
        assert "tool_calls" in result["choices"][0]["message"]
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1
        assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_format_response_with_tool_calls_thought_signature(self):
        """Test formatting response with tool calls preserves thought_signature."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[
                ToolUseBlock(
                    id="call_1",
                    name="get_weather",
                    input={"city": "SF"},
                    extra={"thought_signature": "sig_abc_123"},
                ),
            ],
        )

        result = _openai_serializer.format_response(response)
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["thought_signature"] == "sig_abc_123"

    def test_format_response_with_tool_calls_no_thought_signature(self):
        """Test formatting response without thought_signature omits the key."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[
                ToolUseBlock(id="call_1", name="get_weather", input={"city": "SF"}),
            ],
        )

        result = _openai_serializer.format_response(response)
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert "thought_signature" not in tc

    def test_format_response_with_usage(self):
        """Test formatting response with usage."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[TextBlock(text="Hello")],
            usage=Usage(input_tokens=100, output_tokens=50),
        )

        result = _openai_serializer.format_response(response)
        assert "usage" in result
        assert result["usage"]["prompt_tokens"] == 100
        assert result["usage"]["completion_tokens"] == 50

    def test_streaming_transformer(self):
        """Test that streaming transformer is returned."""
        transformer_cls = openai_protocol.get_streaming_transformer()
        assert transformer_cls is not None

    def test_streaming_transformer_handles_null_input(self):
        """Test that streaming transformer handles None input gracefully."""

        transformer = OpenAIStreamingTransformer(model="test-model", request_id="test-id")

        result = transformer.transform(None)
        assert result is None

    def test_parse_request_with_audio(self):
        """Test parsing request with audio parameters."""
        ChatRequest = openai_protocol.request_model
        request = ChatRequest(
            model="gpt-4-audio",
            messages=[{"role": "user", "content": "hello"}],
            audio={"voice": "alloy"},
            modalities=["text", "audio"],
        )
        unified = _openai_serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.openai is not None
        assert unified.params.openai.audio == {"voice": "alloy"}
        assert unified.params.openai.modalities == ["text", "audio"]

    def test_parse_request_with_reasoning_effort(self):
        """Test parsing request with reasoning_effort."""
        ChatRequest = openai_protocol.request_model
        request = ChatRequest(
            model="o1",
            messages=[{"role": "user", "content": "solve"}],
            reasoning_effort="high",
        )
        unified = _openai_serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.openai is not None
        assert unified.params.openai.reasoning_effort == "high"

    def test_parse_request_with_logit_bias(self):
        """Test parsing request with logit_bias."""
        ChatRequest = openai_protocol.request_model
        request = ChatRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            logit_bias={"1234": -100},
        )
        unified = _openai_serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.openai is not None
        assert unified.params.openai.logit_bias == {"1234": -100}

    def test_parse_request_with_web_search_options(self):
        """Test parsing request with web_search_options."""
        ChatRequest = openai_protocol.request_model
        request = ChatRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "search"}],
            web_search_options={"search_context_size": "high"},
        )
        unified = _openai_serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.openai is not None
        assert unified.params.openai.web_search_options == {"search_context_size": "high"}

    def test_parse_request_with_parallel_tool_calls(self):
        """Test parsing request with parallel_tool_calls."""
        ChatRequest = openai_protocol.request_model
        request = ChatRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
            parallel_tool_calls=False,
        )
        unified = _openai_serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.openai is not None
        assert unified.params.openai.parallel_tool_calls is False

    def test_parse_request_with_prediction(self):
        """Test parsing request with prediction."""
        ChatRequest = openai_protocol.request_model
        request = ChatRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            prediction={"type": "content", "content": "expected"},
        )
        unified = _openai_serializer.parse_request(request.model_dump(exclude_none=True))
        assert unified.params.openai is not None
        assert unified.params.openai.prediction == {"type": "content", "content": "expected"}

    def test_format_response_with_reasoning_content(self):
        """Test formatting response with reasoning_content."""
        from llm_proxy.models.content_blocks.extended import ThinkingBlock

        response = InternalResponse(
            id="resp_123",
            model="o1",
            output=[TextBlock(text="answer"), ThinkingBlock(thinking="thinking process...")],
            usage=Usage(input_tokens=10, output_tokens=20),
        )
        result = _openai_serializer.format_response(response)
        assert result["choices"][0]["message"]["reasoning_content"] == "thinking process..."

    def test_format_response_with_refusal(self):
        """Test formatting response with refusal."""
        from llm_proxy.models.content_blocks.extended import RefusalBlock

        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[RefusalBlock(refusal="I cannot help with that.")],
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        result = _openai_serializer.format_response(response)
        assert result["choices"][0]["message"]["refusal"] == "I cannot help with that."

    def test_format_response_with_system_fingerprint(self):
        """Test formatting response with system_fingerprint."""
        response = InternalResponse(
            id="resp_123",
            model="gpt-4",
            output=[TextBlock(text="hello")],
            provider_info={"system_fingerprint": "fp_abc123"},
            usage=Usage(input_tokens=5, output_tokens=10),
        )
        result = _openai_serializer.format_response(response)
        assert result["system_fingerprint"] == "fp_abc123"

    def test_parse_request_with_assistant_tool_calls(self):
        """Test parsing assistant message with tool_calls.

        This tests the fix for: assistant messages with tool_calls must be
        preserved so that tool result messages have a preceding assistant
        message with tool_calls.
        """

        ChatRequest = openai_protocol.request_model
        request = ChatRequest(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "SF"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "Sunny, 72°F",
                },
                {"role": "user", "content": "Thanks!"},
            ],
        )

        unified = _openai_serializer.parse_request(request.model_dump(exclude_none=True))

        # Check that we have 4 messages
        assert len(unified.conversation.messages) == 4

        # Check user message
        assert unified.conversation.messages[0].role == "user"
        assert len(unified.conversation.messages[0].content) == 1
        assert isinstance(unified.conversation.messages[0].content[0], TextBlock)

        # Check assistant message with tool_calls
        assert unified.conversation.messages[1].role == "assistant"
        assert len(unified.conversation.messages[1].content) == 2
        assert isinstance(unified.conversation.messages[1].content[1], ToolUseBlock)
        assert unified.conversation.messages[1].content[1].id == "call_123"
        assert unified.conversation.messages[1].content[1].name == "get_weather"
        assert unified.conversation.messages[1].content[1].input == {"location": "SF"}

        # Check tool result message
        assert unified.conversation.messages[2].role == "tool"
        assert len(unified.conversation.messages[2].content) == 1
        assert isinstance(unified.conversation.messages[2].content[0], ToolResultBlock)
        assert unified.conversation.messages[2].content[0].tool_use_id == "call_123"

        # Check that converting back preserves tool_calls in assistant message
        openai_messages = format_conversation(unified.conversation)
        assert len(openai_messages) == 4
        assert "tool_calls" in openai_messages[1]
        assert openai_messages[1]["tool_calls"][0]["id"] == "call_123"
        assert openai_messages[2]["role"] == "tool"
        assert openai_messages[2]["tool_call_id"] == "call_123"


class TestOpenAIStreamingTransformerNormalizeChunks:
    """Test suite for OpenAIStreamingTransformer chunk normalization.

    The transformer is fidelity-first: unknown provider fields pass through
    verbatim. Only load-bearing transforms run (model aliasing,
    custom→function tool-call normalization, obfuscation injection), and
    ``_``-prefixed proxy-internal keys plus null values are stripped.
    """

    def test_transform_passes_through_unknown_fields(self):
        """Provider extension fields reach the client verbatim; nulls still go."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello", "reasoning_details": [{"type": "text"}]},
                    "logprobs": None,
                    "finish_reason": None,
                    "token_ids": None,
                    "hidden_states": None,
                }
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 15, "completion_tokens": 5},
            "chutes_verification": "abc123",
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))

        assert "id" in data
        assert "object" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        # Unknown provider extension fields pass through at every level.
        assert data["chutes_verification"] == "abc123"
        assert data["choices"][0]["delta"]["reasoning_details"] == [{"type": "text"}]
        # Usage stays attached to a content chunk only in pending state —
        # it is emitted in its own chunk at finalize (see the strips-usage
        # test below).

        choice = data["choices"][0]
        # Null-valued fields are still stripped.
        assert "logprobs" not in choice
        assert "token_ids" not in choice
        assert "hidden_states" not in choice
        assert "finish_reason" not in choice

    def test_transform_strips_internal_underscore_keys(self):
        """``_``-prefixed proxy-internal keys are stripped at every traversed level."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "_internal_trace": "abc",
            "choices": [
                {
                    "index": 0,
                    "_choice_marker": 1,
                    "delta": {
                        "content": "hi",
                        "_routing": "x",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}", "_debug": 1},
                                "_tc_marker": True,
                            }
                        ],
                    },
                }
            ],
        }

        result = transformer.transform(chunk)

        assert result is not None
        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert "_internal_trace" not in data
        choice = data["choices"][0]
        assert "_choice_marker" not in choice
        assert "_routing" not in choice["delta"]
        tc = choice["delta"]["tool_calls"][0]
        assert "_tc_marker" not in tc
        assert "_debug" not in tc["function"]
        assert tc["function"] == {"name": "f", "arguments": "{}"}

    def test_transform_normalizes_custom_tool_calls(self):
        """Load-bearing transform: custom tool calls become function calls."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "custom",
                                "custom": {"name": "exec", "input": "ls -la"},
                            }
                        ]
                    },
                }
            ],
        }

        result = transformer.transform(chunk)

        assert result is not None
        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        tc = data["choices"][0]["delta"]["tool_calls"][0]
        assert tc["type"] == "function"
        assert "custom" not in tc
        assert tc["function"] == {"name": "exec", "arguments": "ls -la"}

    def test_transform_overrides_model_with_user_facing_alias(self):
        """The provider's model id is masked by the client-requested alias."""

        transformer = OpenAIStreamingTransformer(model="client-alias")

        chunk = {
            "id": "chatcmpl-1",
            "model": "upstream-internal-model-id",
            "choices": [{"index": 0, "delta": {"content": "hi"}}],
        }

        result = transformer.transform(chunk)

        assert result is not None
        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert data["model"] == "client-alias"

    def test_transform_keeps_reasoning_content(self):
        """Test that reasoning_content is kept as a non-standard but allowed field."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [{"index": 0, "delta": {"reasoning_content": "thinking..."}}],
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))

        assert "reasoning_content" in data["choices"][0]["delta"]
        assert data["choices"][0]["delta"]["reasoning_content"] == "thinking..."

    def test_transform_removes_null_values(self):
        """Test that null values are removed from all levels."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [{"index": 0, "delta": {"content": None}, "finish_reason": "stop"}],
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))

        delta = data["choices"][0]["delta"]
        assert delta == {}
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_transform_removes_null_values_without_finish_reason(self):
        """Test that choices with all null fields and no finish_reason are skipped."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [{"index": 0, "delta": {"content": None}}],
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))

        assert "delta" not in data["choices"][0]

    def test_transform_passes_through_unknown_usage_fields(self):
        """Unknown usage fields reach the client; billing reads known keys."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "reasoning_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 80, "audio_tokens": None},
                "non_standard_field": "xxx",
            },
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))

        usage = data["usage"]
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["reasoning_tokens"] == 20
        assert usage["non_standard_field"] == "xxx"
        assert usage["prompt_tokens_details"]["cached_tokens"] == 80
        assert "audio_tokens" not in usage["prompt_tokens_details"]

    def test_transform_strips_usage_from_content_chunks(self):
        """Test that usage is stripped from content chunks."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))

        assert "usage" not in data
        assert data["choices"][0]["delta"]["content"] == "Hello"

    def test_transform_keeps_usage_in_empty_choices_chunk(self):
        """Test that usage is kept when choices is empty (usage-only chunk)."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))

        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 100
        assert data["usage"]["completion_tokens"] == 50
        assert data["usage"]["total_tokens"] == 150
        assert data["choices"] == []

    def test_finalize_emits_cached_usage_when_provider_never_sends_usage_only_chunk(self):
        """Test that finalize emits a usage-only chunk when usage only appeared with content."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [{"index": 0, "delta": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
        transformer.transform(chunk)

        final = transformer.finalize()

        usage_line = final.split("\n\n", 1)[0]
        assert usage_line.startswith("data: ")

        synthesized = orjson.loads(usage_line[6:])
        assert synthesized["choices"] == []
        assert synthesized["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        }
        assert final.endswith("data: [DONE]\n\n")

    def test_transform_filters_empty_chunks(self):
        """Test that empty chunks are filtered."""

        transformer = OpenAIStreamingTransformer(model="test")

        assert transformer.transform(None) is None

    def test_transform_passes_through_extension_only_chunk(self):
        """A chunk carrying only provider extension fields is emitted.

        Fidelity-first pass-through is not gated on choices/usage: unknown
        fields are deliverable content on their own.
        """

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "upstream-name",
            "chutes_verification": "abc123",
        }

        result = transformer.transform(chunk)

        assert result is not None
        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert data["chutes_verification"] == "abc123"
        # Load-bearing model aliasing still applies.
        assert data["model"] == "test"

    def test_transform_drops_envelope_only_chunk(self):
        """A keep-alive chunk reduced to the standard envelope is skipped."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "upstream-name",
            "choices": [],
        }

        assert transformer.transform(chunk) is None

    def test_transform_keeps_tool_calls(self):
        """Test that tool_calls are preserved correctly."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "test", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ],
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))

        tool_calls = data["choices"][0]["delta"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "test"

    def test_transform_filters_incomplete_tool_calls(self):
        """Test that incomplete tool_calls (missing function field) are filtered."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "type": "function"}], "content": ""},
                }
            ],
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))

        delta = data["choices"][0]["delta"]
        assert "tool_calls" not in delta or len(delta.get("tool_calls", [])) == 0

    def test_transform_keeps_partial_tool_calls_with_function(self):
        """Test that tool_calls with partial function data are kept."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "type": "function", "function": {"arguments": '{"'}}
                        ]
                    },
                }
            ],
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))

        tool_calls = data["choices"][0]["delta"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["arguments"] == '{"'

    def test_finalize_sends_done_marker(self):
        """Test that finalize sends the [DONE] marker."""

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="resp_123")
        result = transformer.finalize()

        assert result == "data: [DONE]\n\n"


class TestOpenAIStreamingTransformerMultimodal:
    """Tests for multimodal output in streaming transformer."""

    def test_transform_keeps_images_in_delta(self):
        """Test that delta.images passes through the streaming transformer."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                            }
                        ]
                    },
                }
            ],
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        delta = data["choices"][0]["delta"]
        assert "images" in delta
        assert len(delta["images"]) == 1
        assert delta["images"][0]["type"] == "image_url"
        assert "iVBORw0KGgo=" in delta["images"][0]["image_url"]["url"]

    def test_transform_keeps_audio_in_delta(self):
        """Test that delta.audio passes through the streaming transformer."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "audio": {
                            "data": "AAAA",
                            "transcript": "Hello",
                        }
                    },
                }
            ],
        }

        result = transformer.transform(chunk)

        assert result is not None

        data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        delta = data["choices"][0]["delta"]
        assert "audio" in delta
        assert delta["audio"]["data"] == "AAAA"
        assert delta["audio"]["transcript"] == "Hello"

    def test_accumulate_images_in_finalize(self):
        """Test that images are accumulated and available in finalize."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk1 = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                            }
                        ]
                    },
                }
            ],
        }
        chunk2 = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Here is your image."},
                    "finish_reason": "stop",
                }
            ],
        }

        transformer.transform(chunk1)
        transformer.transform(chunk2)
        transformer.finalize()

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) >= 1
        # At least one ImageBlock should be in the accumulated output

        assert any(isinstance(b, ImageBlock) for b in accumulated)

    def test_accumulate_audio_in_finalize(self):
        """Test that audio is accumulated and available in finalize."""

        transformer = OpenAIStreamingTransformer(model="test")

        chunk1 = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "audio": {
                            "data": "AAAA",
                            "transcript": "Hel",
                        }
                    },
                }
            ],
        }
        chunk2 = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "audio": {
                            "data": "BBBB",
                            "transcript": "lo world",
                        }
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        transformer.transform(chunk1)
        transformer.transform(chunk2)
        transformer.finalize()

        accumulated = transformer.get_accumulated_output()

        audio_blocks = [b for b in accumulated if isinstance(b, AudioBlock)]
        assert len(audio_blocks) == 1, (
            f"Expected 1 concatenated audio block, got {len(audio_blocks)}"
        )
        # Audio data should be concatenated
        assert audio_blocks[0].source.data == "AAAABBBB"
        assert audio_blocks[0].source.transcript == "Hello world"
