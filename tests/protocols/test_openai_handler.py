# tests/protocols/test_openai_handler.py
"""Tests for OpenAI handler exception handling in transform method."""

from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer


class TestOpenAIStreamingTransformerExceptions:
    """Test suite for exception handling in OpenAIStreamingTransformer.transform."""

    def test_transform_valid_dict_passes_through(self):
        """Test that valid dict chunks are processed normally."""
        transformer = OpenAIStreamingTransformer(model="test-model", request_id="test-id")

        valid_chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "test",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }
        result = transformer.transform(valid_chunk)

        assert result is not None
        assert result.startswith("data: ")
        assert "[DONE]" not in result


class TestOpenAIStreamingTransformerAccumulation:
    """Test suite for content accumulation in OpenAIStreamingTransformer."""

    def test_accumulate_text_content(self):
        """Test that text content is accumulated during streaming."""
        from llm_proxy.models.content_blocks import TextBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="test-123")

        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": " world!"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], TextBlock)
        assert accumulated[0].text == "Hello world!"

    def test_accumulate_tool_calls(self):
        """Test that tool calls are accumulated during streaming."""
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="test-456")

        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"loc'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'ation": "SF"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], ToolUseBlock)
        assert accumulated[0].id == "call_123"
        assert accumulated[0].name == "get_weather"
        assert accumulated[0].input == {"location": "SF"}

    def test_custom_tool_call_delta_is_normalized_to_function(self):
        """Custom tool call deltas are normalized to function tool calls in SSE.

        Chat Completions clients reject ``type: "custom"`` tool calls ("unknown
        variant `custom`, expected `function`"), so the transformer rewrites
        them into the ``type: "function"`` shape with the raw input mapped to
        ``function.arguments``.
        """
        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="test-custom")

        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_custom_1",
                                    "type": "custom",
                                    "custom": {"name": "exec", "input": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "custom": {"input": "const r = 1;"}}]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]

        emitted = "".join(t for c in chunks if (t := transformer.transform(c)))
        assert '"type":"function"' in emitted
        assert '"type":"custom"' not in emitted
        assert '"custom":{' not in emitted

        # The tool call is still accumulated for tracing as a CustomToolUseBlock.
        from llm_proxy.models.content_blocks import CustomToolUseBlock

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], CustomToolUseBlock)
        assert accumulated[0].id == "call_custom_1"
        assert accumulated[0].name == "exec"
        assert accumulated[0].input == "const r = 1;"

    def test_accumulate_reasoning_content(self):
        """Test that reasoning content is accumulated during streaming."""
        from llm_proxy.models.content_blocks import TextBlock, ThinkingBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="test-789")

        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "Let me think..."},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {"content": "The answer is 42"}, "finish_reason": None}
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 2
        assert isinstance(accumulated[0], ThinkingBlock)
        assert accumulated[0].thinking == "Let me think..."
        assert isinstance(accumulated[1], TextBlock)
        assert accumulated[1].text == "The answer is 42"

    def test_multiple_tool_calls(self):
        """Test accumulation of multiple tool calls."""
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="test-multi")

        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "tool_a", "arguments": '{"a": 1}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "tool_b", "arguments": '{"b": 2}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 2
        tool_blocks = [b for b in accumulated if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 2
        assert tool_blocks[0].name == "tool_a"
        assert tool_blocks[1].name == "tool_b"

    def test_multiple_tool_calls_with_thought_signature(self):
        """Test accumulation of multiple tool calls with thought_signature."""
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="test-multi")

        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "tool_a", "arguments": '{"a": 1}'},
                                    "thought_signature": "sig_a_123",
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "tool_b", "arguments": '{"b": 2}'},
                                    "thought_signature": "sig_b_456",
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        tool_blocks = [b for b in accumulated if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 2
        assert tool_blocks[0].extra.get("thought_signature") == "sig_a_123"
        assert tool_blocks[1].extra.get("thought_signature") == "sig_b_456"


class TestOpenAIStreamingTransformerStringInput:
    """Test string (SSE) input handling in OpenAIStreamingTransformer.

    OpenAI Responses provider yields SSE strings directly, so the transformer
    must handle string input by parsing SSE data lines and converting them to
    clean OpenAI format.
    """

    def test_handles_sse_string_chunk(self):
        transformer = OpenAIStreamingTransformer(model="test-model", request_id="test-id")

        sse_line = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            '"created":123,"model":"test",'
            '"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        )
        result = transformer.transform(sse_line)
        assert result is not None
        assert result.startswith("data: ")
        assert "[DONE]" not in result

    def test_handles_multiple_sse_lines(self):
        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="test-123")

        sse_line1 = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            '"created":123,"model":"gpt-4",'
            '"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
        )
        sse_line2 = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            '"created":123,"model":"gpt-4",'
            '"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n'
        )
        sse_done = "data: [DONE]\n\n"

        transformer.transform(sse_line1)
        transformer.transform(sse_line2)
        result = transformer.transform(sse_done)

        assert result is None  # [DONE] should be filtered

    def test_ignores_non_data_lines(self):
        transformer = OpenAIStreamingTransformer(model="test-model", request_id="test-id")

        result = transformer.transform("event: response.created\n")
        assert result is None

    def test_handles_empty_string(self):
        transformer = OpenAIStreamingTransformer(model="test-model", request_id="test-id")

        result = transformer.transform("")
        assert result is None

    def test_handles_none_value(self):
        transformer = OpenAIStreamingTransformer(model="test-model", request_id="test-id")

        result = transformer.transform(None)
        assert result is None


class TestOpenAIStreamingTransformerContinuation:
    """Web-search continuation support for the OpenAI streaming transformer."""

    def test_continuation_creates_fresh_transformer(self):
        """continuation() must return a usable transformer instance so the
        web-search continuation loop can pump the follow-up stream through it
        (previously missing — the loop raised
        "Transformer OpenAIStreamingTransformer does not implement
        'continuation'")."""
        cont = OpenAIStreamingTransformer.continuation(
            model="gemini-3.1-flash-lite", request_id="resp_1", start_index=3
        )
        assert isinstance(cont, OpenAIStreamingTransformer)
        assert cont.model == "gemini-3.1-flash-lite"
        assert cont.response_id == "resp_1"
        # Fresh buffers: the continuation must not inherit the original
        # stream's accumulated text/tool state.
        assert cont._text_buffer == ""
        assert cont._tool_calls_buffer == {}
        assert cont.get_accumulated_output() == []

    def test_continuation_transforms_followup_chunks(self):
        """A continuation transformer must transform follow-up chunks into
        SSE frames (the model's answer after the search results)."""
        cont = OpenAIStreamingTransformer.continuation(
            model="gemini-3.1-flash-lite", request_id="resp_1", start_index=0
        )
        frame = cont.transform(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gemini-3.1-flash-lite",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "The capital of France is Paris."},
                        "finish_reason": None,
                    }
                ],
            }
        )
        assert frame is not None
        assert "The capital of France is Paris." in frame
        # Text accumulates in the buffer until the terminal chunk finalizes it.
        assert cont._text_buffer == "The capital of France is Paris."
        cont.transform(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gemini-3.1-flash-lite",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        assert cont.get_accumulated_output()[0].text == "The capital of France is Paris."
