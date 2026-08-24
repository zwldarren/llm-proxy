"""Tests for streaming output accumulation across all protocols."""

import orjson

from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer
from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer
from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer
from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer


class TestOpenAIAccumulation:
    """Test OpenAI streaming transformer accumulation."""

    def test_accumulate_text_content(self):
        """Should accumulate text content during streaming."""
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
        """Should accumulate tool calls during streaming."""
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
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": 'ation": "SF"}'}}]
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


class TestAnthropicAccumulation:
    """Test Anthropic streaming transformer accumulation."""

    def test_accumulate_text_content(self):
        """Should accumulate text content during streaming."""
        from llm_proxy.models.content_blocks import TextBlock

        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="test-456")

        # Anthropic transformer receives OpenAI-format chunks
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
        """Should accumulate tool calls during streaming."""
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="test-789")

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
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": 'ation": "SF"}'}}]
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


class TestGeminiAccumulation:
    """Test Gemini streaming transformer accumulation."""

    def test_accumulate_text_content(self):
        """Should accumulate text content during streaming."""
        from llm_proxy.models.content_blocks import TextBlock

        transformer = GeminiStreamingTransformer(model="gemini-2.0-flash", request_id="test-789")

        chunks = [
            {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": " world!"}]}}]},
            {"candidates": [{"finishReason": "STOP"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], TextBlock)
        assert accumulated[0].text == "Hello world!"

    def test_accumulate_tool_calls(self):
        """Should accumulate tool calls during streaming."""
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = GeminiStreamingTransformer(model="gemini-2.0-flash", request_id="test-012")

        chunks = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"location": "SF"},
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            {"candidates": [{"finishReason": "STOP"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], ToolUseBlock)
        assert accumulated[0].name == "get_weather"
        assert accumulated[0].input == {"location": "SF"}


class TestOpenResponsesAccumulation:
    """Test OpenResponses streaming transformer accumulation."""

    def test_accumulate_text_content(self):
        """Should accumulate text content during streaming."""
        from llm_proxy.models.content_blocks import TextBlock

        transformer = OpenResponsesStreamingTransformer(model="gpt-4", request_id="test-012")

        # OpenResponses takes OpenAI-format string chunks
        chunks = [
            (
                'data: {"choices":[{"index":0,"delta":{"content":"Hello"},'
                '"finish_reason":null}]}\n\n'
            ),
            (
                'data: {"choices":[{"index":0,"delta":{"content":" world!"},'
                '"finish_reason":null}]}\n\n'
            ),
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], TextBlock)
        assert accumulated[0].text == "Hello world!"

    def test_response_completed_includes_reasoning_and_text_content(self):
        """Final response.completed event must retain reasoning and message content."""
        transformer = OpenResponsesStreamingTransformer(model="gpt-4", request_id="test-013")

        chunks = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "Let me think"}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": " about this"}}]},
            {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
            {"choices": [{"index": 0, "delta": {"content": " world!"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]

        events = ""
        for chunk in chunks:
            events += transformer.transform(chunk) or ""

        completed_event = None
        for line in events.split("\n\n"):
            if "response.completed" in line:
                data_prefix = "data: "
                data_start = line.find(data_prefix)
                if data_start != -1:
                    completed_event = orjson.loads(line[data_start + len(data_prefix) :])
                break

        assert completed_event is not None
        output = completed_event["response"]["output"]
        assert len(output) == 2

        reasoning_item = output[0]
        assert reasoning_item["type"] == "reasoning"
        assert reasoning_item["status"] == "completed"
        # Reasoning text is carried in summary parts (industry convention).
        assert reasoning_item["summary"][0]["type"] == "summary_text"
        assert reasoning_item["summary"][0]["text"] == "Let me think about this"

        message_item = output[1]
        assert message_item["type"] == "message"
        assert message_item["status"] == "completed"
        assert message_item["role"] == "assistant"
        assert message_item["phase"] == "final_answer"
        assert message_item["content"][0]["text"] == "Hello world!"

    def test_accumulate_tool_calls(self):
        """Should accumulate tool calls during streaming."""
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = OpenResponsesStreamingTransformer(model="gpt-4", request_id="test-345")

        chunks = [
            (
                'data: {"choices": [{"index": 0, "delta": {"tool_calls": '
                '[{"index": 0, "id": "call_123", "type": "function", '
                '"function": {"name": "get_weather", "arguments": "{\\"loc"}}]}, '
                '"finish_reason": null}]}\n\n'
            ),
            (
                'data: {"choices": [{"index": 0, "delta": {"tool_calls": '
                '[{"index": 0, "function": {"arguments": "ation\\": \\"SF\\"}"}}]}, '
                '"finish_reason": null}]}\n\n'
            ),
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}\n\n',
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], ToolUseBlock)
        assert accumulated[0].id == "call_123"
        assert accumulated[0].name == "get_weather"
        assert accumulated[0].input == {"location": "SF"}
