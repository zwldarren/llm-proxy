"""Tests for OpenAI provider parse_provider_response with structured content.

TDD: Tests written before implementation. Covers the case where an
OpenAI-compatible provider returns message.content as a list of structured
content parts instead of a simple string.
"""

from llm_proxy.models import TextBlock
from llm_proxy.serialization.providers.chat_completions import OpenAIProviderSerializer


class TestStructuredContentParts:
    """parse_provider_response handles structured content parts."""

    def test_string_content_unchanged(self):
        """String content is still parsed correctly."""
        serializer = OpenAIProviderSerializer()
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello, world!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = serializer.parse_provider_response(response, model="gpt-4")
        assert len(result.output) == 1
        assert isinstance(result.output[0], TextBlock)
        assert result.output[0].text == "Hello, world!"

    def test_structured_content_parts_extracted(self):
        """List content parts are extracted to TextBlocks."""
        serializer = OpenAIProviderSerializer()
        response = {
            "id": "chatcmpl-456",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": "World"},
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = serializer.parse_provider_response(response, model="gpt-4")
        texts = [b.text for b in result.output if isinstance(b, TextBlock)]
        assert len(texts) >= 1
        assert "Hello" in "".join(texts)
        assert "World" in "".join(texts)

    def test_structured_content_with_images(self):
        """Image parts in structured content produce TextBlocks with markers."""
        serializer = OpenAIProviderSerializer()
        response = {
            "id": "chatcmpl-789",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Here is an image:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/img.png"},
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = serializer.parse_provider_response(response, model="gpt-4")
        texts = [b.text for b in result.output if isinstance(b, TextBlock)]
        combined = " ".join(texts)
        assert "Here is an image:" in combined

    def test_mixed_string_and_list_not_applicable(self):
        """When content is None, other fields (tool_calls, etc.) still work."""
        serializer = OpenAIProviderSerializer()
        response = {
            "id": "chatcmpl-101",
            "model": "gpt-4",
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
                                "function": {"name": "get_weather", "arguments": '{"loc": "SF"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = serializer.parse_provider_response(response, model="gpt-4")
        assert len(result.output) == 1
        from llm_proxy.models import ToolUseBlock

        assert isinstance(result.output[0], ToolUseBlock)
