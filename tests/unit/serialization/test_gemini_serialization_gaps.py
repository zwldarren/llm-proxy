"""Tests for Gemini serialization gaps.

TDD: Tests written before implementation. Covers:
- Build side: unhandled block types are text-degraded instead of silently dropped
- Parse side: inline_data/file_data parts are converted to ImageBlock
"""

from llm_proxy.models import (
    ConversationContext,
    ImageBlock,
    InternalRequest,
    Message,
    ServerToolUseBlock,
    TextBlock,
)
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.gemini.serializer import GeminiProviderSerializer


class TestGeminiBuildSideUnhandledBlocks:
    """Unhandled blocks are text-degraded when policy is degrade."""

    def _build_parts(self, blocks):
        """Helper to build Gemini parts from content blocks."""
        serializer = GeminiProviderSerializer()
        ctx = BuildContext(unsupported_block_policy="degrade")
        request2 = InternalRequest(
            model="gemini-2.0-flash",
            conversation=ConversationContext(
                messages=[
                    Message(role="user", content=[TextBlock(text="hi")]),
                    Message(role="assistant", content=blocks),
                ]
            ),
        )
        body2 = serializer.build_provider_request(request2, ctx)
        # Extract the assistant message parts
        for content in body2.get("contents", []):
            if content.get("role") == "model":
                return content.get("parts", [])
        return []

    def test_server_tool_use_block_degraded(self):
        """ServerToolUseBlock is text-degraded (not converted to functionCall)."""
        parts = self._build_parts(
            [
                ServerToolUseBlock(id="stu_1", name="web_search", input={"query": "weather"}),
            ]
        )
        assert len(parts) > 0
        text = " ".join(p.get("text", "") for p in parts)
        assert "web_search" in text.lower()

    def test_refusal_block_converted_to_text(self):
        """RefusalBlock is converted to TextBlock and rendered as plain text."""
        from llm_proxy.models import RefusalBlock

        parts = self._build_parts(
            [
                RefusalBlock(refusal="I cannot answer that"),
            ]
        )
        assert len(parts) == 1
        assert parts[0].get("text") == "I cannot answer that"

    def test_redacted_thinking_block_converted_to_thought(self):
        """RedactedThinkingBlock is converted to ThinkingBlock and rendered as thought."""
        from llm_proxy.models import RedactedThinkingBlock

        parts = self._build_parts(
            [
                RedactedThinkingBlock(data="redacted_data"),
            ]
        )
        assert len(parts) == 1
        assert parts[0].get("thought") is True
        assert "[Redacted thinking]" in parts[0].get("text", "")

    def test_custom_tool_use_block_wrapped_as_function_call(self):
        """CustomToolUseBlock is re-wrapped into the {"content": ...} bridge envelope."""
        from llm_proxy.models import CustomToolUseBlock

        parts = self._build_parts(
            [
                CustomToolUseBlock(id="ctu_1", name="custom_fn", input='{"key": "value"}'),
            ]
        )
        assert len(parts) == 1
        func_call = parts[0].get("functionCall")
        assert func_call is not None
        assert func_call["name"] == "custom_fn"
        assert func_call["args"] == {"content": '{"key": "value"}'}

    def test_web_search_tool_result_converted_to_function_response(self):
        """WebSearchToolResultBlock converts to ToolResultBlock -> functionResponse."""
        from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock

        parts = self._build_parts(
            [
                WebSearchToolResultBlock(tool_use_id="ws_1", content="search results here"),
            ]
        )
        assert len(parts) == 1
        assert "functionResponse" in parts[0]

    def test_code_execution_result_converted_to_function_response(self):
        """CodeExecutionToolResultBlock converts to ToolResultBlock -> functionResponse."""
        from llm_proxy.models.content_blocks.anthropic_builtin import CodeExecutionToolResultBlock

        parts = self._build_parts(
            [
                CodeExecutionToolResultBlock(tool_use_id="ce_1", content="code output"),
            ]
        )
        assert len(parts) == 1
        assert "functionResponse" in parts[0]

    def test_tool_reference_block_still_degraded(self):
        """ToolReferenceBlock has no conversion and is text-degraded under degrade policy."""
        from llm_proxy.models.content_blocks.anthropic_builtin import ToolReferenceBlock

        parts = self._build_parts(
            [
                ToolReferenceBlock(tool_id="tool_1", tool_name="my_tool"),
            ]
        )
        assert len(parts) > 0
        text = " ".join(p.get("text", "") for p in parts)
        assert "tool" in text.lower() or "my_tool" in text


class TestGeminiParseSideImageBlock:
    """Gemini response inline_data/file_data is converted to ImageBlock."""

    def test_inline_data_converted_to_image_block(self):
        """inline_data part with mime_type and data becomes ImageBlock."""
        serializer = GeminiProviderSerializer()
        provider_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mime_type": "image/png",
                                    "data": "base64encodeddata",
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
            "modelVersion": "gemini-2.0-flash",
        }
        response = serializer.parse_provider_response(provider_response, model="gemini-2.0-flash")
        assert any(isinstance(b, ImageBlock) for b in response.output)

    def test_file_data_converted_to_image_block(self):
        """file_data part with mime_type and file_uri becomes ImageBlock."""
        serializer = GeminiProviderSerializer()
        provider_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "fileData": {
                                    "mime_type": "image/png",
                                    "file_uri": "https://example.com/img.png",
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
            "modelVersion": "gemini-2.0-flash",
        }
        response = serializer.parse_provider_response(provider_response, model="gemini-2.0-flash")
        assert any(isinstance(b, ImageBlock) for b in response.output)

    def test_image_block_has_correct_source(self):
        """ImageBlock from inline_data has correct source type and data."""
        serializer = GeminiProviderSerializer()
        provider_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mime_type": "image/jpeg",
                                    "data": "somedata",
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
            "modelVersion": "gemini-2.0-flash",
        }
        response = serializer.parse_provider_response(provider_response, model="gemini-2.0-flash")
        image_blocks = [b for b in response.output if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 1
        img = image_blocks[0]
        assert img.source.type == "base64"
        assert img.source.data == "somedata"
        assert img.source.media_type == "image/jpeg"

    def test_image_with_text_and_tool_call(self):
        """ImageBlock coexists with text and tool calls in response."""
        serializer = GeminiProviderSerializer()
        provider_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Here is the image:"},
                            {
                                "inlineData": {
                                    "mime_type": "image/png",
                                    "data": "imgdata",
                                }
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
            "modelVersion": "gemini-2.0-flash",
        }
        response = serializer.parse_provider_response(provider_response, model="gemini-2.0-flash")
        has_text = any(isinstance(b, TextBlock) for b in response.output)
        has_image = any(isinstance(b, ImageBlock) for b in response.output)
        assert has_text
        assert has_image


class TestGeminiMarkdownImageParsing:
    """Nano-banana markdown image parsing for Gemini image models.

    Only ``![alt](data:image/…;base64,…)`` URIs in text are converted
    to Gemini ``inline_data`` parts.  HTTP URLs are never treated as
    images.  This is gated behind ``_is_gemini_image_model``.
    """

    @staticmethod
    def _convert(text: str, model: str = "gemini-2.5-flash-image") -> list[dict]:
        """Build Gemini parts from a single text message for *model*."""
        serializer = GeminiProviderSerializer()
        ctx = BuildContext(model=model, unsupported_block_policy="degrade")
        conv = ConversationContext(messages=[Message(role="user", content=[TextBlock(text=text)])])
        contents, _ = serializer._convert_conversation_to_gemini(conv, ctx)
        return contents[0]["parts"]

    # ── image model (nano banana) tests ──────────────────────────────

    def test_data_uri_converted_to_inline_data(self):
        parts = self._convert(
            "![image](data:image/png;base64,AAAA)",
            model="gemini-2.5-flash-image",
        )
        assert len(parts) == 1
        assert parts[0] == {
            "inline_data": {"mime_type": "image/png", "data": "AAAA"},
        }

    def test_text_before_and_after_image(self):
        parts = self._convert(
            "Before ![img](data:image/jpeg;base64,ZZZZ) After",
            model="gemini-2.5-flash-image",
        )
        assert len(parts) == 3
        assert parts[0] == {"text": "Before "}
        assert parts[1] == {
            "inline_data": {"mime_type": "image/jpeg", "data": "ZZZZ"},
        }
        assert parts[2] == {"text": " After"}

    def test_multiple_images(self):
        parts = self._convert(
            "A ![i1](data:image/png;base64,AA) B ![i2](data:image/gif;base64,BB) C",
            model="gemini-2.5-flash-image",
        )
        assert len(parts) == 5
        assert parts[0] == {"text": "A "}
        assert parts[1] == {"inline_data": {"mime_type": "image/png", "data": "AA"}}
        assert parts[2] == {"text": " B "}
        assert parts[3] == {"inline_data": {"mime_type": "image/gif", "data": "BB"}}
        assert parts[4] == {"text": " C"}

    def test_http_url_not_converted(self):
        """HTTP URLs are NEVER treated as images, even for image models."""
        parts = self._convert(
            "![img](https://example.com/photo.png)",
            model="gemini-2.5-flash-image",
        )
        assert len(parts) == 1
        assert parts[0] == {"text": "![img](https://example.com/photo.png)"}

    def test_invalid_base64_kept_as_text(self):
        """Malformed base64 stays as literal text."""
        parts = self._convert(
            "![img](data:image/png;base64,!!!not-valid-base64!!!)",
            model="gemini-2.5-flash-image",
        )
        assert len(parts) == 1
        assert parts[0]["text"] == "![img](data:image/png;base64,!!!not-valid-base64!!!)"

    def test_non_data_uri_kept_as_text(self):
        """Only data: URIs are parsed; other schemas stay as text."""
        parts = self._convert(
            "![img](file:///path/to/img.png)",
            model="gemini-2.5-flash-image",
        )
        assert len(parts) == 1
        assert parts[0] == {"text": "![img](file:///path/to/img.png)"}

    # ── non-image model tests ────────────────────────────────────────

    def test_data_uri_ignored_for_non_image_model(self):
        """Regular Gemini models do NOT parse markdown images."""
        parts = self._convert(
            "![image](data:image/png;base64,AAAA)",
            model="gemini-2.0-flash",
        )
        assert len(parts) == 1
        assert parts[0] == {
            "text": "![image](data:image/png;base64,AAAA)",
        }

    def test_plain_text_unchanged(self):
        """Plain text without markdown is unchanged for both model types."""
        for model in ("gemini-2.0-flash", "gemini-2.5-flash-image"):
            parts = self._convert("Hello world", model=model)
            assert len(parts) == 1
            assert parts[0] == {"text": "Hello world"}

    def test_image_preview_model_suffix(self):
        """Models ending with -image-preview are treated as image models."""
        parts = self._convert(
            "![img](data:image/png;base64,CC)",
            model="gemini-2.5-flash-image-preview",
        )
        assert len(parts) == 1
        assert parts[0] == {"inline_data": {"mime_type": "image/png", "data": "CC"}}
