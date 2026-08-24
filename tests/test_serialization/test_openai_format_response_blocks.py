"""Tests for OpenAI format_response block downgrades.

TDD: Tests written before implementation. Covers the silent drop of
ImageBlock, DocumentBlock, FileBlock, and ToolResultBlock in
_format_output_to_message().
"""

import orjson
import pytest

from llm_proxy.models import (
    DocumentBlock,
    DocumentSource,
    FileBlock,
    ImageBlock,
    InternalResponse,
    TextBlock,
    ToolResultBlock,
)
from llm_proxy.models.types import ImageSource
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer


@pytest.fixture
def serializer():
    return OpenAIProtocolSerializer()


def _format(serializer, output_blocks):
    """Helper to format blocks through format_response and extract content."""
    response = InternalResponse(id="test-id", model="gpt-4", output=output_blocks)
    result = serializer.format_response(response)
    return result["choices"][0]["message"]["content"] or ""


class TestImageBlockDowngrade:
    """ImageBlock should be degraded to markdown syntax, matching new-api behavior."""

    def test_image_block_as_markdown(self, serializer):
        """ImageBlock produces markdown image syntax, not a content array."""
        block = ImageBlock(
            source=ImageSource(type="base64", data="AAAA", media_type="image/png"),
        )
        content = _format(serializer, [block])
        assert isinstance(content, str), "Content should be a string with markdown"
        assert "![image](" in content
        assert "data:image/png;base64,AAAA" in content

    def test_image_block_with_media_type(self, serializer):
        """ImageBlock media_type should appear in the markdown data URL."""
        block = ImageBlock(
            source=ImageSource(type="base64", data="AAAA", media_type="image/jpeg"),
        )
        content = _format(serializer, [block])
        assert isinstance(content, str)
        assert "data:image/jpeg;base64,AAAA" in content

    def test_image_block_url_type(self, serializer):
        """ImageBlock with URL type should return markdown with the URL."""
        block = ImageBlock(
            source=ImageSource(
                type="url",
                data="https://example.com/image.png",
                media_type="image/png",
            ),
        )
        content = _format(serializer, [block])
        assert isinstance(content, str)
        assert "![image](https://example.com/image.png)" in content

    def test_text_and_image_interleaved(self, serializer):
        """Text and images should be concatenated into a single string with markdown."""
        blocks = [
            TextBlock(text="Here is your image:"),
            ImageBlock(source=ImageSource(type="base64", data="AAAA", media_type="image/png")),
            TextBlock(text="Let me know if you need changes."),
        ]
        content = _format(serializer, blocks)
        assert isinstance(content, str)
        assert "Here is your image:" in content
        assert "![image](data:image/png;base64,AAAA)" in content
        assert "Let me know if you need changes." in content


class TestDocumentBlockDowngrade:
    """DocumentBlock should be downgraded to text, not silently dropped."""

    def test_document_block_downgraded_to_text(self, serializer):
        """DocumentBlock produces '[Document: ...]' text instead of being dropped."""
        block = DocumentBlock(
            source=DocumentSource(type="text", media_type="text/plain", data="Hello"),
        )
        content = _format(serializer, [TextBlock(text="Here is a doc:"), block])
        assert "Here is a doc:" in content
        assert "document" in content.lower() or "Document" in content

    def test_document_block_with_title(self, serializer):
        """DocumentBlock title should appear in the text."""
        block = DocumentBlock(
            source=DocumentSource(type="text", media_type="application/pdf", data="..."),
            title="report.pdf",
        )
        content = _format(serializer, [block])
        assert "report.pdf" in content


class TestFileBlockDowngrade:
    """FileBlock should be downgraded to text, not silently dropped."""

    def test_file_block_downgraded_to_text(self, serializer):
        """FileBlock produces '[File: ...]' text instead of being dropped."""
        block = FileBlock(filename="data.csv")
        content = _format(serializer, [block])
        assert content, "Content should not be empty - FileBlock should produce text"

    def test_file_block_with_filename(self, serializer):
        """FileBlock filename should appear in the text."""
        block = FileBlock(filename="document.pdf", file_id="file_123")
        content = _format(serializer, [block])
        assert "document.pdf" in content or "file_123" in content


class TestToolResultBlockDowngrade:
    """ToolResultBlock should be downgraded to text, not silently dropped."""

    def test_tool_result_block_downgraded_to_text(self, serializer):
        """ToolResultBlock produces text content instead of being dropped."""
        block = ToolResultBlock(
            tool_use_id="toolu_123",
            content="The weather in San Francisco is sunny.",
            is_error=False,
        )
        content = _format(serializer, [TextBlock(text="Tool result:"), block])
        assert "Tool result:" in content
        assert "weather" in content or "San Francisco" in content

    def test_tool_result_with_list_content(self, serializer):
        """ToolResultBlock with list content is downgraded to text."""
        block = ToolResultBlock(
            tool_use_id="toolu_456",
            content=[TextBlock(text="Result line 1"), TextBlock(text="Result line 2")],
        )
        content = _format(serializer, [block])
        assert "Result line 1" in content
        assert "Result line 2" in content


class TestCustomToolUseBlockFormat:
    """CustomToolUseBlock must serialize as a function tool call for Chat clients.

    ``type: "custom"`` only exists in the Responses API. A Chat Completions
    client would reject it with "unknown variant `custom`, expected `function`",
    so the format layer re-wraps the freeform input in the ``{"content": ...}``
    bridge envelope instead.
    """

    def test_custom_tool_use_becomes_function_tool_call(self, serializer):
        """CustomToolUseBlock produces a type='function' tool call."""
        from llm_proxy.models import CustomToolUseBlock

        block = CustomToolUseBlock(
            id="call_custom_1",
            name="exec",
            input="const r = await tools.exec_command({ cmd: 'ls' });",
        )
        response = InternalResponse(id="test-id", model="gpt-4", output=[block])
        result = serializer.format_response(response)

        message = result["choices"][0]["message"]
        assert message.get("tool_calls") is not None
        tool_call = message["tool_calls"][0]
        assert tool_call["id"] == "call_custom_1"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "exec"
        assert orjson.loads(tool_call["function"]["arguments"]) == {
            "content": "const r = await tools.exec_command({ cmd: 'ls' });"
        }
