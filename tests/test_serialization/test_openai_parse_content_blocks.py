"""Tests for OpenAI parse_content_blocks handling of Anthropic-native blocks.

TDD: Tests written before implementation. Covers the gaps in
parse_content_blocks() where Anthropic-native content types are not
recognized and silently dropped.
"""

import pytest

from llm_proxy.models import TextBlock
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer


@pytest.fixture
def serializer():
    return OpenAIProtocolSerializer()


class TestParseDocumentBlock:
    """Document blocks should be parsed (degraded to text)."""

    def test_parse_document_block(self, serializer):
        """Document block is parsed as text."""
        content = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": "Hello world"},
            }
        ]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert "document" in blocks[0].text.lower()


class TestParseToolUseBlock:
    """Tool use blocks should be parsed (degraded to text)."""

    def test_parse_tool_use_block(self, serializer):
        """Tool use block is parsed as text."""
        content = [
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"location": "SF"}}
        ]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert "get_weather" in blocks[0].text


class TestParseServerToolUseBlock:
    """Server tool use blocks should be parsed (degraded to text)."""

    def test_parse_server_tool_use_block(self, serializer):
        """Server tool use block is parsed as text."""
        content = [
            {
                "type": "server_tool_use",
                "id": "stu_1",
                "name": "web_search",
                "input": {"query": "weather"},
            }
        ]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert "web_search" in blocks[0].text


class TestParseThinkingBlock:
    """Thinking blocks should be parsed (degraded to text)."""

    def test_parse_thinking_block(self, serializer):
        """Thinking block is parsed as text."""
        content = [{"type": "thinking", "thinking": "I need to think about this"}]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert "thinking" in blocks[0].text.lower()

    def test_parse_redacted_thinking_block(self, serializer):
        """Redacted thinking block is parsed as text."""
        content = [{"type": "redacted_thinking", "data": "redacted_data_here"}]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert "redacted" in blocks[0].text.lower()


class TestParseToolResultBlock:
    """Tool result blocks should be parsed (degraded to text)."""

    def test_parse_tool_result_block(self, serializer):
        """Tool result block is parsed as text."""
        content = [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "The weather is sunny"}
        ]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert "sunny" in blocks[0].text

    def test_tool_result_with_list_content(self, serializer):
        """Tool result with list content is parsed as text."""
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": [{"type": "text", "text": "Result line"}],
            }
        ]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert "Result line" in blocks[0].text


class TestParseWebSearchBlocks:
    """Web search blocks should be parsed (degraded to text)."""

    def test_parse_web_search_tool_result(self, serializer):
        """Web search tool result block is parsed as text."""
        content = [
            {
                "type": "web_search_tool_result",
                "tool_use_id": "ws_1",
                "content": [{"type": "text", "text": "search result"}],
            }
        ]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) >= 1
        assert any(isinstance(b, TextBlock) for b in blocks)
        assert any("search" in b.text.lower() for b in blocks if isinstance(b, TextBlock))

    def test_parse_web_search_result_content(self, serializer):
        """Web search result content block is parsed as text."""
        content = [{"type": "web_search_result", "url": "https://example.com", "title": "Example"}]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) >= 1
        assert any(isinstance(b, TextBlock) for b in blocks)
        assert any("Example" in b.text for b in blocks if isinstance(b, TextBlock))


class TestParseCodeExecutionBlocks:
    """Code execution blocks should be parsed (degraded to text)."""

    def test_parse_code_execution_result(self, serializer):
        """Code execution tool result block is parsed as text."""
        content = [
            {
                "type": "code_execution_tool_result",
                "tool_use_id": "ce_1",
                "content": [{"type": "text", "text": "code output"}],
            }
        ]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) >= 1
        assert any(isinstance(b, TextBlock) for b in blocks)
        assert any("code" in b.text.lower() for b in blocks if isinstance(b, TextBlock))

    def test_parse_bash_execution_result(self, serializer):
        """Bash execution result block is parsed as text."""
        content = [
            {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "be_1",
                "content": [{"type": "text", "text": "bash output"}],
            }
        ]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) >= 1
        assert any(isinstance(b, TextBlock) for b in blocks)
        assert any("bash" in b.text.lower() for b in blocks if isinstance(b, TextBlock))

    def test_parse_text_editor_result(self, serializer):
        """Text editor execution result block is parsed as text."""
        content = [
            {
                "type": "text_editor_code_execution_tool_result",
                "tool_use_id": "te_1",
                "content": [{"type": "text", "text": "editor output"}],
            }
        ]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) >= 1
        assert any(isinstance(b, TextBlock) for b in blocks)
        assert any("editor" in b.text.lower() for b in blocks if isinstance(b, TextBlock))


class TestParseToolSearchResultBlock:
    """Tool search result blocks should be parsed (degraded to text)."""

    def test_parse_tool_search_result(self, serializer):
        """Tool search result block is parsed as text."""
        content = [
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "ts_1",
                "content": [{"type": "text", "text": "search output"}],
            }
        ]
        blocks = serializer.parse_content_blocks(content)
        assert len(blocks) >= 1
        assert any(isinstance(b, TextBlock) for b in blocks)
        assert any("search" in b.text.lower() for b in blocks if isinstance(b, TextBlock))
