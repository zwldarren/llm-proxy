"""Tests for CustomToolUseBlock handling in Anthropic formatting.

TDD: Tests written before implementation. Covers the silent drop of
CustomToolUseBlock in format_content_blocks().
"""

import pytest

from llm_proxy.models import CustomToolUseBlock
from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer


@pytest.fixture
def serializer():
    return AnthropicProtocolSerializer()


class TestCustomToolUseBlock:
    """CustomToolUseBlock should be formatted, not silently dropped."""

    def test_custom_tool_use_block_formatted(self, serializer):
        """CustomToolUseBlock produces a tool_use block."""
        block = CustomToolUseBlock(
            id="custom_1",
            name="my_custom_tool",
            input='{"key": "value"}',
        )
        # format_content_blocks is a protected method, call through format_response
        from llm_proxy.models import InternalResponse

        response = InternalResponse(id="test", model="claude", output=[block])
        result = serializer.format_response(response)

        content = result.get("content", [])
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"
        assert content[0]["id"] == "custom_1"
        assert content[0]["name"] == "my_custom_tool"
        assert content[0]["input"] == {"key": "value"}

    def test_custom_tool_use_block_string_input(self, serializer):
        """CustomToolUseBlock with non-JSON string input is wrapped."""
        block = CustomToolUseBlock(
            id="custom_2",
            name="string_tool",
            input="plain string input",
        )
        from llm_proxy.models import InternalResponse

        response = InternalResponse(id="test", model="claude", output=[block])
        result = serializer.format_response(response)

        content = result.get("content", [])
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"
        assert content[0]["id"] == "custom_2"
        assert content[0]["name"] == "string_tool"
        # String input should be wrapped in a dict under the same ``input``
        # key the bridged function-tool schema declares, so the model sees a
        # consistent format in history and echoes it back.
        assert isinstance(content[0]["input"], dict)
        assert content[0]["input"] == {"input": "plain string input"}
