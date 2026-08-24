"""Tests for OpenAIToolSearchTool model."""

from llm_proxy.models.tools import OpenAIToolSearchTool, ToolDefinition


class TestOpenAIToolSearchTool:
    def test_is_tool_definition(self):
        assert isinstance(OpenAIToolSearchTool(), ToolDefinition)

    def test_default_name_and_type(self):
        tool = OpenAIToolSearchTool()
        assert tool.name == "tool_search"
        assert tool.type == "tool_search"

    def test_distinct_from_anthropic(self):
        from llm_proxy.models.tools import ToolSearchTool as Ant

        assert not isinstance(OpenAIToolSearchTool(), Ant)
