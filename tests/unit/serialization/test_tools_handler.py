from llm_proxy.models.tools import OpenAIToolSearchTool
from llm_proxy.serialization.openai.components.tools_handler import OpenAIToolsHandler


class TestBuildTools:
    def test_tool_search_to_function(self):
        result = OpenAIToolsHandler().build_tools([OpenAIToolSearchTool()])
        assert len(result) == 1
        f = result[0]["function"]
        assert f["name"] == "tool_search"
        assert "query" in f["parameters"]["properties"]
        assert "limit" in f["parameters"]["properties"]
        assert "query" in f["parameters"]["required"]
