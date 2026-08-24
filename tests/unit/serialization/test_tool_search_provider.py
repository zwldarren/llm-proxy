"""Tests for OpenAIToolSearchTool handling across provider serializers.

Verifies that tool_search is correctly converted to a function tool for
providers that don't natively support the OpenAI Responses tool_search type.
"""

from llm_proxy.models.tools import OpenAIToolSearchTool
from llm_proxy.serialization.anthropic.serializer import AnthropicProviderSerializer
from llm_proxy.serialization.gemini.request_builder import GeminiRequestBuilderMixin
from llm_proxy.serialization.ollama.request_builder import OllamaRequestBuilderMixin

_TOOL_SEARCH_EXPECTED_KEYS = {"query", "limit"}
_TOOL_SEARCH_NAME = "tool_search"


class TestAnthropicToolSearch:
    """Verify Anthropic serializer converts OpenAIToolSearchTool to function tool."""

    def test_tool_search_to_function(self):
        serializer = AnthropicProviderSerializer()
        result = serializer._build_tools([OpenAIToolSearchTool()])
        assert len(result) == 1
        tool = result[0]
        assert tool["name"] == _TOOL_SEARCH_NAME
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert set(tool["input_schema"]["properties"]) == _TOOL_SEARCH_EXPECTED_KEYS
        assert "query" in tool["input_schema"]["required"]

    def test_tool_search_alongside_function_tool(self):
        from llm_proxy.models.tools import FunctionTool

        serializer = AnthropicProviderSerializer()
        func = FunctionTool(name="get_weather", parameters={"type": "object"})
        result = serializer._build_tools([OpenAIToolSearchTool(), func])
        assert len(result) == 2
        names = [t["name"] for t in result]
        assert _TOOL_SEARCH_NAME in names
        assert "get_weather" in names


class TestGeminiToolSearch:
    """Verify Gemini request builder converts OpenAIToolSearchTool to function declaration."""

    def _make_builder(self):
        # GeminiRequestBuilder is a mixin; instantiate via a minimal subclass
        class _TestBuilder(GeminiRequestBuilderMixin):
            def build_request(self, *args, **kwargs):
                raise NotImplementedError

        return _TestBuilder()

    def test_tool_search_to_function(self):
        builder = self._make_builder()
        result = builder._convert_tools_to_gemini([OpenAIToolSearchTool()])
        assert result is not None
        assert "function_declarations" in result
        decls = result["function_declarations"]
        assert len(decls) == 1
        decl = decls[0]
        assert decl["name"] == _TOOL_SEARCH_NAME
        assert "description" in decl
        assert "parameters" in decl
        assert decl["parameters"]["type"] == "object"
        assert set(decl["parameters"]["properties"]) == _TOOL_SEARCH_EXPECTED_KEYS
        assert "query" in decl["parameters"]["required"]

    def test_tool_search_alongside_function_tool(self):
        from llm_proxy.models.tools import FunctionTool

        builder = self._make_builder()
        func = FunctionTool(name="get_weather", parameters={"type": "object"})
        result = builder._convert_tools_to_gemini([OpenAIToolSearchTool(), func])
        assert result is not None
        decls = result["function_declarations"]
        assert len(decls) == 2
        names = [d["name"] for d in decls]
        assert _TOOL_SEARCH_NAME in names
        assert "get_weather" in names


class TestGeminiCustomTool:
    """Verify Gemini request builder converts CustomTool to function declaration."""

    def _make_builder(self):
        class _TestBuilder(GeminiRequestBuilderMixin):
            def build_request(self, *args, **kwargs):
                raise NotImplementedError

        return _TestBuilder()

    def test_custom_tool_to_function(self):
        from llm_proxy.models.tools import CustomTool

        builder = self._make_builder()
        custom = CustomTool(name="my_custom_tool", description="A custom tool")
        result = builder._convert_tools_to_gemini([custom])
        assert result is not None
        assert "function_declarations" in result
        decls = result["function_declarations"]
        assert len(decls) == 1
        decl = decls[0]
        assert decl["name"] == "my_custom_tool"
        assert decl["description"] == "A custom tool"
        assert "parameters" in decl
        assert decl["parameters"]["type"] == "object"
        assert "content" in decl["parameters"]["properties"]
        assert decl["parameters"]["properties"]["content"]["type"] == "string"
        assert "content" in decl["parameters"]["required"]

    def test_custom_tool_without_description(self):
        from llm_proxy.models.tools import CustomTool

        builder = self._make_builder()
        custom = CustomTool(name="bare_tool")
        result = builder._convert_tools_to_gemini([custom])
        assert result is not None
        decl = result["function_declarations"][0]
        assert decl["name"] == "bare_tool"
        assert "description" not in decl

    def test_custom_tool_alongside_function_tool(self):
        from llm_proxy.models.tools import CustomTool, FunctionTool

        builder = self._make_builder()
        custom = CustomTool(name="my_custom")
        func = FunctionTool(name="get_weather", parameters={"type": "object"})
        result = builder._convert_tools_to_gemini([custom, func])
        assert result is not None
        decls = result["function_declarations"]
        assert len(decls) == 2
        names = [d["name"] for d in decls]
        assert "my_custom" in names
        assert "get_weather" in names


class TestOllamaToolSearch:
    """Verify Ollama request builder converts OpenAIToolSearchTool to function tool."""

    def _make_builder(self):
        class _TestBuilder(OllamaRequestBuilderMixin):
            def build_request(self, *args, **kwargs):
                raise NotImplementedError

        return _TestBuilder()

    def test_tool_search_to_function(self):
        builder = self._make_builder()
        result = builder._convert_tools_to_ollama([OpenAIToolSearchTool()])
        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] == _TOOL_SEARCH_NAME
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert set(fn["parameters"]["properties"]) == _TOOL_SEARCH_EXPECTED_KEYS
        assert "query" in fn["parameters"]["required"]

    def test_tool_search_alongside_function_tool(self):
        from llm_proxy.models.tools import FunctionTool

        builder = self._make_builder()
        func = FunctionTool(name="get_weather", parameters={"type": "object"})
        result = builder._convert_tools_to_ollama([OpenAIToolSearchTool(), func])
        assert len(result) == 2
        names = [t["function"]["name"] for t in result]
        assert _TOOL_SEARCH_NAME in names
        assert "get_weather" in names
