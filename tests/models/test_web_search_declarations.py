"""Shared web-search declaration rules (``models/tools``).

``declares_web_search_tool`` is the one predicate behind "search is already
spoken for" — used by ``with_web_search_options`` (Responses / Gemini wire
bodies) and by the Anthropic serializer before it appends the hosted tool.
Getting it wrong in either direction is expensive: a false negative
double-searches upstream (the proxy intercepts *and* the provider executes),
a false positive silently drops the client's search.
"""

import pytest

from llm_proxy.models.tools import (
    AnthropicWebSearchTool,
    FunctionTool,
    HostedTool,
    OpenAIWebSearchTool,
    declares_web_search_tool,
)
from llm_proxy.models.tools.openai_builtin import with_web_search_options


class TestDeclaresWebSearchTool:
    @pytest.mark.parametrize(
        "tool",
        [
            OpenAIWebSearchTool(),
            OpenAIWebSearchTool(type="web_search_preview"),
            OpenAIWebSearchTool(type="web_search_2025_08_26"),
            AnthropicWebSearchTool(type="web_search_20250305"),
            FunctionTool(name="web_search", parameters={}),
            FunctionTool(name="WebSearch", parameters={}),
            {"type": "web_search"},
            {"type": "function", "function": {"name": "web_search"}},
        ],
    )
    def test_recognizes_every_shape(self, tool):
        assert declares_web_search_tool([tool]) is True

    @pytest.mark.parametrize(
        "tools",
        [
            None,
            [],
            [FunctionTool(name="calculator", parameters={})],
            [{"type": "function", "name": "search_docs"}],
        ],
    )
    def test_non_search_declarations(self, tools):
        assert declares_web_search_tool(tools) is False


class TestWithWebSearchOptions:
    def test_no_options_leaves_tools_untouched(self):
        tools = [FunctionTool(name="calculator", parameters={})]

        assert with_web_search_options(tools, None) == tools

    def test_dict_options_append_the_hosted_tool(self):
        tools = with_web_search_options(None, {"search_context_size": "medium"})

        assert len(tools) == 1
        assert isinstance(tools[0], OpenAIWebSearchTool)
        assert tools[0].search_context_size == "medium"

    def test_bare_true_uses_defaults(self):
        tools = with_web_search_options([], True)

        assert tools == [OpenAIWebSearchTool()]

    def test_existing_declaration_wins(self):
        declared = AnthropicWebSearchTool(type="web_search_20250305")

        assert with_web_search_options([declared], {"search_context_size": "low"}) == [declared]

    def test_input_list_is_not_mutated(self):
        tools: list = []

        with_web_search_options(tools, {"search_context_size": "low"})

        assert tools == []


class TestHostedToolFromWire:
    """The one home of the "strip ``type``, keep the rest" intake rule."""

    def test_carries_type_and_remaining_fields(self):
        tool = HostedTool.from_wire({"type": "google_search", "dynamic_threshold": 0.5})

        assert tool.type == "google_search"
        assert tool.name == "google_search"
        assert tool.params == {"dynamic_threshold": 0.5}

    def test_a_bare_type_carries_no_params(self):
        tool = HostedTool.from_wire({"type": "x_search"})

        assert tool.params == {}

    def test_coerces_a_non_string_type(self):
        tool = HostedTool.from_wire({"type": 7})

        assert tool.type == "7"
        assert tool.name == "7"
        assert tool.params == {}
