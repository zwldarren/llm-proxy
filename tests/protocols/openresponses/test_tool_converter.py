"""Tests for convert_responses_tools tool converter module."""

import logging

from llm_proxy.models import CustomTool, FunctionTool
from llm_proxy.models.tools import OpenAIToolSearchTool, OpenAIWebSearchTool
from llm_proxy.protocols.openresponses.serializer import extract_custom_tool_names
from llm_proxy.protocols.openresponses.tool_converter import convert_responses_tools
from llm_proxy.serialization.responses_toolkit.namespace import restore_tool_name


class TestConvertResponsesTools:
    def test_function_passthrough(self):
        tools, ns, preserved = convert_responses_tools([{"type": "function", "name": "f"}])
        assert len(tools) == 1 and isinstance(tools[0], FunctionTool)
        assert ns is None and preserved == []

    def test_tool_search_not_dropped(self):
        tools, _, preserved = convert_responses_tools([{"type": "tool_search"}])
        assert len(tools) == 1 and isinstance(tools[0], OpenAIToolSearchTool)
        assert preserved == []

    def test_tool_search_plus_function_both_survive(self):
        tools, _, preserved = convert_responses_tools(
            [{"type": "tool_search"}, {"type": "function", "name": "f"}]
        )
        assert len(tools) == 2
        assert any(isinstance(t, OpenAIToolSearchTool) for t in tools)
        assert preserved == []

    def test_web_search_converted(self):
        tools, _, preserved = convert_responses_tools([{"type": "web_search"}])
        assert isinstance(tools[0], OpenAIWebSearchTool)
        assert preserved == []

    def test_custom_converted(self):
        tools, _, preserved = convert_responses_tools(
            [
                {
                    "type": "custom",
                    "name": "patch",
                    "format": {"type": "grammar", "grammar": {"definition": "x"}},
                }
            ]
        )
        assert isinstance(tools[0], CustomTool) and tools[0].name == "patch"
        assert preserved == []

    def test_custom_converted_flat_grammar_format(self):
        # Codex sends the flat Responses shape (definition/syntax at the top
        # level of format); it must parse the same as the legacy wrapped shape.
        tools, _, preserved = convert_responses_tools(
            [
                {
                    "type": "custom",
                    "name": "exec",
                    "format": {
                        "type": "grammar",
                        "definition": "start: SOURCE",
                        "syntax": "lark",
                    },
                }
            ]
        )
        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, CustomTool)
        assert tool.format_type == "grammar"
        assert tool.grammar_definition == "start: SOURCE"
        assert tool.grammar_syntax == "lark"
        assert preserved == []

    def test_code_interpreter_preserved_not_dropped(self):
        raw = [{"type": "code_interpreter", "container": {"type": "auto"}}]
        tools, _, preserved = convert_responses_tools(raw)
        # No protocol-agnostic ToolDefinition, but the raw dict is preserved.
        assert tools == []
        assert preserved == raw

    def test_file_search_preserved_not_dropped(self):
        raw = [{"type": "file_search", "vector_store_ids": ["vs1"]}]
        tools, _, preserved = convert_responses_tools(raw)
        assert tools == []
        assert preserved == raw

    def test_namespace_flattened(self):
        tools, ns, preserved = convert_responses_tools(
            [
                {
                    "type": "namespace",
                    "name": "mcp",
                    "tools": [{"type": "function", "name": "t1"}],
                }
            ]
        )
        assert len(tools) == 1
        assert tools[0].name == "mcp__t1"
        assert ns is not None
        assert preserved == []

    def test_namespace_mixed(self):
        tools, ns, preserved = convert_responses_tools(
            [
                {"type": "tool_search"},
                {
                    "type": "namespace",
                    "name": "mcp",
                    "tools": [{"type": "function", "name": "t1"}],
                },
                {"type": "function", "name": "f"},
            ]
        )
        assert len(tools) == 3 and ns is not None
        assert preserved == []

    def test_unknown_preserved_with_warning(self, caplog):
        raw = [{"type": "future", "foo": "bar"}]
        with caplog.at_level(logging.WARNING):
            tools, _, preserved = convert_responses_tools(raw)
        # Unknown types have no ToolDefinition but are preserved verbatim.
        assert tools == []
        assert preserved == raw
        assert any(
            "Preserving Responses API tool of type 'future'" in r.message for r in caplog.records
        )

    def test_namespace_custom_flattened(self):
        tools, ns, preserved = convert_responses_tools(
            [
                {
                    "type": "namespace",
                    "name": "mcp",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "patch",
                            "format": {"type": "grammar", "grammar": {"definition": "x"}},
                        }
                    ],
                }
            ]
        )
        assert len(tools) == 1
        assert isinstance(tools[0], CustomTool)
        assert tools[0].name == "mcp__patch"
        assert ns is not None
        assert restore_tool_name(ns.to_dict(), "mcp__patch") == ("patch", "mcp")
        assert preserved == []

    def test_namespace_built_in_child_skipped_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            tools, ns, preserved = convert_responses_tools(
                [
                    {
                        "type": "namespace",
                        "name": "mcp",
                        "tools": [{"type": "tool_search"}, {"type": "web_search"}],
                    }
                ]
            )
        assert tools == []
        assert ns is not None
        assert preserved == []
        assert any("built-in tools cannot be namespaced" in r.message for r in caplog.records)

    def test_extract_custom_tool_names_includes_namespaced_custom(self):
        raw = [
            {"type": "custom", "name": "patch"},
            {
                "type": "namespace",
                "name": "mcp",
                "tools": [{"type": "custom", "name": "patch"}],
            },
        ]
        assert extract_custom_tool_names(raw) == {"patch", "mcp__patch"}

    def test_computer_use_preserved_with_warning(self, caplog):
        raw = [{"type": "computer_use", "display_width": 1024, "display_height": 768}]
        with caplog.at_level(logging.WARNING):
            tools, _, preserved = convert_responses_tools(raw)
        assert tools == []
        assert preserved == raw
        assert any("computer_use" in r.message for r in caplog.records)

    def test_mcp_preserved_with_warning(self, caplog):
        raw = [{"type": "mcp", "server_label": "s", "server_url": "https://x"}]
        with caplog.at_level(logging.WARNING):
            tools, _, preserved = convert_responses_tools(raw)
        assert tools == []
        assert preserved == raw
        assert any("mcp" in r.message for r in caplog.records)

    def test_mixed_function_and_file_search(self):
        raw = [
            {"type": "function", "name": "f"},
            {"type": "file_search", "vector_store_ids": ["vs1"]},
        ]
        tools, _, preserved = convert_responses_tools(raw)
        assert len(tools) == 1 and isinstance(tools[0], FunctionTool)
        assert preserved == [raw[1]]
