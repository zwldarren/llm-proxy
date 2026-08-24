"""OpenAI tools handler component.

Responsible for building OpenAI-format tool definitions and tool_choice values
from internal tool models.
"""

import logging
from typing import Any

from llm_proxy.models.tools import (
    CustomTool,
    FunctionTool,
    OpenAIToolSearchTool,
    OpenAIWebSearchTool,
    ToolChoice,
    ToolChoiceAllowedTools,
    ToolChoiceCustom,
    ToolChoiceFunction,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


def _build_function_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    """Build a standard OpenAI function-tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class OpenAIToolsHandler:
    """Builds OpenAI Chat Completions tool formats from internal tool models."""

    def build_tools(
        self, tools: list[ToolDefinition], target_endpoint: str = "chat_completions"
    ) -> list[dict[str, Any]]:
        """Convert internal tool list to OpenAI tools format.

        ``target_endpoint`` selects the upstream API shape:
        - ``"responses"`` (OpenAI Responses API): ``custom`` tools are a native
          tool type and are forwarded as ``{"type": "custom", ...}``.
        - ``"chat_completions"``: only ``type: "function"`` tools are valid.
          ``custom`` tools have no equivalent, so they are dropped (with a
          warning) — the same treatment ``tool_search`` receives in the protocol
          converter.
        """
        result: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, FunctionTool):
                func_def: dict[str, Any] = {
                    "name": tool.name,
                    "parameters": tool.parameters,
                }
                if tool.description:
                    func_def["description"] = tool.description
                if tool.strict:
                    func_def["strict"] = tool.strict
                result.append({"type": "function", "function": func_def})
            elif isinstance(tool, OpenAIToolSearchTool):
                # Convert OpenAI Responses tool_search to a standard function
                # tool so that providers only supporting Chat Completions API
                # can recognize and invoke it — same pattern as web_search below.
                result.append(
                    _build_function_tool(
                        name="tool_search",
                        description=(
                            "Search for tools available to the assistant. "
                            "Use this when the user asks for a specific tool "
                            "or capability that you don't currently have."
                        ),
                        properties={
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant tools",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of tools to return (default: 10)",
                                "default": 10,
                            },
                        },
                        required=["query"],
                    )
                )
            elif isinstance(tool, OpenAIWebSearchTool):
                # Convert OpenAI Responses web_search tool to a standard function
                # tool so that providers only supporting Chat Completions API
                # (DeepSeek, OpenRouter, etc.) can recognize and invoke it.
                # When a web search interceptor is configured (e.g. SearXNG),
                # WebSearchStage replaces this tool before it reaches here.
                result.append(
                    _build_function_tool(
                        name="web_search",
                        description=(
                            "Search the web for current information. "
                            "Use this tool when you need up-to-date "
                            "information beyond your knowledge cutoff."
                        ),
                        properties={
                            "query": {
                                "type": "string",
                                "description": "The search query string",
                            },
                        },
                        required=["query"],
                    )
                )
            elif isinstance(tool, CustomTool):
                if target_endpoint == "responses":
                    # OpenAI Responses API supports custom tools natively (flat format).
                    tool_def: dict[str, Any] = {"type": "custom", "name": tool.name}
                    if tool.description:
                        tool_def["description"] = tool.description
                    if tool.format_type:
                        # Responses API custom tools use the flat grammar shape
                        # ({"type": "grammar", "definition": ..., "syntax": ...});
                        # the wrapped {"grammar": {...}} shape is rejected.
                        format_dict: dict[str, Any] = {"type": tool.format_type}
                        if tool.format_type == "grammar" and tool.grammar_definition:
                            format_dict["definition"] = tool.grammar_definition
                            if tool.grammar_syntax:
                                format_dict["syntax"] = tool.grammar_syntax
                        tool_def["format"] = format_dict
                    result.append(tool_def)
                else:
                    # Chat Completions: convert custom tool to a function tool with
                    # a single ``content`` string parameter. Embed grammar in the
                    # description so the model can produce correctly-formatted output.
                    # On the response side, the streaming transformer and
                    # format_response will unwrap the content string back into the
                    # ``custom_tool_call.input`` field expected by Codex.
                    result.append(
                        {
                            "type": "function",
                            "function": self._custom_tool_to_function(tool),
                        }
                    )
            else:
                logger.warning(
                    "Skipping unsupported tool type for OpenAI provider: %s (%s)",
                    type(tool).__name__,
                    getattr(tool, "type", "unknown"),
                )
        return result

    def build_tool_choice(self, tool_choice: Any, target_endpoint: str = "chat_completions") -> Any:
        """Convert internal tool_choice to OpenAI format."""
        match tool_choice:
            case ToolChoice():
                return tool_choice.mode
            case ToolChoiceFunction():
                return {"type": "function", "function": {"name": tool_choice.name}}
            case ToolChoiceCustom():
                if target_endpoint == "responses":
                    return {"type": "custom", "name": tool_choice.name}
                # Custom tool is converted to a function tool for Chat Completions.
                # Force the model to call this specific tool by name.
                return {"type": "function", "function": {"name": tool_choice.name}}
            case ToolChoiceAllowedTools():
                if target_endpoint == "responses":
                    return {
                        "type": "allowed_tools",
                        "tools": tool_choice.allowed_tools.tools,
                        "mode": tool_choice.allowed_tools.mode,
                    }
                # Chat Completions has no allowed_tools concept; the tool set is
                # filtered to the allowed subset by the request builder, so the
                # selection mode alone is the correct tool_choice here.
                return tool_choice.allowed_tools.mode
            case dict():
                return self._build_tool_choice_from_dict(tool_choice)
            case _:
                return tool_choice

    @staticmethod
    def _custom_tool_to_function(tool: Any) -> dict[str, Any]:
        """Convert a CustomTool to a Chat Completions function-tool definition.

        Custom tools (freeform/grammar) have no JSON-schema parameters, so we
        wrap them as a function tool with a single ``content`` string param.
        Grammar definitions are embedded in the description so the model can
        produce correctly-formatted output.

        Mirrors LiteLLM's ``convert_custom_tool_to_function_tool``.
        """
        description = tool.description or ""
        if tool.format_type == "grammar" and tool.grammar_definition:
            syntax = tool.grammar_syntax or ""
            definition = tool.grammar_definition
            description += f"\n\nFormat:\n```{syntax}\n{definition}\n```"
        return {
            "name": tool.name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": (f"The {tool.name} content following the specified format"),
                    }
                },
                "required": ["content"],
            },
        }

    @staticmethod
    def _build_tool_choice_from_dict(tc: dict[str, Any]) -> Any:
        """Build OpenAI tool_choice from a dict value."""
        match tc.get("type"):
            case "auto" | "none" | "required":
                return tc["type"]
            case "any":
                return "required"
            case "tool":
                name = tc.get("name")
                if name:
                    return {"type": "function", "function": {"name": name}}
            case "function":
                func = tc.get("function")
                if func and isinstance(func, dict) and func.get("name"):
                    return {"type": "function", "function": {"name": func["name"]}}
            case "allowed_tools":
                return {
                    "type": "allowed_tools",
                    "allowed_tools": {
                        "mode": tc.get("mode", "auto"),
                        "tools": tc.get("tools", []),
                    },
                }
            case "custom":
                custom = tc.get("custom")
                if custom and isinstance(custom, dict) and custom.get("name"):
                    return {"type": "custom", "custom": {"name": custom["name"]}}
        return tc


__all__ = ["OpenAIToolsHandler"]
