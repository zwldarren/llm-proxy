"""Universal tool definition types for unified protocol format."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolDefinition:
    """Base class for all tool definitions."""

    name: str


@dataclass
class FunctionTool(ToolDefinition):
    """Function tool definition with name, parameters, and optional description."""

    parameters: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    strict: bool = False
    allowed_callers: list[str] | None = None
    defer_loading: bool | None = None
    eager_input_streaming: bool | None = None
    input_examples: list[dict[str, Any]] | None = None


@dataclass
class CustomTool(ToolDefinition):
    """Custom tool definition for provider-specific tool types."""

    description: str | None = None
    format_type: str | None = None
    grammar_definition: str | None = None
    grammar_syntax: str | None = None


# Tool choice types


@dataclass
class ToolChoice:
    """Tool choice specifying the mode of tool usage."""

    mode: Literal["auto", "none", "required", "any"]
    name: str | None = None
    disable_parallel_tool_use: bool | None = None


@dataclass
class ToolChoiceNamed:
    """Tool choice specifying a specific tool by name (Anthropic type='tool')."""

    type: Literal["tool"] = "tool"
    name: str | None = None
    disable_parallel_tool_use: bool | None = None


@dataclass
class ToolChoiceFunction:
    """Tool choice specifying a specific function by name."""

    name: str
    type: Literal["function"] = "function"


@dataclass
class ToolChoiceCustom:
    """Tool choice specifying a specific custom tool by name."""

    name: str
    type: Literal["custom"] = "custom"


@dataclass
class AllowedToolsConfig:
    """Configuration for allowed tools constraint."""

    mode: Literal["auto", "required"]
    tools: list[dict[str, Any]]


@dataclass
class ToolChoiceAllowedTools:
    """Tool choice that constrains available tools to a pre-defined set."""

    allowed_tools: AllowedToolsConfig
    type: Literal["allowed_tools"] = "allowed_tools"


# Type alias for tool choice specification
ToolChoiceSpec = (
    ToolChoice
    | ToolChoiceFunction
    | ToolChoiceCustom
    | ToolChoiceAllowedTools
    | ToolChoiceNamed
    | str
)


def is_web_search_tool_name(name: str) -> bool:
    """Check if a tool name corresponds to web search.

    Normalizes by removing underscores/hyphens and lowercasing
    to handle model variations (e.g., "web_search", "WebSearch",
    "WEB_SEARCH", "websearch").

    Args:
        name: The tool name to check

    Returns:
        True if the name refers to a web search tool
    """
    return name.lower().replace("_", "").replace("-", "") == "websearch"
