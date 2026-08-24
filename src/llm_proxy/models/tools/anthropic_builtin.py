"""Anthropic built-in tool definitions.

These are Anthropic-specific tool types that are NOT exported from the
default namespace. cache_control is declared explicitly on each tool
since the base ToolDefinition (in core.py) no longer includes it.
"""

from dataclasses import dataclass
from typing import Any, Literal

from llm_proxy.models.tools.core import ToolDefinition


@dataclass
class UserLocation:
    """User location for web search tools."""

    type: Literal["approximate"] = "approximate"
    city: str | None = None
    country: str | None = None
    region: str | None = None
    timezone: str | None = None


@dataclass
class BashTool(ToolDefinition):
    """Bash tool for code execution (Anthropic built-in)."""

    name: Literal["bash"] = "bash"
    type: Literal["bash_20250124"] = "bash_20250124"
    cache_control: dict[str, Any] | None = None
    allowed_callers: (
        list[Literal["direct", "code_execution_20250825", "code_execution_20260120"]] | None
    ) = None
    defer_loading: bool | None = None
    input_examples: list[dict[str, Any]] | None = None
    strict: bool | None = None


@dataclass
class CodeExecutionTool(ToolDefinition):
    """Code execution tool (Anthropic built-in)."""

    name: Literal["code_execution"] = "code_execution"
    type: Literal[
        "code_execution_20250522", "code_execution_20250825", "code_execution_20260120"
    ] = "code_execution_20250522"
    cache_control: dict[str, Any] | None = None
    allowed_callers: (
        list[Literal["direct", "code_execution_20250825", "code_execution_20260120"]] | None
    ) = None
    defer_loading: bool | None = None
    strict: bool | None = None


@dataclass
class MemoryTool(ToolDefinition):
    """Memory tool (Anthropic built-in)."""

    name: Literal["memory"] = "memory"
    type: Literal["memory_20250818"] = "memory_20250818"
    cache_control: dict[str, Any] | None = None
    allowed_callers: (
        list[Literal["direct", "code_execution_20250825", "code_execution_20260120"]] | None
    ) = None
    defer_loading: bool | None = None
    input_examples: list[dict[str, Any]] | None = None
    strict: bool | None = None


@dataclass
class TextEditorTool(ToolDefinition):
    """Text editor tool (Anthropic built-in)."""

    name: Literal["str_replace_editor", "str_replace_based_edit_tool"] = "str_replace_editor"
    type: Literal["text_editor_20250124", "text_editor_20250429", "text_editor_20250728"] = (
        "text_editor_20250124"
    )
    cache_control: dict[str, Any] | None = None
    allowed_callers: (
        list[Literal["direct", "code_execution_20250825", "code_execution_20260120"]] | None
    ) = None
    defer_loading: bool | None = None
    input_examples: list[dict[str, Any]] | None = None
    max_characters: int | None = None
    strict: bool | None = None


@dataclass
class ToolSearchTool(ToolDefinition):
    """Tool search tool (Anthropic built-in)."""

    name: Literal["tool_search_tool_bm25", "tool_search_tool_regex"] = "tool_search_tool_bm25"
    type: Literal[
        "tool_search_tool_bm25_20251119",
        "tool_search_tool_bm25",
        "tool_search_tool_regex_20251119",
        "tool_search_tool_regex",
    ] = "tool_search_tool_bm25_20251119"
    cache_control: dict[str, Any] | None = None
    allowed_callers: (
        list[Literal["direct", "code_execution_20250825", "code_execution_20260120"]] | None
    ) = None
    defer_loading: bool | None = None
    strict: bool | None = None


@dataclass
class WebSearchTool(ToolDefinition):
    """Web search tool with optional context size and user location.

    Native Anthropic built-in web_search tool.
    """

    name: Literal["web_search"] = "web_search"
    type: Literal["web_search_20250305", "web_search_20260209"] = "web_search_20250305"
    cache_control: dict[str, Any] | None = None
    allowed_callers: (
        list[Literal["direct", "code_execution_20250825", "code_execution_20260120"]] | None
    ) = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    defer_loading: bool | None = None
    max_uses: int | None = None
    strict: bool | None = None
    user_location: UserLocation | None = None


@dataclass
class WebFetchTool(ToolDefinition):
    """Web fetch tool."""

    name: Literal["web_fetch"] = "web_fetch"
    type: Literal["web_fetch_20250910", "web_fetch_20260209"] = "web_fetch_20250910"
    cache_control: dict[str, Any] | None = None
    allowed_callers: (
        list[Literal["direct", "code_execution_20250825", "code_execution_20260120"]] | None
    ) = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    citations: dict[str, Any] | None = None
    defer_loading: bool | None = None
    max_content_tokens: int | None = None
    max_uses: int | None = None
    strict: bool | None = None
