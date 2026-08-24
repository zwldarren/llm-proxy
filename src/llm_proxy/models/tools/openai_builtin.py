"""OpenAI built-in tool definitions."""

from dataclasses import dataclass
from typing import Any, Literal

from llm_proxy.models.tools.core import ToolDefinition


@dataclass
class WebSearchTool(ToolDefinition):
    """Web search tool (OpenAI built-in).

    Maps to OpenAI Responses API `web_search` / `web_search_preview`
    built-in tool parameters.
    """

    name: Literal["web_search"] = "web_search"
    type: Literal["web_search", "web_search_preview"] = "web_search"
    search_context_size: Literal["low", "medium", "high"] | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: dict[str, Any] | None = None
    external_web_access: bool | None = None
    return_token_budget: Literal["default", "unlimited"] | None = None
    search_content_types: list[str] | None = None
    image_settings: dict[str, Any] | None = None
    max_uses: int | None = None


@dataclass
class CodeInterpreterTool(ToolDefinition):
    """Code interpreter tool (OpenAI built-in)."""

    name: Literal["code_interpreter"] = "code_interpreter"
    type: Literal["code_interpreter"] = "code_interpreter"


@dataclass
class FileSearchTool(ToolDefinition):
    """File search tool (OpenAI built-in)."""

    name: Literal["file_search"] = "file_search"
    type: Literal["file_search"] = "file_search"
    vector_store_ids: list[str] | None = None
    max_num_results: int | None = None
    ranking_options: dict[str, Any] | None = None


@dataclass
class OpenAIToolSearchTool(ToolDefinition):
    """OpenAI Responses API hosted tool-search. Converted to function tool for Chat providers."""

    name: str = "tool_search"
    type: str = "tool_search"


__all__ = [
    "CodeInterpreterTool",
    "FileSearchTool",
    "OpenAIToolSearchTool",
    "WebSearchTool",
]
