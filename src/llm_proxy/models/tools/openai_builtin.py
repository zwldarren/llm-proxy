"""OpenAI built-in tool definitions."""

from dataclasses import dataclass
from typing import Any, Literal

from llm_proxy.models.tools.core import (
    ToolDefinition,
    declares_web_search_tool,
)


@dataclass(frozen=True)
class WebSearchTool(ToolDefinition):
    """Web search tool (OpenAI built-in).

    Maps to OpenAI Responses API `web_search` / `web_search_preview`
    built-in tool parameters. ``type`` is forwarded verbatim because
    providers also ship dated variants of the same hosted tool (DeepSeek
    accepts ``web_search_2025_08_26``), exactly like the Anthropic-side
    ``web_search_20250305`` family.

    ``max_uses`` is not an OpenAI Responses field — it belongs to the
    Anthropic builtin and the OpenRouter dialect, which read it from this
    model; the OpenAI Responses serializer never emits it.
    """

    name: Literal["web_search"] = "web_search"
    type: str = "web_search"
    search_context_size: Literal["low", "medium", "high"] | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: dict[str, Any] | None = None
    external_web_access: bool | None = None
    return_token_budget: Literal["default", "unlimited"] | None = None
    search_content_types: list[str] | None = None
    image_settings: dict[str, Any] | None = None
    max_uses: int | None = None


@dataclass(frozen=True)
class CodeInterpreterTool(ToolDefinition):
    """Code interpreter tool (OpenAI built-in)."""

    name: Literal["code_interpreter"] = "code_interpreter"
    type: Literal["code_interpreter"] = "code_interpreter"


@dataclass(frozen=True)
class FileSearchTool(ToolDefinition):
    """File search tool (OpenAI built-in)."""

    name: Literal["file_search"] = "file_search"
    type: Literal["file_search"] = "file_search"
    vector_store_ids: list[str] | None = None
    max_num_results: int | None = None
    ranking_options: dict[str, Any] | None = None


@dataclass(frozen=True)
class OpenAIToolSearchTool(ToolDefinition):
    """OpenAI Responses API hosted tool-search. Converted to function tool for Chat providers."""

    name: str = "tool_search"
    type: str = "tool_search"


def normalized_user_location(tool: WebSearchTool) -> dict[str, Any] | None:
    """The tool's ``user_location`` in the wire shape every vendor shares.

    OpenAI, Anthropic and OpenRouter document the same object
    (``type``/``city``/``country``/``region``/``timezone``), so the proxy emits
    one normalized form and drops the keys the client left unset instead of
    forwarding explicit nulls.
    """
    if tool.user_location is None:
        return None
    location: dict[str, Any] = {"type": tool.user_location.get("type") or "approximate"}
    for key in ("city", "country", "region", "timezone"):
        value = tool.user_location.get(key)
        if value is not None:
            location[key] = value
    return location


def with_web_search_options(tools: list[Any] | None, options: Any) -> list[Any]:
    """Tools list carrying the hosted ``web_search`` tool that ``options`` implies.

    ``web_search_options`` is the OpenAI-standard "please search the web"
    signal on Chat Completions, where ``gpt-5-search-api`` consumes it in
    place. Providers that answer in a protocol without such a field (Responses
    bodies, Gemini ``generateContent``) have to translate it into the hosted
    tool it stands for — otherwise a search request silently reaches the
    upstream with no search at all, which is exactly what happens whenever the
    proxy has no web-search interceptor of its own to fall back on.

    A declared web-search tool wins over the option: a hosted declaration is
    already explicit, and a *function* tool named ``web_search`` is either the
    proxy's own interception bridge (the interceptor owns the search) or a
    client that executes the search itself. Neither wants a second copy.

    Args:
        tools: The request's tool list (may be ``None`` or empty).
        options: ``params.openai.web_search_options`` — a dict of controls or
            a bare ``True`` for defaults.

    Returns:
        A new list; the input list is never mutated.
    """
    resolved = list(tools or [])
    if not options:
        return resolved
    if declares_web_search_tool(resolved):
        return resolved
    if isinstance(options, bool):
        return [*resolved, WebSearchTool()]
    if not isinstance(options, dict):
        return resolved
    user_location = options.get("user_location")
    return [
        *resolved,
        WebSearchTool(
            search_context_size=options.get("search_context_size"),
            user_location=user_location if isinstance(user_location, dict) else None,
        ),
    ]


__all__ = [
    "CodeInterpreterTool",
    "FileSearchTool",
    "OpenAIToolSearchTool",
    "WebSearchTool",
    "normalized_user_location",
    "with_web_search_options",
]
