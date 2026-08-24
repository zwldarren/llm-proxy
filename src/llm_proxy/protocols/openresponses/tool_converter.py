"""Tool converter for OpenResponses protocol.

Converts raw Responses API tool dictionaries into protocol-agnostic
``ToolDefinition`` objects, handling namespace flattening.
"""

import logging
from typing import Any

from llm_proxy.models.tools.core import CustomTool, FunctionTool, ToolDefinition
from llm_proxy.models.tools.openai_builtin import OpenAIToolSearchTool, WebSearchTool
from llm_proxy.serialization.responses_toolkit.namespace import NamespaceMapping

logger = logging.getLogger(__name__)


def _parse_tool(tool: dict[str, Any]) -> FunctionTool:
    """Parse a function-type tool dict into a FunctionTool."""
    src = tool.get("function", tool)
    return FunctionTool(
        name=src.get("name", ""),
        description=src.get("description"),
        parameters=src.get("parameters", {"type": "object"}),
        strict=tool.get("strict", False),
    )


def _build_web_search_tool(tool_dict: dict[str, Any]) -> WebSearchTool:
    """Build a WebSearchTool from a web_search or web_search_preview tool dict."""
    tool_type = tool_dict.get("type", "web_search")
    loc = tool_dict.get("user_location")
    allowed_domains = tool_dict.get("allowed_domains")
    blocked_domains = tool_dict.get("blocked_domains")
    filters = tool_dict.get("filters")
    if isinstance(filters, dict):
        if allowed_domains is None:
            allowed_domains = filters.get("allowed_domains")
        if blocked_domains is None:
            blocked_domains = filters.get("blocked_domains")

    return WebSearchTool(
        name="web_search",
        type=tool_type,
        user_location=(
            {
                "type": loc.get("type", "approximate"),
                "city": loc.get("city"),
                "country": loc.get("country"),
                "region": loc.get("region"),
                "timezone": loc.get("timezone"),
            }
            if isinstance(loc, dict)
            else None
        ),
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        search_context_size=tool_dict.get("search_context_size"),
        external_web_access=tool_dict.get("external_web_access"),
        return_token_budget=tool_dict.get("return_token_budget"),
        search_content_types=tool_dict.get("search_content_types"),
        image_settings=tool_dict.get("image_settings"),
        max_uses=tool_dict.get("max_uses"),
    )


def _build_custom_tool(tool_dict: dict[str, Any]) -> CustomTool:
    """Build a CustomTool from a custom-type tool dict."""
    name = tool_dict.get("name", "")
    description = tool_dict.get("description")
    format_info = tool_dict.get("format", {}) or {}
    is_grammar = format_info.get("type") == "grammar"
    # Accept both the flat Responses shape ({"type": "grammar",
    # "definition": ..., "syntax": ...}) and the legacy wrapped shape
    # ({"type": "grammar", "grammar": {...}}).
    grammar_info = format_info.get("grammar") or format_info if is_grammar else {}
    return CustomTool(
        name=name,
        description=description,
        format_type=format_info.get("type"),
        grammar_definition=grammar_info.get("definition") if is_grammar else None,
        grammar_syntax=grammar_info.get("syntax") if is_grammar else None,
    )


# Responses API built-in tool types that have no protocol-agnostic ToolDefinition
# equivalent. They are preserved verbatim in InternalRequest.extra so that a
# native Responses provider (e.g. OpenAI) can forward them as-is, while other
# providers can decide to drop them with an explicit warning.
_RESPONSES_BUILTIN_PASSTHROUGH_TYPES: frozenset[str] = frozenset(
    {"file_search", "code_interpreter"}
)


def convert_responses_tools(
    raw_tools: list[dict[str, Any]],
) -> tuple[list[ToolDefinition], NamespaceMapping | None, list[dict[str, Any]]]:
    """Convert raw Responses API tool dicts into ToolDefinition objects.

    Handles:
    - ``function`` → ``FunctionTool``
    - ``tool_search`` → ``OpenAIToolSearchTool``
    - ``web_search`` / ``web_search_preview`` → ``OpenAIWebSearchTool``
    - ``custom`` → ``CustomTool``
    - ``namespace`` → child tools are flattened and a ``NamespaceMapping`` is returned
    - ``file_search`` / ``code_interpreter`` and any other Responses-only or
      unknown tool type (``computer_use`` / ``mcp`` / future types) are NOT
      silently dropped: they have no protocol-agnostic ``ToolDefinition``
      equivalent, so they are preserved verbatim in the returned
      ``preserved_tools`` list so a native Responses provider can forward them.
      A warning is logged for types that are not universally supported.

    Returns:
        A tuple of (list of ToolDefinition, NamespaceMapping or None,
        list of preserved raw tool dicts).
    """
    tools: list[ToolDefinition] = []
    namespace_mapping: NamespaceMapping | None = None
    preserved_tools: list[dict[str, Any]] = []

    for tool in raw_tools:
        tool_dict = tool if isinstance(tool, dict) else {}
        tool_type = tool_dict.get("type")

        if tool_type in _RESPONSES_BUILTIN_PASSTHROUGH_TYPES:
            # file_search / code_interpreter are valid Responses API built-in
            # tools but have no protocol-agnostic ToolDefinition. Preserve the
            # raw definition so a native Responses provider (OpenAI) can forward
            # it; non-native providers will drop it (with their own warning).
            logger.debug(
                "Preserving Responses built-in tool %r in extra for native "
                "provider forwarding; non-Responses providers will drop it.",
                tool_type,
            )
            preserved_tools.append(tool_dict)
            continue
        elif tool_type in ("web_search", "web_search_preview"):
            tools.append(_build_web_search_tool(tool_dict))
        elif tool_type == "function":
            tools.append(_parse_tool(tool_dict))
        elif tool_type == "custom":
            tools.append(_build_custom_tool(tool_dict))
        elif tool_type == "tool_search":
            tools.append(OpenAIToolSearchTool(name="tool_search", type="tool_search"))
        elif tool_type == "namespace":
            ns_name = tool_dict.get("name", "")
            child_tools = tool_dict.get("tools") or []
            if not ns_name:
                logger.warning("Skipping namespace tool with missing or empty 'name'")
                continue
            if not child_tools:
                logger.warning("Skipping namespace %r with empty 'tools' list", ns_name)
                continue
            if namespace_mapping is None:
                namespace_mapping = NamespaceMapping()
            for child in child_tools:
                child_dict = child if isinstance(child, dict) else {}
                child_type = child_dict.get("type")
                if child_type == "function":
                    fn = _parse_tool(child_dict)
                    if not fn.name:
                        logger.warning(
                            "Skipping unnamed function tool inside namespace %r",
                            ns_name,
                        )
                        continue
                    flat_name = namespace_mapping.flatten(ns_name, fn.name)
                    fn.name = flat_name
                    tools.append(fn)
                elif child_type == "custom":
                    custom_tool = _build_custom_tool(child_dict)
                    if not custom_tool.name:
                        logger.warning(
                            "Skipping unnamed custom tool inside namespace %r",
                            ns_name,
                        )
                        continue
                    flat_name = namespace_mapping.flatten(ns_name, custom_tool.name)
                    custom_tool.name = flat_name
                    tools.append(custom_tool)
                elif child_type in ("web_search", "web_search_preview", "tool_search"):
                    logger.warning(
                        "Skipping namespace %r child tool of built-in type %r: "
                        "built-in tools cannot be namespaced.",
                        ns_name,
                        child_type,
                    )
                else:
                    logger.warning(
                        "Skipping namespace %r child tool of unsupported type %r",
                        ns_name,
                        child_type,
                    )
        else:
            # Unknown / Responses-only tool type (e.g. computer_use, mcp,
            # future built-ins). Preserve the raw definition so a native
            # Responses provider can still forward it, but warn that
            # non-OpenAI providers cannot execute it and will drop it.
            logger.warning(
                "Preserving Responses API tool of type %r in extra; it has no "
                "Chat Completions equivalent and non-OpenAI providers will drop it.",
                tool_type,
            )
            preserved_tools.append(tool_dict)

    return tools, namespace_mapping, preserved_tools
