"""Protocol-agnostic message extraction for the routing engine.

Returns the request messages as a list of ``{"role", "content"}`` dicts,
preserving the original ``content`` shape (strings, multimodal arrays, tool
blocks). Signal helpers flatten content to text when they need it; smart
routing does not transform multimodal or tool-call payloads.

Supports both ``/v1/chat/completions`` (``messages`` field) and
``/v1/responses`` (``input`` field as ``str | list[ItemParam]``).
"""

import json
from typing import Any


def extract_messages_for_routing(request: Any) -> list[dict[str, Any]]:
    # ── /v1/chat/completions path: ``messages`` field ──
    messages = getattr(request, "messages", None)
    if messages:
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = _get(msg, "role", "user")
            content = _get(msg, "content", "")
            out.append({"role": role, "content": content})
        return out

    # ── /v1/responses path: ``input`` field (str | list[ItemParam]) ──
    inp = getattr(request, "input", None)
    if inp is None:
        return []

    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]

    if isinstance(inp, list):
        out: list[dict[str, Any]] = []
        for item in inp:
            item_dict = (
                dict(item)
                if hasattr(item, "model_dump")
                else (item if isinstance(item, dict) else {})
            )
            item_type = item_dict.get("type", "")

            if item_type == "message":
                role = item_dict.get("role", "user")
                content = item_dict.get("content", "")
            elif item_type == "reasoning":
                role = "assistant"
                # Reasoning content is in 'content' field as list of output_text blocks
                content_list = item_dict.get("content", [])
                content = _extract_text_from_content_list(content_list)
            elif item_type == "function_call":
                role = "assistant"
                # Function call: name in 'name', arguments in 'arguments'
                name = item_dict.get("name", "")
                args = item_dict.get("arguments", "{}")
                content = f"{name}({args})"
            elif item_type == "function_call_output":
                role = "tool"
                # Function output: `output` is a plain string or an array of
                # structured content items (Codex content_items form).
                content = function_call_output_to_text(item_dict.get("output", ""))
            elif item_type in ("custom_tool_call", "local_shell_call", "tool_search_call"):
                role = "assistant"
                content = _tool_call_like_summary(item_dict)
            elif item_type in ("custom_tool_call_output", "tool_search_output"):
                role = "tool"
                content = _tool_output_like_summary(item_dict)
            elif item_type == "agent_message":
                role = "user"
                content = _agent_message_text(item_dict.get("content", []))
            elif item_type in (
                "item_reference",
                "web_search_call",
                "image_generation_call",
                "additional_tools",
                "compaction",
                "compaction_summary",
                "compaction_trigger",
                "context_compaction",
            ):
                # No useful routing text / no Chat Completions equivalent.
                continue
            else:
                # Unknown type, try generic extraction
                role = item_dict.get("role", "user")
                content = item_dict.get("content", item_dict.get("output", ""))

            out.append({"role": role, "content": content})
        return out

    return []


def _extract_text_from_content_list(content_list: Any) -> str:
    """Extract text from a list of content blocks."""
    if not content_list:
        return ""
    if isinstance(content_list, str):
        return content_list
    if isinstance(content_list, list):
        parts = []
        for part in content_list:
            if isinstance(part, dict):
                # Handle output_text, input_text, etc.
                part_type = part.get("type", "")
                if "text" in part_type:
                    parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return str(content_list) if content_list else ""


def function_call_output_to_text(output: Any) -> str:
    """Flatten a ``function_call_output.output`` value to plain text.

    Per the OpenAI Responses API (and the Codex wire format), ``output`` may be
    either a plain string or an array of structured content items
    (``input_text`` / ``input_image`` / ``encrypted_content``). Text is extracted
    from ``input_text`` items and joined with newlines; image and encrypted
    content items are skipped (they carry no renderable text for the downstream
    Chat Completions tool result). This mirrors Codex's
    ``function_call_output_content_items_to_text``.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: list[str] = []
        for part in output:
            if isinstance(part, dict):
                if part.get("type") == "input_text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(output) if output else ""


def _tool_call_like_summary(item_dict: dict[str, Any]) -> str:
    """Best-effort text summary of a tool-call-like item for routing signals."""
    item_type = item_dict.get("type")
    if item_type == "custom_tool_call":
        return f"{item_dict.get('name', '')}({item_dict.get('input', '{}')})"
    if item_type == "local_shell_call":
        return f"local_shell({item_dict.get('action', {})})"
    if item_type == "tool_search_call":
        return f"tool_search({item_dict.get('arguments', {})})"
    return ""


def _tool_output_like_summary(item_dict: dict[str, Any]) -> str:
    """Best-effort text summary of a tool-output-like item for routing signals."""
    item_type = item_dict.get("type")
    if item_type in ("custom_tool_call_output", "local_shell_call_output"):
        return function_call_output_to_text(item_dict.get("output", ""))
    if item_type == "tool_search_output":
        tools = item_dict.get("tools")
        return json.dumps(tools) if tools else ""
    return ""


def _agent_message_text(content: Any) -> str:
    """Extract readable text from an agent_message content array for routing."""
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "input_text":
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
