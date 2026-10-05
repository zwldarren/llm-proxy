"""Wire-shape-agnostic reading and writing of OpenAI tool definitions.

Tool definitions arrive in two shapes: the Responses API flat shape
(controls at the top level) and the Chat Completions nested shape (the
definition under ``function``/``custom``). Both are read so the
programmatic / async / tool-search controls survive the hop to a native
OpenAI upstream instead of being silently dropped, and written back out
when a Responses target is built.

The ``allowed_callers`` / ``defer_loading`` / ``async`` controls exist on
both the official ``FunctionTool`` and ``CustomToolParam`` schemas, and
``output_schema`` only on ``FunctionTool``.
"""

from typing import Any

from llm_proxy.models.tools.core import CustomTool, FunctionTool


def read_tool_controls(tool: dict[str, Any]) -> dict[str, Any]:
    """Extract the tool controls shared by function and custom tools.

    Returns kwargs ready to be splatted into a ``FunctionTool`` or
    ``CustomTool`` constructor.
    """
    return {
        "allowed_callers": tool.get("allowed_callers"),
        "defer_loading": tool.get("defer_loading"),
        "async_": tool.get("async"),
    }


def write_tool_controls(tool_def: dict[str, Any], tool: FunctionTool | CustomTool) -> None:
    """Emit the shared Responses tool controls onto a provider tool dict.

    Inverse of ``read_tool_controls``: only controls the client actually set
    are written, so an absent ``async`` (or an empty ``allowed_callers``) never
    reaches the upstream as a null/empty field. Only the native Responses API
    understands these; emitting them is what keeps programmatic / async /
    tool-search tool calling working through the proxy.
    """
    if tool.allowed_callers:
        tool_def["allowed_callers"] = tool.allowed_callers
    if tool.defer_loading is not None:
        tool_def["defer_loading"] = tool.defer_loading
    if tool.async_ is not None:
        tool_def["async"] = tool.async_


def parse_function_tool(tool: dict[str, Any]) -> FunctionTool:
    """Parse a function-type tool dict into a ``FunctionTool``.

    The definition may sit at the top level (Responses shape) or under
    ``function`` (Chat Completions shape). ``strict`` preserves an explicit
    ``null`` on the Responses shape, falls back to the nested Chat value, and
    defaults to ``False``. ``output_schema`` is a Responses-only control, so
    only the top level is read.
    """
    src = tool.get("function", tool)
    # Responses puts ``strict`` on the top level; Chat nests it under
    # ``function``. An explicit ``strict: null`` on the Responses shape is
    # preserved rather than defaulted away.
    strict = tool["strict"] if "strict" in tool else src.get("strict", False)
    return FunctionTool(
        name=src.get("name", ""),
        description=src.get("description"),
        parameters=src.get("parameters", {"type": "object"}),
        strict=strict,
        output_schema=tool.get("output_schema"),
        **read_tool_controls(tool),
    )
