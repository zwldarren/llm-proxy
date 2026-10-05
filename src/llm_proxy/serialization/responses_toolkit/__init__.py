"""Shared Responses-shaped tooling.

Helpers understood by both the OpenResponses protocol module and
provider-family serializers: tool-name namespaces and Responses item helpers.
"""

from llm_proxy.serialization.responses_toolkit.items import (
    extract_reasoning_text,
    extract_summary_text,
    generate_item_id,
)
from llm_proxy.serialization.responses_toolkit.namespace import (
    NamespaceMapping,
    flatten_history_tool_name,
    restore_tool_name,
)
from llm_proxy.serialization.responses_toolkit.tools import (
    parse_function_tool,
    read_tool_controls,
    write_tool_controls,
)

__all__ = [
    "NamespaceMapping",
    "flatten_history_tool_name",
    "restore_tool_name",
    "generate_item_id",
    "extract_reasoning_text",
    "extract_summary_text",
    "parse_function_tool",
    "read_tool_controls",
    "write_tool_controls",
]
