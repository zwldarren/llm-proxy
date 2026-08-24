"""Shared Responses-shaped tooling.

Helpers understood by both the OpenResponses protocol module and
provider-family serializers: tool-name namespaces and Responses item helpers.
"""

from llm_proxy.serialization.responses_toolkit.items import (
    _extract_reasoning_text,
    _extract_summary_text,
    generate_item_id,
)
from llm_proxy.serialization.responses_toolkit.namespace import (
    NamespaceMapping,
    flatten_history_tool_name,
    restore_tool_name,
)

__all__ = [
    "NamespaceMapping",
    "flatten_history_tool_name",
    "restore_tool_name",
    "generate_item_id",
    "_extract_reasoning_text",
    "_extract_summary_text",
]
