"""OpenResponses protocol module.

Everything about the OpenResponses protocol lives here: endpoint registration
(handler.py), request/response conversion (serializer.py, tool_converter.py),
streaming (streaming*.py), and store=true persistence (store.py). Shared
Responses-shaped helpers used by other families live in
``llm_proxy.serialization.responses_toolkit``.
"""

from llm_proxy.protocols.openresponses.handler import openresponses_protocol
from llm_proxy.protocols.openresponses.replay import replay_stored_response
from llm_proxy.protocols.openresponses.serializer import (
    OpenResponsesProtocolSerializer,
    conversation_to_input_items,
)
from llm_proxy.protocols.registry import register_protocol

__all__ = [
    "OpenResponsesProtocolSerializer",
    "conversation_to_input_items",
    "openresponses_protocol",
    "replay_stored_response",
]

register_protocol(openresponses_protocol)
