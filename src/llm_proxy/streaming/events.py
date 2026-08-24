# src/llm_proxy/streaming/events.py
"""StreamEvent for unified streaming."""

from dataclasses import dataclass, field
from typing import Any, Literal

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models.content_blocks import ContentBlock
from llm_proxy.models.types import Usage

StreamEventType = Literal[
    "text_start",
    "text_delta",
    "text_done",
    "tool_call_start",
    "tool_call_delta",
    "tool_call_done",
    "thinking_start",
    "thinking_delta",
    "thinking_done",
    "usage",
    "response_start",
    "error",
    "done",
]


@dataclass
class StreamEvent:
    """Event emitted during streaming responses.

    Attributes:
        type: The type of streaming event
        content: Text content for text/thinking/tool_call delta events
        block: ContentBlock for done events (ToolUseBlock, ThinkingBlock)
        usage: Usage information for usage events
        error: ProviderError for error events
        response_id: Response ID for response_start events
        model: Model name for response_start events
        index: Index of the content block being streamed
        metadata: Additional metadata for the event
    """

    type: StreamEventType
    content: str | None = None
    block: ContentBlock | None = None
    usage: Usage | None = None
    error: ProviderError | None = None
    response_id: str | None = None
    model: str | None = None
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
