"""Unified streaming infrastructure for LLM Proxy.

Public names are PEP 810 lazy imports: each symbol defers its submodule until
first use, so `import llm_proxy.streaming` on a cold path does not build the
transformer, handler and SSE-builder modules.
"""

lazy from llm_proxy.streaming.accumulation import (
    CanonicalDelta,
    Coexistence,
    CursorPolicy,
    SegmentAccumulator,
    SegmentClosed,
    SegmentEvent,
    SegmentOpened,
    SegmentPolicy,
    ToolVariant,
)
lazy from llm_proxy.streaming.events import StreamEvent
lazy from llm_proxy.streaming.handler import StreamingHandler, StreamingResponseConfig
lazy from llm_proxy.streaming.sse_builder import SSEBuilder, create_sse_error
lazy from llm_proxy.streaming.transformer import PendingTerminalState, StreamingTransformer

__all__ = [
    "CanonicalDelta",
    "Coexistence",
    "CursorPolicy",
    "PendingTerminalState",
    "SegmentAccumulator",
    "SegmentClosed",
    "SegmentEvent",
    "SegmentOpened",
    "SegmentPolicy",
    "SSEBuilder",
    "StreamEvent",
    "StreamingHandler",
    "StreamingResponseConfig",
    "StreamingTransformer",
    "ToolVariant",
    "create_sse_error",
]
