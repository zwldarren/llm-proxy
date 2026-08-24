"""Tests for StreamEvent and the StreamEventType union."""

from typing import get_args

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models.content_blocks import ToolUseBlock
from llm_proxy.models.types import Usage
from llm_proxy.streaming.events import StreamEvent, StreamEventType

_EXPECTED_EVENT_TYPES = {
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
}


class TestStreamEventTypeUnion:
    """The StreamEventType Literal must enumerate every supported event."""

    def test_all_expected_types_present(self):
        """Every documented event type is part of the union."""
        assert set(get_args(StreamEventType)) == _EXPECTED_EVENT_TYPES

    def test_union_is_closed(self):
        """No undocumented event types slip into the union."""
        assert set(get_args(StreamEventType)) <= _EXPECTED_EVENT_TYPES


class TestStreamEventConstruction:
    """StreamEvent must round-trip its payload fields correctly."""

    @pytest.mark.parametrize("event_type", sorted(_EXPECTED_EVENT_TYPES))
    def test_event_requires_only_type(self, event_type):
        """Each event type can be constructed with no optional payload."""
        event = StreamEvent(type=event_type)
        assert event.type == event_type
        assert event.content is None
        assert event.block is None
        assert event.usage is None
        assert event.error is None
        assert event.response_id is None
        assert event.model is None
        assert event.index == 0
        assert event.metadata == {}

    def test_metadata_default_factory_is_per_instance(self):
        """Each event gets its own metadata dict (no shared mutable default)."""
        a = StreamEvent(type="done")
        b = StreamEvent(type="done")
        a.metadata["k"] = "v"
        assert "k" not in b.metadata

    def test_text_delta_carries_content(self):
        """Text delta events carry incremental text content."""
        event = StreamEvent(type="text_delta", content="hello", index=2)
        assert event.content == "hello"
        assert event.index == 2

    def test_tool_call_done_carries_block(self):
        """Tool call done events carry the completed ContentBlock."""
        block = ToolUseBlock(id="call_1", name="get_weather", input={"city": "SF"})
        event = StreamEvent(type="tool_call_done", block=block)
        assert event.block is block

    def test_usage_event_carries_usage(self):
        """Usage events carry a Usage payload."""
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        event = StreamEvent(type="usage", usage=usage)
        assert event.usage is not None
        assert event.usage is usage
        assert event.usage.total_tokens == 30

    def test_response_start_carries_id_and_model(self):
        """response_start events carry the response id and model name."""
        event = StreamEvent(type="response_start", response_id="resp_1", model="gpt-4")
        assert event.response_id == "resp_1"
        assert event.model == "gpt-4"

    def test_error_event_carries_provider_error(self):
        """Error events carry a ProviderError."""
        error = ProviderError("boom", provider_name="openai")
        event = StreamEvent(type="error", error=error)
        assert event.error is not None
        assert event.error is error
        assert event.error.message == "boom"

    def test_optional_metadata_is_preserved(self):
        """Arbitrary metadata can be attached and is preserved."""
        meta = {"finish_reason": "stop"}
        event = StreamEvent(type="done", metadata=meta)
        assert event.metadata == meta
