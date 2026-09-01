"""Regression tests for Anthropic native streaming usage extraction."""

from llm_proxy.core.conversion import NativePassthroughHandler
from llm_proxy.core.request_type import RequestType
from llm_proxy.observability.event_context import EventContext

handler = NativePassthroughHandler()


def test_anthropic_message_start_usage_extraction():
    """Native Anthropic message_start usage should populate EventContext."""
    frame = (
        "event: message_start\ndata: "
        '{"type":"message_start","message":'
        '{"id":"msg_123","type":"message","role":"assistant","model":"claude-3",'
        '"content":[],"stop_reason":null,"stop_sequence":null,"usage":'
        '{"input_tokens":1000,"output_tokens":0,"cache_read_input_tokens":300,'
        '"cache_creation_input_tokens":200}}}\n\n'
    )
    ctx = EventContext(
        request_id="r1",
        trace_id="t1",
        model="claude-3",
        request_type=RequestType.CHAT,
    )
    handler.maybe_capture_native_streaming_usage(frame, ctx)
    assert ctx.prompt_tokens == 1500  # 1000 + 300 + 200
    assert ctx.cache_read_input_tokens == 300
    assert ctx.cache_creation_input_tokens == 200
    # Both prompt and completion are known (completion is explicitly 0).
    assert ctx.total_tokens == 1500


def test_anthropic_message_delta_output_usage_extraction():
    """Native Anthropic message_delta output usage should update EventContext."""
    frame = (
        "event: message_delta\ndata: "
        '{"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":500}}\n\n'
    )
    ctx = EventContext(
        request_id="r1",
        trace_id="t1",
        model="claude-3",
        request_type=RequestType.CHAT,
    )
    handler.maybe_capture_native_streaming_usage(frame, ctx)
    assert ctx.completion_tokens == 500
    # total_tokens is only computed once both prompt and completion are known;
    # prompt_tokens is still missing in this isolated test.
    assert ctx.total_tokens is None


def test_anthropic_combined_start_and_delta_usage():
    """Full Anthropic streaming flow: message_start + message_delta on same EventContext."""
    ctx = EventContext(
        request_id="r1",
        trace_id="t1",
        model="claude-3",
        request_type=RequestType.CHAT,
    )
    start_frame = (
        "event: message_start\ndata: "
        '{"type":"message_start","message":'
        '{"id":"msg_123","type":"message","role":"assistant","model":"claude-3",'
        '"content":[],"stop_reason":null,"stop_sequence":null,"usage":'
        '{"input_tokens":1000,"output_tokens":0,"cache_read_input_tokens":300,'
        '"cache_creation_input_tokens":200}}}\n\n'
    )
    delta_frame = (
        "event: message_delta\ndata: "
        '{"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":500}}\n\n'
    )
    handler.maybe_capture_native_streaming_usage(start_frame, ctx)
    handler.maybe_capture_native_streaming_usage(delta_frame, ctx)
    # True prompt tokens = input_tokens + cache_read + cache_creation
    assert ctx.prompt_tokens == 1500  # 1000 + 300 + 200
    assert ctx.completion_tokens == 500
    assert ctx.cache_read_input_tokens == 300
    assert ctx.cache_creation_input_tokens == 200
    assert ctx.total_tokens == 2000  # 1500 + 500


def test_anthropic_no_space_sse_delimiters_usage_extraction():
    """Kimi Code-style upstreams emit SSE without a space after the field
    colon (``event:message_delta`` / ``data:{...}``). Per the SSE spec that
    is equivalent to ``event: message_delta``; usage must still be captured.
    """
    frame = (
        "event:message_delta\n"
        'data:{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
        '"usage":{"input_tokens":294,"cache_creation_input_tokens":0,'
        '"cache_read_input_tokens":96512,"output_tokens":63,'
        '"output_tokens_details":{"thinking_tokens":41}}}\n\n'
    )
    ctx = EventContext(
        request_id="r1",
        trace_id="t1",
        model="kimi-k2",
        request_type=RequestType.CHAT,
    )
    handler.maybe_capture_native_streaming_usage(frame, ctx)
    assert ctx.prompt_tokens == 96806
    assert ctx.completion_tokens == 63
    assert ctx.total_tokens == 96869
    assert ctx.cache_read_input_tokens == 96512
    assert ctx.cache_creation_input_tokens == 0


def test_anthropic_no_space_message_start_usage_extraction():
    """message_start frames with no-space delimiters must also be parsed."""
    frame = (
        "event:message_start\n"
        'data:{"type":"message_start","message":{"id":"msg_123","type":"message",'
        '"role":"assistant","model":"kimi-k2","usage":{"input_tokens":1000,'
        '"output_tokens":0,"cache_read_input_tokens":300,'
        '"cache_creation_input_tokens":200}}}\n\n'
    )
    ctx = EventContext(
        request_id="r1",
        trace_id="t1",
        model="kimi-k2",
        request_type=RequestType.CHAT,
    )
    handler.maybe_capture_native_streaming_usage(frame, ctx)
    assert ctx.prompt_tokens == 1500
    assert ctx.total_tokens == 1500
