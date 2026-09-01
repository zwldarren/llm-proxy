"""Tests for the spec-tolerant SSE field parsers.

Raw-upstream consumers must accept both SSE spellings: ``data: {...}`` and
the no-space form Kimi Code's Anthropic Messages endpoint sends
(``data:{...}``). See ``llm_proxy.streaming.sse_parse``.
"""

from llm_proxy.streaming.sse_parse import (
    contains_sse_event,
    iter_sse_data_events,
    parse_sse_data_line,
    split_sse_field,
    strip_sse_data_prefix,
)


class TestParseSseDataLine:
    def test_spaced_form(self):
        assert parse_sse_data_line('data: {"a":1}') == '{"a":1}'

    def test_no_space_form(self):
        """Kimi Code-style ``data:{...}`` framing parses identically."""
        assert parse_sse_data_line('data:{"a":1}') == '{"a":1}'

    def test_done_marker_both_spellings(self):
        assert parse_sse_data_line("data: [DONE]") == "[DONE]"
        assert parse_sse_data_line("data:[DONE]") == "[DONE]"

    def test_empty_payload_returns_none(self):
        assert parse_sse_data_line("data:") is None
        assert parse_sse_data_line("data: ") is None

    def test_non_data_lines_return_none(self):
        assert parse_sse_data_line("event: message_delta") is None
        assert parse_sse_data_line(": keep-alive") is None
        assert parse_sse_data_line("id: 1") is None
        assert parse_sse_data_line("") is None

    def test_only_first_colon_splits(self):
        """Only the field-name colon splits; JSON-internal colons survive."""
        line = 'data: {"choices":[{"delta":{"content":"a: b"}}]}'
        assert "a: b" in parse_sse_data_line(line)

    def test_similar_field_names_not_matched(self):
        assert parse_sse_data_line("database: x") is None


class TestSplitSseField:
    def test_event_line_both_spellings(self):
        assert split_sse_field("event: message_start") == ("event", "message_start")
        assert split_sse_field("event:message_start") == ("event", "message_start")

    def test_strips_at_most_one_leading_space(self):
        """The spec strips exactly one optional leading space."""
        assert split_sse_field("data:  x") == ("data", " x")

    def test_no_colon_yields_whole_line_as_field(self):
        assert split_sse_field("garbage") == ("garbage", "")


class TestStripSseDataPrefix:
    def test_data_line_returns_payload(self):
        assert strip_sse_data_prefix('data: {"a":1}') == '{"a":1}'
        assert strip_sse_data_prefix('data:{"a":1}') == '{"a":1}'

    def test_done_marker_both_spellings(self):
        assert strip_sse_data_prefix("data: [DONE]") == "[DONE]"
        assert strip_sse_data_prefix("data:[DONE]") == "[DONE]"

    def test_non_data_lines_pass_through(self):
        assert strip_sse_data_prefix("event: message_delta") == "event: message_delta"
        assert strip_sse_data_prefix(": keep-alive") == ": keep-alive"
        assert strip_sse_data_prefix("") == ""

    def test_empty_data_payload_passes_through(self):
        assert strip_sse_data_prefix("data:") == "data:"


class TestContainsSseEvent:
    def test_both_spellings(self):
        assert contains_sse_event("event: message_start\ndata: {}", "message_start")
        assert contains_sse_event("event:message_start\ndata:{}", "message_start")

    def test_absent_event(self):
        assert not contains_sse_event("event: message_delta\ndata: {}", "message_start")
        assert not contains_sse_event("data: {}", "message_start")


class TestIterSseDataEvents:
    def test_yields_event_type_and_payload(self):
        chunk = 'event: message_start\ndata: {"a":1}\n\nevent: message_delta\ndata: {"b":2}'
        assert list(iter_sse_data_events(chunk)) == [
            ("message_start", {"a": 1}),
            ("message_delta", {"b": 2}),
        ]

    def test_no_space_spelling(self):
        chunk = 'event:message_start\ndata:{"a":1}'
        assert list(iter_sse_data_events(chunk)) == [("message_start", {"a": 1})]

    def test_malformed_data_keeps_pending_event(self):
        chunk = 'event: message_start\ndata: not-json\ndata: {"a":1}'
        assert list(iter_sse_data_events(chunk)) == [("message_start", {"a": 1})]

    def test_data_without_event_yields_none(self):
        assert list(iter_sse_data_events('data: {"a":1}')) == [(None, {"a": 1})]
