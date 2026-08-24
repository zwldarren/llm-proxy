"""Tests for the Responses SSE aggregation fallback (stream:false → SSE body)."""

import orjson
import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.providers.openai.sse_fallback import (
    aggregate_sse_to_response_object,
    body_looks_like_sse,
    parse_json_or_sse,
)


def _sse_block(event_type: str, payload: dict) -> str:
    # Real Responses API frames carry the type both on the ``event:`` line and
    # inside the data payload's ``type`` field.
    frame = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {orjson.dumps(frame).decode()}\n\n"


def _full_sse_stream(*, with_item_events: bool = True) -> str:
    parts = [
        _sse_block("response.created", {"response": {"id": "resp_1", "status": "in_progress"}}),
    ]
    if with_item_events:
        parts.append(
            _sse_block(
                "response.output_item.done",
                {
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    },
                },
            )
        )
    parts.append(
        _sse_block(
            "response.completed",
            {
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "status": "completed",
                    "model": "gpt-5",
                    "output": [],
                    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                }
            },
        )
    )
    parts.append("data: [DONE]\n\n")
    return "".join(parts)


class TestBodyLooksLikeSse:
    def test_sse_body_detected(self):
        assert body_looks_like_sse(_full_sse_stream()) is True

    def test_sse_without_event_lines_detected(self):
        assert body_looks_like_sse('data: {"type": "response.created"}\n\n') is True

    def test_json_body_not_detected(self):
        assert body_looks_like_sse('{"id": "resp_1", "object": "response"}') is False

    def test_empty_body_not_detected(self):
        assert body_looks_like_sse("") is False

    def test_leading_blank_lines_tolerated(self):
        assert body_looks_like_sse("\n\ndata: {}\n\n") is True


class TestResponsesSseToResponseValue:
    def test_aggregates_items_into_terminal_response(self):
        result = aggregate_sse_to_response_object(_full_sse_stream())
        assert result["id"] == "resp_1"
        assert result["status"] == "completed"
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "message"
        assert result["usage"]["input_tokens"] == 5

    def test_keeps_terminal_output_when_no_item_events(self):
        stream = _full_sse_stream(with_item_events=False).replace(
            '"output":[]', '"output":[{"type":"message","id":"m"}]'
        )
        result = aggregate_sse_to_response_object(stream)
        # No output_item.done events → the terminal event's own output survives.
        assert result["output"] == [{"type": "message", "id": "m"}]

    def test_crlf_framing_tolerated(self):
        stream = _full_sse_stream().replace("\n", "\r\n")
        result = aggregate_sse_to_response_object(stream)
        assert result["id"] == "resp_1"
        assert len(result["output"]) == 1

    def test_incomplete_terminal_event_accepted(self):
        stream = _full_sse_stream().replace("response.completed", "response.incomplete")
        result = aggregate_sse_to_response_object(stream)
        assert result["id"] == "resp_1"

    def test_event_line_fallback_when_payload_omits_type(self):
        """Upstreams whose data payload omits ``type`` still aggregate (the
        ``event:`` line supplies the event type)."""
        stream = 'event: response.completed\ndata: {"response": {"id": "resp_9", "output": []}}\n\n'
        result = aggregate_sse_to_response_object(stream)
        assert result["id"] == "resp_9"

    def test_failed_event_raises_provider_error(self):
        stream = _sse_block(
            "response.failed",
            {"response": {"error": {"message": "boom", "type": "server_error"}}},
        )
        with pytest.raises(ProviderError, match="boom"):
            aggregate_sse_to_response_object(stream)

    def test_missing_terminal_event_raises_provider_error(self):
        stream = _sse_block("response.created", {"response": {"id": "resp_1"}})
        with pytest.raises(ProviderError, match="no response.completed event"):
            aggregate_sse_to_response_object(stream)


class _FakeResponse:
    """Mimics an upstream response whose json() fails (SSE body, JSON content-type)."""

    def __init__(self, text: str, json_raises: bool):
        self.text = text
        self._json_raises = json_raises

    def json(self):
        if self._json_raises:
            raise orjson.JSONDecodeError("unexpected character", "", 0)
        return orjson.loads(self.text)


class TestParseJsonOrSse:
    def test_parses_plain_json(self):
        response = _FakeResponse('{"id": "resp_1"}', json_raises=False)
        assert parse_json_or_sse(response) == {"id": "resp_1"}

    def test_aggregates_sse_when_json_fails(self):
        response = _FakeResponse(_full_sse_stream(), json_raises=True)
        result = parse_json_or_sse(response)
        assert result["id"] == "resp_1"
        assert len(result["output"]) == 1

    def test_reraises_decode_error_for_non_sse_body(self):
        response = _FakeResponse("<html>Bad Gateway</html>", json_raises=True)
        with pytest.raises(orjson.JSONDecodeError):
            parse_json_or_sse(response)
