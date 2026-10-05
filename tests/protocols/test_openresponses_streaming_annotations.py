"""Streaming url_citation annotations survive the converted path.

The OpenAI Responses upstream emits ``response.output_text.annotation.added``
during streaming. The converted OpenResponses path used to drop it — the
``annotations`` array on every emitted ``output_text`` part was hardcoded empty.
These tests pin the annotation on the upstream event -> canonical chunk ->
OpenResponses event chain, and on the completed part / terminal snapshot.
"""

import orjson

from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer
from llm_proxy.serialization.openai.streaming_converter import OpenAIResponsesChunkConverter

ANNOTATION = {
    "type": "url_citation",
    "url": "https://example.com/a",
    "start_index": 0,
    "end_index": 5,
    "title": "Example",
}


def _parse_events(events: str) -> list[dict]:
    parsed: list[dict] = []
    for line in events.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        parsed.append(orjson.loads(payload))
    return parsed


def _chunk(delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": "resp_1",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-6-luna",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


class TestUpstreamAnnotationEvent:
    def test_annotation_added_becomes_annotation_delta(self):
        converter = OpenAIResponsesChunkConverter(model="gpt-6-luna", request_id="resp_1")
        converter._response_id = "resp_1"
        converter._created_at = 1234567890

        chunk = converter.convert_chunk(
            {
                "event_type": "response.output_text.annotation.added",
                "type": "response.output_text.annotation.added",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "annotation_index": 0,
                "annotation": ANNOTATION,
            }
        )

        assert chunk is not None
        assert chunk["choices"][0]["delta"]["annotations"] == [ANNOTATION]

    def test_annotation_added_without_payload_is_dropped(self):
        converter = OpenAIResponsesChunkConverter(model="gpt-6-luna")
        assert (
            converter.convert_chunk(
                {
                    "event_type": "response.output_text.annotation.added",
                    "annotation": None,
                }
            )
            is None
        )


class TestAnnotationStreamingEvents:
    """Drive the canonical chunks through the OpenResponses streaming transformer."""

    def _events(self) -> list[dict]:
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )

        clear_format_context()
        set_format_context({})
        transformer = OpenResponsesStreamingTransformer(model="gpt-6-luna", request_id="resp_1")
        out: list[dict] = []
        for chunk in (
            _chunk({"content": "hello"}),
            _chunk({"annotations": [ANNOTATION]}),
            _chunk({}, finish_reason="stop"),
        ):
            emitted = transformer.transform(chunk)
            if emitted:
                out.extend(_parse_events(emitted))
        clear_format_context()
        return out

    def test_annotation_added_event_is_emitted(self):
        added = [e for e in self._events() if e["type"] == "response.output_text.annotation.added"]
        assert len(added) == 1
        assert added[0]["annotation"] == ANNOTATION
        assert added[0]["annotation_index"] == 0
        assert added[0]["content_index"] == 0

    def test_content_part_done_carries_annotations(self):
        done = [
            e
            for e in self._events()
            if e["type"] == "response.content_part.done" and e["part"]["type"] == "output_text"
        ]
        assert done
        assert done[-1]["part"]["annotations"] == [ANNOTATION]

    def test_output_item_done_carries_annotations(self):
        done = [e for e in self._events() if e["type"] == "response.output_item.done"]
        parts = [p for p in done[-1]["item"]["content"] if p["type"] == "output_text"]
        assert parts and parts[0]["annotations"] == [ANNOTATION]

    def test_terminal_snapshot_carries_annotations(self):
        completed = [e for e in self._events() if e["type"] == "response.completed"]
        assert completed
        items = [item for item in completed[-1]["response"]["output"] if item["type"] == "message"]
        parts = [p for p in items[0]["content"] if p["type"] == "output_text"]
        assert parts and parts[0]["annotations"] == [ANNOTATION]
        assert parts[0]["text"] == "hello"


class TestNoAnnotationsUnchanged:
    def test_parts_keep_empty_annotations_array(self):
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )

        clear_format_context()
        set_format_context({})
        transformer = OpenResponsesStreamingTransformer(model="gpt-6-luna", request_id="resp_1")
        events: list[dict] = []
        for chunk in (_chunk({"content": "hi"}), _chunk({}, finish_reason="stop")):
            emitted = transformer.transform(chunk)
            if emitted:
                events.extend(_parse_events(emitted))
        clear_format_context()

        assert not [e for e in events if e["type"] == "response.output_text.annotation.added"]
        done = [
            e
            for e in events
            if e["type"] == "response.content_part.done" and e["part"]["type"] == "output_text"
        ]
        assert done[-1]["part"]["annotations"] == []
