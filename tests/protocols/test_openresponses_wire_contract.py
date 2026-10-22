"""The emitted stream is checked against the published Open Responses schema.

Hand-written assertions encode what the implementation happens to produce: add a
field to the builder and the test keeps passing because the expected dict was
edited alongside it. This module validates the wire objects against the
*authoritative* schema instead, so a field the spec requires but the proxy
stopped sending (or renamed) fails without anyone having to notice.

The fixture is the schema closure of the events this endpoint emits, copied
verbatim from the Open Responses OpenAPI (`tests/protocols/fixtures/
openresponses_schema_contract.json` records the source URL and commit).

Regenerate after a spec release::

    curl -sSL <source URL from the fixture> > openapi.json

then re-run the trim described by the fixture's ``note`` (walk ``$ref`` from the
``*StreamingEvent`` schemas plus ``ResponseResource``/``Usage``/``ItemField``).
"""

import json
from pathlib import Path
from typing import Any

import orjson
import pytest

from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer

FIXTURE = Path(__file__).parent / "fixtures" / "openresponses_schema_contract.json"
CONTRACT = json.loads(FIXTURE.read_text(encoding="utf-8"))
SCHEMAS: dict[str, Any] = CONTRACT["schemas"]

#: JSON type name -> Python types that satisfy it (``bool`` is an ``int``).
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _events(stream: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for block in stream.split("\n\n"):
        lines = block.split("\n")
        data = [line[6:] for line in lines if line.startswith("data: ")]
        if not data or data[0] == "[DONE]":
            continue
        event_type = [line[7:] for line in lines if line.startswith("event: ")]
        parsed.append(
            {"event": event_type[0] if event_type else None, "data": orjson.loads(data[0])}
        )
    return parsed


def _check(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> None:
    """Validate ``value`` against one schema node, collecting every mismatch."""
    if not schema:
        return
    if "$ref" in schema:
        _check(value, SCHEMAS[schema["$ref"].rsplit("/", 1)[-1]], path, problems)
        return
    for union in ("anyOf", "oneOf"):
        branches = schema.get(union)
        if branches:
            best: list[str] | None = None
            for branch in branches:
                trial: list[str] = []
                _check(value, branch, path, trial)
                if not trial:
                    return
                if best is None or len(trial) < len(best):
                    best = trial
            problems.extend(best or [])
            return
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} not in {schema['enum']}")
        return
    expected = schema.get("type")
    if expected is not None:
        names = expected if isinstance(expected, list) else [expected]
        allowed: tuple[type, ...] = ()
        for name in names:
            allowed += _JSON_TYPES[name]
        # ``bool`` is an ``int`` subclass, but never a valid integer field.
        if isinstance(value, bool) and "boolean" not in names:
            problems.append(f"{path}: boolean where {names} expected")
            return
        if not isinstance(value, allowed):
            problems.append(f"{path}: {type(value).__name__} where {names} expected")
            return
    if isinstance(value, dict):
        for required in schema.get("required", []):
            if required not in value:
                problems.append(f"{path}: missing required field {required!r}")
        for key, sub_schema in schema.get("properties", {}).items():
            if key in value:
                _check(value[key], sub_schema, f"{path}.{key}", problems)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _check(item, schema["items"], f"{path}[{index}]", problems)


def _emit_completed_stream() -> str:
    """Reasoning + annotated text + tool call + a trailing usage chunk."""
    transformer = OpenResponsesStreamingTransformer(model="test-model", request_id="resp_contract")
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "reasoning_content": "think"},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com",
                                "title": "Example",
                                "start_index": 0,
                                "end_index": 5,
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 10},
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        },
    ]
    stream = "".join(transformer.transform(chunk) or "" for chunk in chunks)
    return stream + transformer.finalize()


def _emit_incomplete_stream() -> str:
    transformer = OpenResponsesStreamingTransformer(
        model="test-model", request_id="resp_contract_incomplete"
    )
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "Partial"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        },
    ]
    stream = "".join(transformer.transform(chunk) or "" for chunk in chunks)
    return stream + transformer.finalize()


def _emit_error_stream() -> str:
    from llm_proxy.core.exceptions import ProviderError

    transformer = OpenResponsesStreamingTransformer(
        model="test-model", request_id="resp_contract_err"
    )
    stream = (
        transformer.transform(
            {"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}
        )
        or ""
    )
    return stream + "".join(
        transformer.error_frames(
            ProviderError(message="upstream boom", error_type="rate_limit_error", provider_name="p")
        )
    )


STREAMS = {
    "completed": _emit_completed_stream,
    "incomplete": _emit_incomplete_stream,
    "error": _emit_error_stream,
}


@pytest.mark.parametrize("stream_name", STREAMS)
def test_emitted_events_satisfy_the_published_schema(stream_name: str) -> None:
    events = _events(STREAMS[stream_name]())
    problems: list[str] = []

    by_type = {name: schema for name, schema in SCHEMAS.items() if name.endswith("StreamingEvent")}
    event_schemas = {schema["properties"]["type"]["enum"][0]: schema for schema in by_type.values()}

    for index, event in enumerate(events):
        payload = event["data"]
        # The ``event:`` field MUST match ``type`` in the body (spec: Streaming
        # HTTP Responses).
        if event["event"] != payload.get("type"):
            problems.append(
                f"event[{index}]: header {event['event']!r} != body {payload.get('type')!r}"
            )
        schema = event_schemas.get(payload.get("type"))
        if schema is None:
            problems.append(f"event[{index}]: {payload.get('type')!r} is not a spec event")
            continue
        _check(payload, schema, f"event[{index}]", problems)

    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("stream_name", STREAMS)
def test_stream_ordering_and_terminal_event(stream_name: str) -> None:
    events = _events(STREAMS[stream_name]())
    sequence = [event["data"]["sequence_number"] for event in events]
    assert sequence == sorted(sequence)
    assert len(set(sequence)) == len(sequence), "sequence_number repeats"

    terminal_types = {"response.completed", "response.incomplete", "response.failed"}
    terminals = [
        event["data"]["type"] for event in events if event["data"]["type"] in terminal_types
    ]
    assert len(terminals) == 1
    # Spec: the stream's terminal response event is last (``[DONE]`` follows it
    # and carries no event type).
    assert events[-1]["data"]["type"] == terminals[0]
    if stream_name == "error":
        # Spec: any error incurred while streaming is followed by
        # ``response.failed``.
        assert events[-2]["data"]["type"] == "error"


def test_terminal_usage_matches_the_published_usage_schema() -> None:
    events = _events(_emit_completed_stream())
    terminal = events[-1]["data"]["response"]
    problems: list[str] = []
    _check(terminal["usage"], {"$ref": "#/components/schemas/Usage"}, "usage", problems)
    assert not problems, "\n".join(problems)
    # The OpenAI ResponseUsage carries both cache counters; Codex deserializes
    # ``cached_tokens`` and ``reasoning_tokens`` as required fields.
    assert terminal["usage"]["input_tokens_details"] == {
        "cached_tokens": 40,
        "cache_write_tokens": 10,
    }
    assert terminal["usage"]["output_tokens_details"] == {"reasoning_tokens": 5}
    assert terminal["usage"]["total_tokens"] == (
        terminal["usage"]["input_tokens"] + terminal["usage"]["output_tokens"]
    )
