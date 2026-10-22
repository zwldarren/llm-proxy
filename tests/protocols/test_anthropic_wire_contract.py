"""The Anthropic-shaped bodies are checked against the published Claude schema.

Covers the non-streaming ``Message``, every streaming event body, the
``count_tokens`` response and the error envelope. The fixture is the ``$ref``
closure of those schemas, copied verbatim from the published Claude OpenAPI
(the Stainless "documented" spec the Anthropic SDKs are generated from);
``tests/protocols/fixtures/anthropic_schema_contract.json`` records the source
URL. Regenerate with ``tests/protocols/fixtures/trim_schema.py``.

Hand-written assertions pin the parts clients read from the Anthropic wire, so a
spec change that makes our body pass while dropping a fact still fails.
"""

from typing import Any

import pytest
from schema_validation import load_contract, parse_sse, validate

from llm_proxy.models import InternalResponse, TextBlock
from llm_proxy.models.types import Usage
from llm_proxy.protocols.anthropic.handler import CountTokensResponse
from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer
from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer
from llm_proxy.serialization.anthropic.streaming_converter import AnthropicChunkConverter

CONTRACT = load_contract("anthropic_schema_contract")
SCHEMAS: dict[str, Any] = CONTRACT["schemas"]

MODEL = "claude-test"
USAGE = Usage(
    input_tokens=1050,
    output_tokens=44,
    total_tokens=1094,
    cache_read_input_tokens=100,
    cache_creation_input_tokens=50,
    reasoning_tokens=12,
)


def _message() -> dict[str, Any]:
    response = InternalResponse(
        id="msg_contract",
        model=MODEL,
        output=[TextBlock(text="Hello")],
        finish_reason="stop",
        usage=USAGE,
    )
    return AnthropicProtocolSerializer().format_response(response)


def _frames() -> list[dict[str, Any]]:
    """Native upstream frames for one turn: text, a thinking block, a tool call."""
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_contract",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 900,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 50,
                    "output_tokens": 1,
                },
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":"SF"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {
                "output_tokens": 44,
                "output_tokens_details": {"thinking_tokens": 12},
            },
        },
        {"type": "message_stop"},
    ]


def _stream() -> list[dict[str, Any]]:
    """Round-trip native frames through the converter into the Anthropic wire."""
    converter = AnthropicChunkConverter(model=MODEL, request_id="msg_contract")
    transformer = AnthropicStreamingTransformer(model=MODEL, request_id="msg_contract")
    chunks = [chunk for frame in _frames() if (chunk := converter.convert_chunk(frame)) is not None]
    chunks.extend(converter.finalize_chunks())
    stream = (
        "".join(transformer.transform(chunk) or "" for chunk in chunks) + transformer.finalize()
    )
    return parse_sse(stream)


def test_message_body_satisfies_the_published_schema() -> None:
    body = _message()
    validate(body, "Message", SCHEMAS)
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "Hello"
    # Anthropic's ``input_tokens`` is exclusive of the cache counters, and the
    # body reports the same split the streaming ``message_delta`` does.
    assert body["usage"]["input_tokens"] == 900
    assert body["usage"]["cache_read_input_tokens"] == 100
    assert body["usage"]["cache_creation_input_tokens"] == 50
    assert body["usage"]["output_tokens"] == 44


def test_stream_events_satisfy_the_published_schema() -> None:
    events = _stream()
    assert events, "the stream emitted no events"
    problems: list[str] = []
    from schema_validation import check

    for index, event in enumerate(events):
        payload = event["data"]
        if event["event"] != payload.get("type"):
            problems.append(
                f"event[{index}]: header {event['event']!r} != body {payload.get('type')!r}"
            )
        check(
            payload,
            {"$ref": "#/components/schemas/MessageStreamEvent"},
            f"event[{index}]",
            problems,
            SCHEMAS,
        )
    assert not problems, "\n".join(problems)


def test_stream_start_message_and_delta_usage_satisfy_the_nested_schemas() -> None:
    events = {event["data"]["type"]: event["data"] for event in _stream()}
    # ``MessageStartEvent.message`` is a full ``Message`` (empty content).
    validate(events["message_start"]["message"], "Message", SCHEMAS)
    # The terminal usage is the delta variant, not the full ``Usage``.
    validate(events["message_delta"]["usage"], "MessageDeltaUsage", SCHEMAS)

    assert [block["type"] for block in events["message_start"]["message"]["content"]] == []
    # Anthropic splits the bill: input on ``message_start``, output on
    # ``message_delta``. Both halves must be present or a client's context/usage
    # display is wrong.
    #
    # ``message_start`` reports zero input here because this composition is the
    # *converted* tier: the transformer emits ``message_start`` on the first
    # content chunk, and the upstream's usage chunk only arrives after it. An
    # Anthropic-speaking upstream that states the count at ``message_start`` is
    # served verbatim (every such adapter lists ``anthropic`` in
    # ``native_protocols``), so the real wire keeps it. Do not "fix" the zero by
    # hoisting usage onto the first chunk: that leaks a premature usage block
    # onto every other protocol's stream.
    assert events["message_start"]["message"]["usage"]["input_tokens"] == 0
    delta = events["message_delta"]["usage"]
    assert delta["output_tokens"] == 44
    assert delta["input_tokens"] == 900
    assert delta["cache_read_input_tokens"] == 100
    assert delta["cache_creation_input_tokens"] == 50


def test_count_tokens_body_satisfies_the_published_schema() -> None:
    body = CountTokensResponse(input_tokens=42).model_dump()
    validate(body, "CountMessageTokensResponse", SCHEMAS)
    assert body["input_tokens"] == 42


@pytest.mark.parametrize(
    "error_type",
    ["api_error", "invalid_request_error", "rate_limit_error", "overloaded_error"],
)
def test_error_envelope_satisfies_the_published_schema(error_type: str) -> None:
    from llm_proxy.api.error_responses import ErrorResponseBuilder

    body = ErrorResponseBuilder.create_anthropic_error(message="boom", error_type=error_type)
    validate(body, "ErrorResponse", SCHEMAS)
    assert body["type"] == "error"
    assert body["error"]["type"] == error_type
    assert body["error"]["message"] == "boom"


def test_fixture_records_where_it_came_from() -> None:
    assert CONTRACT["source"].startswith("https://app.stainless.com/api/spec/documented/anthropic/")
    # The Stainless-documented document carries no commit/version, so the
    # retrieval date is the reproducibility anchor.
    assert CONTRACT["retrieved"]
    missing: set[str] = set()
    for schema in SCHEMAS.values():
        stack: list[Any] = [schema]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if isinstance(node.get("$ref"), str):
                    name = node["$ref"].rsplit("/", 1)[-1]
                    if name not in SCHEMAS:
                        missing.add(name)
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    assert not missing, f"fixture is not closed: {sorted(missing)}"
