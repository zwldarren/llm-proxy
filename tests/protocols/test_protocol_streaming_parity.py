"""Every protocol's streaming usage must agree with its non-streaming usage.

The OpenResponses parity test compares whole bodies because that protocol emits
one snapshot object. The other protocols express the same bill differently:

- OpenAI Chat Completions: inclusive ``prompt_tokens`` plus
  ``prompt_tokens_details``, sent in a trailing chunk (forced
  ``stream_options.include_usage``).
- Anthropic: exclusive ``input_tokens`` with cache counters beside it, split
  across ``message_start`` (input) and ``message_delta`` (output).

So the comparison is on the canonical **amounts** — input total, output, cache
read, cache write, reasoning — read from each protocol's own wire shape. Field
placement is the protocol's business; the numbers are not.

This module is the cross-protocol counterpart to
``test_openresponses_streaming_parity.py``. Every protocol that has both a
streaming and a non-streaming body should register a case here: a new protocol
without one is a guarantee gap, not a covered path.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import orjson
import pytest

from llm_proxy.models import InternalResponse, TextBlock
from llm_proxy.models.types import (
    CompletionTokensDetails,
    PromptTokensDetails,
    Usage,
)
from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer
from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer
from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer
from llm_proxy.protocols.openresponses.serializer import OpenResponsesProtocolSerializer
from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer
from llm_proxy.serialization.anthropic.streaming_converter import AnthropicChunkConverter

MODEL = "test-model"

# The same turn for every protocol: one text block and one bill with every
# counter non-zero, so a dropped field cannot hide behind a zero value.
TEXT = "Hello"
USAGE = Usage(
    input_tokens=1050,
    output_tokens=44,
    total_tokens=1094,
    cache_read_input_tokens=100,
    cache_creation_input_tokens=50,
    reasoning_tokens=12,
    prompt_tokens_details=PromptTokensDetails(cached_tokens=100),
    completion_tokens_details=CompletionTokensDetails(reasoning_tokens=12),
)
EXPECTED = {
    "input_total": 1050,
    "output": 44,
    "cache_read": 100,
    "cache_write": 50,
    "reasoning": 12,
}

#: Canonical Chat Completions chunks for the turn, with usage after
#: ``finish_reason`` the way converted providers deliver it.
CANONICAL_CHUNKS = [
    {
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": TEXT}, "finish_reason": None}
        ]
    },
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    {
        "choices": [],
        "usage": {
            "prompt_tokens": 1050,
            "completion_tokens": 44,
            "total_tokens": 1094,
            "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 50},
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    },
]


def _int(value: Any) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _openai_facts(usage: dict[str, Any]) -> dict[str, int]:
    """OpenAI/Chat-Completions dialect: prompt tokens are inclusive."""
    prompt = usage.get("prompt_tokens_details") or {}
    completion = usage.get("completion_tokens_details") or {}
    return {
        "input_total": _int(usage.get("prompt_tokens")),
        "output": _int(usage.get("completion_tokens")),
        "cache_read": _int(prompt.get("cached_tokens") or usage.get("cache_read_input_tokens")),
        "cache_write": _int(
            prompt.get("cache_write_tokens") or usage.get("cache_creation_input_tokens")
        ),
        "reasoning": _int(completion.get("reasoning_tokens") or usage.get("reasoning_tokens")),
    }


def _anthropic_facts(usage: dict[str, Any]) -> dict[str, int]:
    """Anthropic dialect: ``input_tokens`` is exclusive of the cache counters."""
    return {
        "input_total": _int(usage.get("input_tokens"))
        + _int(usage.get("cache_read_input_tokens"))
        + _int(usage.get("cache_creation_input_tokens")),
        "output": _int(usage.get("output_tokens")),
        "cache_read": _int(usage.get("cache_read_input_tokens")),
        "cache_write": _int(usage.get("cache_creation_input_tokens")),
        "reasoning": _int(
            ((usage.get("output_tokens_details") or {}).get("thinking_tokens"))
            or usage.get("reasoning_tokens")
        ),
    }


def _openresponses_facts(usage: dict[str, Any]) -> dict[str, int]:
    """OpenResponses dialect: inclusive ``input_tokens`` plus details objects."""
    prompt = usage.get("input_tokens_details") or {}
    output = usage.get("output_tokens_details") or {}
    return {
        "input_total": _int(usage.get("input_tokens")),
        "output": _int(usage.get("output_tokens")),
        "cache_read": _int(prompt.get("cached_tokens")),
        "cache_write": _int(prompt.get("cache_write_tokens")),
        "reasoning": _int(output.get("reasoning_tokens")),
    }


@dataclass
class Case:
    """One protocol's streaming path and the readers for both wire shapes."""

    #: Registered protocol name (``registry.list_protocols()``), so the coverage
    #: test can tell a new streaming protocol from a covered one.
    protocol: str

    formatter: Callable[[InternalResponse], dict[str, Any]]
    transformer: Callable[[], Any]
    converter: Callable[[], Any] | None
    stream_facts: Callable[[list[dict[str, Any]]], dict[str, int]]
    body_facts: Callable[[dict[str, Any]], dict[str, int]]
    stream_text: Callable[[list[dict[str, Any]]], str]
    body_text: Callable[[dict[str, Any]], str]


def _openai_stream_facts(events: list[dict[str, Any]]) -> dict[str, int]:
    for event in reversed(events):
        usage = event.get("usage")
        if isinstance(usage, dict):
            return _openai_facts(usage)
    return _openai_facts({})


def _anthropic_stream_facts(events: list[dict[str, Any]]) -> dict[str, int]:
    merged: dict[str, Any] = {}
    for event in events:
        if event.get("type") == "message_start":
            merged.update((event.get("message") or {}).get("usage") or {})
        elif event.get("type") == "message_delta":
            merged.update(event.get("usage") or {})
    return _anthropic_facts(merged)


def _openresponses_stream_facts(events: list[dict[str, Any]]) -> dict[str, int]:
    for event in reversed(events):
        if event.get("type") in ("response.completed", "response.incomplete"):
            return _openresponses_facts((event.get("response") or {}).get("usage") or {})
    return _openresponses_facts({})


def _openai_stream_text(events: list[dict[str, Any]]) -> str:
    return "".join(
        str((event["choices"][0].get("delta") or {}).get("content") or "")
        for event in events
        if event.get("choices")
    )


def _openai_body_text(body: dict[str, Any]) -> str:
    return str(body["choices"][0]["message"].get("content") or "")


def _anthropic_stream_text(events: list[dict[str, Any]]) -> str:
    return "".join(
        str((event.get("delta") or {}).get("text") or "")
        for event in events
        if event.get("type") == "content_block_delta"
    )


def _anthropic_body_text(body: dict[str, Any]) -> str:
    return "".join(
        block.get("text") or ""
        for block in body.get("content") or []
        if block.get("type") == "text"
    )


def _openresponses_stream_text(events: list[dict[str, Any]]) -> str:
    return "".join(
        str(event.get("delta") or "")
        for event in events
        if event.get("type") == "response.output_text.delta"
    )


def _openresponses_body_text(body: dict[str, Any]) -> str:
    return "".join(
        part.get("text") or ""
        for item in body.get("output") or []
        for part in item.get("content") or []
    )


def _anthropic_frames() -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "model": MODEL,
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
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": TEXT}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 44, "output_tokens_details": {"thinking_tokens": 12}},
        },
        {"type": "message_stop"},
    ]


def _internal() -> InternalResponse:
    return InternalResponse(
        id="resp_parity",
        model=MODEL,
        output=[TextBlock(text=TEXT)],
        finish_reason="stop",
        usage=USAGE,
    )


CASES: dict[str, Case] = {
    "openai_chat_completions": Case(
        protocol="openai",
        formatter=OpenAIProtocolSerializer().format_response,
        transformer=lambda: OpenAIStreamingTransformer(model=MODEL, request_id="chatcmpl_parity"),
        converter=None,
        stream_facts=_openai_stream_facts,
        body_facts=lambda body: _openai_facts(body.get("usage") or {}),
        stream_text=_openai_stream_text,
        body_text=_openai_body_text,
    ),
    "anthropic_messages": Case(
        protocol="anthropic",
        formatter=AnthropicProtocolSerializer().format_response,
        transformer=lambda: AnthropicStreamingTransformer(model=MODEL, request_id="msg_parity"),
        converter=lambda: AnthropicChunkConverter(model=MODEL, request_id="msg_parity"),
        stream_facts=_anthropic_stream_facts,
        body_facts=lambda body: _anthropic_facts(body.get("usage") or {}),
        stream_text=_anthropic_stream_text,
        body_text=_anthropic_body_text,
    ),
    "openresponses": Case(
        protocol="openresponses",
        formatter=OpenResponsesProtocolSerializer().format_response,
        transformer=lambda: OpenResponsesStreamingTransformer(
            model=MODEL, request_id="resp_parity"
        ),
        converter=None,
        stream_facts=_openresponses_stream_facts,
        body_facts=lambda body: _openresponses_facts(body.get("usage") or {}),
        stream_text=_openresponses_stream_text,
        body_text=_openresponses_body_text,
    ),
}


def _parse_events(stream: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in stream.split("\n\n"):
        data = [line[6:] for line in block.split("\n") if line.startswith("data: ")]
        if not data or data[0] == "[DONE]":
            continue
        events.append(orjson.loads(data[0]))
    return events


def _stream_events(case: Case) -> list[dict[str, Any]]:
    converter = case.converter() if case.converter is not None else None
    transformer = case.transformer()
    events: list[dict[str, Any]] = []
    for frame in _anthropic_frames() if converter is not None else CANONICAL_CHUNKS:
        chunk = converter.convert_chunk(frame) if converter is not None else frame
        if chunk is None:
            continue
        events.extend(_parse_events(transformer.transform(chunk) or ""))
    if converter is not None:
        for chunk in converter.finalize_chunks():
            events.extend(_parse_events(transformer.transform(chunk) or ""))
    events.extend(_parse_events(transformer.finalize()))
    return events


@pytest.mark.parametrize("case_name", CASES)
def test_streaming_usage_matches_non_streaming_usage(case_name: str) -> None:
    """The bill a streamed client can read equals the non-streaming body's bill."""
    case = CASES[case_name]
    streamed = case.stream_facts(_stream_events(case))
    formatted = case.body_facts(case.formatter(_internal()))
    assert streamed == EXPECTED
    assert formatted == EXPECTED


@pytest.mark.parametrize("case_name", CASES)
def test_streaming_text_matches_non_streaming_text(case_name: str) -> None:
    """The content, not just the bill, survives both paths."""
    case = CASES[case_name]
    streamed = case.stream_text(_stream_events(case))
    written = case.body_text(case.formatter(_internal()))
    assert streamed == written == TEXT


def test_every_streaming_protocol_has_a_parity_case() -> None:
    """A protocol with a streaming transformer but no case here is a gap.

    The registry is the list of what the proxy actually serves; importing the
    handlers makes their registration run. If this fails, add a ``Case`` (and,
    if the protocol has none, real parity work) rather than deleting the test:
    the point is that nobody has to remember.
    """
    import llm_proxy.protocols.anthropic.handler  # noqa: F401
    import llm_proxy.protocols.openai.handler  # noqa: F401
    import llm_proxy.protocols.openresponses.handler  # noqa: F401
    from llm_proxy.protocols import registry

    streaming = {
        name
        for name in registry.list_protocols()
        if registry.get_protocol(name) is not None
        and registry.get_protocol(name).get_streaming_transformer() is not None
    }
    covered = {case.protocol for case in CASES.values()}
    assert streaming == covered, {
        "uncovered": sorted(streaming - covered),
        "stale": sorted(covered - streaming),
    }
