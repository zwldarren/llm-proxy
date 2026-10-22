"""Streaming and non-streaming must express the same OpenResponses body.

Two code paths build the client-visible response: the streaming transformer's
deferred terminal snapshot (``OpenResponsesStreamingTransformer.finalize``) and
``OpenResponsesProtocolSerializer.format_response``. They are separate code, so
they can drift — they already did: ``input_tokens_details.cache_write_tokens``
reached the non-streaming body only, and before the terminal snapshot was
deferred, streaming reported all-zero usage while non-streaming reported the
real bill.

This module pins the two together for the same response. It does not restate
what each field should be; it asserts that both paths agree, so a counter added
to one side, a detail dropped by a converter, or a snapshot built too early
fails here. ``tests/protocols/test_openresponses_wire_contract.py`` pins the
shared shape to the published schema.

The streaming side is driven through the provider converters, not through
hand-built canonical chunks, so the chunk orderings that actually reach the
transformer are the ones under test — including the trailing usage-only chunk
that Chat Completions, Gemini and Interactions emit *after* ``finish_reason``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import orjson
import pytest

from llm_proxy.models import InternalResponse, TextBlock, ToolUseBlock
from llm_proxy.models.types import (
    CompletionTokensDetails,
    PromptTokensDetails,
    Usage,
)
from llm_proxy.protocols.openresponses.handler import clear_format_context, set_format_context
from llm_proxy.protocols.openresponses.serializer import OpenResponsesProtocolSerializer
from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer
from llm_proxy.serialization.anthropic.streaming_converter import AnthropicChunkConverter
from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer

MODEL = "test-model"
RESPONSE_ID = "resp_parity"


@dataclass
class Wire:
    """One provider's native stream plus the response it stands for.

    ``converter`` is the provider chunk converter factory (``None`` when the
    frames already are canonical Chat Completions chunks).
    """

    frames: list[dict[str, Any]]
    internal: InternalResponse
    converter: Callable[[], Any] | None = None
    extra_frames: list[str] = field(default_factory=list)


def _usage(
    *,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int | None = None,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
    audio_input_tokens: int = 0,
    audio_output_tokens: int = 0,
    image_input_tokens: int = 0,
) -> Usage:
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens if total_tokens is not None else input_tokens + output_tokens,
        cache_read_input_tokens=cached_tokens,
        cache_creation_input_tokens=cache_write_tokens,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=cached_tokens,
            audio_tokens=audio_input_tokens or None,
            image_tokens=image_input_tokens or None,
        ),
        completion_tokens_details=CompletionTokensDetails(
            reasoning_tokens=reasoning_tokens,
            audio_tokens=audio_output_tokens or None,
        ),
    )


def _chat_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
    audio_input_tokens: int = 0,
    audio_output_tokens: int = 0,
    image_input_tokens: int = 0,
) -> dict[str, Any]:
    prompt_details: dict[str, Any] = {
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
    }
    if audio_input_tokens:
        prompt_details["audio_tokens"] = audio_input_tokens
    if image_input_tokens:
        prompt_details["image_tokens"] = image_input_tokens
    completion_details: dict[str, Any] = {"reasoning_tokens": reasoning_tokens}
    if audio_output_tokens:
        completion_details["audio_tokens"] = audio_output_tokens
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_details": prompt_details,
        "completion_tokens_details": completion_details,
    }


def _chat_completions_text() -> Wire:
    return Wire(
        frames=[
            {
                "id": "chatcmpl-1",
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "model": MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
            # Chat Completions sends usage in a chunk after finish_reason
            # (stream_options.include_usage); the snapshot must fold it in.
            {
                "id": "chatcmpl-1",
                "model": MODEL,
                "choices": [],
                "usage": _chat_usage(
                    input_tokens=1000,
                    output_tokens=44,
                    total_tokens=1044,
                    cached_tokens=100,
                    cache_write_tokens=50,
                    reasoning_tokens=12,
                ),
            },
        ],
        internal=InternalResponse(
            id=RESPONSE_ID,
            model=MODEL,
            output=[TextBlock(text="Hello")],
            finish_reason="stop",
            usage=_usage(
                input_tokens=1000,
                output_tokens=44,
                total_tokens=1044,
                cached_tokens=100,
                cache_write_tokens=50,
                reasoning_tokens=12,
            ),
        ),
    )


def _chat_completions_tool_call() -> Wire:
    return Wire(
        frames=[
            {
                "id": "chatcmpl-2",
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"SF"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-2",
                "model": MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
            {
                "id": "chatcmpl-2",
                "model": MODEL,
                "choices": [],
                "usage": _chat_usage(input_tokens=500, output_tokens=30, total_tokens=530),
            },
        ],
        internal=InternalResponse(
            id=RESPONSE_ID,
            model=MODEL,
            output=[ToolUseBlock(id="call_1", name="get_weather", input={"city": "SF"})],
            finish_reason="tool_calls",
            usage=_usage(input_tokens=500, output_tokens=30, total_tokens=530),
        ),
    )


def _gemini_text() -> Wire:
    """Gemini reports usage in a separate, choices-less chunk after finish."""

    def convert() -> GeminiStreamingTransformer:
        return GeminiStreamingTransformer(model=MODEL, request_id=RESPONSE_ID)

    return Wire(
        frames=[
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "Hello"}]},
                        "finishReason": "STOP",
                    }
                ]
            },
            {
                "usageMetadata": {
                    "promptTokenCount": 4321,
                    "candidatesTokenCount": 76,
                    "totalTokenCount": 4397,
                    "cachedContentTokenCount": 4096,
                    "thoughtsTokenCount": 40,
                }
            },
        ],
        internal=InternalResponse(
            id=RESPONSE_ID,
            model=MODEL,
            output=[TextBlock(text="Hello")],
            finish_reason="stop",
            usage=_usage(
                input_tokens=4321,
                output_tokens=76,
                total_tokens=4397,
                cached_tokens=4096,
                reasoning_tokens=40,
            ),
        ),
        converter=convert,
    )


def _anthropic_text() -> Wire:
    """Anthropic carries finish and usage on ``message_stop``."""

    def convert() -> AnthropicChunkConverter:
        return AnthropicChunkConverter(model=MODEL, request_id=RESPONSE_ID)

    return Wire(
        frames=[
            {
                "type": "message_start",
                "message": {
                    "id": RESPONSE_ID,
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
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 44, "output_tokens_details": {"thinking_tokens": 12}},
            },
            {"type": "message_stop"},
        ],
        internal=InternalResponse(
            id=RESPONSE_ID,
            model=MODEL,
            output=[TextBlock(text="Hello")],
            finish_reason="end_turn",
            usage=_usage(
                input_tokens=1050,
                output_tokens=44,
                total_tokens=1094,
                cached_tokens=100,
                cache_write_tokens=50,
                reasoning_tokens=12,
            ),
        ),
        converter=convert,
    )


def _chat_completions_incomplete() -> Wire:
    return Wire(
        frames=[
            {
                "id": "chatcmpl-3",
                "model": MODEL,
                "choices": [{"index": 0, "delta": {"content": "Partial"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-3",
                "model": MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            },
            {
                "id": "chatcmpl-3",
                "model": MODEL,
                "choices": [],
                "usage": _chat_usage(input_tokens=7, output_tokens=3, total_tokens=10),
            },
        ],
        internal=InternalResponse(
            id=RESPONSE_ID,
            model=MODEL,
            output=[TextBlock(text="Partial")],
            finish_reason="length",
            usage=_usage(input_tokens=7, output_tokens=3, total_tokens=10),
        ),
    )


def _chat_completions_audio_details() -> Wire:
    """An audio-capable model reports detail splits the canonical record keeps nested.

    ``audio_tokens`` and ``image_tokens`` have no flat canonical scalar, so the
    non-streaming body gets them from ``PromptTokensDetails`` while the streaming
    snapshot has to fold them out of the wire's details block. Nothing carried
    them across before, so one turn reported two different breakdowns.
    """
    return Wire(
        frames=[
            {
                "id": "chatcmpl-4",
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-4",
                "model": MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
            {
                "id": "chatcmpl-4",
                "model": MODEL,
                "choices": [],
                "usage": _chat_usage(
                    input_tokens=210,
                    output_tokens=60,
                    total_tokens=270,
                    audio_input_tokens=150,
                    image_input_tokens=20,
                    audio_output_tokens=40,
                ),
            },
        ],
        internal=InternalResponse(
            id=RESPONSE_ID,
            model=MODEL,
            output=[TextBlock(text="Hello")],
            finish_reason="stop",
            usage=_usage(
                input_tokens=210,
                output_tokens=60,
                total_tokens=270,
                audio_input_tokens=150,
                image_input_tokens=20,
                audio_output_tokens=40,
            ),
        ),
    )


CASES = {
    "chat_completions_text": _chat_completions_text,
    "chat_completions_tool_call": _chat_completions_tool_call,
    "chat_completions_audio_details": _chat_completions_audio_details,
    "gemini_text": _gemini_text,
    "anthropic_text": _anthropic_text,
    "chat_completions_incomplete": _chat_completions_incomplete,
}


def _terminal_snapshot(wire: Wire) -> dict[str, Any]:
    """Run the provider frames through converter -> transformer -> finalize."""
    converter = wire.converter() if wire.converter is not None else None
    transformer = OpenResponsesStreamingTransformer(model=MODEL, request_id=RESPONSE_ID)
    events: list[dict[str, Any]] = []
    for frame in wire.frames:
        chunk = converter.convert_chunk(frame) if converter is not None else frame
        if chunk is None:
            continue
        events.extend(_parse_events(transformer.transform(chunk) or ""))
    if converter is not None:
        for chunk in converter.finalize_chunks():
            events.extend(_parse_events(transformer.transform(chunk) or ""))
    events.extend(_parse_events(transformer.finalize()))

    terminals = [
        event
        for event in events
        if event["type"] in ("response.completed", "response.incomplete", "response.failed")
    ]
    assert len(terminals) == 1, [event["type"] for event in events]
    # Terminal last (``[DONE]`` carries no event type).
    assert events[-1]["type"] == terminals[0]["type"]
    return terminals[0]["response"]


def _parse_events(stream: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in stream.split("\n\n"):
        data = [line[6:] for line in block.split("\n") if line.startswith("data: ")]
        if not data or data[0] == "[DONE]":
            continue
        events.append(orjson.loads(data[0]))
    return events


def _normalize_output(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop per-request values: ids and empty logprobs are not semantics."""
    normalized: list[dict[str, Any]] = []
    for item in items:
        clean = {key: value for key, value in item.items() if key != "id"}
        if isinstance(clean.get("content"), list):
            clean["content"] = [
                {
                    key: value
                    for key, value in part.items()
                    if key not in ("logprobs", "annotations")
                }
                for part in clean["content"]
            ]
        normalized.append(clean)
    return normalized


@pytest.mark.parametrize("case", CASES)
def test_streaming_terminal_matches_non_streaming_body(case: str) -> None:
    """The streaming snapshot and the formatted body describe the same response."""
    set_format_context({"store": True})
    try:
        wire = CASES[case]()
        streamed = _terminal_snapshot(wire)
        formatted = OpenResponsesProtocolSerializer().format_response(wire.internal)
    finally:
        clear_format_context()

    assert streamed["usage"] == formatted["usage"]
    assert streamed["status"] == formatted["status"]
    assert streamed["incomplete_details"] == formatted["incomplete_details"]
    assert _normalize_output(streamed["output"]) == _normalize_output(formatted["output"])
    assert streamed["model"] == formatted["model"]
    assert streamed["store"] == formatted["store"]

    # Every field the non-streaming body carries must be on the snapshot too:
    # two expressions of one response, not two dialects.
    volatile = {"id", "created_at", "completed_at"}
    assert set(streamed) == set(formatted)
    for key in set(formatted) - volatile - {"output", "usage"}:
        assert streamed[key] == formatted[key], key


@pytest.mark.parametrize("case", CASES)
def test_usage_totals_are_self_consistent(case: str) -> None:
    """Every body the proxy emits adds up, whichever path built it."""
    wire = CASES[case]()
    for body in (
        _terminal_snapshot(wire),
        OpenResponsesProtocolSerializer().format_response(wire.internal),
    ):
        usage = body["usage"]
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
        assert set(usage["input_tokens_details"]) >= {
            "cached_tokens",
            "cache_write_tokens",
        }
        assert "reasoning_tokens" in usage["output_tokens_details"]


def test_detail_splits_without_a_canonical_scalar_reach_the_snapshot() -> None:
    """``audio_tokens``/``image_tokens`` have no flat field on the canonical record.

    They ride ``PromptTokensDetails``/``CompletionTokensDetails``, which only the
    non-streaming formatter reads directly; the snapshot has to fold them out of
    the wire's details block. Both effects are named here rather than left to the
    equality assertion above, so a future reader can see what the case is for.
    """
    body = _terminal_snapshot(_chat_completions_audio_details())

    assert body["usage"]["input_tokens_details"]["audio_tokens"] == 150
    assert body["usage"]["input_tokens_details"]["image_tokens"] == 20
    assert body["usage"]["output_tokens_details"]["audio_tokens"] == 40
