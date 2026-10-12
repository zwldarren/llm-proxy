"""Table-driven tests for the segment accumulator (ADR-0031).

One scenario matrix runs against the engine's interface — ``feed`` /
``finish`` / ``blocks`` / ``cursor`` / ``supersede`` — instead of once per
per-protocol copy: the matrix encodes the drift rulings (nothing silently
discarded), the eviction rule, the media turn-buffering and the error modes.
Policy variants (tool coexistence, cursor) get their own rows where their
behavior differs.
"""

from typing import Any

import pytest

from llm_proxy.models.content_blocks import (
    AudioBlock,
    ContentBlock,
    CustomToolUseBlock,
    ImageBlock,
    RedactedThinkingBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llm_proxy.streaming.accumulation import (
    Coexistence,
    CursorPolicy,
    SegmentAccumulator,
    SegmentClosed,
    SegmentOpened,
    SegmentPolicy,
)


def assemble(
    deltas: list[dict[str, Any]],
    policy: SegmentPolicy | None = None,
    *,
    finish: bool = True,
) -> list[ContentBlock]:
    """Feed a delta sequence and return the assembled blocks."""
    accumulator = SegmentAccumulator(policy)
    for delta in deltas:
        accumulator.feed(delta)
    if finish:
        accumulator.finish()
    return accumulator.blocks()


# ── the scenario matrix (default policy) ─────────────────────────────────


@pytest.mark.parametrize(
    ("name", "deltas", "expected"),
    [
        (
            "text_only",
            [{"content": "Hello"}],
            [TextBlock(text="Hello")],
        ),
        (
            "text_continues_across_deltas",
            [{"content": "Hel"}, {"content": "lo"}],
            [TextBlock(text="Hello")],
        ),
        (
            "reasoning_then_text",
            [{"reasoning_content": "hmm"}, {"content": "Hi"}],
            [ThinkingBlock(thinking="hmm", signature=None), TextBlock(text="Hi")],
        ),
        (
            "reasoning_alias_field",
            [{"reasoning": "via-alias"}, {"content": "Hi"}],
            [ThinkingBlock(thinking="via-alias", signature=None), TextBlock(text="Hi")],
        ),
        (
            "reasoning_with_signature",
            [{"reasoning_content": "hmm", "reasoning_signature": "sig"}],
            [ThinkingBlock(thinking="hmm", signature="sig")],
        ),
        (
            "signature_fragments_concatenate",
            [
                {"reasoning_content": "a", "reasoning_signature": "s1"},
                {"reasoning_content": "b", "reasoning_signature": "s2"},
            ],
            [ThinkingBlock(thinking="ab", signature="s1s2")],
        ),
        (
            "signature_only_segment_kept",
            [{"reasoning_signature": "sig"}],
            [ThinkingBlock(thinking="", signature="sig")],
        ),
        (
            "reasoning_details_snapshot",
            [
                {"reasoning_content": "a", "reasoning_details": [{"type": "text"}]},
                {"reasoning_content": "b", "reasoning_details": [{"type": "summary"}]},
            ],
            [
                ThinkingBlock(
                    thinking="ab",
                    signature=None,
                    extra={"reasoning_details": [{"type": "summary"}]},
                )
            ],
        ),
        (
            "reasoning_after_text_kept",
            [
                {"content": "answer"},
                {"reasoning_content": "late thought"},
            ],
            [TextBlock(text="answer"), ThinkingBlock(thinking="late thought", signature=None)],
        ),
        (
            "redacted_reasoning_prefers_encrypted_payload",
            [
                {
                    "reasoning_content": "[redacted]",
                    "reasoning_is_redacted": True,
                    "encrypted_content": "opaque-blob",
                }
            ],
            [RedactedThinkingBlock(data="opaque-blob")],
        ),
        (
            "redacted_after_thinking_is_a_new_block",
            [
                {"reasoning_content": "think", "reasoning_signature": "sig"},
                {
                    "reasoning_content": "[redacted]",
                    "reasoning_is_redacted": True,
                    "encrypted_content": "opaque",
                },
            ],
            [
                ThinkingBlock(thinking="think", signature="sig"),
                RedactedThinkingBlock(data="opaque"),
            ],
        ),
        (
            "encrypted_content_joins_reasoning",
            [
                {"encrypted_content": "blob"},
                {"reasoning_content": "thought"},
            ],
            [ThinkingBlock(thinking="thought", encrypted_content="blob")],
        ),
        (
            "unclaimed_encrypted_blob_dropped",
            [{"encrypted_content": "blob"}, {"content": "text"}],
            [TextBlock(text="text")],
        ),
        (
            "tool_args_split_across_deltas",
            [
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get", "arguments": ""},
                        }
                    ]
                },
                {"tool_calls": [{"index": 0, "function": {"arguments": '{"x":'}}]},
                {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
            ],
            [ToolUseBlock(id="call_1", name="get", input={"x": 1})],
        ),
        (
            "parallel_tool_calls_flush_in_index_order",
            [
                {
                    "tool_calls": [
                        {"index": 1, "id": "b", "function": {"name": "second", "arguments": "{}"}}
                    ]
                },
                {
                    "tool_calls": [
                        {"index": 0, "id": "a", "function": {"name": "first", "arguments": "{}"}}
                    ]
                },
            ],
            [
                ToolUseBlock(id="a", name="first", input={}),
                ToolUseBlock(id="b", name="second", input={}),
            ],
        ),
        (
            "unparseable_tool_args_fall_back_to_empty_dict",
            [
                {
                    "tool_calls": [
                        {"index": 0, "id": "c", "function": {"name": "f", "arguments": "{oops"}}
                    ]
                }
            ],
            [ToolUseBlock(id="c", name="f", input={})],
        ),
        (
            "thought_signature_lands_in_extra",
            [
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c",
                            "function": {"name": "f", "arguments": "{}"},
                            "thought_signature": "tsig",
                        }
                    ]
                }
            ],
            [ToolUseBlock(id="c", name="f", input={}, extra={"thought_signature": "tsig"})],
        ),
        (
            "custom_tool_input_never_parsed",
            [
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c1",
                            "type": "custom",
                            "custom": {"name": "patch", "input": "not-json"},
                        }
                    ]
                }
            ],
            [CustomToolUseBlock(id="c1", name="patch", input="not-json")],
        ),
        (
            "tool_call_missing_id_or_name_dropped",
            [
                {"tool_calls": [{"index": 0, "function": {"name": "f", "arguments": "{}"}}]},
                {"tool_calls": [{"index": 1, "id": "c", "function": {"arguments": "{}"}}]},
            ],
            [],
        ),
        (
            "refusal_reaches_canonical_model",
            [{"refusal": "I cannot help with that."}],
            [RefusalBlock(refusal="I cannot help with that.")],
        ),
        (
            "images_only_turn_survives_finish",
            [{"images": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}],
            [ImageBlock(source=Any)],
        ),
        (
            "audio_turn_becomes_one_block",
            [
                {"audio": {"data": "AAAA"}},
                {"audio": {"transcript": "hello"}},
            ],
            [
                AudioBlock(
                    source=Any,
                )
            ],
        ),
        (
            "media_is_turn_buffered_not_evicted",
            [
                {"content": "before"},
                {"images": [{"type": "image_url", "image_url": {"url": "https://x/1.png"}}]},
                {"content": "after"},
            ],
            [
                TextBlock(text="beforeafter"),
                ImageBlock(source=Any),
            ],
        ),
        (
            "inline_image_part_appends_markdown_with_separator",
            [
                {"content": "see "},
                {"content": [{"type": "image_url", "image_url": {"url": "https://x/2.png"}}]},
            ],
            [TextBlock(text="see \n![image](https://x/2.png)")],
        ),
        (
            "inline_image_part_without_text",
            [{"content": [{"type": "image_url", "image_url": {"url": "https://x/3.png"}}]}],
            [TextBlock(text="![image](https://x/3.png)")],
        ),
    ],
)
def test_scenario_matrix(
    name: str, deltas: list[dict[str, Any]], expected: list[ContentBlock]
) -> None:
    blocks = assemble(deltas)
    assert len(blocks) == len(expected), f"{name}: {blocks!r}"
    for got, want in zip(blocks, expected, strict=True):
        assert type(got) is type(want), f"{name}: {type(got).__name__} != {type(want).__name__}"
        for attr in vars(want):
            if attr in ("source",):
                # Sources are constructed objects; type identity is the check.
                continue
            assert getattr(got, attr) == getattr(want, attr), (
                f"{name}.{attr}: {getattr(got, attr)!r}"
            )


# ── eviction and event shape ─────────────────────────────────────────────


def test_eviction_closes_other_kinds_in_canonical_order() -> None:
    accumulator = SegmentAccumulator()
    accumulator.feed({"content": "hi"})
    accumulator.feed({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f"}}]})
    # The reasoning delta closes the still-open tool segment (text was closed
    # by the tool call's own arrival), then opens reasoning.
    events = accumulator.feed({"reasoning_content": "hmm"})
    kinds = [type(event).__name__ for event in events]
    assert kinds == ["SegmentClosed", "SegmentOpened", "SegmentAppended"]
    closed_kinds = [event.kind for event in events if isinstance(event, SegmentClosed)]
    assert closed_kinds == ["tool"]


def test_events_carry_the_triggered_delta() -> None:
    accumulator = SegmentAccumulator()
    events = accumulator.feed({"content": "hi"})
    assert isinstance(events[0], SegmentOpened)
    assert events[0].delta.text == "hi"
    assert events[0].index is None  # cursor policy NONE


def test_finish_is_idempotent() -> None:
    accumulator = SegmentAccumulator()
    accumulator.feed({"content": "hi"})
    first = accumulator.finish()
    assert len(first) == 1
    assert accumulator.finish() == ()


def test_feed_after_finish_reopens_and_appends() -> None:
    accumulator = SegmentAccumulator()
    accumulator.feed({"content": "one"})
    accumulator.finish()
    accumulator.feed({"content": "two"})
    accumulator.finish()
    assert accumulator.blocks() == [TextBlock(text="one"), TextBlock(text="two")]


def test_empty_finish_produces_no_events() -> None:
    assert SegmentAccumulator().finish() == ()


def test_blocks_projection_includes_open_segments() -> None:
    accumulator = SegmentAccumulator()
    accumulator.feed({"content": "streaming"})
    projected = accumulator.blocks()
    assert projected == [TextBlock(text="streaming")]
    # The projection is a snapshot: the open segment still closes normally.
    accumulator.finish()
    assert accumulator.blocks() == [TextBlock(text="streaming")]


def test_blocks_projection_excludes_partial_tool_and_media() -> None:
    accumulator = SegmentAccumulator()
    accumulator.feed({"content": "hi"})
    accumulator.feed(
        {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": '{"a":'}}]}
    )
    accumulator.feed({"images": [{"type": "image_url", "image_url": {"url": "https://x/4.png"}}]})
    assert accumulator.blocks() == [TextBlock(text="hi")]


def test_blocks_returns_a_copy() -> None:
    accumulator = SegmentAccumulator()
    accumulator.feed({"content": "hi"})
    accumulator.finish()
    blocks = accumulator.blocks()
    blocks.clear()
    assert len(accumulator.blocks()) == 1


def test_supersede_replaces_a_closed_block() -> None:
    accumulator = SegmentAccumulator()
    accumulator.feed(
        {
            "tool_calls": [
                {"index": 0, "id": "c", "function": {"name": "websearch", "arguments": "{}"}}
            ]
        }
    )
    accumulator.finish()
    replacement = ToolUseBlock(id="c", name="websearch", input={})
    accumulator.supersede(0, replacement)
    assert accumulator.blocks() == [replacement]


def test_supersede_out_of_range_raises() -> None:
    accumulator = SegmentAccumulator()
    with pytest.raises(IndexError):
        accumulator.supersede(0, TextBlock(text="x"))


# ── malformed input never raises ─────────────────────────────────────────


@pytest.mark.parametrize(
    "delta",
    [
        "not-a-dict",
        42,
        None,
        {},
        {"content": 123},
        {"content": [{"type": "text"}]},
        {"content": [None, "ok"]},
        {"tool_calls": "nope"},
        {
            "tool_calls": [
                None,
                "x",
                {"index": True, "id": "c", "function": {"name": "f", "arguments": "{}"}},
            ]
        },
        {"tool_calls": [{"index": 0, "function": "not-a-dict"}]},
        {"images": "nope"},
        {"images": [None, "x"]},
        {"audio": "nope"},
        {"audio": {"data": 1}},
        {"reasoning_content": 123},
        {"reasoning_details": "nope"},
        {"reasoning_details": [None, "x"]},
    ],
)
def test_malformed_members_are_skipped(delta: Any) -> None:
    accumulator = SegmentAccumulator()
    accumulator.feed(delta)  # must not raise
    accumulator.finish()


def test_non_dict_content_parts_are_skipped_but_text_survives() -> None:
    blocks = assemble([{"content": [None, {"type": "text", "text": "ok"}]}])
    assert blocks == [TextBlock(text="ok")]


# ── policy variants ──────────────────────────────────────────────────────


def test_sequential_tools_close_the_previous_call() -> None:
    policy = SegmentPolicy(tool_coexistence=Coexistence.SEQUENTIAL)
    accumulator = SegmentAccumulator(policy)
    first = accumulator.feed(
        {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "one", "arguments": "{}"}}]}
    )
    # The second call's start closes the first, opens the second.
    second = accumulator.feed(
        {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "two", "arguments": "{}"}}]}
    )
    finish_events = accumulator.finish()
    blocks = accumulator.blocks()
    assert [b.id for b in blocks] == ["a", "b"]
    all_events = first + second + finish_events
    closes = [event for event in all_events if isinstance(event, SegmentClosed)]
    opens = [event for event in all_events if isinstance(event, SegmentOpened)]
    assert len(closes) == 2
    assert len(opens) == 2


def test_next_free_cursor_counts_closed_blocks() -> None:
    policy = SegmentPolicy(cursor=CursorPolicy.NEXT_FREE)
    accumulator = SegmentAccumulator(policy)
    assert accumulator.cursor() == 0
    accumulator.feed({"content": "hi"})
    events = accumulator.feed({"reasoning_content": "hmm"})
    assert accumulator.cursor() == 1
    closed = next(event for event in events if isinstance(event, SegmentClosed))
    opened = next(event for event in events if isinstance(event, SegmentOpened))
    assert closed.index == 0  # the text block that just closed landed at 0
    assert opened.index == 1  # the reasoning segment opens at the next free index
    accumulator.finish()
    assert accumulator.cursor() == 2
    # Result blocks emitted at explicit indices do not move the cursor.
    accumulator.supersede(0, TextBlock(text="replaced"))
    assert accumulator.cursor() == 2


def test_none_cursor_reports_none() -> None:
    accumulator = SegmentAccumulator()
    accumulator.feed({"content": "hi"})
    accumulator.finish()
    assert accumulator.cursor() is None


def test_closed_events_report_the_index_the_block_lands_at() -> None:
    accumulator = SegmentAccumulator(SegmentPolicy(cursor=CursorPolicy.NEXT_FREE))
    accumulator.feed({"content": "hi"})
    events = list(accumulator.feed({"reasoning_content": "hmm"}))
    events.extend(accumulator.finish())
    closes = [event for event in events if isinstance(event, SegmentClosed)]
    opens = [event for event in events if isinstance(event, SegmentOpened)]
    # Text lands at 0, reasoning at 1 — the close reports the same index the
    # matching open announced, so the pair can be joined on it.
    assert [event.index for event in closes] == [0, 1]
    assert [event.index for event in opens] == [1]
    assert [event.kind for event in closes] == ["text", "reasoning"]


def test_signature_origin_rides_the_reasoning_lane() -> None:
    blocks = assemble(
        [
            {
                "reasoning_content": "hmm",
                "reasoning_signature": "sig",
                "signature_origin": "gemini",
            }
        ]
    )
    assert blocks == [ThinkingBlock(thinking="hmm", signature="sig", signature_origin="gemini")]


def test_unknown_signature_origin_is_dropped() -> None:
    blocks = assemble(
        [{"reasoning_content": "hmm", "reasoning_signature": "sig", "signature_origin": "nope"}]
    )
    assert blocks == [ThinkingBlock(thinking="hmm", signature="sig", signature_origin=None)]
