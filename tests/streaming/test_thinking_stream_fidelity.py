"""Regression tests for streaming interleaved-thinking fidelity.

Covers two audit findings where the streaming path diverged from non-stream:

* redacted_thinking: the provider-side Anthropic chunk converter replaced the
  opaque ``data`` payload with the "[redacted]" display placeholder, so
  streaming clients could never echo a valid redacted block back (non-stream
  preserved it);
* multi-item encrypted reasoning: a single global capture slot on both the
  provider-side Responses converter and the OpenResponses protocol transformer
  collapsed interleaved reasoning items onto the first ``encrypted_content``
  blob.
"""

import json

from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer
from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer
from llm_proxy.serialization.anthropic.streaming_converter import AnthropicChunkConverter
from llm_proxy.serialization.openai.streaming_converter import (
    OpenAIResponsesChunkConverter,
)


def _sse_payloads(events: str) -> list[dict]:
    """Extract JSON payloads from an SSE events string."""
    payloads = []
    for raw_line in events.splitlines():
        payload = raw_line.strip()
        if not payload.startswith("data:"):
            continue
        payload = payload[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        payloads.append(json.loads(payload))
    return payloads


class TestAnthropicRedactedStreamPassthrough:
    """redacted_thinking.data must survive the streaming conversion chain."""

    def test_provider_converter_carries_opaque_data_in_delta(self):
        converter = AnthropicChunkConverter(model="claude-x", request_id="msg_1")
        chunk = converter.convert_chunk(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": "OPAQUE-DATA"},
            }
        )
        assert chunk is not None
        delta = chunk["choices"][0]["delta"]
        assert delta["reasoning_content"] == "[redacted]"
        assert delta["reasoning_is_redacted"] is True
        assert delta["encrypted_content"] == "OPAQUE-DATA"

    def test_protocol_transformer_emits_real_redacted_data(self):
        """AnthropicChunkConverter -> AnthropicStreamingTransformer chain."""
        converter = AnthropicChunkConverter(model="claude-x", request_id="msg_1")
        transformer = AnthropicStreamingTransformer(model="claude-x", request_id="msg_1")

        frames = [
            {"type": "message_start", "message": {"id": "msg_1", "usage": {}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": "OPAQUE-DATA"},
            },
            {"type": "content_block_stop", "index": 0},
        ]
        emitted = ""
        for frame in frames:
            chunk = converter.convert_chunk(frame)
            if chunk is not None:
                out = transformer.transform(chunk)
                if out:
                    emitted += out
        # Close the still-open redacted block so it lands in the accumulation.
        transformer.finalize()

        payloads = _sse_payloads(emitted)
        starts = [
            p
            for p in payloads
            if p.get("type") == "content_block_start"
            and p.get("content_block", {}).get("type") == "redacted_thinking"
        ]
        assert starts, "a redacted_thinking content_block_start should be emitted"
        assert starts[0]["content_block"]["data"] == "OPAQUE-DATA"

        # The accumulated output (request log / store) must carry the real
        # payload too, not the "[redacted]" placeholder.
        from llm_proxy.models import RedactedThinkingBlock

        redacted = [
            b for b in transformer._accumulated_output if isinstance(b, RedactedThinkingBlock)
        ]
        assert redacted and redacted[0].data == "OPAQUE-DATA"


class TestProviderResponsesMultiItemEncrypted:
    """OpenAI Responses provider converter: one encrypted blob per item."""

    def _converter(self) -> OpenAIResponsesChunkConverter:
        converter = OpenAIResponsesChunkConverter(model="o3", request_id="resp_1")
        converter.convert_chunk(
            {"event_type": "response.created", "response": {"id": "resp_1", "model": "o3"}}
        )
        return converter

    def test_interleaved_items_attach_their_own_blob_to_first_delta(self):
        converter = self._converter()
        converter.convert_chunk(
            {
                "event_type": "response.output_item.added",
                "item": {"id": "rs_1", "type": "reasoning", "encrypted_content": "ENC_A"},
            }
        )
        chunk_a = converter.convert_chunk(
            {"event_type": "response.reasoning_text.delta", "item_id": "rs_1", "delta": "plan A"}
        )
        # A function call interleaves between the two reasoning items.
        converter.convert_chunk(
            {
                "event_type": "response.output_item.added",
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                },
            }
        )
        converter.convert_chunk(
            {
                "event_type": "response.output_item.added",
                "item": {"id": "rs_2", "type": "reasoning", "encrypted_content": "ENC_B"},
            }
        )
        chunk_b = converter.convert_chunk(
            {"event_type": "response.reasoning_text.delta", "item_id": "rs_2", "delta": "plan B"}
        )

        assert chunk_a["choices"][0]["delta"]["encrypted_content"] == "ENC_A"
        assert chunk_b["choices"][0]["delta"]["encrypted_content"] == "ENC_B"

    def test_encrypted_only_item_emits_blob_at_item_done(self):
        """An encrypted-only reasoning item (empty summary, no deltas) must
        still put its blob on the wire when the item completes."""
        converter = self._converter()
        converter.convert_chunk(
            {
                "event_type": "response.output_item.added",
                "item": {"id": "rs_1", "type": "reasoning", "encrypted_content": "ENC_A"},
            }
        )
        chunk = converter.convert_chunk(
            {
                "event_type": "response.output_item.done",
                "item": {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "ENC_A",
                },
            }
        )
        assert chunk is not None
        assert chunk["encrypted_content"] == "ENC_A"

    def test_second_items_done_fallback_not_blocked_by_first_item(self):
        """Two summary-only items (no deltas): both summaries and both blobs
        must be emitted; a single global "already emitted" flag would drop
        the second item entirely."""
        converter = self._converter()
        for iid, text, enc in (("rs_1", "plan A", "ENC_A"), ("rs_2", "plan B", "ENC_B")):
            converter.convert_chunk(
                {
                    "event_type": "response.output_item.added",
                    "item": {"id": iid, "type": "reasoning"},
                }
            )
            chunk = converter.convert_chunk(
                {
                    "event_type": "response.output_item.done",
                    "item": {
                        "id": iid,
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": text}],
                        "encrypted_content": enc,
                    },
                }
            )
            assert chunk is not None
            delta = chunk["choices"][0]["delta"]
            assert delta["reasoning_content"] == text
            assert delta["encrypted_content"] == enc

    def test_completed_fallback_emits_each_unemitted_blob(self):
        converter = self._converter()
        converter.convert_chunk(
            {
                "event_type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC_A"},
                        {"type": "reasoning", "id": "rs_2", "encrypted_content": "ENC_B"},
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 10},
                },
            }
        )
        blobs = [
            c["encrypted_content"] for c in converter.finalize_chunks() if "encrypted_content" in c
        ]
        assert blobs == ["ENC_A", "ENC_B"]

    def test_completed_fallback_dedupes_delta_emitted_blobs(self):
        converter = self._converter()
        converter.convert_chunk(
            {
                "event_type": "response.output_item.added",
                "item": {"id": "rs_1", "type": "reasoning", "encrypted_content": "ENC_A"},
            }
        )
        converter.convert_chunk(
            {"event_type": "response.reasoning_text.delta", "item_id": "rs_1", "delta": "plan A"}
        )
        converter.convert_chunk(
            {
                "event_type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC_A"},
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 10},
                },
            }
        )
        blobs = [
            c["encrypted_content"] for c in converter.finalize_chunks() if "encrypted_content" in c
        ]
        assert blobs == []

    def test_completed_fallback_not_gated_on_completed_status(self):
        converter = self._converter()
        converter.convert_chunk(
            {
                "event_type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "output": [
                        {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC_A"},
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 10},
                },
            }
        )
        blobs = [
            c["encrypted_content"] for c in converter.finalize_chunks() if "encrypted_content" in c
        ]
        assert blobs == ["ENC_A"]


class TestOpenResponsesTransformerMultiItemEncrypted:
    """OpenResponses protocol transformer: per-item encrypted attribution."""

    def test_interleaved_items_keep_their_own_blobs(self):
        transformer = OpenResponsesStreamingTransformer(model="o3", request_id="resp_1")
        transformer.state.include_reasoning_encrypted = True

        chunks = [
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "plan A",
                            "encrypted_content": "ENC_A",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            # A tool call interleaves between the two reasoning items.
            {
                "id": "c2",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "c3",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "plan B",
                            "encrypted_content": "ENC_B",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "c4",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
        ]

        emitted = ""
        for chunk in chunks:
            out = transformer.transform(chunk)
            if out:
                emitted += out

        reasoning_done = [
            p["item"]
            for p in _sse_payloads(emitted)
            if p.get("type") == "response.output_item.done"
            and p.get("item", {}).get("type") == "reasoning"
        ]
        assert len(reasoning_done) == 2
        assert reasoning_done[0].get("encrypted_content") == "ENC_A"
        assert reasoning_done[1].get("encrypted_content") == "ENC_B"

    def test_late_top_level_blob_attributed_to_reasoning_item(self):
        """A blob arriving as a top-level chunk (response.completed fallback)
        after the reasoning deltas must still land on the reasoning item."""
        transformer = OpenResponsesStreamingTransformer(model="o3", request_id="resp_1")
        transformer.state.include_reasoning_encrypted = True

        chunks = [
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "plan A"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "c2",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "encrypted_content": "ENC_LATE",
            },
            {
                "id": "c3",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]

        emitted = ""
        for chunk in chunks:
            out = transformer.transform(chunk)
            if out:
                emitted += out

        reasoning_items = [
            p["item"]
            for p in _sse_payloads(emitted)
            if p.get("type") == "response.output_item.done"
            and p.get("item", {}).get("type") == "reasoning"
        ]
        assert reasoning_items
        assert reasoning_items[0].get("encrypted_content") == "ENC_LATE"

    def test_repeated_blob_is_deduped(self):
        """The same blob arriving twice (delta + terminal chunk) must not be
        attributed to two different items."""
        transformer = OpenResponsesStreamingTransformer(model="o3", request_id="resp_1")
        transformer.state.include_reasoning_encrypted = True

        chunks = [
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "plan A",
                            "encrypted_content": "ENC_A",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            # A tool call separates the two reasoning items.
            {
                "id": "c2",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            # Item B carries no encrypted payload of its own.
            {
                "id": "c3",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "plan B"},
                        "finish_reason": None,
                    }
                ],
            },
            # The terminal chunk repeats item A's blob; without dedupe it
            # would be misattributed to item B.
            {
                "id": "c4",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "encrypted_content": "ENC_A",
            },
            {
                "id": "c5",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]

        emitted = ""
        for chunk in chunks:
            out = transformer.transform(chunk)
            if out:
                emitted += out

        reasoning_items = [
            p["item"]
            for p in _sse_payloads(emitted)
            if p.get("type") == "response.output_item.done"
            and p.get("item", {}).get("type") == "reasoning"
        ]
        assert len(reasoning_items) == 2
        assert reasoning_items[0].get("encrypted_content") == "ENC_A"
        assert "encrypted_content" not in reasoning_items[1]

    def test_redacted_payload_bypasses_include_gate(self):
        """A redacted-thinking payload (encrypted_content + reasoning_is_redacted)
        must be attached to the reasoning item even without the include flag:
        like an Anthropic signature, it is integrity data required for the
        next-turn echo, not spec-gated encrypted reasoning."""
        transformer = OpenResponsesStreamingTransformer(model="claude-x", request_id="resp_1")

        chunks = [
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "claude-x",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "[redacted]",
                            "reasoning_is_redacted": True,
                            "encrypted_content": "OPAQUE-DATA",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "c2",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "claude-x",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]

        emitted = ""
        for chunk in chunks:
            out = transformer.transform(chunk)
            if out:
                emitted += out

        reasoning_items = [
            p["item"]
            for p in _sse_payloads(emitted)
            if p.get("type") == "response.output_item.done"
            and p.get("item", {}).get("type") == "reasoning"
        ]
        assert reasoning_items
        assert reasoning_items[0].get("encrypted_content") == "OPAQUE-DATA"

    def test_genuine_blob_still_include_gated(self):
        """Without include, a genuine (non-redacted) encrypted blob is not
        captured and must not leak onto the wire."""
        transformer = OpenResponsesStreamingTransformer(model="o3", request_id="resp_1")

        chunks = [
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "plan A",
                            "encrypted_content": "ENC_A",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "c2",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "o3",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]

        emitted = ""
        for chunk in chunks:
            out = transformer.transform(chunk)
            if out:
                emitted += out

        reasoning_items = [
            p["item"]
            for p in _sse_payloads(emitted)
            if p.get("type") == "response.output_item.done"
            and p.get("item", {}).get("type") == "reasoning"
        ]
        assert reasoning_items
        assert "encrypted_content" not in reasoning_items[0]


class TestAnthropicInterleavedThinkingAfterTool:
    """A ``thinking -> tool_use -> thinking -> tool_use`` stream must stay
    well-formed: the canonical stream carries no boundary chunk when a tool
    block stops, so the transformer has to close the open tool block itself
    before starting the next thinking block. Regression for an interleaved
    turn that reused a content-block index and never stopped the tool block.
    """

    def _frames(self) -> list[dict]:
        return [
            {"type": "message_start", "message": {"id": "msg_1", "usage": {}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "plan"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "SIG0"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"q":"x"}'},
            },
            {"type": "content_block_stop", "index": 1},
            # Interleaved thinking after the tool call.
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "thinking_delta", "thinking": "after"},
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "signature_delta", "signature": "SIG2"},
            },
            {"type": "content_block_stop", "index": 2},
            {
                "type": "content_block_start",
                "index": 3,
                "content_block": {"type": "tool_use", "id": "t2", "name": "search", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 3,
                "delta": {"type": "input_json_delta", "partial_json": '{"q":"y"}'},
            },
            {"type": "content_block_stop", "index": 3},
        ]

    def _run(self) -> list[dict]:
        converter = AnthropicChunkConverter(model="claude-x", request_id="msg_1")
        transformer = AnthropicStreamingTransformer(model="claude-x", request_id="msg_1")
        emitted = ""
        for frame in self._frames():
            chunk = converter.convert_chunk(frame)
            if chunk is not None:
                out = transformer.transform(chunk)
                if out:
                    emitted += out
        transformer.finalize()
        return _sse_payloads(emitted)

    def test_every_block_gets_its_own_stop_at_its_own_index(self):
        payloads = self._run()
        started: dict[int, str] = {}
        stopped: set[int] = set()
        for payload in payloads:
            if payload.get("type") == "content_block_start":
                # A new block must never start at an index without a stop
                # in between.
                assert payload["index"] not in started, (
                    f"index {payload['index']} reused without a stop"
                )
                started[payload["index"]] = payload["content_block"]["type"]
            elif payload.get("type") == "content_block_stop":
                assert payload["index"] in started
                stopped.add(payload["index"])
        # The two tool blocks and the two thinking blocks all opened.
        assert [started[i] for i in sorted(started)] == [
            "thinking",
            "tool_use",
            "thinking",
            "tool_use",
        ]
        # The interleaved tool block (index 1) was stopped before index 2 began.
        assert 1 in stopped

    def test_signatures_follow_their_own_thinking_block(self):
        payloads = self._run()
        signature_by_index = {
            p["index"]: p["delta"]["signature"]
            for p in payloads
            if p.get("type") == "content_block_delta"
            and p.get("delta", {}).get("type") == "signature_delta"
        }
        assert signature_by_index == {0: "SIG0", 2: "SIG2"}


class TestAnthropicSignatureOnlySegments:
    """A thinking signature with no text must survive streaming accumulation.

    Non-stream preserved signature-only blocks already; the stream dropped them
    and could leak the stale signature onto a later unrelated block.
    """

    def test_signature_only_chunk_accumulates_a_signed_block(self):
        transformer = AnthropicStreamingTransformer(model="claude-x", request_id="msg_1")
        out = transformer.transform(
            {
                "choices": [
                    {"index": 0, "delta": {"reasoning_signature": "SIG"}, "finish_reason": "stop"}
                ]
            }
        )
        assert "signature_delta" in (out or "")
        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert accumulated[0].thinking == ""
        assert accumulated[0].signature == "SIG"

    def test_signature_does_not_leak_into_the_next_block(self):
        transformer = AnthropicStreamingTransformer(model="claude-x", request_id="msg_1")
        frames = [
            {
                "choices": [
                    {"index": 0, "delta": {"reasoning_signature": "S1"}, "finish_reason": None}
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
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "f", "arguments": "{}"},
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
                        "delta": {"reasoning_content": "second", "reasoning_signature": "S2"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ]
        for frame in frames:
            transformer.transform(frame)
        accumulated = transformer.get_accumulated_output()
        assert [type(b).__name__ for b in accumulated] == [
            "ThinkingBlock",
            "ToolUseBlock",
            "ThinkingBlock",
        ]
        assert [(b.thinking, b.signature) for b in accumulated if hasattr(b, "thinking")] == [
            ("", "S1"),
            ("second", "S2"),
        ]


class TestProviderResponsesEncryptedDedupe:
    """A repeated encrypted blob must reach the wire only once."""

    def _converter(self) -> OpenAIResponsesChunkConverter:
        converter = OpenAIResponsesChunkConverter(model="o3", request_id="resp_1")
        converter.convert_chunk(
            {"event_type": "response.created", "response": {"id": "resp_1", "model": "o3"}}
        )
        return converter

    def test_final_fallback_dedupes_by_value_across_item_ids(self):
        converter = self._converter()
        converter.convert_chunk(
            {
                "event_type": "response.output_item.added",
                "item": {"id": "rs_1", "type": "reasoning", "encrypted_content": "ENC"},
            }
        )
        emitted = converter.convert_chunk(
            {"event_type": "response.reasoning_text.delta", "item_id": "rs_1", "delta": "plan"}
        )
        assert emitted["choices"][0]["delta"]["encrypted_content"] == "ENC"

        # The completed snapshot reports the same blob under a different id.
        converter.convert_chunk(
            {
                "event_type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "id": "rs_9", "encrypted_content": "ENC"},
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }
        )
        blobs = [
            c["encrypted_content"] for c in converter.finalize_chunks() if "encrypted_content" in c
        ]
        assert blobs == []
