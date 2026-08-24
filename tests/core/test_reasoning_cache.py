"""Tests for the reasoning cache write paths (core.reasoning_cache).

The cache restores reasoning text (keyed by tool-call id, scoped by
response id) when a later openresponses materialization needs it. Every
response path — parsed, wire-reuse verbatim, and native passthrough — must
feed it; these tests pin the raw-body extractors the verbatim paths use.
"""

import types

import pytest

from llm_proxy.core import reasoning_cache
from llm_proxy.core.conversion import NativePassthroughHandler
from llm_proxy.core.reasoning_cache import (
    cache_reasoning_from_chat_completion_body,
    cache_reasoning_from_responses_output,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    reasoning_cache.clear()
    yield
    reasoning_cache.clear()


class TestCacheReasoningFromResponsesOutput:
    def test_reasoning_before_call(self):
        output = [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "chain"}],
            },
            {"type": "function_call", "call_id": "call_1", "name": "f"},
        ]
        cache_reasoning_from_responses_output(output, "resp_1")
        assert reasoning_cache.get("call_1") == "chain"

    def test_call_before_reasoning_backfills(self):
        """A model may emit the function_call before its reasoning."""
        output = [
            {"type": "function_call", "call_id": "call_1", "name": "f"},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "late chain"}],
            },
        ]
        cache_reasoning_from_responses_output(output, "resp_1")
        assert reasoning_cache.get("call_1") == "late chain"

    def test_missing_response_id_is_noop(self):
        output = [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "x"}]},
            {"type": "function_call", "call_id": "call_1"},
        ]
        cache_reasoning_from_responses_output(output, "")
        assert reasoning_cache.get("call_1") is None

    def test_no_reasoning_is_noop(self):
        cache_reasoning_from_responses_output(
            [{"type": "function_call", "call_id": "call_1"}], "resp_1"
        )
        assert reasoning_cache.get("call_1") is None


class TestCacheReasoningFromChatCompletionBody:
    def test_pairs_reasoning_with_tool_calls(self):
        body = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "thinking hard",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
        }
        cache_reasoning_from_chat_completion_body(body)
        assert reasoning_cache.get("call_1") == "thinking hard"

    def test_raw_reasoning_field_accepted(self):
        """The pre-rename provider field name still feeds the cache."""
        body = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning": "raw field",
                        "tool_calls": [{"id": "call_2", "type": "function", "function": {}}],
                    }
                }
            ],
        }
        cache_reasoning_from_chat_completion_body(body)
        assert reasoning_cache.get("call_2") == "raw field"

    def test_reasoning_without_tool_calls_not_cached(self):
        """Cache entries are keyed by call_id; bare reasoning has no key."""
        body = {
            "id": "chatcmpl-1",
            "choices": [{"message": {"role": "assistant", "reasoning_content": "x"}}],
        }
        cache_reasoning_from_chat_completion_body(body)
        # Nothing to assert against a key — the cache must simply stay empty.
        assert reasoning_cache.get("call_1") is None


class TestNativeStreamReasoningCache:
    """The native openresponses stream bypasses the chunk transformer, so the
    terminal snapshot is the only place reasoning can enter the cache."""

    def test_terminal_snapshot_feeds_cache(self):
        transformer = types.SimpleNamespace(
            state=types.SimpleNamespace(final_response_payload=None)
        )
        chunk = (
            "event: response.completed\n"
            'data: {"type":"response.completed","response":{"id":"resp_9",'
            '"output":['
            '{"type":"reasoning","summary":[{"type":"summary_text","text":"chain"}]},'
            '{"type":"function_call","call_id":"call_9","name":"f"}'
            "]}}\n\n"
        )

        NativePassthroughHandler.maybe_capture_native_openresponses(chunk, transformer, None)

        assert reasoning_cache.get("call_9") == "chain"
        # Snapshot bookkeeping still happens alongside the cache write.
        assert transformer.state.final_response_payload["id"] == "resp_9"
