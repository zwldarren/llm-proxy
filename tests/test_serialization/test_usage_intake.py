"""Usage intake: one adapter per wire shape, one accumulator per stream (ADR-0025)."""

from __future__ import annotations

import pytest

from llm_proxy.models.types import Usage
from llm_proxy.serialization.usage import UsageAccumulator, add_usage, parse_usage

# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


class TestChatDialect:
    def test_parses_a_full_body(self) -> None:
        usage = parse_usage(
            "chat",
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        )

        assert usage == Usage(input_tokens=100, output_tokens=20, total_tokens=120)

    def test_no_usage_block_is_not_a_zero_usage(self) -> None:
        assert parse_usage("chat", {"id": "chatcmpl-1", "choices": []}) is None

    def test_nested_cache_alias_is_lifted_and_cleared(self) -> None:
        usage = parse_usage(
            "chat",
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_tokens_details": {
                        "cached_tokens": 80,
                        "cache_write_tokens": 15,
                        "audio_tokens": 5,
                    },
                }
            },
        )

        assert usage is not None
        assert usage.cache_read_input_tokens == 80
        assert usage.cache_creation_input_tokens == 15
        # One fact, one field: the nested copies do not coexist with the flat ones.
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens is None
        assert usage.prompt_tokens_details.cache_write_tokens is None
        # Unrelated nested facts survive the lift.
        assert usage.prompt_tokens_details.audio_tokens == 5

    def test_flat_cache_read_wins_over_the_nested_copy(self) -> None:
        usage = parse_usage(
            "chat",
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cache_read_input_tokens": 70,
                    "prompt_tokens_details": {"cached_tokens": 999},
                }
            },
        )

        assert usage is not None
        assert usage.cache_read_input_tokens == 70
        # The nested copy was the block's only fact, so the lifted block is gone
        # rather than left as an all-``None`` object a truthiness test would read
        # as "has nested facts".
        assert usage.prompt_tokens_details is None

    def test_deepseek_top_level_cache_hits_fold_in(self) -> None:
        usage = parse_usage(
            "chat",
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 0,
                    "prompt_cache_hit_tokens": 60,
                }
            },
        )

        assert usage is not None
        assert usage.cache_read_input_tokens == 60

    def test_sglang_top_level_reasoning_lands_in_the_canonical_field(self) -> None:
        """Thinking tokens are one fact with one home: the flat field."""
        usage = parse_usage(
            "chat",
            {"usage": {"prompt_tokens": 10, "completion_tokens": 7, "reasoning_tokens": 5}},
        )

        assert usage is not None
        assert usage.reasoning_tokens == 5
        # The dialect alias is cleared, not left as a second expression.
        assert usage.completion_tokens_details is None

    def test_nested_reasoning_lands_in_the_canonical_field(self) -> None:
        usage = parse_usage(
            "chat",
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 7,
                    "completion_tokens_details": {"reasoning_tokens": 5, "audio_tokens": 2},
                }
            },
        )

        assert usage is not None
        assert usage.reasoning_tokens == 5
        # A detail with no canonical field of its own survives the lift.
        assert usage.completion_tokens_details is not None
        assert usage.completion_tokens_details.audio_tokens == 2
        assert usage.completion_tokens_details.reasoning_tokens is None

    def test_an_explicit_nested_count_wins_over_the_top_level_one(self) -> None:
        usage = parse_usage(
            "chat",
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 7,
                    "reasoning_tokens": 5,
                    "completion_tokens_details": {"reasoning_tokens": 9},
                }
            },
        )

        assert usage is not None
        assert usage.reasoning_tokens == 9

    def test_provider_reported_cost_and_web_search(self) -> None:
        usage = parse_usage(
            "chat",
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "nanogpt_cost": 0.0042,
                    "server_tool_use": {"web_search_requests": 3},
                }
            },
        )

        assert usage is not None
        assert usage.provider_reported_cost == pytest.approx(0.0042)
        assert usage.web_search_requests == 3

    def test_openrouter_cost_is_a_fallback_for_nanogpt_cost(self) -> None:
        usage = parse_usage("chat", {"usage": {"cost": 0.01}})

        assert usage is not None
        assert usage.provider_reported_cost == pytest.approx(0.01)

    def test_a_stream_chunk_is_the_same_envelope(self) -> None:
        usage = parse_usage(
            "chat",
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

        assert usage is not None
        assert usage.input_tokens == 5


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class TestAnthropicDialect:
    def test_message_start_makes_input_tokens_inclusive_of_cache(self) -> None:
        usage = parse_usage(
            "anthropic",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {
                        "input_tokens": 400,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 800,
                        "cache_creation_input_tokens": 100,
                    },
                },
            },
        )

        assert usage is not None
        assert usage.input_tokens == 1300
        assert usage.output_tokens == 1
        assert usage.cache_read_input_tokens == 800
        assert usage.cache_creation_input_tokens == 100
        assert usage.total_tokens == 1301

    def test_message_delta_carries_usage_at_the_top_level(self) -> None:
        usage = parse_usage(
            "anthropic",
            {"type": "message_delta", "usage": {"output_tokens": 42}},
        )

        assert usage is not None
        assert usage.output_tokens == 42

    def test_thinking_tokens_become_reasoning_tokens(self) -> None:
        usage = parse_usage(
            "anthropic",
            {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "output_tokens_details": {"thinking_tokens": 7},
                }
            },
        )

        assert usage is not None
        assert usage.reasoning_tokens == 7

    def test_non_usage_event_is_none(self) -> None:
        assert (
            parse_usage("anthropic", {"type": "content_block_delta", "delta": {"text": "x"}})
            is None
        )

    def test_absent_cache_counters_are_not_reported_as_zero(self) -> None:
        """``None`` means "the wire did not say"; a zero means it said zero."""
        usage = parse_usage("anthropic", {"type": "message_delta", "usage": {"output_tokens": 40}})

        assert usage is not None
        assert usage.cache_read_input_tokens is None
        assert usage.cache_creation_input_tokens is None

    def test_a_reported_zero_cache_counter_stays_zero(self) -> None:
        usage = parse_usage(
            "anthropic",
            {"usage": {"input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 0}},
        )

        assert usage is not None
        assert usage.cache_read_input_tokens == 0


# ---------------------------------------------------------------------------
# OpenResponses
# ---------------------------------------------------------------------------


class TestResponsesDialect:
    def test_parses_a_body(self) -> None:
        usage = parse_usage(
            "responses",
            {
                "id": "resp_1",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 8,
                    "total_tokens": 58,
                    "input_tokens_details": {"cached_tokens": 30, "image_tokens": 2},
                    "output_tokens_details": {"reasoning_tokens": 4},
                },
            },
        )

        assert usage is not None
        assert usage.input_tokens == 50
        assert usage.cache_read_input_tokens == 30
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.image_tokens == 2
        # Thinking is the canonical flat field; the rest of the details block went
        # with the alias.
        assert usage.reasoning_tokens == 4
        assert usage.completion_tokens_details is None

    def test_stream_terminal_event_nests_usage_under_response(self) -> None:
        usage = parse_usage(
            "responses",
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "usage": {"input_tokens": 9, "output_tokens": 3}},
            },
        )

        assert usage is not None
        assert usage.input_tokens == 9
        assert usage.output_tokens == 3
        # No wire total: the record adds the parts rather than reporting zero.
        assert usage.total_tokens == 12

    def test_web_search_calls_bill_without_any_usage_object(self) -> None:
        usage = parse_usage(
            "responses",
            {"id": "resp_1", "output": [{"type": "web_search_call"}, {"type": "message"}]},
        )

        assert usage is not None
        assert usage.web_search_requests == 1
        assert usage.input_tokens == 0

    def test_a_body_bills_its_search_items_alongside_the_usage_block(self) -> None:
        """The Responses wire has no usage-level search count: the items are the count.

        The parity ratchet caught this against the shipped
        ``parse_usage_from_response``: a body that carried both a usage block and
        web-search items billed zero searches, because the adapter only looked
        for a usage-level field the Responses API never sends.
        """
        usage = parse_usage(
            "responses",
            {
                "id": "resp_1",
                "usage": {"input_tokens": 10, "output_tokens": 1},
                "output": [{"type": "web_search_call"}, {"type": "message"}],
            },
        )

        assert usage is not None
        assert usage.web_search_requests == 1
        assert usage.input_tokens == 10

    def test_a_reported_search_count_wins_over_the_items(self) -> None:
        usage = parse_usage(
            "responses",
            {
                "usage": {"input_tokens": 10, "output_tokens": 1, "web_search_requests": 4},
                "output": [{"type": "web_search_call"}],
            },
        )

        assert usage is not None
        assert usage.web_search_requests == 4

    def test_no_usage_and_no_billable_events_is_none(self) -> None:
        assert parse_usage("responses", {"id": "resp_1", "output": [{"type": "message"}]}) is None


# ---------------------------------------------------------------------------
# Realtime
# ---------------------------------------------------------------------------


class TestRealtimeDialect:
    def test_response_done_singular_detail_keys(self) -> None:
        usage = parse_usage(
            "realtime",
            {
                "type": "response.done",
                "response": {
                    "id": "resp_1",
                    "usage": {
                        "input_tokens": 120,
                        "output_tokens": 30,
                        "input_token_details": {"cached_tokens": 20, "audio_tokens": 10},
                        "output_token_details": {"audio_tokens": 25},
                    },
                },
            },
        )

        assert usage is not None
        assert usage.input_tokens == 120
        assert usage.output_tokens == 30
        assert usage.cache_read_input_tokens == 20
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.audio_tokens == 10
        assert usage.completion_tokens_details is not None
        assert usage.completion_tokens_details.audio_tokens == 25


class TestOllamaDialect:
    """Ollama's counters, and the cache count that is a subset of the input one."""

    def test_absent_counters_make_no_record(self) -> None:
        assert parse_usage("ollama", {"model": "llama3", "done": True}) is None

    def test_either_counter_alone_is_a_report(self) -> None:
        only_prompt = parse_usage("ollama", {"prompt_eval_count": 11})
        assert only_prompt is not None
        assert (only_prompt.input_tokens, only_prompt.output_tokens) == (11, 0)

        only_output = parse_usage("ollama", {"eval_count": 7})
        assert only_output is not None
        assert (only_output.input_tokens, only_output.output_tokens) == (0, 7)

    def test_a_reported_zero_is_a_count_not_a_missing_field(self) -> None:
        usage = parse_usage("ollama", {"prompt_eval_count": 0, "eval_count": 50})
        assert usage is not None
        assert (usage.input_tokens, usage.output_tokens) == (0, 50)

    def test_the_cache_count_is_clamped_into_the_input_total(self) -> None:
        """A provider breaking "cache is a subset of input" yields a sane count.

        ``prompt_eval_cached_count`` is a subset of ``prompt_eval_count``; a count
        outside ``[0, input]`` would otherwise be priced as tokens that were never
        part of the prompt.
        """
        oversized = parse_usage("ollama", {"prompt_eval_count": 5, "prompt_eval_cached_count": 99})
        assert oversized is not None
        assert oversized.cache_read_input_tokens == 5

        negative = parse_usage("ollama", {"prompt_eval_count": 5, "prompt_eval_cached_count": -3})
        assert negative is not None
        assert negative.cache_read_input_tokens == 0

    def test_absent_cache_count_is_not_a_reported_zero(self) -> None:
        usage = parse_usage("ollama", {"prompt_eval_count": 11, "eval_count": 2})
        assert usage is not None
        assert usage.cache_read_input_tokens is None

    def test_the_total_is_the_two_counters_summed(self) -> None:
        usage = parse_usage("ollama", {"prompt_eval_count": 11, "eval_count": 2})
        assert usage is not None
        assert usage.total_tokens == 13


# ---------------------------------------------------------------------------
# The accumulator
# ---------------------------------------------------------------------------


class TestUsageAccumulator:
    def test_a_split_turn_merges_rather_than_sums(self) -> None:
        accumulator = UsageAccumulator("anthropic")
        accumulator.observe(
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 50,
                    }
                },
            }
        )
        accumulator.observe(
            {"type": "message_delta", "usage": {"input_tokens": 100, "output_tokens": 40}}
        )

        usage = accumulator.result()

        assert usage is not None
        assert usage.input_tokens == 150  # 100 + the 50 cache read, once
        assert usage.output_tokens == 40  # the later snapshot's value, not 1 + 40
        assert usage.cache_read_input_tokens == 50  # the delta omitted it; it survives

    def test_a_cumulative_counter_never_decreases(self) -> None:
        accumulator = UsageAccumulator("anthropic")
        accumulator.observe(
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 100, "cache_read_input_tokens": 50}},
            }
        )
        # A later event restates the prompt accounting without the cache
        # counters: the cached prompt must not drop out of the inclusive count.
        accumulator.observe({"type": "message_delta", "usage": {"input_tokens": 100}})

        usage = accumulator.result()

        assert usage is not None
        assert usage.input_tokens == 150
        assert usage.cache_read_input_tokens == 50

    def test_nothing_observed_is_none(self) -> None:
        assert UsageAccumulator("chat").result() is None

    def test_chunks_without_usage_leave_no_record(self) -> None:
        accumulator = UsageAccumulator("chat")
        accumulator.observe({"choices": [{"delta": {"content": "hello"}}]})

        assert accumulator.result() is None

    def test_billable_events_alone_still_produce_a_record(self) -> None:
        accumulator = UsageAccumulator("responses")
        accumulator.observe(
            {"type": "response.output_item.added", "item": {"type": "web_search_call"}}
        )
        accumulator.observe(
            {"type": "response.output_item.added", "item": {"type": "web_search_call"}}
        )

        usage = accumulator.result()

        assert usage is not None
        assert usage.web_search_requests == 2
        assert usage.input_tokens == 0

    def test_provider_reported_count_wins_over_the_counted_events(self) -> None:
        accumulator = UsageAccumulator("responses")
        accumulator.observe(
            {"type": "response.output_item.added", "item": {"type": "web_search_call"}}
        )
        accumulator.observe(
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 5, "web_search_requests": 7}},
            }
        )

        usage = accumulator.result()

        assert usage is not None
        assert usage.web_search_requests == 7

    def test_result_is_repeatable(self) -> None:
        accumulator = UsageAccumulator("chat")
        accumulator.observe({"usage": {"prompt_tokens": 3, "completion_tokens": 1}})

        assert accumulator.result() == accumulator.result()

    def test_result_is_isolated_from_later_observations(self) -> None:
        accumulator = UsageAccumulator("chat")
        accumulator.observe({"usage": {"prompt_tokens": 3, "completion_tokens": 1}})
        first = accumulator.result()
        assert first is not None

        accumulator.observe({"usage": {"prompt_tokens": 30, "completion_tokens": 10}})
        first.prompt_tokens_details = None  # a caller's own edit must not propagate

        assert first.input_tokens == 3
        assert accumulator.result() is not None
        assert accumulator.result().input_tokens == 30

    def test_a_restated_terminal_output_array_does_not_double_the_tally(self) -> None:
        accumulator = UsageAccumulator("responses")
        accumulator.observe(
            {"type": "response.output_item.added", "item": {"type": "web_search_call"}}
        )
        # The terminal event restates every item it already announced, and
        # carries the usage block that confirms the count.
        accumulator.observe(
            {
                "type": "response.completed",
                "response": {
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                    "output": [{"type": "web_search_call"}, {"type": "message"}],
                },
            }
        )

        usage = accumulator.result()

        assert usage is not None
        assert usage.web_search_requests == 1

    def test_a_synthesized_total_tracks_the_merged_counters(self) -> None:
        """Anthropic sends no wire total, so a per-event total is a partial sum."""
        accumulator = UsageAccumulator("anthropic")
        accumulator.observe(
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 50,
                    }
                },
            }
        )
        accumulator.observe({"type": "message_delta", "usage": {"output_tokens": 40}})

        usage = accumulator.result()

        assert usage is not None
        assert usage.input_tokens == 150
        assert usage.output_tokens == 40
        assert usage.total_tokens == 190

    def test_dialect_is_declared_not_guessed(self) -> None:
        with pytest.raises(ValueError, match="unknown usage dialect"):
            parse_usage("gemini", {"usage": {}})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Merge rules
# ---------------------------------------------------------------------------


class TestMergeRules:
    def test_add_usage_sums_independent_calls(self) -> None:
        target = Usage(input_tokens=10, output_tokens=2, total_tokens=12, cache_read_input_tokens=4)
        source = Usage(input_tokens=5, output_tokens=3, total_tokens=8, cache_read_input_tokens=1)

        add_usage(target, source)

        assert target.input_tokens == 15
        assert target.output_tokens == 5
        assert target.total_tokens == 20
        assert target.cache_read_input_tokens == 5

    def test_add_usage_leaves_unknown_fields_alone(self) -> None:
        target = Usage(input_tokens=10, output_tokens=2)
        source = Usage(input_tokens=5, output_tokens=3)

        add_usage(target, source)

        assert target.cache_read_input_tokens is None
        assert target.reasoning_tokens is None

    def test_add_usage_sums_two_reported_costs(self) -> None:
        """Both calls were billed; a cost that covers one of them under-reports."""
        target = Usage(input_tokens=10, provider_reported_cost=0.01)
        source = Usage(input_tokens=5, provider_reported_cost=0.02)

        add_usage(target, source)

        assert target.provider_reported_cost == pytest.approx(0.03)
