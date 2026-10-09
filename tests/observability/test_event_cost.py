"""Tests for the shared event-cost finalization helper."""

from unittest.mock import AsyncMock, MagicMock, patch

from llm_proxy.billing.cost import CostBreakdown
from llm_proxy.billing.tokens import token_usage_from
from llm_proxy.models.types import PromptTokensDetails, Usage
from llm_proxy.observability.cost import (
    calculate_event_cost,
    extract_streaming_usage,
    finalize_event_cost,
)
from llm_proxy.observability.event_context import EventContext


def _context(**kwargs):
    return EventContext(request_id="req-1", trace_id="trace-1", model="gpt-4", **kwargs)


class TestCalculateEventCost:
    async def test_skips_when_cost_already_set(self):
        context = _context(cost_usd=0.5, usage=Usage(input_tokens=10, output_tokens=5))
        await calculate_event_cost(context, config_manager=MagicMock())
        # Unchanged — no re-calculation.
        assert context.cost_usd == 0.5

    async def test_provider_reported_cost_takes_precedence(self):
        context = _context(
            provider_reported_cost=0.42, usage=Usage(input_tokens=10, output_tokens=5)
        )
        with patch("llm_proxy.observability.cost.calculate_cost", new=AsyncMock()) as mock_calc:
            await calculate_event_cost(context, config_manager=MagicMock())
            mock_calc.assert_not_called()
        assert context.cost_usd == 0.42

    async def test_calculates_from_pricing_db(self):
        context = _context(usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15))
        config_manager = MagicMock()
        with patch(
            "llm_proxy.observability.cost.calculate_cost",
            new=AsyncMock(return_value=CostBreakdown(cost_usd=0.012, cache_savings_usd=0.001)),
        ) as mock_calc:
            await calculate_event_cost(context, config_manager)
            mock_calc.assert_awaited_once()
        assert context.cost_usd == 0.012
        assert context.cache_savings_usd == 0.001

    async def test_no_token_data_skips_calculation(self):
        context = _context()
        with patch("llm_proxy.observability.cost.calculate_cost", new=AsyncMock()) as mock_calc:
            await calculate_event_cost(context, config_manager=MagicMock())
            mock_calc.assert_not_called()
        assert context.cost_usd is None

    async def test_no_config_manager_skips_calculation(self):
        context = _context(usage=Usage(input_tokens=10, output_tokens=5))
        with patch("llm_proxy.observability.cost.calculate_cost", new=AsyncMock()) as mock_calc:
            await calculate_event_cost(context, config_manager=None)
            mock_calc.assert_not_called()
        assert context.cost_usd is None


class TestExtractStreamingUsage:
    async def test_noop_without_transformer(self):
        context = _context()
        await extract_streaming_usage(context)
        assert context.usage is None

    async def test_pulls_usage_from_transformer(self):
        usage = Usage(input_tokens=7, output_tokens=3, total_tokens=10)
        transformer = MagicMock()
        transformer.get_usage.return_value = usage

        context = _context()
        context.transformer = transformer
        await extract_streaming_usage(context)
        assert context.usage is not None
        assert context.usage.input_tokens == 7
        assert context.usage.output_tokens == 3

    async def test_a_non_record_is_ignored(self):
        """A transformer that hands back something else is not a usage record."""
        transformer = MagicMock()
        transformer.get_usage.return_value = {"input_tokens": 7}

        context = _context()
        context.transformer = transformer
        await extract_streaming_usage(context)
        assert context.usage is None

    async def test_ignores_transformer_without_get_usage(self):
        context = _context()
        context.transformer = MagicMock(spec=[])  # no get_usage attribute
        await extract_streaming_usage(context)
        assert context.usage is None


class TestFinalizeEventCost:
    async def test_extracts_usage_then_calculates_cost(self):
        usage = Usage(input_tokens=7, output_tokens=3, total_tokens=10)
        transformer = MagicMock()
        transformer.get_usage.return_value = usage

        context = _context()
        context.transformer = transformer

        with patch(
            "llm_proxy.observability.cost.calculate_cost",
            new=AsyncMock(return_value=CostBreakdown(cost_usd=0.02)),
        ):
            await finalize_event_cost(context, config_manager=MagicMock())

        assert context.usage is not None
        assert context.usage.input_tokens == 7
        assert context.cost_usd == 0.02


class TestUpdateUsageKeepsOneFactPerField:
    """The record carries each fact once, in the field its dialect used.

    ``update_usage`` no longer rewrites a dialect's nested detail into the flat
    field: whether a nested expression stands in for the flat one is billing's
    read-model question (``token_usage_from``), not the record's, and the record
    keeps what the provider said (ADR-0025).
    """

    def test_nested_cache_write_detail_is_kept(self):
        context = _context()
        context.update_usage(
            Usage(
                input_tokens=100,
                output_tokens=10,
                prompt_tokens_details=PromptTokensDetails(cache_write_tokens=40),
            )
        )
        assert context.usage is not None
        assert context.usage.prompt_tokens_details is not None
        assert context.usage.prompt_tokens_details.cache_write_tokens == 40
        # Billing reads it as the cache-write dimension.
        assert token_usage_from(context.usage).cache_creation_input_tokens == 40

    def test_nested_cached_tokens_detail_is_kept(self):
        context = _context()
        context.update_usage(
            Usage(
                input_tokens=100,
                output_tokens=10,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=30),
            )
        )
        assert context.usage is not None
        assert context.usage.prompt_tokens_details is not None
        assert context.usage.prompt_tokens_details.cached_tokens == 30
        assert token_usage_from(context.usage).cached_prompt_tokens == 30

    def test_flat_cache_write_wins_at_the_read_model(self):
        context = _context()
        context.update_usage(
            Usage(
                input_tokens=100,
                output_tokens=10,
                cache_creation_input_tokens=7,
                prompt_tokens_details=PromptTokensDetails(cache_write_tokens=40),
            )
        )
        assert token_usage_from(context.usage).cache_creation_input_tokens == 7

    def test_a_later_observation_cannot_shrink_the_record(self):
        """Usage arrives cumulatively: a sparse frame adds facts, never removes."""
        context = _context()
        context.update_usage(Usage(input_tokens=100, output_tokens=10, cache_read_input_tokens=40))
        context.update_usage(Usage(input_tokens=5))
        assert context.usage is not None
        assert context.usage.input_tokens == 100
        assert context.usage.cache_read_input_tokens == 40

    def test_the_callers_record_is_not_mutated(self):
        """A caller may still hold (and log) the record it observed."""
        observed = Usage(input_tokens=100, output_tokens=10)
        context = _context()
        context.update_usage(observed)
        context.update_usage(Usage(input_tokens=500, output_tokens=50))
        assert observed.input_tokens == 100
        assert context.usage is not None
        assert context.usage.input_tokens == 500


class TestCalculateEventCostEstimation:
    async def test_estimates_from_request_messages_when_usage_missing(self):
        """Providers that report no billable usage fall back to tiktoken
        estimation from the request messages instead of dropping the cost."""
        context = _context()
        context.request_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello world"}],
        }
        with patch(
            "llm_proxy.observability.cost.calculate_cost",
            new=AsyncMock(return_value=CostBreakdown(cost_usd=0.001)),
        ) as mock_calc:
            await calculate_event_cost(context, MagicMock())
            mock_calc.assert_awaited_once()
            assert mock_calc.await_args.kwargs["messages"] == [
                {"role": "user", "content": "hello world"}
            ]
        assert context.cost_usd == 0.001

    async def test_estimates_completion_text_from_transformer_output(self):
        """Streaming: when no usage chunk arrives, completion tokens are
        estimated from the transformer's accumulated output blocks."""
        from llm_proxy.models.content_blocks.core import TextBlock

        context = _context()
        context.request_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello world"}],
        }
        transformer = MagicMock()
        transformer.get_accumulated_output.return_value = [
            TextBlock(text="hello "),
            TextBlock(text="world"),
        ]
        context.transformer = transformer
        with patch(
            "llm_proxy.observability.cost.calculate_cost",
            new=AsyncMock(return_value=CostBreakdown(cost_usd=0.002)),
        ) as mock_calc:
            await calculate_event_cost(context, MagicMock())
            mock_calc.assert_awaited_once()
            assert mock_calc.await_args.kwargs["completion_text"] == "hello world"
        assert context.cost_usd == 0.002

    async def test_estimates_completion_text_from_response_body(self):
        """Non-streaming: completion tokens are estimated from the formatted
        response body when the provider omitted usage."""
        context = _context()
        context.request_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello world"}],
        }
        context.response_body = {
            "choices": [{"message": {"role": "assistant", "content": "hi there"}}]
        }
        with patch(
            "llm_proxy.observability.cost.calculate_cost",
            new=AsyncMock(return_value=CostBreakdown(cost_usd=0.003)),
        ) as mock_calc:
            await calculate_event_cost(context, MagicMock())
            mock_calc.assert_awaited_once()
            assert mock_calc.await_args.kwargs["completion_text"] == "hi there"
        assert context.cost_usd == 0.003

    async def test_openresponses_estimates_from_input_and_output(self):
        """OpenResponses: prompt tokens come from the ``input`` items and
        completion tokens from the ``output`` message items."""
        context = _context()
        context.request_body = {
            "model": "gpt-5.2",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello world"}]}],
        }
        context.response_body = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi there"}],
                }
            ]
        }
        with patch(
            "llm_proxy.observability.cost.calculate_cost",
            new=AsyncMock(return_value=CostBreakdown(cost_usd=0.004)),
        ) as mock_calc:
            await calculate_event_cost(context, MagicMock())
            mock_calc.assert_awaited_once()
            assert mock_calc.await_args.kwargs["messages"] == [
                {"role": "user", "content": [{"type": "input_text", "text": "hello world"}]}
            ]
            assert mock_calc.await_args.kwargs["completion_text"] == "hi there"
        assert context.cost_usd == 0.004

    async def test_openresponses_string_input_wrapped_as_message(self):
        """OpenResponses: a plain string ``input`` is wrapped into a single
        user message for prompt-token estimation."""
        context = _context()
        context.request_body = {"model": "gpt-5.2", "input": "hello world"}
        with patch(
            "llm_proxy.observability.cost.calculate_cost",
            new=AsyncMock(return_value=CostBreakdown(cost_usd=0.005)),
        ) as mock_calc:
            await calculate_event_cost(context, MagicMock())
            mock_calc.assert_awaited_once()
            assert mock_calc.await_args.kwargs["messages"] == [
                {"role": "user", "content": "hello world"}
            ]
        assert context.cost_usd == 0.005

    async def test_no_messages_no_usage_still_skips(self):
        """Non-chat requests (e.g. embeddings) are never estimated: their
        ``input`` field is the text to embed, not a conversation."""
        from llm_proxy.core.request_type import RequestType

        context = _context(request_type=RequestType.EMBEDDING)
        context.request_body = {"model": "gpt-4", "input": "text-embedding request"}
        with patch("llm_proxy.observability.cost.calculate_cost", new=AsyncMock()) as mock_calc:
            await calculate_event_cost(context, config_manager=MagicMock())
            mock_calc.assert_not_called()
        assert context.cost_usd is None
