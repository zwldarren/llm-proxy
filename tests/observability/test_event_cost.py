"""Tests for the shared event-cost finalization helper."""

from unittest.mock import AsyncMock, MagicMock, patch

from llm_proxy.billing.cost import CostBreakdown
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
        context = _context(cost_usd=0.5, prompt_tokens=10, completion_tokens=5)
        await calculate_event_cost(context, config_manager=MagicMock())
        # Unchanged — no re-calculation.
        assert context.cost_usd == 0.5

    async def test_provider_reported_cost_takes_precedence(self):
        context = _context(provider_reported_cost=0.42, prompt_tokens=10, completion_tokens=5)
        with patch("llm_proxy.observability.cost.calculate_cost", new=AsyncMock()) as mock_calc:
            await calculate_event_cost(context, config_manager=MagicMock())
            mock_calc.assert_not_called()
        assert context.cost_usd == 0.42

    async def test_calculates_from_pricing_db(self):
        context = _context(prompt_tokens=10, completion_tokens=5, total_tokens=15)
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
        context = _context(prompt_tokens=10, completion_tokens=5)
        with patch("llm_proxy.observability.cost.calculate_cost", new=AsyncMock()) as mock_calc:
            await calculate_event_cost(context, config_manager=None)
            mock_calc.assert_not_called()
        assert context.cost_usd is None


class TestExtractStreamingUsage:
    async def test_noop_without_transformer(self):
        context = _context()
        await extract_streaming_usage(context)
        assert context.prompt_tokens is None

    async def test_pulls_usage_from_transformer(self):
        usage = MagicMock()
        usage.input_tokens = 7
        usage.output_tokens = 3
        usage.total_tokens = 10
        usage.cache_read_input_tokens = None
        usage.cache_creation_input_tokens = None

        transformer = MagicMock()
        transformer.get_usage.return_value = usage

        context = _context()
        context.transformer = transformer
        await extract_streaming_usage(context)
        assert context.prompt_tokens == 7
        assert context.completion_tokens == 3

    async def test_ignores_transformer_without_get_usage(self):
        context = _context()
        context.transformer = MagicMock(spec=[])  # no get_usage attribute
        await extract_streaming_usage(context)
        assert context.prompt_tokens is None


class TestFinalizeEventCost:
    async def test_extracts_usage_then_calculates_cost(self):
        usage = MagicMock()
        usage.input_tokens = 7
        usage.output_tokens = 3
        usage.total_tokens = 10
        usage.cache_read_input_tokens = None
        usage.cache_creation_input_tokens = None

        transformer = MagicMock()
        transformer.get_usage.return_value = usage

        context = _context()
        context.transformer = transformer

        with patch(
            "llm_proxy.observability.cost.calculate_cost",
            new=AsyncMock(return_value=CostBreakdown(cost_usd=0.02)),
        ):
            await finalize_event_cost(context, config_manager=MagicMock())

        assert context.prompt_tokens == 7
        assert context.cost_usd == 0.02


class TestUpdateUsageCacheWrite:
    def test_cache_write_tokens_map_to_cache_creation(self):
        """OpenAI-style prompt_tokens_details.cache_write_tokens is billed at
        the cache-write rate, same as Anthropic cache_creation_input_tokens.
        The mapping happens at the billing seam (extract_tokens_from_usage),
        not in update_usage (ADR-0006: tolerance in exactly one place)."""
        from llm_proxy.billing.tokens import extract_tokens_from_usage
        from llm_proxy.models.types import PromptTokensDetails, Usage

        context = _context()
        context.update_usage(
            Usage(
                input_tokens=100,
                output_tokens=10,
                prompt_tokens_details=PromptTokensDetails(cache_write_tokens=40),
            )
        )
        # Dialect value carried as-is; the flat Anthropic field stays unset.
        assert context.cache_write_tokens == 40
        assert context.cache_creation_input_tokens is None
        # Billing seam maps nested cache_write_tokens to cache_creation_input_tokens.
        token_usage = extract_tokens_from_usage(context.to_usage_dict())
        assert token_usage.cache_creation_input_tokens == 40

    def test_flat_cache_creation_wins_over_details(self):
        """Anthropic flat cache_creation_input_tokens takes precedence over
        the details-level fallback."""
        from llm_proxy.billing.tokens import extract_tokens_from_usage
        from llm_proxy.models.types import PromptTokensDetails, Usage

        context = _context()
        context.update_usage(
            Usage(
                input_tokens=100,
                output_tokens=10,
                cache_creation_input_tokens=7,
                prompt_tokens_details=PromptTokensDetails(cache_write_tokens=40),
            )
        )
        token_usage = extract_tokens_from_usage(context.to_usage_dict())
        assert token_usage.cache_creation_input_tokens == 7


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
