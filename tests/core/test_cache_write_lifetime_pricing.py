"""Cache writes priced by cache lifetime (ADR-0066).

Upstreams that let a request ask for a longer-lived prompt cache charge more for
the write: OpenAI bills a default 30-minute write at 1.25x the input rate and a
12-hour one at 2.5x. These tests pin the three places that has to hold: the wire
count is read and carried as a nested detail of the write total, billing prices
the two parts apart, and the client gets the split back.
"""

import pytest

from llm_proxy.billing.cost import PricingRates, _calculate_cache_cost
from llm_proxy.billing.tokens import TokenUsage, token_usage_from
from llm_proxy.models.types import PromptTokensDetails, Usage
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer
from llm_proxy.protocols.openresponses.usage_payload import build_usage_payload
from llm_proxy.serialization.usage import parse_usage

# $1/M input makes every expectation readable in micro-dollars.
INPUT = 1.0
SHORT_WRITE = 1.25
LONG_WRITE = 2.5


def _write_usage(**overrides: object) -> TokenUsage:
    values: dict[str, object] = {
        "prompt_tokens": 2000,
        "cache_creation_input_tokens": 1000,
    }
    values.update(overrides)
    return TokenUsage(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Wire parsing
# ---------------------------------------------------------------------------


class TestParsingTheLifetimeSplit:
    def test_chat_reads_the_long_write_count(self) -> None:
        usage = parse_usage(
            "chat",
            {
                "usage": {
                    "prompt_tokens": 12000,
                    "completion_tokens": 10,
                    "total_tokens": 12010,
                    "prompt_tokens_details": {
                        "cached_tokens": 4000,
                        "cache_write_tokens": 7000,
                        "cache_write_tokens_details": {
                            "ttl_30m_tokens": 3000,
                            "ttl_12h_tokens": 4000,
                        },
                    },
                }
            },
        )

        assert usage is not None
        assert usage.cache_read_input_tokens == 4000
        # The write total stays one flat number; the lifetime split is nested.
        assert usage.cache_creation_input_tokens == 7000
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cache_write_long_tokens == 4000

    def test_responses_reads_the_long_write_count(self) -> None:
        usage = parse_usage(
            "responses",
            {
                "usage": {
                    "input_tokens": 9000,
                    "output_tokens": 5,
                    "input_tokens_details": {
                        "cached_tokens": 1000,
                        "cache_write_tokens": 5000,
                        "cache_write_tokens_details": {"ttl_12h_tokens": 5000},
                    },
                }
            },
        )

        assert usage is not None
        assert usage.cache_creation_input_tokens == 5000
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cache_write_long_tokens == 5000

    def test_no_split_block_means_no_long_writes(self) -> None:
        usage = parse_usage(
            "chat",
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cache_write_tokens": 10},
                }
            },
        )

        assert usage is not None
        assert usage.cache_creation_input_tokens == 10
        assert usage.prompt_tokens_details is None

    def test_a_long_count_without_a_write_total_is_dropped(self) -> None:
        """Self-contradicting wire data must not price writes that never happened."""
        usage = parse_usage(
            "chat",
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {
                        "cache_write_tokens_details": {"ttl_12h_tokens": 40},
                    },
                }
            },
        )

        assert usage is not None
        assert usage.cache_creation_input_tokens is None
        assert usage.prompt_tokens_details is None

    def test_an_unusable_long_count_is_cleared_from_the_details(self) -> None:
        """A reported-zero write clears the long count rather than keeping it.

        The flat ``0`` is itself a measurement ("this turn wrote nothing"), so it
        survives; what cannot survive is a long count sitting under a zero write.
        """
        usage = parse_usage(
            "chat",
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "cache_creation_input_tokens": 0,
                    "prompt_tokens_details": {
                        "cache_write_tokens": 0,
                        "cache_write_tokens_details": {"ttl_12h_tokens": 40},
                    },
                }
            },
        )

        assert usage is not None
        assert usage.cache_creation_input_tokens == 0
        assert usage.prompt_tokens_details is None

    def test_a_long_count_larger_than_the_write_is_clamped(self) -> None:
        usage = parse_usage(
            "chat",
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {
                        "cache_write_tokens": 10,
                        "cache_write_tokens_details": {"ttl_12h_tokens": 99},
                    },
                }
            },
        )

        assert usage is not None
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cache_write_long_tokens == 10


# ---------------------------------------------------------------------------
# The billing read model
# ---------------------------------------------------------------------------


class TestBillingReadModel:
    def test_long_writes_reach_the_read_model(self) -> None:
        usage = Usage(
            input_tokens=1000,
            cache_creation_input_tokens=600,
            prompt_tokens_details=PromptTokensDetails(cache_write_long_tokens=400),
        )

        token_usage = token_usage_from(usage)

        assert token_usage.cache_creation_input_tokens == 600
        assert token_usage.cache_write_long_tokens == 400

    def test_long_writes_are_clamped_to_the_write_total(self) -> None:
        usage = Usage(
            input_tokens=1000,
            cache_creation_input_tokens=300,
            prompt_tokens_details=PromptTokensDetails(cache_write_long_tokens=900),
        )

        assert token_usage_from(usage).cache_write_long_tokens == 300

    def test_no_details_block_means_no_long_writes(self) -> None:
        usage = Usage(input_tokens=1000, cache_creation_input_tokens=300)

        assert token_usage_from(usage).cache_write_long_tokens == 0


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class TestPricingTheSplit:
    def test_the_two_parts_bill_at_their_own_rates(self) -> None:
        adjustment, _ = _calculate_cache_cost(
            _write_usage(cache_write_long_tokens=400),
            PricingRates(
                input_cost_per_1m=INPUT,
                cached_write_cost_per_1m=SHORT_WRITE,
                cached_write_long_cost_per_1m=LONG_WRITE,
            ),
        )

        # 600 short writes at 1.25 plus 400 long writes at 2.5, against 1000 at 1.
        expected = (600 / 1_000_000) * SHORT_WRITE + (400 / 1_000_000) * LONG_WRITE
        expected -= (1000 / 1_000_000) * INPUT
        assert abs(adjustment - expected) < 1e-12

    def test_without_a_long_rate_every_write_bills_at_the_short_rate(self) -> None:
        adjustment, _ = _calculate_cache_cost(
            _write_usage(cache_write_long_tokens=400),
            PricingRates(input_cost_per_1m=INPUT, cached_write_cost_per_1m=SHORT_WRITE),
        )

        # The split is not priced apart, so the long count is only a detail.
        expected = (1000 / 1_000_000) * SHORT_WRITE - (1000 / 1_000_000) * INPUT
        assert abs(adjustment - expected) < 1e-12

    def test_only_a_long_rate_configured_bills_every_write_there(self) -> None:
        adjustment, _ = _calculate_cache_cost(
            _write_usage(),
            PricingRates(input_cost_per_1m=INPUT, cached_write_long_cost_per_1m=LONG_WRITE),
        )

        expected = (1000 / 1_000_000) * LONG_WRITE - (1000 / 1_000_000) * INPUT
        assert abs(adjustment - expected) < 1e-12

    def test_no_rate_at_all_leaves_the_tokens_at_the_input_rate(self) -> None:
        adjustment, savings = _calculate_cache_cost(
            _write_usage(cache_write_long_tokens=400),
            PricingRates(input_cost_per_1m=INPUT),
        )

        assert adjustment == 0.0
        assert savings == 0.0

    def test_cache_savings_nets_out_the_write_premium(self) -> None:
        """A write above the input rate costs money, so the figure goes negative."""
        _, savings = _calculate_cache_cost(
            _write_usage(),
            PricingRates(
                input_cost_per_1m=INPUT,
                cached_write_cost_per_1m=SHORT_WRITE,
                cached_write_long_cost_per_1m=LONG_WRITE,
            ),
        )

        premium = (1000 / 1_000_000) * (SHORT_WRITE - INPUT)
        assert abs(savings + premium) < 1e-12

    def test_a_read_that_outweighs_the_write_stays_positive(self) -> None:
        _, savings = _calculate_cache_cost(
            _write_usage(
                prompt_tokens=20000,
                cache_creation_input_tokens=1000,
                cache_read_input_tokens=18000,
            ),
            PricingRates(
                input_cost_per_1m=INPUT,
                cached_read_cost_per_1m=0.1,
                cached_write_cost_per_1m=SHORT_WRITE,
            ),
        )

        read_saving = (18000 / 1_000_000) * (INPUT - 0.1)
        write_premium = (1000 / 1_000_000) * (SHORT_WRITE - INPUT)
        assert abs(savings - (read_saving - write_premium)) < 1e-12
        assert savings > 0


# ---------------------------------------------------------------------------
# What the client reads back
# ---------------------------------------------------------------------------


class TestClientEmission:
    def test_openresponses_builder_emits_the_split(self) -> None:
        payload = build_usage_payload(
            input_tokens=12000,
            output_tokens=10,
            cached_tokens=4000,
            cache_write_tokens=7000,
            cache_write_long_tokens=4000,
        )

        details = payload["input_tokens_details"]
        assert details["cache_write_tokens"] == 7000
        assert details["cache_write_tokens_details"] == {"ttl_12h_tokens": 4000}

    def test_openresponses_builder_omits_an_empty_split(self) -> None:
        payload = build_usage_payload(input_tokens=10, output_tokens=1, cache_write_tokens=5)

        assert "cache_write_tokens_details" not in payload["input_tokens_details"]


@pytest.mark.parametrize(
    ("long_tokens", "expected"),
    [(4000, {"ttl_12h_tokens": 4000}), (None, None)],
)
def test_chat_emission_carries_the_split(long_tokens: int | None, expected: dict | None) -> None:
    from llm_proxy.models import InternalResponse, TextBlock

    response = InternalResponse(
        id="chatcmpl-1",
        model="gpt-6.2-sol",
        output=[TextBlock(text="hi")],
        usage=Usage(
            input_tokens=12000,
            output_tokens=10,
            cache_creation_input_tokens=7000,
            prompt_tokens_details=PromptTokensDetails(cache_write_long_tokens=long_tokens),
        ),
    )

    body = OpenAIProtocolSerializer().format_response(response)

    details = body["usage"]["prompt_tokens_details"]
    assert details["cache_write_tokens"] == 7000
    assert details.get("cache_write_tokens_details") == expected
