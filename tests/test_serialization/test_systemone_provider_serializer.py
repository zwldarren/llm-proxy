"""Tests for the System One provider-serializer defaults.

Both TypeSafe and OpenRouter use the shared defaults on ``ProviderSerializer``;
``OpenAIProviderSerializer`` is the concrete stand-in here because it adds no
System One override of its own.
"""

from typing import Any

from llm_proxy.models.systemone import InternalSystemOneRequest
from llm_proxy.models.types import Usage
from llm_proxy.serialization.providers.chat_completions import OpenAIProviderSerializer
from llm_proxy.serialization.providers.field_utils import reported_cost

QUESTION = {"type": "noul", "instructions": "Does this convey urgency?"}


def _request(**extra: Any) -> InternalSystemOneRequest:
    return InternalSystemOneRequest(
        model="jev-latest",
        state="Help! My payouts have been failing for 3 days.",
        questions={"is_urgent": QUESTION},
        extra=dict(extra),
    )


class TestReportedCost:
    def test_reads_positive_numbers(self):
        assert reported_cost({"cost": 0.00002}) == 0.00002
        assert reported_cost({"cost": 2}) == 2.0

    def test_rejects_bool_zero_negative_string_and_non_dict(self):
        # ``bool`` is an ``int`` subclass: ``cost: true`` must not bill 1.0 USD.
        assert reported_cost({"cost": True}) is None
        assert reported_cost({"cost": False}) is None
        assert reported_cost({"cost": 0}) is None
        assert reported_cost({"cost": -1}) is None
        assert reported_cost({"cost": "0.5"}) is None
        assert reported_cost({}) is None
        assert reported_cost(None) is None


class TestSystemOneDefaults:
    def test_builds_the_whole_body(self):
        body = OpenAIProviderSerializer().build_provider_systemone_request(_request())
        assert body == {
            "model": "jev-latest",
            "state": "Help! My payouts have been failing for 3 days.",
            "questions": {"is_urgent": QUESTION},
        }

    def test_parses_usage_and_echoes_the_raw_block(self):
        raw_usage = {"input_tokens": 476, "output_tokens": 70, "cost": 0.00002}
        result = OpenAIProviderSerializer().parse_provider_systemone_response(
            {
                "model": "typesafe/jev-1.13-20260917",
                "answers": {"is_urgent": {"type": "noul", "noul": 1.0}},
                "usage": raw_usage,
            },
            model="jev-latest",
        )
        assert result.model == "typesafe/jev-1.13-20260917"
        assert result.usage == Usage(input_tokens=476, output_tokens=70)
        assert result.provider_info["systemone_usage"] == raw_usage
        assert result.provider_info["openrouter_cost"] == 0.00002

    def test_missing_usage_parses_as_none(self):
        result = OpenAIProviderSerializer().parse_provider_systemone_response(
            {"model": "jev-1.13.0", "answers": {}}, model="jev-latest"
        )
        assert result.usage is None
        assert "systemone_usage" not in result.provider_info
        assert "openrouter_cost" not in result.provider_info
