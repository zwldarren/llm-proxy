"""The service tier a request asks for is captured for billing.

Billing prices a turn from the ``service_tier_pricing`` entry its tier names, so
the requested tier has to ride along from the parsed request until the cost is
computed. Both dialects that carry the field express it as a per-protocol
parameter block.
"""

from llm_proxy.core.processing.unified import _requested_service_tier
from llm_proxy.models import InternalRequest
from llm_proxy.models.conversation import ConversationContext
from llm_proxy.models.params import (
    AnthropicSpecificParams,
    GenerationParams,
    OpenAISpecificParams,
)


def _request(params: GenerationParams | None = None) -> InternalRequest:
    return InternalRequest(
        model="m",
        conversation=ConversationContext(),
        params=params or GenerationParams(),
    )


def test_no_params_reports_none():
    assert _requested_service_tier(_request()) is None


def test_openai_service_tier_is_captured():
    request = _request(GenerationParams(openai=OpenAISpecificParams(service_tier="flex")))

    assert _requested_service_tier(request) == "flex"


def test_anthropic_service_tier_is_captured():
    request = _request(
        GenerationParams(anthropic=AnthropicSpecificParams(service_tier="standard_only"))
    )

    assert _requested_service_tier(request) == "standard_only"


def test_blank_tier_is_dropped():
    request = _request(GenerationParams(openai=OpenAISpecificParams(service_tier="  ")))

    assert _requested_service_tier(request) is None
