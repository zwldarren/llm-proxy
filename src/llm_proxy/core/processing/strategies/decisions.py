"""Decisions evaluation processing strategy."""

from typing import Any

from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.strategies.base import ProcessingStrategy
from llm_proxy.core.request_type import RequestType


class DecisionsStrategy(ProcessingStrategy):
    """Strategy for Decisions evaluation requests (OpenAI's /v1/decisions).

    Decisions is synchronous and non-streaming: the request is a single
    evaluation, so there is no streaming marker branch here.
    """

    request_type = RequestType.DECISIONS
    trace_name = "llm-proxy-decisions-request"

    async def execute(
        self, unified_request: Any, adapter: BaseAdapter, context: RequestContext
    ) -> Any:
        return await adapter.decisions(unified_request)


__all__ = ["DecisionsStrategy"]
