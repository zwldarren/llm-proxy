"""Rerank processing strategy."""

from typing import Any

from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.strategies.base import ProcessingStrategy
from llm_proxy.core.request_type import RequestType


class RerankStrategy(ProcessingStrategy):
    """Strategy for rerank requests.

    Rerank is synchronous and non-streaming: one query is scored against a
    fixed document list, so there is no streaming marker branch here.
    """

    request_type = RequestType.RERANK
    trace_name = "llm-proxy-rerank-request"

    async def execute(
        self, unified_request: Any, adapter: BaseAdapter, context: RequestContext
    ) -> Any:
        return await adapter.rerank(unified_request)


__all__ = ["RerankStrategy"]
