"""Embedding processing strategy."""

from typing import Any

from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.strategies.base import ProcessingStrategy
from llm_proxy.core.request_type import RequestType


class EmbeddingStrategy(ProcessingStrategy):
    """Strategy for embedding requests."""

    request_type = RequestType.EMBEDDING
    trace_name = "llm-proxy-embedding-request"

    async def execute(
        self, unified_request: Any, adapter: BaseAdapter, context: RequestContext
    ) -> Any:
        return await adapter.embeddings(unified_request)
