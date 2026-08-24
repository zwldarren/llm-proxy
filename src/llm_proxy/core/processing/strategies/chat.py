"""Chat completion processing strategy."""

from typing import Any

from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.strategies.base import (
    ProcessingStrategy,
    StreamingResponseMarker,
)
from llm_proxy.core.request_type import RequestType


class ChatStrategy(ProcessingStrategy):
    """Strategy for chat completion requests."""

    request_type = RequestType.CHAT
    trace_name = "llm-proxy-request"

    async def execute(
        self, unified_request: Any, adapter: BaseAdapter, context: RequestContext
    ) -> Any:
        if unified_request.stream:
            return StreamingResponseMarker(unified_request, adapter)
        return await adapter.chat_completion(unified_request)
