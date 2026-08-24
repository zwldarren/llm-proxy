"""Image generation and edit processing strategies."""

from typing import Any

from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.strategies.base import (
    ProcessingStrategy,
    StreamingResponseMarker,
)
from llm_proxy.core.request_type import RequestType


class ImageStrategy(ProcessingStrategy):
    """Strategy for image generation requests."""

    request_type = RequestType.IMAGE_GENERATION
    trace_name = "llm-proxy-image-request"

    async def execute(
        self, unified_request: Any, adapter: BaseAdapter, context: RequestContext
    ) -> Any:
        if unified_request.stream:
            return StreamingResponseMarker(unified_request, adapter)
        return await adapter.image_generation(unified_request)


class ImageEditStrategy(ProcessingStrategy):
    """Strategy for image edit requests."""

    request_type = RequestType.IMAGE_EDIT
    trace_name = "llm-proxy-image-edit-request"

    async def execute(
        self, unified_request: Any, adapter: BaseAdapter, context: RequestContext
    ) -> Any:
        if unified_request.stream:
            return StreamingResponseMarker(unified_request, adapter)
        return await adapter.image_edit(unified_request)
