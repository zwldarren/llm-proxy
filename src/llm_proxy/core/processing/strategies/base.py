"""Base processing strategy and the streaming-response marker."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_proxy.protocols.serializer_base import ProtocolSerializer

import orjson
from fastapi import Response

from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.processing.base import RequestContext


class StreamingResponseMarker:
    """Marker to indicate a streaming response is needed.

    Returned by ChatStrategy when request has stream=True.
    Handled specially by UnifiedProcessor.
    """

    def __init__(self, request: Any, adapter: BaseAdapter):
        self.request = request
        self.adapter = adapter


class ProcessingStrategy:
    """Base class for request processing strategies.

    Subclasses override request_type, trace_name, and execute().
    format_response() uses ProtocolSerializer directly.
    """

    async def execute(
        self, unified_request: Any, adapter: BaseAdapter, context: RequestContext
    ) -> Any:
        raise NotImplementedError

    async def format_response(
        self, response: Any, serializer: ProtocolSerializer, protocol_name: str
    ) -> Response:
        # Native passthrough: the adapter stashed the raw upstream body on the
        # response; emit it verbatim instead of re-serializing parsed blocks.
        raw_body = getattr(response, "provider_info", {}).get("_raw_response_body")
        if raw_body is not None:
            return Response(content=orjson.dumps(raw_body), media_type="application/json")
        # Use serializer directly for response formatting
        response_dict = serializer.format_response(response)
        return Response(content=orjson.dumps(response_dict), media_type="application/json")
