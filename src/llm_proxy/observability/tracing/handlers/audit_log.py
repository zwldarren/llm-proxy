"""Audit log handler for database logging via the unified capture layer.

This handler connects the pipeline's lifecycle events to the Log intake module
(ADR-0020). It owns only the capture that needs the live stream — arming the
``EventContext`` for streaming and buffering raw chunks when the operator asked
for raw capture — and hands finished situations to the intake verbs, which own
classification, masking, dispatch and the usage-record fan-out.
"""

from typing import TYPE_CHECKING

from llm_proxy.observability.log_intake import (
    record_request_end,
    record_request_error,
    record_stream_end,
)
from llm_proxy.observability.tracing.handlers.base import TracingHandler

if TYPE_CHECKING:
    from llm_proxy.models import InternalRequest, InternalResponse
    from llm_proxy.observability.event_context import EventContext


class AuditLogHandler(TracingHandler):
    """Writes request logs to the database by driving the Log intake module.

    Respects the sampling decision recorded on the ``EventContext``; usage
    records are always written, independently of sampling.
    """

    provider_name = "audit_log"

    async def on_request_end(
        self,
        request: InternalRequest,
        response: InternalResponse,
        context: EventContext,
    ) -> None:
        """Record a completed non-streaming request."""
        if not self._enabled:
            return
        await record_request_end(request, response, context)

    async def on_error(
        self,
        request: InternalRequest,
        error: Exception,
        context: EventContext,
    ) -> None:
        """Record a failed non-streaming request.

        A streaming error is recorded by :meth:`on_stream_end`, which owns the
        stream's accumulated data.
        """
        if not self._enabled:
            return
        if context.is_streaming:
            return
        await record_request_error(request, error, context)

    async def on_stream_start(
        self,
        request: InternalRequest,
        context: EventContext,
    ) -> None:
        """Mark the context as streaming."""
        if not self._enabled:
            return
        context.is_streaming = True

    async def on_stream_chunk(
        self,
        request: InternalRequest,
        chunk: str,
        context: EventContext,
    ) -> None:
        """Buffer a chunk when sampling allows raw capture."""
        if not self._enabled:
            return
        context.capture_streaming_chunk(chunk)

    async def on_stream_end(
        self,
        request: InternalRequest,
        context: EventContext,
        error: Exception | None = None,
    ) -> None:
        """Record a finished stream (success or error)."""
        if not self._enabled:
            return
        await record_stream_end(request, context, error)
