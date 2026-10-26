"""The lifecycle adapter every exporter-backed tracing backend shares.

ADR-0044 keeps the Langfuse-style architecture for backends that need a live
event: a span must be open before the response headers are written (the trace id
they echo), TTFT must be recorded while the stream runs, and a per-request
content render is only available from the typed request/response. The handler is
therefore a thin adapter over an exporter object, and every backend's adapter is
the same sequence of calls:

    on_request_start -> open
    on_request_end   -> finish
    on_error         -> fail
    on_stream_*      -> stream_opened / stream_progress / finish | fail

The exporter (``observability/sinks/``) owns every SDK call and the payload
shaping. Subclasses only declare their provider metadata and build their
exporter in ``create_handler``.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from llm_proxy.observability.observation.collector import capture_request_end, collect
from llm_proxy.observability.tracing.handlers.base import TracingHandler

if TYPE_CHECKING:
    from llm_proxy.observability.event_context import EventContext

#: How long a released backend stays alive so requests already in flight can end
#: the span they opened. After it, the provider and its export client are closed.
RELEASE_GRACE_SECONDS = 300.0


class ExporterBackedHandler(TracingHandler):
    """Tracing handler whose backend lives in an exporter object.

    Subclasses set :attr:`_exporter` to an object implementing ``open``,
    ``finish``, ``fail``, ``stream_opened``, ``needs_ttft``, ``stream_progress``,
    ``get_trace_id``, ``get_observation_id``, ``flush`` and ``close`` — see
    ``observability/sinks/langfuse.py`` and ``observability/sinks/otlp.py``.
    """

    _exporter: Any

    def __init__(self, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._pending_closes: set[asyncio.Task[None]] = set()

    async def on_request_start(self, request: Any, context: EventContext) -> None:
        if not self._enabled:
            return
        self._exporter.open(collect(context), request)

    async def on_request_end(
        self,
        request: Any,
        response: Any,
        context: EventContext,
    ) -> None:
        # Finalize the capture (usage, status, cost) so this exporter sees the
        # same facts the log row will; the intake verb repeats it idempotently.
        record = await capture_request_end(context, response=response)
        self._exporter.finish(record, request=request, response=response)

    async def on_error(
        self,
        request: Any,
        error: Exception,
        context: EventContext,
    ) -> None:
        record = await capture_request_end(context, error=error)
        self._exporter.fail(record, error, request=request)

    async def on_stream_start(self, request: Any, context: EventContext) -> None:
        self._exporter.stream_opened(collect(context))

    async def on_stream_chunk(
        self,
        request: Any,
        chunk: str,
        context: EventContext,
    ) -> None:
        # The snapshot is taken at most once per stream: ``needs_ttft`` is false
        # as soon as the first-token update landed, so the per-chunk hook pays
        # only a dict lookup after that.
        if not self._exporter.needs_ttft(context.request_id):
            return
        self._exporter.stream_progress(collect(context))

    async def on_stream_end(
        self,
        request: Any,
        context: EventContext,
        error: Exception | None = None,
    ) -> None:
        # Streaming folds the transformer's usage here, before the audit handler
        # runs: user handlers are registered ahead of the system ones, and the
        # export must not miss the final usage report.
        record = await capture_request_end(context, error=error, streaming=True)
        if error is not None:
            self._exporter.fail(record, error, request=request, streaming=True)
        else:
            self._exporter.finish(record, request=request)

    # -- trace identity ---------------------------------------------------

    def get_trace_id(self) -> str | None:
        return self._exporter.get_trace_id()

    def get_observation_id(self) -> str | None:
        return self._exporter.get_observation_id()

    def get_trace_header_name(self) -> str:
        return "x-trace-id"

    # -- lifetime ---------------------------------------------------------

    async def release(self) -> None:
        """Flush now, then tear the backend down once in-flight requests drain.

        Called when the owning registry is dropped (the user changed their
        tracing config, or a peer worker did). Requests already in flight still
        hold this handler and must be able to end the span they opened, so the
        close is deferred by :data:`RELEASE_GRACE_SECONDS` instead of running
        here; without it every config change would strand the provider's
        batching thread and export client for the process lifetime.
        """
        await self._exporter.flush()
        asyncio.get_running_loop().call_later(RELEASE_GRACE_SECONDS, self._close_after_release)

    def _close_after_release(self) -> None:
        task = asyncio.ensure_future(self._exporter.close())
        self._pending_closes.add(task)
        task.add_done_callback(self._pending_closes.discard)

    async def shutdown(self) -> None:
        self._enabled = False
        await self._exporter.close()


__all__ = ["ExporterBackedHandler"]
