"""Langfuse handler lifecycle across the streaming task boundary.

The streaming lifecycle splits a request across asyncio tasks: ``on_request_start``
runs inside the pipeline task (``keepalive.await_with_disconnect_monitor`` wraps the
pipeline in ``ensure_future``), while ``on_stream_chunk``/``on_stream_end`` run in
the task Starlette spawned to consume the response body (see
``StreamingResponse.__call__`` under ASGI spec < 2.4). Handler state stored in a
``ContextVar`` set by the pipeline task is invisible to that other task, so the
generation was never ended and the trace was never exported (regression: all
streamed requests silently vanished from Langfuse). State must be reachable
across those tasks.
"""

import asyncio

import pytest

from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.tracing.handlers.providers.langfuse.handler import (
    LangfuseTracingHandler,
)


class _FakeGeneration:
    """Captures update/end calls instead of talking to the Langfuse OTLP layer."""

    def __init__(self, trace_id: str = "t" * 32):
        self.trace_id = trace_id
        self.id = "obs-1"
        self.updates: list[dict] = []
        self.ended = False

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended = True

    def start_observation(self, **kwargs) -> _FakeGeneration:
        child = _FakeGeneration()
        child.ended = True
        return child


class _FakeClient:
    def __init__(self) -> None:
        self.generation = _FakeGeneration()

    def start_as_current_observation(self, **kwargs):  # noqa: ANN201 - context manager
        holder = self

        class _CM:
            def __init__(self) -> None:
                self.generation = _FakeGeneration()
                holder.generation = self.generation

            def __enter__(self):  # noqa: ANN001, ANN202
                return self.generation

            def __exit__(self, *args: object) -> bool:
                return False  # end_on_exit=False: stays open until .end()

        return _CM()


def _handler() -> tuple[LangfuseTracingHandler, _FakeClient]:
    client = _FakeClient()
    handler = LangfuseTracingHandler(
        enabled=True,
        name="langfuse",
        client=client,
        base_url="https://cloud.langfuse.com",
    )
    return handler, client


@pytest.mark.asyncio
async def test_stream_end_in_another_task_still_ends_the_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce the production task boundary: end hooks run in a fresh task."""

    # context.start_as_current_observation is created under a context the
    # pipeline never sees otherwise; propagate_attributes touches OTel context,
    # which behaves identically with no active span.
    monkeypatch.setattr(
        "llm_proxy.observability.sinks.langfuse.propagate_attributes",
        lambda **kwargs: _noop_cm(),
    )

    handler, client = _handler()
    request = object()
    context = EventContext(
        request_id="rid-cross-task",
        trace_id="a" * 32,
        model="mock-model",
        metadata={"endpoint": "/v1/chat/completions"},
        is_streaming=True,
    )

    # on_request_start: pipeline task
    await handler.on_request_start(request, context)
    assert client.generation is not None
    assert handler._exporter._request_states.get("rid-cross-task") is not None

    # on_stream_end: a task that never saw the ContextVar write — the exact
    # shape of the Starlette-streamed-response failure mode.
    async def _remote_end() -> None:
        await handler.on_stream_end(request, context)

    task = asyncio.ensure_future(_remote_end())
    await task

    assert client.generation.ended, "generation must be ended from the stream task"
    assert handler._exporter._request_states.get("rid-cross-task") is None


def _noop_cm():
    from contextlib import nullcontext

    return nullcontext()


@pytest.mark.asyncio
async def test_error_end_in_another_task_still_ends_the_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llm_proxy.observability.sinks.langfuse.propagate_attributes",
        lambda **kwargs: _noop_cm(),
    )
    handler, client = _handler()
    request = object()
    context = EventContext(
        request_id="rid-error",
        trace_id="b" * 32,
        model="mock-model",
        metadata={"endpoint": "/v1/chat/completions"},
    )
    await handler.on_request_start(request, context)

    await asyncio.ensure_future(handler.on_error(request, RuntimeError("boom"), context))
    assert client.generation.ended
