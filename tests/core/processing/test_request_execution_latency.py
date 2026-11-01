"""Regression tests for provider-latency observations during execution."""

from unittest.mock import AsyncMock, MagicMock

from llm_proxy.core.processing.stages.request_execution import RetryExecutor
from llm_proxy.core.processing.strategies import StreamingResponseMarker


async def _execute(streaming: bool):
    result = StreamingResponseMarker(MagicMock(), MagicMock()) if streaming else MagicMock()
    state = MagicMock()
    state.strategy.execute = AsyncMock(return_value=result)
    state.event_context.latency_ms = 12.0
    context = MagicMock()
    context.web_search_interceptor = None
    executor = RetryExecutor(param_override_service=MagicMock())

    actual, error = await executor.execute(state, context)

    assert actual is result
    assert error is None
    return context


async def test_non_streaming_execution_records_total_latency():
    context = await _execute(streaming=False)

    context.orchestrator.record_last_success.assert_called_once_with(12.0)


async def test_streaming_marker_does_not_record_total_latency():
    context = await _execute(streaming=True)

    context.orchestrator.record_last_success.assert_called_once_with(None)
