"""Behavior of the Log intake verbs (ADR-0020), through the interface.

One verb per situation, one place that assembles the row and fans out the
records. These tests drive the endpoint-lifecycle verbs with a real
``EventContext`` and assert on what reaches the background writers — the seam a
caller crosses — rather than on the module's internal row builders.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.models.types import Usage
from llm_proxy.observability import log_intake, store
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.log_intake import (
    record_request_end,
    record_request_error,
    record_stream_end,
)


def _context(**overrides) -> EventContext:
    fields: dict = {
        "request_id": "req-1",
        "trace_id": "trace-1",
        "model": "fast",
        "provider": "openai",
        "log_type": "endpoint",
        "is_api_endpoint": True,
        "metadata": {"endpoint": "/v1/chat/completions", "method": "POST"},
        "usage": Usage(input_tokens=100, output_tokens=20, total_tokens=120),
        "cost_usd": 0.002,
    }
    fields.update(overrides)
    return EventContext(**fields)


class _Capture:
    """Captures the log rows and usage records the verbs dispatch."""

    def __init__(self) -> None:
        self.logs: list = []
        self.usage: list = []
        logs, usage = self.logs, self.usage

        class _LogService:
            def __init__(self, config):  # noqa: ARG002
                pass

            def create_log_background(self, data):
                logs.append(data)

        class _UsageService:
            def create_usage_background(self, data):
                usage.append(data)

        self._log_service_cls = _LogService
        self._usage_service_cls = _UsageService

    def drive(self, coro) -> None:
        """Run one verb coroutine with the writers replaced by this capture."""
        log_intake.configure(config=LoggingConfig(log_input_output=True))
        with (
            patch.object(store, "RequestLogService", self._log_service_cls),
            patch.object(store, "UsageService", self._usage_service_cls),
        ):
            asyncio.run(coro)


def test_request_end_writes_one_classified_row_and_one_usage_record():
    capture = _Capture()
    context = _context()

    capture.drive(record_request_end(MagicMock(), SimpleNamespace(usage=None), context))

    assert len(capture.logs) == 1
    assert len(capture.usage) == 1
    row = capture.logs[0]
    assert row.status_code == 200
    assert row.trace_id == "trace-1", "the row must carry the trace the backends recorded"
    assert row.endpoint == "/v1/chat/completions"
    assert row.method == "POST"
    assert row.event_type == "data_access"
    assert row.action_category == "create"
    assert row.outcome == "success"
    assert row.prompt_tokens == 100
    assert row.log_metadata["streaming"] is False
    assert row.server_hostname
    assert row.service_name == "llm-proxy"

    usage = capture.usage[0]
    assert usage.request_id == "req-1"
    assert usage.total_tokens == 120
    assert usage.cost_usd == 0.002


def test_a_reported_zero_is_stored_as_zero_not_null():
    """A turn that reports a counter as ``0`` reported it — the whole prompt came
    from cache, the turn had produced no output tokens yet — so the column keeps the
    report. Only a turn with no usage record at all leaves the token columns NULL."""
    capture = _Capture()
    context = _context(usage=Usage(input_tokens=40, output_tokens=0, cache_read_input_tokens=40))

    capture.drive(record_request_end(MagicMock(), SimpleNamespace(usage=None), context))

    row = capture.logs[0]
    assert row.prompt_tokens == 40
    assert row.completion_tokens == 0
    assert row.cache_read_input_tokens == 40
    assert row.total_tokens == 40
    usage = capture.usage[0]
    assert usage.prompt_tokens == 40
    assert usage.completion_tokens == 0


def test_no_usage_record_leaves_the_token_columns_null():
    capture = _Capture()
    context = _context(usage=None)

    capture.drive(record_request_end(MagicMock(), SimpleNamespace(usage=None), context))

    row = capture.logs[0]
    assert row.prompt_tokens is None
    assert row.completion_tokens is None
    assert row.total_tokens is None
    assert row.cache_read_input_tokens is None


def test_request_error_writes_an_error_row_without_forcing_cost():
    capture = _Capture()
    context = _context(cost_usd=None)

    capture.drive(record_request_error(MagicMock(), ValueError("upstream broke"), context))

    row = capture.logs[0]
    assert row.status_code == 500
    assert row.outcome == "error"
    assert row.error_message
    assert row.response_body == {"error": True, "message": "Request failed"}
    # The error path does not force cost calculation.
    assert row.cost_usd is None
    assert len(capture.usage) == 1


def test_stream_end_writes_a_stream_row_and_usage_record():
    capture = _Capture()
    context = _context(is_streaming=True)

    capture.drive(record_stream_end(MagicMock(), context, None))

    row = capture.logs[0]
    assert row.status_code == 200
    assert row.log_metadata["streaming"] is True
    assert row.log_metadata["response_body_truncated"] is False
    assert len(capture.usage) == 1
    assert capture.usage[0].is_streaming is True
