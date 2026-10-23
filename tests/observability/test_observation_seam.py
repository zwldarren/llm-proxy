"""The observation/sink seam: one capture, one policy, one emit (ADR-0044).

These tests pin the two invariants the seam exists for:

- the collector snapshots the request once and no sink reads ``EventContext``;
- every sink receives a payload its policy allowed, and cannot widen it.

The parity tests at the bottom are the migration guard: feeding the new path the
same request must reproduce the rows the endpoint lifecycle writes today, field
for field. They are the reason the seam can be introduced without changing
behaviour.
"""

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

from legacy_row_reference import (
    build_error_row,
    build_stream_row,
    build_success_row,
    build_usage_record,
)
from sink_rows import sink_usage_record

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.models.types import Usage
from llm_proxy.observability import log_intake, store
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.log_intake import record_request_end
from llm_proxy.observability.observation.bus import ObservationBus
from llm_proxy.observability.observation.collector import capture_request_end, collect
from llm_proxy.observability.observation.database import database_bus
from llm_proxy.observability.observation.model import ObservationKind, RequestObservation
from llm_proxy.observability.observation.policy import (
    FACTS_ONLY,
    ContentPolicy,
    SinkPolicy,
    narrow_mode,
    prepare_content,
    sensitive_keys_for,
)
from llm_proxy.observability.observation.projection import apply_sink_policy
from llm_proxy.observability.sinks.database_log import DatabaseLogSink

CONFIG = LoggingConfig(log_input_output=True, mask_sensitive_data=True)


def _context(**overrides) -> EventContext:
    fields: dict = {
        "request_id": "req-seam",
        "trace_id": "trace-seam",
        "model": "fast",
        "internal_model": "fast-internal",
        "provider": "openai",
        "provider_model_name": "gpt-fast",
        "request_type": "chat",
        "log_type": "endpoint",
        "is_api_endpoint": False,
        "should_capture_full_body": True,
        "should_capture_raw_stream": False,
        "should_log_input_output": True,
        "mask_sensitive_data": True,
        "sensitive_keys": frozenset({"api_key"}),
        "start_time": time.perf_counter() - 0.123,
        "start_timestamp": 1_700_000_000.0,
        "response_status_code": 200,
        "request_headers": {
            "authorization": "Bearer sk-secret-token",
            "content-type": "application/json",
        },
        "request_body": {"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
        "response_headers": {"content-type": "application/json"},
        "response_body": {"id": "chatcmpl-1"},
        "usage": Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        "cost_usd": 0.002,
        "cache_savings_usd": 0.0001,
        "first_chunk_time": None,
        "metadata": {
            "endpoint": "/v1/chat/completions",
            "method": "POST",
            "routing": {"complexity": 0.4, "tier": "standard"},
        },
        "fallback_attempts": [
            {"provider": "slow", "attempt": 1, "status_code": 503, "error_message": "down"}
        ],
        "retry_attempts": [],
    }
    fields.update(overrides)
    return EventContext(**fields)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


def test_collector_snapshots_facts_into_the_normalized_sections():
    context = _context()
    observation = collect(context)

    assert observation.kind is ObservationKind.REQUEST
    assert observation.core.request_id == "req-seam"
    assert observation.core.trace_id == "trace-seam"
    assert observation.core.endpoint == "/v1/chat/completions"
    assert observation.identity.user is None
    assert observation.routing.model == "fast"
    assert observation.routing.fallbacks == (
        {"provider": "slow", "attempt": 1, "status_code": 503, "error_message": "down"},
    )
    assert observation.usage.usage is not None
    assert observation.usage.usage.total_tokens == 15
    assert observation.timing.latency_ms is not None
    assert observation.outcome.status_code == 200
    assert observation.content is not None
    assert observation.content.captured is True


def test_collector_copies_the_usage_record_so_the_observation_is_not_aliased():
    context = _context()
    observation = collect(context)

    assert observation.usage.usage is not context.usage
    assert observation.usage.usage == context.usage


def test_collector_records_the_failure_it_was_given():
    context = _context(response_status_code=500)
    error = ValueError("boom")

    observation = collect(context, error=error)

    assert observation.outcome.failed is True
    assert observation.outcome.error_message == "boom"


def test_collector_is_the_only_module_reading_event_context():
    """A sink that imports the capture object has reintroduced the coupling."""
    import ast
    import inspect

    from llm_proxy.observability.sinks import database_log, database_usage

    for module in (database_log, database_usage):
        tree = ast.parse(inspect.getsource(module))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "EventContext" not in imported, f"{module.__name__} reads EventContext"


def test_capture_request_end_folds_its_facts_once_for_two_callers():
    """Two lifecycle adapters finalize the same request; the folds are idempotent."""
    context = _context(response_status_code=None, usage=None)
    response = SimpleNamespace(usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15))

    first = asyncio.run(capture_request_end(context, response=response))
    second = asyncio.run(capture_request_end(context, response=response))

    assert first.usage.usage == second.usage.usage == response.usage
    assert second.outcome.status_code == 200
    assert second.outcome.failed is False


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_a_sink_cannot_widen_the_requests_content_mode():
    assert narrow_mode("none", "full") == "none"
    assert narrow_mode("masked", "full") == "masked"
    assert narrow_mode("full", "masked") == "masked"
    assert narrow_mode("full", "none") == "none"


def test_prepare_content_returns_none_for_the_none_mode():
    assert prepare_content({"a": "b"}, mode="none") is None


def test_prepare_content_masks_sensitive_keys_only_when_masked():
    payload = {"api_key": "sk-supersecret-value", "model": "fast"}

    masked = prepare_content(payload, mode="masked", sensitive_keys=frozenset({"api_key"}))
    verbatim = prepare_content(payload, mode="full", sensitive_keys=frozenset({"api_key"}))

    assert masked["model"] == "fast"
    assert masked["api_key"] != payload["api_key"]
    assert verbatim == payload


def test_prepare_content_caps_a_payload_it_cannot_store():
    capped = prepare_content({"big": "x" * 100}, mode="full", max_bytes=10)

    assert capped == {"_truncated": True, "size": capped["size"]}
    assert capped["size"] > 10


def test_a_facts_only_sink_never_receives_a_body():
    observation = collect(_context())

    projected = apply_sink_policy(observation, FACTS_ONLY)

    assert projected is not None
    assert projected.content is not None
    assert projected.content.withheld is True
    assert projected.content.request_body is None
    assert projected.content.response_body is None


def test_a_content_sink_gets_masked_bodies_when_the_deployment_masks():
    observation = collect(_context())
    observation = replace(
        observation,
        content=replace(observation.content, request_body={"api_key": "sk-supersecret-value"}),
    )

    projected = apply_sink_policy(observation, SinkPolicy(content="full"))

    assert projected is not None
    assert projected.content is not None
    assert projected.content.request_body["api_key"] != "sk-supersecret-value"


def test_a_projection_masks_the_streamed_tool_blocks():
    """Streamed tool calls carry arguments, so they are content like a body."""
    observation = collect(_context())
    observation = replace(
        observation,
        content=replace(
            observation.content,
            stream_output_blocks=[
                {"type": "tool_use", "name": "search", "input": {"api_key": "sk-supersecret"}}
            ],
        ),
    )

    projected = apply_sink_policy(observation, SinkPolicy(content="full"))

    assert projected is not None
    assert projected.content is not None
    assert "sk-supersecret" not in str(projected.content.stream_output_blocks)


def test_a_projection_caps_the_raw_stream_without_rewriting_it():
    """Raw capture is verbatim by contract: a policy may only shorten it."""
    context = _context(should_capture_raw_stream=True)
    context.capture_streaming_chunk("data: {" + '"content": "x"}' * 10 + "}\n\n")
    observation = collect(context)
    raw = observation.content.raw_stream
    assert raw

    projected = apply_sink_policy(observation, SinkPolicy(content="full", max_bytes=16))

    assert projected is not None
    assert projected.content is not None
    assert projected.content.raw_stream == raw[:16]
    assert projected.content.raw_stream_truncated is True


def test_the_record_sample_decides_whether_a_sink_is_called():
    observation = collect(_context())

    assert apply_sink_policy(observation, SinkPolicy(record=False)) is None
    assert apply_sink_policy(observation, SinkPolicy(sample_record=0.0)) is None


def test_an_invalid_record_rate_keeps_the_record():
    """A negative or NaN rate is misconfiguration, not "drop every record"."""
    observation = collect(_context())

    assert apply_sink_policy(observation, SinkPolicy(sample_record=-1.0)) is not None
    assert apply_sink_policy(observation, SinkPolicy(sample_record=float("nan"))) is not None


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, name: str, kinds: frozenset[ObservationKind]) -> None:
        self.name = name
        self.kinds = kinds
        self.received: list[RequestObservation] = []
        self.flushed = 0
        self.closed = 0

    async def emit(self, observation: RequestObservation) -> None:
        self.received.append(observation)

    async def flush(self) -> None:
        self.flushed += 1

    async def close(self) -> None:
        self.closed += 1


class _Failing:
    name = "failing"
    kinds = frozenset({ObservationKind.REQUEST})

    async def emit(self, observation: RequestObservation) -> None:
        raise RuntimeError("backend down")

    async def flush(self) -> None:
        raise RuntimeError("backend down")

    async def close(self) -> None:
        raise RuntimeError("backend down")


def test_bus_delivers_only_to_subscribed_kinds():
    bus = ObservationBus()
    requests = _Recorder("requests", frozenset({ObservationKind.REQUEST}))
    tools = _Recorder("tools", frozenset({ObservationKind.MCP}))
    bus.register(requests)
    bus.register(tools)

    asyncio.run(bus.emit(collect(_context())))

    assert len(requests.received) == 1
    assert tools.received == []


def test_bus_isolates_a_failing_sink(caplog):
    bus = ObservationBus()
    recorder = _Recorder("recorder", frozenset({ObservationKind.REQUEST}))
    bus.register(_Failing())
    bus.register(recorder)

    delivered = asyncio.run(bus.emit(collect(_context())))

    assert delivered == 1
    assert len(recorder.received) == 1
    # The failure is diagnosable: which sink, which request, and the traceback.
    assert "'failing'" in caplog.text
    assert caplog.records[-1].exc_info is not None


def test_bus_flush_and_close_reach_every_sink():
    bus = ObservationBus()
    sink = _Recorder("recorder", frozenset({ObservationKind.REQUEST}))
    bus.register(sink)
    bus.register(_Failing())

    asyncio.run(bus.flush())
    asyncio.run(bus.close())

    assert sink.flushed == 1
    assert sink.closed == 1


# ---------------------------------------------------------------------------
# Golden parity: the new path reproduces today's rows
# ---------------------------------------------------------------------------


def _parity_row(context: EventContext, *, error: Exception | None = None, streaming=False):
    """The row the new path produces for a request the lifecycle already captured."""
    observation = collect(context, error=error)
    projected = apply_sink_policy(observation, SinkPolicy(content="full"))
    assert projected is not None
    sink = DatabaseLogSink(config_provider=lambda: CONFIG, dispatch=lambda row: True)
    return sink.build_row(projected)


def _assert_same_row(actual, expected) -> None:
    """Field-for-field parity, with the one live value neutralised.

    ``response_time_ms`` is read from ``perf_counter()`` at access time, so the two
    builds sample the clock milliseconds apart. Comparing it would flake across an
    integer boundary for a value neither side decides; every other field must match
    exactly.
    """
    assert actual.response_time_ms is not None
    assert expected.response_time_ms is not None
    assert replace(expected, response_time_ms=actual.response_time_ms) == actual


def test_database_log_sink_reproduces_the_success_row():
    context = _context()
    log_intake.configure(config=CONFIG)

    actual = _parity_row(context)
    expected = build_success_row(context, CONFIG)

    _assert_same_row(actual, expected)
    # ``trace_id`` reaches the row only through the assembler both sides share, so
    # a field dropped there would fall back to ``None`` on both and this suite
    # would stay green. Pin the migration's new field directly.
    assert actual.trace_id == expected.trace_id == "trace-seam"


def test_database_log_sink_reproduces_the_error_row():
    error = ValueError("boom")
    context = _context(
        response_status_code=500,
        error_message=str(error),
        error_stack_trace="traceback",
        error_details={"error_type": "ValueError"},
    )
    log_intake.configure(config=CONFIG)

    _assert_same_row(_parity_row(context, error=error), build_error_row(context, CONFIG))


def test_database_log_sink_reproduces_the_stream_row():
    context = _context(
        is_streaming=True,
        assembled_response_body={"id": "chatcmpl-1"},
        metadata={
            "endpoint": "/v1/chat/completions",
            "method": "POST",
            "streaming": True,
        },
    )
    log_intake.configure(config=CONFIG)

    _assert_same_row(_parity_row(context), build_stream_row(context, CONFIG))


def test_a_streaming_row_ignores_a_stale_response_body():
    """``record_stream_end`` overwrote ``response_body`` with the stream body.

    The sink reads the buffered or reassembled body instead, so a value left on
    ``response_body`` by a streaming capture never becomes the row's body.
    """
    context = _context(
        is_streaming=True,
        response_body={"id": "not-the-stream-body"},
        metadata={"endpoint": "/v1/chat/completions", "method": "POST"},
    )
    log_intake.configure(config=CONFIG)

    actual = _parity_row(context)

    assert actual.response_body == {"streaming": True, "_assembled": False}
    _assert_same_row(actual, build_stream_row(context, CONFIG))


def test_database_log_sink_reproduces_the_body_less_row():
    """Bodies off is the privacy path: markers and masked headers must match."""
    config = LoggingConfig(log_input_output=False, mask_sensitive_data=True)
    context = _context(
        should_log_input_output=False,
        should_capture_full_body=False,
        response_body=None,
    )
    log_intake.configure(config=config)
    observation = collect(context)
    projected = apply_sink_policy(observation, SinkPolicy(content="full"))
    assert projected is not None
    sink = DatabaseLogSink(config_provider=lambda: config, dispatch=lambda row: True)

    expected = build_success_row(context, config)
    actual = sink.build_row(projected)

    assert actual.request_body == expected.request_body == {"_bodies_disabled": True}
    assert actual.response_body == expected.response_body == {"_bodies_disabled": True}
    assert actual.request_headers["authorization"] != "Bearer sk-secret-token"
    _assert_same_row(actual, expected)


def test_database_usage_sink_reproduces_the_usage_record():
    context = _context()

    # Through the shared helper, so the live path it claims to mirror is the one
    # exercised (``sink_usage_record`` = collect -> FACTS_ONLY -> build_record).
    assert sink_usage_record(context) == build_usage_record(context)


def test_the_content_policy_defaults_to_none_without_a_config():
    assert ContentPolicy.resolve(None).mode == "none"


def test_database_log_sink_reproduces_the_body_less_error_row():
    """An error row keeps its constant outcome body when payloads are off."""
    config = LoggingConfig(log_input_output=False, mask_sensitive_data=True)
    error = ValueError("boom")
    context = _context(
        should_log_input_output=False,
        should_capture_full_body=False,
        request_body={"messages": []},
        response_body=None,
        response_status_code=500,
        error_message=str(error),
        error_stack_trace="traceback",
        error_details={
            "status_code": 500,
            "response_body": "UPSTREAM SECRET",
            "original_error": {"message": "UPSTREAM SECRET"},
        },
    )
    log_intake.configure(config=config)
    observation = collect(context, error=error)
    projected = apply_sink_policy(observation, SinkPolicy(content="full"))
    assert projected is not None
    sink = DatabaseLogSink(config_provider=lambda: config, dispatch=lambda row: True)

    actual = sink.build_row(projected)
    expected = build_error_row(context, config)

    assert (
        actual.response_body
        == expected.response_body
        == {
            "error": True,
            "message": "Request failed",
        }
    )
    assert actual.request_body == expected.request_body == {"_bodies_disabled": True}
    assert actual.log_metadata["error_details"]["response_body"] == {"_bodies_disabled": True}
    _assert_same_row(actual, expected)


def test_database_log_sink_reproduces_the_sampled_out_stream_row():
    """A stream that lost the sampling draw stores the sentinel, not content."""
    config = LoggingConfig(log_input_output=True, mask_sensitive_data=True)
    context = _context(
        should_capture_full_body=False,
        should_capture_raw_stream=False,
        is_streaming=True,
        response_body=None,
        metadata={"endpoint": "/v1/chat/completions", "method": "POST"},
    )
    log_intake.configure(config=config)
    observation = collect(context)
    projected = apply_sink_policy(observation, SinkPolicy(content="full"))
    assert projected is not None
    sink = DatabaseLogSink(config_provider=lambda: config, dispatch=lambda row: True)

    actual = sink.build_row(projected)
    expected = build_stream_row(context, config)

    assert actual.request_body == expected.request_body == {"_sampled_out": True}
    assert actual.response_body == expected.response_body == {"_sampled_out": True}
    _assert_same_row(actual, expected)


def test_the_live_row_masks_the_builtin_keys_the_frozen_reference_does_not():
    """A deliberate endpoint-row change, pinned instead of hidden.

    The store masks with ``sensitive_keys_for`` — the built-in key set union the
    operator's ``logging.sensitive_keys`` — while the pre-seam builders masked with
    the configured list alone. The parity contexts above set ``sensitive_keys``
    explicitly (the production wiring), which is what lets them claim
    field-for-field equality; this test states the difference they cannot see.
    """
    config = LoggingConfig(
        log_input_output=True,
        mask_sensitive_data=True,
        sensitive_keys=["api_key"],
    )
    context = _context(
        sensitive_keys=sensitive_keys_for(config),
        request_body={
            "model": "fast",
            "key": "sk-builtin-value",
            "api_key": "sk-configured-value",
        },
    )
    log_intake.configure(config=config)

    actual = _parity_row(context)
    expected = build_success_row(context, config)

    # ``key`` is a built-in the operator never configured: masked on the live row,
    # verbatim in the frozen pre-seam row.
    assert actual.request_body["key"] != "sk-builtin-value"
    assert expected.request_body["key"] == "sk-builtin-value"
    # A configured key is masked by both.
    assert actual.request_body["api_key"] == expected.request_body["api_key"]


def test_the_database_bus_carries_both_stores():
    names = {sink.name for sink in database_bus().sinks}

    assert names == {"database_log", "database_usage"}


def _capture_stores(monkeypatch, rows: list, usage: list) -> None:
    """Point the two database sinks at in-memory stores."""

    class _LogService:
        def __init__(self, config):  # noqa: ARG002
            pass

        def create_log_background(self, data):
            rows.append(data)

    class _UsageService:
        def create_usage_background(self, data):
            usage.append(data)

    monkeypatch.setattr(store, "RequestLogService", _LogService)
    monkeypatch.setattr(store, "UsageService", _UsageService)


def test_endpoint_verbs_fan_out_through_the_database_bus(monkeypatch):
    """The production path: one observation, two stores, plus any extra sink."""
    bus = database_bus()
    seen: list[RequestObservation] = []
    rows: list = []
    usage: list = []

    class _Spy:
        name = "spy"
        kinds = frozenset({ObservationKind.REQUEST})

        async def emit(self, observation: RequestObservation) -> None:
            seen.append(observation)

        async def flush(self) -> None:
            pass

        async def close(self) -> None:
            pass

    spy = _Spy()
    bus.register(spy)
    _capture_stores(monkeypatch, rows, usage)
    log_intake.configure(config=CONFIG)
    try:
        asyncio.run(record_request_end(MagicMock(), SimpleNamespace(usage=None), _context()))
    finally:
        bus.unregister(spy)

    assert len(seen) == 1
    assert seen[0].kind is ObservationKind.REQUEST
    assert len(rows) == 1
    assert len(usage) == 1
    assert rows[0].request_id == seen[0].core.request_id


def test_a_failing_sink_does_not_cost_the_audit_row(monkeypatch, caplog):
    """A broken backend is loud but not fatal: the other sinks still write."""
    bus = database_bus()
    rows: list = []
    usage: list = []

    class _Boom:
        name = "boom"
        kinds = frozenset({ObservationKind.REQUEST})

        async def emit(self, observation: RequestObservation) -> None:
            raise RuntimeError("backend down")

        async def flush(self) -> None:
            pass

        async def close(self) -> None:
            pass

    boom = _Boom()
    bus.register(boom)
    _capture_stores(monkeypatch, rows, usage)
    log_intake.configure(config=CONFIG)
    try:
        asyncio.run(record_request_end(MagicMock(), SimpleNamespace(usage=None), _context()))
    finally:
        bus.unregister(boom)

    assert len(rows) == 1
    assert len(usage) == 1
    assert "'boom'" in caplog.text
