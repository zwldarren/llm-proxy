"""Tests for the OTLP exporter and the OpenTelemetry / LangSmith providers.

The two backends share one exporter (``observability/sinks/otlp.py``) and one
lifecycle adapter (``tracing/handlers/exporter_handler.py``); the interesting
decisions are the attribute mapping, the per-request span bookkeeping across the
streaming task boundary, and the two providers' settings validation.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import SpanKind, StatusCode

import llm_proxy.observability.sinks.otlp as otlp_module
from llm_proxy.core.request_type import RequestType
from llm_proxy.models.params import (
    AnthropicSpecificParams,
    CommonParams,
    GenerationParams,
)
from llm_proxy.models.types import PromptTokensDetails, Usage
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.observation.collector import collect
from llm_proxy.observability.sinks.otlp import (
    LANGSMITH_OTLP_ENDPOINT,
    OtlpExporter,
    build_tracer_provider,
    operation_name,
)
from llm_proxy.observability.tracing.handlers import get_handler_type
from llm_proxy.observability.tracing.handlers.providers.langsmith.handler import (
    LangsmithTracingHandler,
)
from llm_proxy.observability.tracing.handlers.providers.otel.handler import (
    OtlpTracingHandler,
)

TRACE_ID = "a" * 32


# ── fixtures and helpers ────────────────────────────────────────────────


@pytest.fixture
def memory() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(memory: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    return provider


def _request() -> MagicMock:
    message = MagicMock()
    message.role = "user"
    message.content = [{"type": "text", "text": "hello"}]
    message.name = None
    conversation = MagicMock()
    conversation.system_messages = []
    conversation.messages = [message]

    request = MagicMock()
    request.conversation = conversation
    request.model = "fast"
    request.request_type = RequestType.CHAT
    request.stream = True
    request.tools = None
    request.tool_choice = None
    request.params = None
    return request


def _response() -> MagicMock:
    response = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "world"
    response.output = [text_block]
    response.finish_reason = "stop"
    response.model = "fast-2026"
    return response


def _context(**overrides) -> EventContext:
    fields: dict = {
        "request_id": "req-1",
        "trace_id": TRACE_ID,
        "model": "fast",
        "provider": "openai",
        "provider_model_name": "fast-2026",
        "session_id": "sess-1",
        "user_id": "alice",
        "api_key_name": "team-key",
        "request_type": RequestType.CHAT,
        "metadata": {"endpoint": "/v1/chat/completions", "method": "POST"},
        "is_streaming": True,
    }
    fields.update(overrides)
    return EventContext(**fields)


def _exporter(provider: TracerProvider, **overrides) -> OtlpExporter:
    return OtlpExporter(provider=provider, **overrides)


# ── attribute mapping ───────────────────────────────────────────────────


def test_operation_names_use_the_genai_semconv_spelling():
    assert operation_name(RequestType.CHAT) == "chat"
    assert operation_name(RequestType.EMBEDDING) == "embeddings"
    assert operation_name(RequestType.IMAGE_GENERATION) == "generate_content"
    # Types the conventions do not name keep the proxy's own operation.
    assert operation_name(RequestType.RERANK) == "rerank"
    assert operation_name(None) == "chat"


def test_finish_exports_one_genai_span(provider, memory):
    exporter = _exporter(provider)
    request, response = _request(), _response()
    context = _context(
        usage=Usage(
            input_tokens=5,
            output_tokens=7,
            total_tokens=12,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=2),
        ),
        cost_usd=0.002,
    )
    record = collect(context)

    exporter.open(record, request)
    assert exporter.get_trace_id() == TRACE_ID
    assert exporter.get_observation_id() is not None
    assert len(exporter.get_observation_id()) == 16

    exporter.finish(record, request=request, response=response)

    (span,) = memory.get_finished_spans()
    assert span.name == "chat fast"
    assert span.kind is SpanKind.CLIENT
    attributes = span.attributes
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.request.model"] == "fast"
    assert attributes["gen_ai.provider.name"] == "openai"
    assert attributes["gen_ai.response.model"] == "fast-2026"
    assert attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert attributes["gen_ai.usage.input_tokens"] == 5
    assert attributes["gen_ai.usage.output_tokens"] == 7
    assert attributes["gen_ai.usage.total_tokens"] == 12
    assert attributes["gen_ai.usage.cache_read.input_tokens"] == 2
    assert attributes["gen_ai.conversation.id"] == "sess-1"
    assert attributes["llm_proxy.request_id"] == "req-1"
    assert attributes["llm_proxy.cost_usd"] == 0.002
    assert attributes["llm_proxy.provider"] == "openai"
    assert "hello" in attributes["input.value"]
    assert "world" in attributes["output.value"]
    assert span.status.status_code == StatusCode.UNSET
    # The finished request leaves no state behind.
    assert exporter._request_states == {}


def test_content_none_withholds_payloads_but_keeps_facts(provider, memory):
    exporter = _exporter(provider, send_content="none")
    request, response = _request(), _response()
    context = _context(usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3))
    record = collect(context)

    exporter.open(record, request)
    exporter.finish(record, request=request, response=response)

    (span,) = memory.get_finished_spans()
    assert "input.value" not in span.attributes
    assert "output.value" not in span.attributes
    assert span.attributes["gen_ai.usage.total_tokens"] == 3


def test_content_inherit_follows_the_log_privacy_switch(provider, memory):
    exporter = _exporter(provider, send_content="inherit")
    request, response = _request(), _response()
    context = _context(should_log_input_output=False)
    record = collect(context)

    exporter.open(record, request)
    exporter.finish(record, request=request, response=response)

    (span,) = memory.get_finished_spans()
    assert "input.value" not in span.attributes
    assert "output.value" not in span.attributes


def test_langsmith_backend_adds_run_correlation_attributes(provider, memory):
    exporter = _exporter(provider, backend="langsmith")
    request, response = _request(), _response()
    record = collect(_context())

    exporter.open(record, request)
    exporter.finish(record, request=request, response=response)

    (span,) = memory.get_finished_spans()
    attributes = span.attributes
    # LangSmith reads the pre-semconv provider spelling and its own attributes.
    assert attributes["gen_ai.system"] == "openai"
    assert attributes["langsmith.trace.session_id"] == "sess-1"
    assert attributes["langsmith.metadata.request_id"] == "req-1"
    assert "llm-proxy" in attributes["langsmith.span.tags"]
    # Its ingest maps the legacy flattened message spellings onto inputs/outputs.
    assert attributes["gen_ai.prompt.0.role"] == "user"
    assert "hello" in attributes["gen_ai.prompt.0.content"]
    assert attributes["gen_ai.completion.0.role"] == "assistant"
    assert attributes["gen_ai.completion.0.content"] == "world"


def test_generic_backend_does_not_emit_langsmith_attributes(provider, memory):
    exporter = _exporter(provider)
    request, response = _request(), _response()
    record = collect(_context())

    exporter.open(record, request)
    exporter.finish(record, request=request, response=response)

    (span,) = memory.get_finished_spans()
    assert "gen_ai.system" not in span.attributes
    assert not any(key.startswith("langsmith.") for key in span.attributes)


def test_content_attributes_are_capped_in_bytes(provider, memory):
    exporter = _exporter(provider)
    request, response = _request(), _response()
    # Four-byte characters overflow a character-counted cap.
    response.output[0].text = "😀" * 5000
    record = collect(_context())

    exporter.open(record, request)
    exporter.finish(record, request=request, response=response)

    (span,) = memory.get_finished_spans()
    output = span.attributes["output.value"]
    assert len(output.encode("utf-8")) <= otlp_module._MAX_CONTENT_BYTES
    assert output.endswith("…")


def test_a_long_conversation_stays_inside_the_span_attribute_limit(provider, memory):
    exporter = _exporter(provider, backend="langsmith")
    request, response = _request(), _response()
    request.conversation.messages = [request.conversation.messages[0]] * 200
    record = collect(_context())

    exporter.open(record, request)
    exporter.finish(record, request=request, response=response)

    (span,) = memory.get_finished_spans()
    flattened = [
        key for key in span.attributes if key.startswith("gen_ai.prompt.") and key.endswith(".role")
    ]
    assert len(flattened) == otlp_module._MAX_FLATTENED_MESSAGES
    # The SDK drops attributes past its 128-attribute span limit, and the
    # dropped ones are the content the operator asked for.
    assert len(span.attributes) < 128


def test_a_passthrough_response_exports_the_body_it_carries(
    provider, memory, make_passthrough_response
):
    """A native-passthrough response parses no blocks; its trace keeps the body.

    Without the fallback the observation would carry a prompt and no answer: the
    client-provider wire reuse path deliberately leaves ``output`` empty.
    """
    exporter = _exporter(provider)
    request = _request()
    response = make_passthrough_response()
    body = response.provider_info["_raw_response_body"]
    record = collect(_context())

    exporter.open(record, request)
    exporter.finish(record, request=request, response=response)

    (span,) = memory.get_finished_spans()
    assert json.loads(span.attributes["output.value"]) == body


def test_provider_specific_top_k_is_exported(provider, memory):
    exporter = _exporter(provider)
    request, response = _request(), _response()
    request.params = GenerationParams(
        common=CommonParams(temperature=0.2),
        anthropic=AnthropicSpecificParams(top_k=40),
    )
    record = collect(_context())

    exporter.open(record, request)
    exporter.finish(record, request=request, response=response)

    (span,) = memory.get_finished_spans()
    assert span.attributes["gen_ai.request.top_k"] == 40
    assert span.attributes["gen_ai.request.temperature"] == 0.2


# ── lifecycle ───────────────────────────────────────────────────────────


def test_ttft_is_recorded_once_as_attribute_and_event(provider, memory):
    exporter = _exporter(provider)
    request = _request()
    context = _context()
    context.first_chunk_time = datetime.now(UTC) + timedelta(seconds=1)

    exporter.open(collect(context), request)
    exporter.stream_opened(collect(context))
    assert exporter.needs_ttft("req-1") is True

    exporter.stream_progress(collect(context))
    assert exporter.needs_ttft("req-1") is False

    exporter.stream_progress(collect(context))
    exporter.finish(collect(context), request=request)

    (span,) = memory.get_finished_spans()
    assert span.attributes["llm_proxy.ttft_ms"] == context.ttft_ms
    assert span.attributes["llm_proxy.streaming"] is True
    first_token_events = [event for event in span.events if event.name == "gen_ai.first_token"]
    assert len(first_token_events) == 1


def test_fail_marks_the_span_as_an_error(provider, memory):
    exporter = _exporter(provider)
    request = _request()
    record = collect(_context())

    exporter.open(record, request)
    exporter.fail(record, RuntimeError("upstream exploded"), request=request)

    (span,) = memory.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert "upstream exploded" in str(span.status.description)
    assert span.attributes["error.type"] == "RuntimeError"
    assert any(event.name == "exception" for event in span.events)
    assert exporter._request_states == {}


def test_a_caller_trace_id_is_continued_when_it_is_a_trace_id(provider, memory):
    exporter = _exporter(provider)
    request = _request()

    exporter.open(collect(_context(trace_id=TRACE_ID)), request)

    assert exporter.get_trace_id() == TRACE_ID
    exporter.finish(collect(_context(trace_id=TRACE_ID)), request=request)
    (span,) = memory.get_finished_spans()
    assert format(span.get_span_context().trace_id, "032x") == TRACE_ID


def test_a_langsmith_span_starts_its_own_root_run(provider, memory):
    """LangSmith must receive a root run, not an orphan child span.

    Its ingest only materializes a trace whose root span arrives: a span parented
    to a run that never comes is accepted with HTTP 200 and then dropped, so the
    export looks healthy while the project stays empty (ADR-0048).
    """
    exporter = _exporter(provider, backend="langsmith")
    request = _request()

    exporter.open(collect(_context(trace_id=TRACE_ID)), request)
    exporter.finish(collect(_context(trace_id=TRACE_ID)), request=request)

    (span,) = memory.get_finished_spans()
    assert span.parent is None
    # The caller's correlation id survives as an attribute instead of as a parent.
    assert span.attributes["llm_proxy.correlation_id"] == TRACE_ID


def test_a_non_trace_correlation_id_is_not_adopted(provider, memory):
    exporter = _exporter(provider)
    request = _request()

    exporter.open(collect(_context(trace_id="not-a-trace-id")), request)

    trace_id = exporter.get_trace_id()
    assert trace_id is not None and trace_id != "not-a-trace-id"
    assert len(trace_id) == 32
    exporter.finish(collect(_context(trace_id="not-a-trace-id")), request=request)
    memory.clear()


def test_a_sampled_out_span_publishes_no_trace_id(memory):
    provider = TracerProvider(sampler=ALWAYS_OFF)
    provider.add_span_processor(SimpleSpanProcessor(memory))
    exporter = _exporter(provider)
    request = _request()
    record = collect(_context())

    exporter.open(record, request)
    exporter.finish(record, request=request)

    # The header must not point at a trace nobody will ever see.
    assert exporter.get_trace_id() is None
    assert memory.get_finished_spans() == ()


async def test_flush_and_close_drive_the_provider():
    provider = MagicMock()
    exporter = _exporter(provider)

    await exporter.flush()
    provider.force_flush.assert_called_once()
    provider.shutdown.assert_not_called()

    await exporter.close()
    provider.shutdown.assert_called_once()
    await exporter.close()
    provider.shutdown.assert_called_once()


class _LifecycleExporter:
    """Records whether release flushed and whether a close followed."""

    def __init__(self) -> None:
        self.flushed = 0
        self.closed = 0

    async def flush(self) -> None:
        self.flushed += 1

    async def close(self) -> None:
        self.closed += 1


async def test_release_flushes_now_and_closes_after_the_grace_period(monkeypatch):
    from llm_proxy.observability.tracing.handlers import exporter_handler

    handler = OtlpTracingHandler(provider=MagicMock())
    exporter = _LifecycleExporter()
    monkeypatch.setattr(handler, "_exporter", exporter)
    monkeypatch.setattr(exporter_handler, "RELEASE_GRACE_SECONDS", 0)

    await handler.release()
    assert exporter.flushed == 1
    assert exporter.closed == 0

    # The deferred close is what stops every config change from stranding the
    # provider's batching thread and export client.
    await asyncio.sleep(0.05)
    assert exporter.closed == 1


# ── provider construction ───────────────────────────────────────────────


class _RecordingExporter:
    """Stands in for the httpx/gRPC span exporter so no socket is opened."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def export(self, spans):  # noqa: ANN001, ANN201 - exporter protocol
        return None

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002
        return True


def test_build_tracer_provider_wires_http(monkeypatch):
    created: list[_RecordingExporter] = []

    def factory(**kwargs) -> _RecordingExporter:
        exporter = _RecordingExporter(**kwargs)
        created.append(exporter)
        return exporter

    monkeypatch.setattr(otlp_module, "HttpSpanExporter", factory)
    provider = build_tracer_provider(
        endpoint="https://collector.example:4318/v1/traces",
        headers={"Authorization": "Bearer token"},
        timeout=3.0,
        service_name="proxy-test",
    )
    try:
        assert created[0].kwargs["endpoint"] == "https://collector.example:4318/v1/traces"
        assert created[0].kwargs["headers"] == {"Authorization": "Bearer token"}
        assert created[0].kwargs["timeout"] == 3.0
        assert provider.resource.attributes["service.name"] == "proxy-test"
    finally:
        provider.shutdown()


def test_build_tracer_provider_lowercases_grpc_metadata(monkeypatch):
    created: list[_RecordingExporter] = []

    def factory(**kwargs) -> _RecordingExporter:
        exporter = _RecordingExporter(**kwargs)
        created.append(exporter)
        return exporter

    monkeypatch.setattr(otlp_module, "GrpcSpanExporter", factory)
    provider = build_tracer_provider(
        endpoint="https://collector.example:4317",
        protocol="grpc",
        headers={"Authorization": "Bearer token"},
        insecure=False,
    )
    try:
        # gRPC metadata keys are case-sensitive and must be lowercase.
        assert created[0].kwargs["headers"] == {"authorization": "Bearer token"}
        assert created[0].kwargs["insecure"] is False
    finally:
        provider.shutdown()


# ── provider settings ───────────────────────────────────────────────────


def test_otel_handler_is_registered():
    assert get_handler_type("otel") is OtlpTracingHandler
    assert get_handler_type("langsmith") is LangsmithTracingHandler


def test_otel_handler_metadata_declares_its_fields():
    assert OtlpTracingHandler.provider_name == "otel"
    assert OtlpTracingHandler.required_settings == ["endpoint"]
    fields = {field["name"] for field in OtlpTracingHandler.field_metadata}
    assert {"endpoint", "protocol", "headers", "insecure", "send_content"} <= fields


def test_an_unset_insecure_setting_is_left_to_the_endpoint_scheme():
    assert OtlpTracingHandler.parse_insecure({}) is None
    assert OtlpTracingHandler.parse_insecure({"insecure": ""}) is None
    assert OtlpTracingHandler.parse_insecure({"insecure": False}) is False
    assert OtlpTracingHandler.parse_insecure({"insecure": True}) is True
    assert OtlpTracingHandler.parse_insecure({"insecure": "yes"}) is True


def test_a_plaintext_grpc_endpoint_keeps_the_sdk_tls_inference(monkeypatch):
    captured: dict = {}

    def fake_build(**kwargs) -> TracerProvider:
        captured.update(kwargs)
        return TracerProvider()

    monkeypatch.setattr(
        "llm_proxy.observability.tracing.handlers.providers.otel.handler.build_tracer_provider",
        fake_build,
    )
    OtlpTracingHandler.create_handler({"endpoint": "http://collector:4317", "protocol": "grpc"})

    # The SDK infers "plain text" from http:// only while insecure is unset.
    assert captured["insecure"] is None


def test_otel_handler_validate_config():
    assert OtlpTracingHandler.validate_config({"endpoint": "https://collector:4318/v1/traces"})
    assert not OtlpTracingHandler.validate_config({})
    assert not OtlpTracingHandler.validate_config({"endpoint": ""})
    assert not OtlpTracingHandler.validate_config(
        {"endpoint": "https://collector:4318/v1/traces", "protocol": "carrier-pigeon"}
    )
    assert not OtlpTracingHandler.validate_config(
        {"endpoint": "https://collector:4318/v1/traces", "send_content": "everything"}
    )
    assert not OtlpTracingHandler.validate_config(
        {"endpoint": "https://collector:4318/v1/traces", "sample_rate": 2.0}
    )


def test_otel_handler_requires_an_endpoint():
    with pytest.raises(ValueError, match="endpoint"):
        OtlpTracingHandler.create_handler({})


def test_langsmith_handler_metadata_declares_its_fields():
    assert LangsmithTracingHandler.provider_name == "langsmith"
    assert LangsmithTracingHandler.backend == "langsmith"
    assert LangsmithTracingHandler.required_settings == ["api_key"]
    fields = {field["name"] for field in LangsmithTracingHandler.field_metadata}
    assert {"api_key", "project", "endpoint", "tenant_id", "send_content"} <= fields


def test_langsmith_handler_validate_config():
    assert LangsmithTracingHandler.validate_config({"api_key": "lsv2-key"})
    assert not LangsmithTracingHandler.validate_config({})
    assert not LangsmithTracingHandler.validate_config({"api_key": ""})
    assert not LangsmithTracingHandler.validate_config({"api_key": "lsv2-key", "sample_rate": -0.1})


def test_langsmith_handler_requires_an_api_key():
    with pytest.raises(ValueError, match="api_key"):
        LangsmithTracingHandler.create_handler({})


def test_langsmith_handler_targets_the_gateway_with_credential_headers(monkeypatch):
    captured: dict = {}

    def fake_build(**kwargs) -> TracerProvider:
        captured.update(kwargs)
        return TracerProvider()

    monkeypatch.setattr(
        "llm_proxy.observability.tracing.handlers.providers.langsmith.handler."
        "build_tracer_provider",
        fake_build,
    )
    handler = LangsmithTracingHandler.create_handler(
        {"api_key": "lsv2-key", "project": "team", "tenant_id": "ws-1"}
    )

    assert handler.enabled
    assert captured["endpoint"] == LANGSMITH_OTLP_ENDPOINT
    assert captured["protocol"] == "http"
    assert captured["headers"]["x-api-key"] == "lsv2-key"
    assert captured["headers"]["Langsmith-Project"] == "team"
    assert captured["headers"]["X-Tenant-Id"] == "ws-1"
    assert handler._exporter.backend == "langsmith"


def test_langsmith_handler_defaults_the_project_name(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        "llm_proxy.observability.tracing.handlers.providers.langsmith.handler."
        "build_tracer_provider",
        lambda **kwargs: captured.update(kwargs) or TracerProvider(),
    )

    LangsmithTracingHandler.create_handler({"api_key": "lsv2-key"})

    assert captured["headers"]["Langsmith-Project"] == "default"


# ── the handler drives the exporter ─────────────────────────────────────


async def test_handler_lifecycle_exports_a_span(provider, memory):
    handler = OtlpTracingHandler(provider=provider)
    request, response = _request(), _response()
    context = _context(usage=Usage(input_tokens=4, output_tokens=6, total_tokens=10))

    await handler.on_request_start(request, context)
    await handler.on_stream_start(request, context)
    await handler.on_stream_end(request, context)  # noqa: F841 - completes the stream

    # Streaming finish has no response object; the output comes from the capture.
    spans = memory.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "chat fast"
    assert spans[0].attributes["gen_ai.usage.total_tokens"] == 10
    assert handler._exporter._request_states == {}

    nonstream_context = _context(request_id="req-2", trace_id="b" * 32)
    await handler.on_request_start(request, nonstream_context)
    await handler.on_request_end(request, response, nonstream_context)
    assert len(memory.get_finished_spans()) == 2


async def test_handler_error_path_records_the_exception(provider, memory):
    handler = LangsmithTracingHandler(provider=provider, backend="langsmith")
    request = _request()
    context = _context()

    await handler.on_request_start(request, context)
    await handler.on_error(request, RuntimeError("boom"), context)

    (span,) = memory.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["gen_ai.system"] == "openai"
