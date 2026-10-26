"""OTLP exporter: one span per request, exported through the OpenTelemetry SDK.

Two tracing backends share this exporter:

- ``otel`` — any OTLP collector (Tempo, Jaeger, Phoenix, Grafana, …), over
  HTTP/protobuf or gRPC;
- ``langsmith`` — LangSmith's OTLP/HTTP ingest gateway, which accepts traces
  only and reads the same ``gen_ai.*`` attributes plus a few ``langsmith.*``
  ones of its own.

The request lifecycle lives in
``tracing/handlers/providers/{otel,langsmith}/handler.py``; everything that
talks to the SDK lives here. The exporter reads no ``EventContext``: it receives
the immutable :class:`RequestObservation` plus the typed wire request/response
(the latter rendered by ``observability/content/registry.py``), exactly like the
Langfuse exporter (ADR-0044).

It is a *live* exporter, not a one-call sink: a span must be open before the
response headers are written (they echo its trace id) and TTFT is recorded while
the stream still runs. Per-request state is keyed by request id on the instance
rather than kept in a ``ContextVar``, because streaming hooks run in a different
asyncio task than the pipeline that opened the span.
"""

import asyncio
import os
import re
from collections import OrderedDict
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal

import orjson
lazy from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcSpanExporter,
)
lazy from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpSpanExporter,
)
lazy from opentelemetry.sdk.resources import Resource
lazy from opentelemetry.sdk.trace import TracerProvider
lazy from opentelemetry.sdk.trace.export import BatchSpanProcessor
lazy from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
lazy from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
    set_span_in_context,
)

from llm_proxy.core.request_type import RequestType, coerce_request_type
from llm_proxy.observability.content.registry import format_output_blocks
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.observation.model import RequestObservation
from llm_proxy.observability.observation.policy import SendContent, cap_payload
from llm_proxy.observability.sinks.send_content import SendContentProjector

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

logger = get_logger(__name__)

#: How the spans leave the process. LangSmith's ingest is HTTP/protobuf only;
#: a generic collector may speak either.
OtlpProtocol = Literal["http", "grpc"]

#: Which attribute vocabulary beyond the shared ``gen_ai.*`` set to emit.
OtlpBackend = Literal["otel", "langsmith"]

#: LangSmith's OTLP/HTTP traces endpoint. The exporter uses this URL verbatim
#: (``OTLPSpanExporter`` does not append ``/v1/traces``), and the region is the
#: operator's choice: ``eu``/``apac``/``aws`` variants and self-hosted endpoints
#: (``https://<host>/api/v1/otel/traces``) differ only in the host.
LANGSMITH_OTLP_ENDPOINT: Final = "https://api.smith.langchain.com/otel/v1/traces"

#: LangSmith's per-project routing header. Without it traces land in the
#: account's ``default`` project.
LANGSMITH_PROJECT_HEADER: Final = "Langsmith-Project"

#: A canonical OTel trace id — the only caller-supplied correlation id this
#: exporter can continue. Anything else (a UUID with dashes, an arbitrary
#: correlation string) cannot be a span's trace id.
_TRACE_ID_RE = re.compile(r"[0-9a-fA-F]{32}\Z")

#: Characters kept per content attribute before the byte budget applies. The
#: truncation shape itself lives in the shared ``cap_payload``.
_MAX_CONTENT_CHARS = 16384

#: Bytes one content attribute may occupy. OTLP bounds a message in bytes, so a
#: character cap alone still lets four-byte characters overshoot the 16 KiB the
#: spec promises (ADR-0048).
_MAX_CONTENT_BYTES = 16384

_ELLIPSIS = "…"

#: Messages flattened per content side. Two attributes per message keep a long
#: conversation inside the SDK's 128-attribute span limit alongside the request,
#: result and fact attributes; 20 matches the content registry's own preview bound.
_MAX_FLATTENED_MESSAGES = 20

#: Provider-specific parameter groups that carry ``top_k`` (see
#: ``_provider_top_k``); ``CommonParams`` does not.
_TOP_K_PARAM_GROUPS: Final = ("anthropic", "gemini")

#: Finished requests keep their ids for the header write that follows the end
#: hook, in the same task. Bounded so a long-running proxy does not accumulate
#: one entry per traced request for the process lifetime.
_MAX_TRACE_ID_ENTRIES = 1024

#: The GenAI operation an endpoint maps to. Values are the OpenTelemetry
#: semconv spellings where one exists; request types the conventions do not
#: name keep the proxy's own name (``rerank``, ``speech``, …).
_OPERATION_NAMES: Final[Mapping[RequestType, str]] = frozendict(
    {
        RequestType.CHAT: "chat",
        RequestType.EMBEDDING: "embeddings",
        RequestType.IMAGE_GENERATION: "generate_content",
        RequestType.IMAGE_EDIT: "generate_content",
    }
)


def operation_name(request_type: object) -> str:
    """The ``gen_ai.operation.name`` for a request type (``chat`` when unknown)."""
    coerced = coerce_request_type(request_type)
    if coerced is None:
        return "chat"
    return _OPERATION_NAMES.get(coerced, coerced.value)


def build_tracer_provider(
    *,
    endpoint: str,
    protocol: OtlpProtocol = "http",
    headers: Mapping[str, str] | None = None,
    insecure: bool | None = None,
    timeout: float | None = None,
    service_name: str = "llm-proxy",
    sample_rate: float | None = None,
) -> TracerProvider:
    """Build the process-local ``TracerProvider`` for one tracing backend.

    The provider is per handler instance, not process-wide: two users exporting
    to different collectors must never cross-export each other's spans, and a
    configured sampling rate belongs to one project.
    """
    from llm_proxy.version import get_version

    attributes: dict[str, Any] = {
        "service.name": service_name,
        "service.version": get_version(),
    }
    provider = TracerProvider(
        resource=Resource.create(attributes),
        sampler=TraceIdRatioBased(sample_rate) if sample_rate is not None else None,
    )
    span_exporter: Any
    if protocol == "grpc":
        span_exporter = GrpcSpanExporter(
            endpoint=endpoint,
            headers={key.lower(): str(value) for key, value in headers.items()}
            if headers
            else None,
            insecure=insecure,
            timeout=timeout,
        )
    else:
        span_exporter = HttpSpanExporter(
            endpoint=endpoint,
            headers=dict(headers) if headers else None,
            timeout=timeout,
        )
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    return provider


def _unix_ns(seconds: float) -> int:
    return int(seconds * 1_000_000_000)


def _datetime_ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _encode_json(value: Any) -> str:
    try:
        return orjson.dumps(
            value,
            default=str,
            option=orjson.OPT_NON_STR_KEYS,
        ).decode("utf-8")
    except TypeError, ValueError:
        return str(value)


def _cap_text(text: str) -> str:
    """Bound one attribute: the shared character cap, then the byte budget."""
    capped = str(cap_payload(text, max_chars=_MAX_CONTENT_CHARS))
    encoded = capped.encode("utf-8")
    if len(encoded) <= _MAX_CONTENT_BYTES:
        return capped
    budget = _MAX_CONTENT_BYTES - len(_ELLIPSIS.encode("utf-8"))
    return encoded[:budget].decode("utf-8", "ignore") + _ELLIPSIS


def _json_text(value: Any) -> str:
    """JSON-encode an attribute value, bounded to the content budget."""
    return _cap_text(_encode_json(value))


def _content_text(value: Any) -> str:
    """Render one message part as text: strings verbatim, everything else JSON."""
    return _cap_text(value if isinstance(value, str) else _encode_json(value))


def _legacy_content(prefix: str, value: Any, *, role: str) -> dict[str, Any]:
    """Flatten a rendered payload into LangSmith's message attributes.

    LangSmith's OTLP ingest reads the pre-semconv ``gen_ai.prompt.{n}.role`` /
    ``gen_ai.prompt.{n}.content`` spellings and renders them as the run's
    ``inputs.messages`` / ``outputs.messages``. A payload that is not a message
    list (rerank documents, an embedding batch, image URLs) is one JSON string
    under the prefix instead.
    """
    messages: Any = value
    if isinstance(value, Mapping):
        if "messages" in value:
            messages = value["messages"]
        elif "role" in value:
            # One rendered message (a non-streaming output is a single assistant
            # message, not a conversation).
            messages = [value]
        else:
            messages = None
    if isinstance(messages, list) and messages and all(isinstance(m, Mapping) for m in messages):
        attributes: dict[str, Any] = {}
        for index, message in enumerate(messages[:_MAX_FLATTENED_MESSAGES]):
            attributes[f"{prefix}.{index}.role"] = str(message.get("role") or role)
            attributes[f"{prefix}.{index}.content"] = _content_text(message.get("content"))
        return attributes
    return {prefix: _json_text(value)}


def _context_facts(record: RequestObservation) -> dict[str, Any]:
    """Facts every trace backend shows beside the GenAI attributes.

    One mapping, then each exporter namespaces it (``llm_proxy.*`` attributes,
    ``langsmith.metadata.*``): the decision of *which* facts describe a request
    is made once, in ``RequestObservation``.
    """
    streaming = record.content is not None and record.content.streaming
    facts: dict[str, Any] = {
        "request_id": record.core.request_id,
        "correlation_id": record.core.trace_id,
        "endpoint": record.core.endpoint or None,
        "request_type": str(record.core.request_type) if record.core.request_type else None,
        "provider": record.routing.provider,
        "model": record.routing.model,
        "provider_model_name": record.routing.provider_model_name,
        "streaming": streaming or None,
        "status_code": record.outcome.status_code,
        "error_message": record.outcome.error_message,
        "user": record.identity.user,
        "api_key_name": record.identity.api_key_name,
        "auth_method": record.identity.auth_method,
        "session_id": record.identity.session_id,
        "fallback_count": len(record.routing.fallbacks) or None,
        "retry_count": len(record.routing.retries) or None,
        "latency_ms": round(record.timing.latency_ms, 3)
        if record.timing.latency_ms is not None
        else None,
        "ttft_ms": record.timing.ttft_ms,
        "cost_usd": record.usage.cost_usd,
        "cache_savings_usd": record.usage.cache_savings_usd,
        "provider_reported_cost": record.usage.provider_reported_cost,
        "web_search_requests": record.usage.web_search_requests,
        "images_generated": record.usage.images_generated,
        "tts_characters": record.usage.tts_characters,
    }
    return {key: value for key, value in facts.items() if value is not None}


def _usage_attributes(record: RequestObservation) -> dict[str, Any]:
    """``gen_ai.usage.*`` attributes from the canonical usage record.

    The GenAI conventions define the input/output counters as *inclusive* (the
    input count already contains cached tokens), unlike Langfuse's exclusive
    buckets, so the counters are emitted as reported and the detail dimensions
    sit beside them.
    """
    usage = record.usage.usage
    if usage is None:
        return {}
    attributes: dict[str, Any] = {}
    if usage.input_tokens is not None:
        attributes["gen_ai.usage.input_tokens"] = usage.input_tokens
    if usage.output_tokens is not None:
        attributes["gen_ai.usage.output_tokens"] = usage.output_tokens
    if usage.total_tokens is not None:
        attributes["gen_ai.usage.total_tokens"] = usage.total_tokens
    details = usage.prompt_tokens_details
    cache_read = usage.cache_read_input_tokens or (
        details.cached_tokens if details is not None else None
    )
    if cache_read:
        attributes["gen_ai.usage.cache_read.input_tokens"] = cache_read
    if usage.cache_creation_input_tokens:
        attributes["gen_ai.usage.cache_creation.input_tokens"] = usage.cache_creation_input_tokens
    if usage.reasoning_tokens:
        attributes["gen_ai.usage.reasoning.output_tokens"] = usage.reasoning_tokens
    return attributes


def _provider_top_k(params: Any) -> int | None:
    """``top_k`` from whichever provider-specific parameter group sets it.

    ``top_k`` is not part of ``CommonParams``: only the Anthropic and Gemini
    dialects carry it, and the semconv spelling is the same for both.
    """
    for group in _TOP_K_PARAM_GROUPS:
        group_params = getattr(params, group, None)
        top_k = getattr(group_params, "top_k", None)
        if top_k is not None:
            return top_k
    return None


def _model_parameter_attributes(params: Any) -> dict[str, Any]:
    """``gen_ai.request.*`` attributes from the request's generation parameters."""
    common = getattr(params, "common", None)
    if common is None:
        return {}
    mapping = {
        "temperature": "gen_ai.request.temperature",
        "top_p": "gen_ai.request.top_p",
        "max_tokens": "gen_ai.request.max_tokens",
        "frequency_penalty": "gen_ai.request.frequency_penalty",
        "presence_penalty": "gen_ai.request.presence_penalty",
        "seed": "gen_ai.request.seed",
        "stop": "gen_ai.request.stop_sequences",
    }
    attributes: dict[str, Any] = {}
    for field_name, attribute_name in mapping.items():
        value = getattr(common, field_name, None)
        if value is not None:
            attributes[attribute_name] = value
    top_k = _provider_top_k(params)
    if top_k is not None:
        attributes["gen_ai.request.top_k"] = top_k
    return attributes


def _request_attributes(record: RequestObservation, request: Any) -> dict[str, Any]:
    """Everything known about the call before it finishes."""
    model = record.routing.model or getattr(request, "model", None)
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": operation_name(
            record.core.request_type or getattr(request, "request_type", None)
        ),
        "gen_ai.request.model": model,
        "gen_ai.provider.name": record.routing.provider,
        "gen_ai.conversation.id": record.identity.session_id,
    }
    attributes.update(_model_parameter_attributes(getattr(request, "params", None)))
    return {key: value for key, value in attributes.items() if value is not None}


def _result_attributes(
    record: RequestObservation,
    request: Any,
    response: Any | None,
) -> dict[str, Any]:
    """Everything the finished call adds: response model, usage, cost."""
    attributes: dict[str, Any] = {}
    response_model = getattr(response, "model", None) or record.routing.provider_model_name
    if response_model:
        attributes["gen_ai.response.model"] = str(response_model)
    finish_reason = getattr(response, "finish_reason", None)
    if finish_reason:
        attributes["gen_ai.response.finish_reasons"] = [str(finish_reason)]
    attributes.update(_usage_attributes(record))
    return {key: value for key, value in attributes.items() if value is not None}


def _parent_context(trace_id: str | None) -> Any:
    """A parent context that continues a caller-supplied trace, when usable.

    The proxy accepts ``x-trace-id`` (and ``x-langfuse-trace-id``) so a caller can
    nest this request's span inside the trace it already started. Only a
    canonical 32-hex id can be continued; anything else would be silently
    replaced by the SDK with a new trace anyway, so it is ignored here.
    """
    if not trace_id or not _TRACE_ID_RE.prefixmatch(trace_id):
        return None
    trace_id_int = int(trace_id, 16)
    if trace_id_int == 0:
        return None
    span_id_int = int.from_bytes(os.urandom(8), "big") or 1
    context = SpanContext(
        trace_id=trace_id_int,
        span_id=span_id_int,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return set_span_in_context(NonRecordingSpan(context))


@dataclass
class _RequestState:
    """Per-request span, keyed by request id on the exporter."""

    span: Span
    ttft_recorded: bool = False


class OtlpExporter(SendContentProjector):
    """Shape observations into one OTLP span and drive the OpenTelemetry SDK."""

    def __init__(
        self,
        *,
        provider: TracerProvider,
        backend: OtlpBackend = "otel",
        name: str = "otlp",
        send_content: SendContent = "full",
    ) -> None:
        self.name = name
        self.backend = backend
        self.send_content = send_content
        self._provider: TracerProvider | None = provider
        self._tracer: Tracer = provider.get_tracer("llm-proxy")
        # The id of the request the *current* context opened is needed by the
        # header-writing seam, which asks this handler for a trace id without
        # passing one. It runs in the same task that opened the span, so a
        # ContextVar is the right key there; the per-request span and ids maps
        # below are ordinary dicts because streaming hooks run in another task.
        self._active_request_id: ContextVar[str | None] = ContextVar(
            f"otlp_active_request_id_{id(self)}", default=None
        )
        self._request_states: dict[str, _RequestState] = {}
        self._request_trace_ids: OrderedDict[str, tuple[str, str]] = OrderedDict()

    # -- per-request state ------------------------------------------------

    def _set_trace_ids(self, request_id: str, trace_id: str, span_id: str) -> None:
        self._request_trace_ids[request_id] = (trace_id, span_id)
        self._request_trace_ids.move_to_end(request_id)
        while len(self._request_trace_ids) > _MAX_TRACE_ID_ENTRIES:
            self._request_trace_ids.popitem(last=False)

    def get_trace_id(self) -> str | None:
        request_id = self._active_request_id.get()
        if request_id is None:
            return None
        ids = self._request_trace_ids.get(request_id)
        return ids[0] if ids is not None else None

    def get_observation_id(self) -> str | None:
        request_id = self._active_request_id.get()
        if request_id is None:
            return None
        ids = self._request_trace_ids.get(request_id)
        return ids[1] if ids is not None else None

    # -- attributes -------------------------------------------------------

    @staticmethod
    def _set_attributes(span: Span, attributes: Mapping[str, Any]) -> None:
        for key, value in attributes.items():
            if value is None:
                continue
            span.set_attribute(key, value)

    def _backend_attributes(self, record: RequestObservation) -> dict[str, Any]:
        """Facts namespaced for this backend, plus the LangSmith-only extras."""
        facts = _context_facts(record)
        attributes: dict[str, Any] = {f"llm_proxy.{key}": value for key, value in facts.items()}
        if self.backend != "langsmith":
            return attributes

        # LangSmith reads the pre-semconv provider spelling and its own run
        # correlation attributes; unmapped OTLP attributes are dropped by its
        # ingest (see the exporter module docstring).
        attributes["gen_ai.system"] = record.routing.provider or "llm_proxy"
        attributes.update({f"langsmith.metadata.{key}": value for key, value in facts.items()})
        if record.identity.session_id:
            attributes["langsmith.trace.session_id"] = record.identity.session_id
        tags = [
            tag
            for tag in (
                "llm-proxy",
                record.core.request_type,
                record.routing.provider,
                record.routing.model,
            )
            if tag
        ]
        if tags:
            attributes["langsmith.span.tags"] = [str(tag) for tag in tags]
        return attributes

    def _content_attributes(
        self,
        record: RequestObservation,
        request: Any,
        response: Any | None,
    ) -> dict[str, Any]:
        """The projected input/output, if this backend may see it.

        Rendering goes through the request type's profile (``content/registry``)
        and then through this backend's content mode, so a backend never grows a
        second masking rule (ADR-0044).
        """
        profile = self.profile(record, request)
        input_value = self.prepare(profile.build_input(request), record)
        if response is not None:
            raw_output: Any = profile.build_output(response)
        else:
            blocks = record.content.stream_output_blocks if record.content is not None else None
            raw_output = format_output_blocks(blocks) if blocks else None
        output_value = self.prepare(raw_output, record) if raw_output is not None else None

        attributes: dict[str, Any] = {}
        if input_value is not None:
            # ``input.value``/``output.value`` are the OpenInference spellings
            # most trace UIs render; ``gen_ai.*`` carries the structured fields.
            attributes["input.value"] = _json_text(input_value)
            attributes["input.mime_type"] = "application/json"
            if self.backend == "langsmith":
                attributes.update(_legacy_content("gen_ai.prompt", input_value, role="user"))
        if output_value is not None:
            attributes["output.value"] = _json_text(output_value)
            attributes["output.mime_type"] = "application/json"
            if self.backend == "langsmith":
                attributes.update(
                    _legacy_content("gen_ai.completion", output_value, role="assistant")
                )
        return attributes

    # -- lifecycle --------------------------------------------------------

    def open(self, record: RequestObservation, request: Any) -> None:
        """Open the request's span before the response headers are written."""
        request_id = record.core.request_id
        self._active_request_id.set(request_id)
        span: Span | None = None
        try:
            span = self._tracer.start_span(
                self._span_name(record, request),
                context=_parent_context(record.core.trace_id),
                kind=SpanKind.CLIENT,
                start_time=_unix_ns(record.core.started_at),
            )
            self._request_states[request_id] = _RequestState(span)
            span_context = span.get_span_context()
            # A sampled-out span has a randomly generated context that is never
            # exported; echoing its trace id would point the caller at nothing.
            if span.is_recording() and span_context.is_valid:
                self._set_trace_ids(
                    request_id,
                    format(span_context.trace_id, "032x"),
                    format(span_context.span_id, "016x"),
                )
            attributes = _request_attributes(record, request)
            attributes.update(self._backend_attributes(record))
            attributes.update(self._content_attributes(record, request, response=None))
            self._set_attributes(span, attributes)
        except Exception as e:
            self._request_states.pop(request_id, None)
            self._request_trace_ids.pop(request_id, None)
            if span is not None:
                span.end()
            logger.error(f"{self.name} exporter failed to start span: {e}", exc_info=True)

    def _span_name(self, record: RequestObservation, request: Any) -> str:
        operation = operation_name(
            record.core.request_type or getattr(request, "request_type", None)
        )
        model = record.routing.model or getattr(request, "model", None)
        return f"{operation} {model}" if model else operation

    def stream_opened(self, record: RequestObservation) -> None:
        self._active_request_id.set(record.core.request_id)
        state = self._request_states.get(record.core.request_id)
        if state is None:
            return
        try:
            self._set_attributes(state.span, self._backend_attributes(record))
        except Exception as e:
            logger.error(f"{self.name} exporter failed to mark stream start: {e}", exc_info=True)

    def needs_ttft(self, request_id: str) -> bool:
        """Whether this stream's TTFT update still has to be sent."""
        state = self._request_states.get(request_id)
        return state is not None and not state.ttft_recorded

    def stream_progress(self, record: RequestObservation) -> None:
        """Record TTFT once per stream, as an attribute and a span event."""
        self._active_request_id.set(record.core.request_id)
        state = self._request_states.get(record.core.request_id)
        if state is None or state.ttft_recorded:
            return
        if record.timing.ttft_ms is None or record.timing.first_chunk_time is None:
            return
        try:
            self._set_attributes(state.span, self._backend_attributes(record))
            state.span.add_event(
                "gen_ai.first_token",
                attributes={"llm_proxy.ttft_ms": record.timing.ttft_ms},
                timestamp=_datetime_ns(record.timing.first_chunk_time),
            )
            state.ttft_recorded = True
        except Exception as e:
            logger.error(f"{self.name} exporter failed to record TTFT: {e}", exc_info=True)

    def finish(
        self,
        record: RequestObservation,
        *,
        request: Any,
        response: Any | None = None,
    ) -> None:
        """Fill in output, usage and cost, then end the span."""
        self._active_request_id.set(record.core.request_id)
        state = self._request_states.pop(record.core.request_id, None)
        if state is None:
            return
        try:
            attributes = _request_attributes(record, request)
            attributes.update(_result_attributes(record, request, response))
            attributes.update(self._backend_attributes(record))
            attributes.update(self._content_attributes(record, request, response))
            self._set_attributes(state.span, attributes)
        except Exception as e:
            logger.error(f"{self.name} exporter failed to record result: {e}", exc_info=True)
        finally:
            state.span.end()

    def fail(
        self,
        record: RequestObservation,
        error: Exception,
        *,
        request: Any | None = None,
        streaming: bool = False,
    ) -> None:
        """Mark the span as failed, attach the exception, and end it."""
        self._active_request_id.set(record.core.request_id)
        state = self._request_states.pop(record.core.request_id, None)
        if state is None:
            return
        try:
            attributes = _request_attributes(record, request)
            attributes.update(self._backend_attributes(record))
            attributes["error.type"] = type(error).__name__
            if streaming:
                attributes["llm_proxy.streaming"] = True
            self._set_attributes(state.span, attributes)
            state.span.record_exception(error)
            state.span.set_status(Status(StatusCode.ERROR, str(error)))
        except Exception as e:
            logger.error(f"{self.name} exporter failed to record error: {e}", exc_info=True)
        finally:
            state.span.end()

    # -- client lifetime --------------------------------------------------

    async def flush(self) -> None:
        provider = self._provider
        if provider is None:
            return
        try:
            await asyncio.to_thread(provider.force_flush)
        except Exception as e:
            logger.warning(f"Failed to flush {self.name} tracer provider: {e}")

    async def close(self) -> None:
        provider = self._provider
        self._provider = None
        if provider is None:
            return
        try:
            await asyncio.to_thread(provider.force_flush)
            await asyncio.to_thread(provider.shutdown)
        except Exception as e:
            logger.error(f"Failed to shutdown {self.name} tracer provider: {e}")


__all__ = [
    "LANGSMITH_OTLP_ENDPOINT",
    "LANGSMITH_PROJECT_HEADER",
    "OtlpBackend",
    "OtlpExporter",
    "OtlpProtocol",
    "build_tracer_provider",
    "operation_name",
]
