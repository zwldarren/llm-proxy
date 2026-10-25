"""Langfuse exporter: one observation per request, shaped for the SDK.

The request lifecycle lives in
``tracing/handlers/providers/langfuse/handler.py``; everything that talks to
Langfuse lives here. The exporter reads no ``EventContext``: it receives the
immutable :class:`RequestObservation` plus the typed wire request/response (which
the content registry renders). That keeps the capture object out of the backend,
and the content policy in exactly one place (ADR-0044).

It is a *live* exporter, not a one-call sink: Langfuse needs a span open before
the response headers are written (the trace id they echo) and TTFT recorded while
the stream runs. A future step can reduce this to ``emit(observation)`` once the
content registry renders from the observation's captured bodies instead of the
typed internals.
"""

import asyncio
import contextlib
import re
import threading
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

lazy from langfuse import Langfuse, propagate_attributes
lazy from langfuse.types import TraceContext
lazy from opentelemetry.sdk.trace import TracerProvider
lazy from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from llm_proxy.core.request_type import coerce_request_type
from llm_proxy.observability.content.registry import (
    _endpoint_display_name,
    _format_output_for_langfuse,
    extract_tool_uses,
    observation_profile,
)
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.observation.model import RequestObservation
from llm_proxy.observability.observation.policy import (
    ContentMode,
    SendContent,
    prepare_content,
    sensitive_keys_for,
)

if TYPE_CHECKING:
    from llm_proxy.models.params import GenerationParams

logger = get_logger(__name__)

#: Langfuse trace ids are 32-char lowercase hex (an OTel trace id). Anything
#: else (a proxy-generated UUID, an arbitrary correlation id) cannot be used as
#: a ``trace_context`` and must not be handed to the SDK, which raises on it.
_LANGFUSE_TRACE_ID_RE = re.compile(r"[0-9a-f]{32}\Z")

#: How many finished requests keep their trace ids. The response header that
#: echoes them is written immediately after the end hook, in the same task, so
#: this only has to cover the gap between ``finish``/``fail`` and that write.
#: Bounding it keeps a long-running proxy from accumulating one tuple per traced
#: request for the lifetime of the process.
_MAX_TRACE_ID_ENTRIES = 1024

#: Langfuse validates propagated string attributes (user_id, session_id,
#: trace_name, tags): US-ASCII, at most 200 characters. Invalid values are
#: dropped server-side with a warning, so drop them here instead.
_MAX_PROPAGATED_ATTRIBUTE_LENGTH = 200


def _sanitize_propagated_attribute(value: str | None) -> str | None:
    """Return a value accepted by ``propagate_attributes``, or ``None``."""
    if not value or len(value) > _MAX_PROPAGATED_ATTRIBUTE_LENGTH:
        return None
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return None
    return value


def _langfuse_trace_context(trace_id: str | None) -> TraceContext | None:
    """Build the SDK ``trace_context`` for a caller-provided Langfuse trace id.

    The proxy accepts ``x-langfuse-trace-id`` (or ``x-trace-id``) so a caller can
    nest this request's observation inside the trace it already started. Only a
    canonical 32-char lowercase hex id is usable; the SDK logs and then raises on
    anything else, so non-Langfuse correlation ids are ignored.
    """
    if trace_id and _LANGFUSE_TRACE_ID_RE.prefixmatch(trace_id):
        return {"trace_id": trace_id}
    return None


def build_usage_details(record: RequestObservation) -> dict[str, int] | None:
    """Build Langfuse ``usage_details`` as mutually exclusive buckets.

    Langfuse treats every key as a non-overlapping bucket: ``input`` must exclude
    ``input_*`` values and ``output`` must exclude ``output_*`` values, or the
    tokens are counted (and priced) twice. The proxy's own token counts are
    inclusive — ``prompt_tokens`` already contains cache reads/writes and audio
    tokens, ``completion_tokens`` already contains reasoning and audio tokens —
    so each detail is subtracted from ``input``/``output`` before being emitted
    under its own bucket (see the Langfuse token & cost tracking contract).

    Image tokens stay inside ``input``: Langfuse's image usage key varies by
    model definition, so carving them out could silently price them at zero.
    """
    usage = record.usage.usage
    if usage is None:
        return None

    # Canonical fields only (CONTEXT.md "Canonical usage record"): the record holds
    # one field per fact, and its own counters already include the detail tokens.
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    cache_read = usage.cache_read_input_tokens or (
        prompt_details.cached_tokens if prompt_details is not None else None
    )
    cache_write = usage.cache_creation_input_tokens
    audio_in = prompt_details.audio_tokens if prompt_details is not None else None
    audio_out = completion_details.audio_tokens if completion_details is not None else None
    reasoning = usage.reasoning_tokens

    details: dict[str, int] = {}
    if usage.input_tokens is not None:
        exclusive_input = usage.input_tokens - (
            (cache_read or 0) + (cache_write or 0) + (audio_in or 0)
        )
        details["input"] = max(exclusive_input, 0)
    if usage.output_tokens is not None:
        exclusive_output = usage.output_tokens - ((audio_out or 0) + (reasoning or 0))
        details["output"] = max(exclusive_output, 0)
    if usage.total_tokens is not None:
        details["total"] = usage.total_tokens

    if cache_read:
        details["cache_read_input_tokens"] = cache_read
    if cache_write:
        details["cache_creation_input_tokens"] = cache_write
    if audio_in:
        details["input_audio_tokens"] = audio_in
    if audio_out:
        details["output_audio_tokens"] = audio_out
    if reasoning:
        details["output_reasoning_tokens"] = reasoning

    return details if details else None


def build_cost_details(record: RequestObservation) -> dict[str, float] | None:
    """Build Langfuse ``cost_details`` from the observation's cost.

    Only ``total`` is emitted: the proxy computes a single USD total and does not
    split it per usage bucket. The provider-reported cost is informational (it is
    the same total when it wins cost precedence) and is carried in metadata
    instead of as a phantom cost bucket.
    """
    if record.usage.cost_usd is None:
        return None
    return {"total": record.usage.cost_usd}


def build_embedding_usage_details(record: RequestObservation) -> dict[str, int] | None:
    """Build Langfuse ``usage_details`` for an embedding observation.

    An embedding call bills input tokens only, so the ``output`` buckets a
    ``generation`` reports would show up as zero rows that price nothing (and
    that the call never produced).
    """
    details = build_usage_details(record)
    if details is None:
        return None
    return {key: value for key, value in details.items() if not key.startswith("output")}


def build_metadata(record: RequestObservation) -> dict[str, Any]:
    """Build Langfuse observation metadata from the observation."""
    metadata: dict[str, Any] = {
        "request_id": record.core.request_id,
        "trace_id": record.core.trace_id,
    }
    if record.routing.provider:
        metadata["provider"] = record.routing.provider
    if record.routing.model:
        metadata["model"] = record.routing.model
    if record.core.request_type:
        metadata["request_type"] = str(record.core.request_type)

    endpoint = record.core.endpoint
    if endpoint:
        metadata["endpoint"] = endpoint
        endpoint_name = _endpoint_display_name(endpoint)
        if endpoint_name:
            metadata["endpoint_name"] = endpoint_name

    streaming = record.content is not None and record.content.streaming
    if streaming:
        metadata["stream"] = True
    if record.timing.ttft_ms is not None:
        metadata["ttft_ms"] = record.timing.ttft_ms
    if record.timing.latency_ms is not None:
        metadata["latency_ms"] = round(record.timing.latency_ms, 3)
    if record.outcome.error_message:
        metadata["error_message"] = record.outcome.error_message

    # Token/usage facts that are not emitted as ``usage_details`` buckets (see
    # ``build_usage_details``) stay visible here.
    usage = record.usage.usage
    image_input = (
        usage.prompt_tokens_details.image_tokens
        if usage is not None and usage.prompt_tokens_details is not None
        else None
    )
    if image_input is not None:
        metadata["image_input_tokens"] = image_input
    if record.usage.cache_savings_usd is not None:
        metadata["cache_savings_usd"] = record.usage.cache_savings_usd
    if record.usage.provider_reported_cost is not None:
        metadata["provider_reported_cost"] = record.usage.provider_reported_cost
    if record.routing.provider_model_name:
        metadata["provider_model_name"] = record.routing.provider_model_name

    return metadata


def build_model_parameters(params: GenerationParams | None) -> dict[str, Any] | None:
    """Build model parameters dict for Langfuse generation.

    Extracts common generation parameters (temperature, max_tokens, top_p,
    stop sequences, etc.) from the GenerationParams object.
    """
    if params is None:
        return None

    common = params.common
    if common is None:
        return None

    result: dict[str, Any] = {}
    for key in (
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "seed",
    ):
        val = getattr(common, key, None)
        if val is not None:
            result[key] = val

    return result if result else None


@dataclass
class _RequestState:
    """Per-request Langfuse state stored on the exporter.

    Entries are keyed by request id and removed when the request ends. Plain
    dicts are required here rather than ``ContextVar``s: the request's start hook
    runs in the pipeline task (``keepalive.await_with_*`` wraps the pipeline in
    ``ensure_future``) while a streaming request's chunk/end hooks run in the task
    Starlette spawned for the response body — a different context, which never
    sees a ``ContextVar.set()`` done in the pipeline task. Keying one
    instance-level dict by request id survives that boundary; a dict mutation
    between ``await``s on a single event loop is atomic.
    """

    observation: Any
    ttft_recorded: bool = False


#: Per-project TracerProvider holding a configured sampling rate. The SDK installs
#: a sampler on the process-wide OpenTelemetry TracerProvider, so two users on
#: different Langfuse projects would otherwise share whichever rate was configured
#: first (and the later project's rate would be silently ignored).
_SAMPLED_TRACER_PROVIDERS: dict[str, TracerProvider] = {}
_SAMPLED_TRACER_PROVIDERS_LOCK = threading.Lock()


def sampled_tracer_provider(public_key: str, sample_rate: float) -> TracerProvider:
    """Return the process-wide sampled ``TracerProvider`` for a Langfuse project.

    Only used when a rate below 1.0 is configured: without sampling the shared
    default provider behaves identically. Keyed by public key so repeated handler
    rebuilds (config changes) reuse one provider instead of leaking a span-processor
    thread each time.
    """
    with _SAMPLED_TRACER_PROVIDERS_LOCK:
        provider = _SAMPLED_TRACER_PROVIDERS.get(public_key)
        if provider is None:
            provider = TracerProvider(sampler=TraceIdRatioBased(sample_rate))
            _SAMPLED_TRACER_PROVIDERS[public_key] = provider
        return provider


class LangfuseExporter:
    """Shape observations and drive the Langfuse SDK for one handler instance."""

    #: The operator's content choice: ``full`` exports verbatim content,
    #: ``inherit`` follows the request's policy, ``none`` never exports content.
    send_content: SendContent

    def __init__(
        self,
        *,
        client: Langfuse,
        base_url: str,
        version: str = "1.0",
        send_content: SendContent = "full",
        name: str = "langfuse",
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.client = client
        self.version = version
        self.send_content = send_content
        # The id of the request whose generation the *current* context created
        # stays in a ContextVar because it is only read back in the same task
        # that set it (the header-writing seam after ``on_request_end``).
        self._active_request_id: ContextVar[str | None] = ContextVar(
            f"langfuse_active_request_id_{id(self)}", default=None
        )
        self._request_states: dict[str, _RequestState] = {}
        self._request_trace_ids: OrderedDict[str, tuple[str, str]] = OrderedDict()

    # -- content policy ---------------------------------------------------

    def _effective_mode(self, record: RequestObservation) -> ContentMode:
        """What this exporter may export for this request.

        ``full`` (the default) exports verbatim content. ``inherit`` follows the
        request's captured logging policy — the same one the database log obeys —
        so a deployment that wants the body switches to govern tracing too can
        choose it. ``none`` never exports content.
        """
        if self.send_content == "none":
            return "none"
        if self.send_content == "full":
            return "full"
        return record.policy.mode

    @staticmethod
    def _profile(record: RequestObservation, request: Any) -> Any:
        """The request type's profile, preferring what the request itself says.

        ``record.core.request_type`` is ``None`` when the pipeline never stamped
        one; an explicit request type (e.g. an embedding) must still win, and a
        missing one must fall back to the metadata-only profile rather than to
        the log row's "chat" default.
        """
        request_type = getattr(request, "request_type", None) or coerce_request_type(
            record.core.request_type
        )
        return observation_profile(request_type)

    def _prepare(self, value: Any, record: RequestObservation) -> Any | None:
        """Project a payload through this request's effective content mode."""
        mode = self._effective_mode(record)
        keys = record.policy.sensitive_keys if mode == "masked" else frozenset()
        if mode == "masked" and not keys:
            keys = sensitive_keys_for(None)
        return prepare_content(value, mode=mode, sensitive_keys=keys)

    # -- per-request state ------------------------------------------------

    def _set_active_request_id(self, request_id: str) -> None:
        self._active_request_id.set(request_id)

    def _set_state(self, request_id: str, state: _RequestState) -> None:
        self._request_states[request_id] = state

    def _pop_observation(self, request_id: str) -> Any | None:
        state = self._request_states.pop(request_id, None)
        return state.observation if state else None

    def _set_trace_ids(self, request_id: str, trace_id: str, observation_id: str) -> None:
        self._request_trace_ids[request_id] = (trace_id, observation_id)
        self._request_trace_ids.move_to_end(request_id)
        while len(self._request_trace_ids) > _MAX_TRACE_ID_ENTRIES:
            self._request_trace_ids.popitem(last=False)

    def _discard_trace_ids(self, request_id: str) -> None:
        self._request_trace_ids.pop(request_id, None)

    def _get_current_trace_ids(self) -> tuple[str, str] | None:
        request_id = self._active_request_id.get()
        if request_id is None:
            return None
        return self._request_trace_ids.get(request_id)

    def get_trace_id(self) -> str | None:
        ids = self._get_current_trace_ids()
        return ids[0] if ids is not None else None

    def get_observation_id(self) -> str | None:
        ids = self._get_current_trace_ids()
        return ids[1] if ids is not None else None

    # -- lifecycle --------------------------------------------------------

    def _span_name(self, record: RequestObservation, request: Any) -> str:
        """Build a span name from the requested endpoint path.

        Falls back to the operation name and model if the endpoint is unknown.
        """
        endpoint_name = _endpoint_display_name(record.core.endpoint)
        if endpoint_name:
            return endpoint_name
        request_type = coerce_request_type(record.core.request_type)
        operation = request_type.value if request_type is not None else "request"
        model = getattr(request, "model", None)
        return f"{operation} {model or 'unknown'}"

    def open(self, record: RequestObservation, request: Any) -> None:
        """Open the Langfuse observation for this request.

        Runs at request start because the trace/observation ids are needed while
        the response headers are written, and the stream's TTFT is recorded on
        this same observation.
        """
        self._set_active_request_id(record.core.request_id)
        client = self.client
        if client is None:
            return
        try:
            span_name = self._span_name(record, request)
            # One observation profile per request type: which Langfuse
            # observation type to open and how to render this request's input.
            profile = self._profile(record, request)
            input_data = self._prepare(profile.build_input(request), record)
            model = getattr(request, "model", None)
            model_parameters = build_model_parameters(getattr(request, "params", None))

            observation_kwargs: dict[str, Any] = {
                "name": span_name,
                "input": input_data,
                "model": model,
                "model_parameters": model_parameters,
                "version": self.version,
                "end_on_exit": False,
                "trace_context": _langfuse_trace_context(record.core.trace_id),
            }

            # Trace-level attributes (name, user, session) are not accepted by
            # ``observation.update()`` in SDK v4 — it swallows unknown keyword
            # arguments — so they are propagated when the span is created.
            #
            # ``end_on_exit=False`` keeps the observation open after the context
            # manager exits; it is ended explicitly in ``finish``/``fail``.
            # The two ``as_type`` call sites are kept literal on purpose: the
            # SDK's overloads are keyed on the literal, and a variable or ternary
            # there matches no overload at all.
            start_observation = client.start_as_current_observation
            if profile.as_type == "embedding":
                observation = start_observation(as_type="embedding", **observation_kwargs)
            else:
                observation = start_observation(as_type="generation", **observation_kwargs)

            with (
                propagate_attributes(
                    trace_name=span_name,
                    user_id=_sanitize_propagated_attribute(record.identity.user),
                    session_id=_sanitize_propagated_attribute(record.identity.session_id),
                ),
                observation as observation,
            ):
                self._set_state(record.core.request_id, _RequestState(observation))
                self._set_trace_ids(record.core.request_id, observation.trace_id, observation.id)
        except Exception as e:
            self._pop_observation(record.core.request_id)
            self._discard_trace_ids(record.core.request_id)
            logger.error(f"Langfuse exporter failed to start generation: {e}", exc_info=True)

    def stream_opened(self, record: RequestObservation) -> None:
        self._set_active_request_id(record.core.request_id)
        state = self._request_states.get(record.core.request_id)
        if state is None:
            return
        try:
            metadata = build_metadata(record)
            metadata["streaming"] = True
            state.observation.update(metadata=metadata)
        except Exception as e:
            logger.error(f"Langfuse exporter failed to mark stream start: {e}", exc_info=True)

    def needs_ttft(self, request_id: str) -> bool:
        """Whether the TTFT update still has to be sent for this stream."""
        state = self._request_states.get(request_id)
        return state is not None and not state.ttft_recorded

    def stream_progress(self, record: RequestObservation) -> None:
        """Record TTFT and the completion start time once per stream."""
        self._set_active_request_id(record.core.request_id)
        state = self._request_states.get(record.core.request_id)
        if state is None or state.ttft_recorded:
            return
        if record.timing.ttft_ms is None or record.timing.first_chunk_time is None:
            return
        try:
            metadata = build_metadata(record)
            metadata["ttft_ms"] = record.timing.ttft_ms
            state.observation.update(
                metadata=metadata,
                completion_start_time=record.timing.first_chunk_time,
            )
            state.ttft_recorded = True
        except Exception as e:
            logger.error(f"Langfuse exporter failed to record stream chunk: {e}", exc_info=True)

    def finish(
        self,
        record: RequestObservation,
        *,
        request: Any,
        response: Any | None = None,
    ) -> None:
        """Fill in output, usage, cost and metadata, then end the observation.

        ``response`` is the typed non-streaming result; a streaming request
        carries its accumulated blocks on the observation instead.
        """
        self._set_active_request_id(record.core.request_id)
        request_id = record.core.request_id
        # Deliberately not gated on the handler being enabled: a handler released
        # while this request was in flight must still end the observation it
        # already created, or the trace is never exported.
        observation = self._pop_observation(request_id)
        if observation is None:
            return
        try:
            profile = self._profile(record, request)
            if response is not None:
                output_data = self._prepare(profile.build_output(response), record)
                tool_blocks = getattr(response, "output", None)
            else:
                tool_blocks = (
                    record.content.stream_output_blocks if record.content is not None else None
                )
                output_data = self._prepare(self._format_stream_output(tool_blocks), record)

            usage_details = (
                build_embedding_usage_details(record)
                if profile.as_type == "embedding"
                else build_usage_details(record)
            )
            cost_details = build_cost_details(record)
            metadata = build_metadata(record)

            model = getattr(response, "model", None) or getattr(request, "model", None)

            update_kwargs: dict[str, Any] = {"metadata": metadata}
            if output_data is not None:
                update_kwargs["output"] = output_data
            if usage_details is not None:
                update_kwargs["usage_details"] = usage_details
            if cost_details is not None:
                update_kwargs["cost_details"] = cost_details
            if model:
                update_kwargs["model"] = model

            observation.update(**update_kwargs)
            self._record_tool_observations(
                observation,
                extract_tool_uses(tool_blocks),
                record,
            )
            observation.end()
        except Exception as e:
            self._fail_update("end observation", e, observation)

    def fail(
        self,
        record: RequestObservation,
        error: Exception,
        *,
        streaming: bool = False,
    ) -> None:
        """Mark the observation as failed and end it."""
        self._set_active_request_id(record.core.request_id)
        observation = self._pop_observation(record.core.request_id)
        if observation is None:
            return
        try:
            metadata = build_metadata(record)
            metadata["error_type"] = type(error).__name__
            if streaming:
                metadata["streaming"] = True
            observation.update(
                level="ERROR",
                status_message=str(error),
                metadata=metadata,
            )
            observation.end()
        except Exception as e:
            self._fail_update("record error", e, observation)

    # -- shaping helpers --------------------------------------------------

    @staticmethod
    def _format_stream_output(output_blocks: list[Any] | None) -> dict[str, Any] | None:
        """Render accumulated streaming blocks as the observation's output."""
        if not output_blocks:
            return None
        try:
            return _format_output_for_langfuse(output_blocks)
        except TypeError, ValueError:
            return None

    def _record_tool_observations(
        self,
        observation: Any,
        tool_uses: list[dict[str, Any]],
        record: RequestObservation,
    ) -> None:
        """Emit one Langfuse ``tool`` observation per tool call in the response.

        The proxy is stateless across requests, so it only observes the LLM's
        tool *invocations* (name + arguments), not the client-side execution
        results. Recording these as dedicated ``tool`` observations makes them
        appear in the Langfuse trace tree / agent graph and tool filtering
        instead of being buried inside the observation output.
        """
        if not tool_uses or self._effective_mode(record) == "none":
            return
        for tool_use in tool_uses:
            name = tool_use.get("name") or "tool"
            try:
                tool = observation.start_observation(
                    name=name,
                    as_type="tool",
                    input=self._prepare(tool_use.get("input"), record),
                    metadata={
                        "tool_call_id": tool_use.get("id"),
                        # Proxy does not observe tool results; they arrive in a
                        # subsequent request, so output is intentionally unset.
                        "result_observed": False,
                    },
                )
                tool.end()
            except Exception as e:
                logger.debug(f"Langfuse exporter failed to record tool observation: {e}")

    def _fail_update(self, stage: str, error: Exception, observation: Any) -> None:
        """Log a failed observation update and still end the observation.

        A second failure while ending must not mask the original error, so it is
        suppressed after the log.
        """
        logger.error(f"Langfuse exporter failed to {stage}: {error}", exc_info=True)
        with contextlib.suppress(Exception):
            observation.end()

    # -- client lifetime --------------------------------------------------

    async def flush(self) -> None:
        client = self.client
        if client is None:
            return
        try:
            await asyncio.to_thread(client.flush)
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse client for {self.name}: {e}")

    async def close(self) -> None:
        client = self.client
        self.client = None
        if client is None:
            return
        try:
            await asyncio.to_thread(client.flush)
            await asyncio.to_thread(client.shutdown)
        except Exception as e:
            logger.error(f"Failed to shutdown Langfuse client for {self.name}: {e}")


__all__ = [
    "LangfuseExporter",
    "build_cost_details",
    "build_embedding_usage_details",
    "build_metadata",
    "build_model_parameters",
    "build_usage_details",
    "sampled_tracer_provider",
]
