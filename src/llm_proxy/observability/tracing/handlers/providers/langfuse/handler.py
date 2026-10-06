"""Langfuse tracing handler.

Exports traces to Langfuse using the official Langfuse Python SDK. Each request
is recorded as a ``generation`` observation with structured input, output,
usage, and metadata. Per-request state is stored in ``ContextVar`` dictionaries
keyed by ``context.request_id`` so the handler remains safe under concurrent
async load.
"""

import asyncio
import contextlib
import re
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langfuse import Langfuse, propagate_attributes
from langfuse.types import TraceContext
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from llm_proxy.core.request_type import RequestType
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.tracing.handlers.base import TracingHandler
from llm_proxy.observability.tracing.handlers.providers.langfuse.attributes import (
    _endpoint_display_name,
    _format_output_for_langfuse,
    build_cost_details,
    build_metadata,
    build_model_parameters,
    build_request_input_data,
    build_response_output_data,
    build_usage_details,
    extract_tool_uses,
)

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.models import InternalRequest, InternalResponse
    from llm_proxy.observability.event_context import EventContext

logger = get_logger(__name__)

#: Langfuse trace ids are 32-char lowercase hex (an OTel trace id). Anything
#: else (a proxy-generated UUID, an arbitrary correlation id) cannot be used as
#: a ``trace_context`` and must not be handed to the SDK, which raises on it.
_LANGFUSE_TRACE_ID_RE = re.compile(r"[0-9a-f]{32}\Z")

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
    if trace_id and _LANGFUSE_TRACE_ID_RE.match(trace_id):
        return {"trace_id": trace_id}
    return None


@dataclass
class _RequestState:
    """Per-request Langfuse state stored in the handler's ContextVar.

    Keeping this per-request (instead of a handler-wide set) means one request's
    bookkeeping can never leak into another's under concurrent async load.
    """

    generation: Any
    ttft_recorded: bool = False


#: Per-project TracerProvider holding a configured sampling rate. The SDK installs
#: a sampler on the process-wide OpenTelemetry TracerProvider, so two users on
#: different Langfuse projects would otherwise share whichever rate was configured
#: first (and the later project's rate would be silently ignored).
_SAMPLED_TRACER_PROVIDERS: dict[str, TracerProvider] = {}
_SAMPLED_TRACER_PROVIDERS_LOCK = threading.Lock()


def _sampled_tracer_provider(public_key: str, sample_rate: float) -> TracerProvider:
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


class LangfuseTracingHandler(TracingHandler):
    """Tracing handler that exports request traces to Langfuse.

    Uses the official Langfuse Python SDK. A request is represented as a single
    ``generation`` observation. Streaming requests accumulate output and usage
    on the same observation. Self-hosted Langfuse is supported by providing a
    custom ``base_url``.
    """

    provider_name = "langfuse"
    required_settings = ["public_key", "secret_key"]
    optional_settings = ["base_url", "timeout", "sample_rate", "version"]
    description = "Export traces to Langfuse. Requires public and secret keys."
    field_metadata = [
        {
            "name": "public_key",
            "type": "text",
            "required": True,
            "description": "Langfuse public key",
        },
        {
            "name": "secret_key",
            "type": "password",
            "required": True,
            "description": "Langfuse secret key",
        },
        {
            "name": "base_url",
            "type": "text",
            "required": False,
            "default": "https://cloud.langfuse.com",
            "description": "Langfuse base URL (cloud region or self-hosted)",
        },
        {
            "name": "timeout",
            "type": "number",
            "required": False,
            "description": "Langfuse SDK request timeout in seconds",
        },
        {
            "name": "sample_rate",
            "type": "number",
            "required": False,
            "description": "Sampling rate (0.0-1.0) for trace export",
        },
        {
            "name": "version",
            "type": "text",
            "required": False,
            "default": "1.0",
            "description": "Version tag for traces",
        },
    ]

    _DEFAULT_BASE_URL = "https://cloud.langfuse.com"

    @classmethod
    def _resolve_host(cls, settings: dict[str, Any]) -> str:
        base_url = settings.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            return base_url.rstrip("/")
        return cls._DEFAULT_BASE_URL

    @classmethod
    def validate_config(cls, settings: dict[str, Any]) -> bool:
        """Validate that public and secret keys are non-empty strings."""
        public_key = settings.get("public_key")
        secret_key = settings.get("secret_key")
        return (
            isinstance(public_key, str)
            and bool(public_key)
            and isinstance(secret_key, str)
            and bool(secret_key)
        )

    @classmethod
    def create_handler(
        cls,
        settings: dict[str, Any],
        config_manager: DatabaseConfigManager | None = None,
    ) -> LangfuseTracingHandler:
        """Create a Langfuse handler from configuration.

        Args:
            settings: Provider-specific settings dictionary
            config_manager: Optional config manager (unused, kept for API compatibility)

        Returns:
            Configured LangfuseTracingHandler instance

        Raises:
            ValueError: If public_key or secret_key is missing or empty
        """
        public_key = settings.get("public_key")
        secret_key = settings.get("secret_key")
        if not isinstance(public_key, str) or not public_key:
            raise ValueError(
                "Langfuse handler requires a non-empty string 'public_key' in settings"
            )
        if not isinstance(secret_key, str) or not secret_key:
            raise ValueError(
                "Langfuse handler requires a non-empty string 'secret_key' in settings"
            )

        base_url = cls._resolve_host(settings)
        timeout = settings.get("timeout")
        sample_rate = settings.get("sample_rate")

        client_kwargs: dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
            "base_url": base_url,
        }
        if timeout is not None:
            timeout_val = float(timeout)
            if timeout_val <= 0:
                raise ValueError("timeout must be a positive number")
            client_kwargs["timeout"] = timeout_val
        if sample_rate is not None:
            sample_rate_val = float(sample_rate)
            if not (0.0 <= sample_rate_val <= 1.0):
                raise ValueError("sample_rate must be between 0.0 and 1.0")
            client_kwargs["sample_rate"] = sample_rate_val
            if sample_rate_val < 1.0:
                # Scope the sampler to this project; the SDK's default would put it
                # on the process-wide provider and apply it to every other project.
                client_kwargs["tracer_provider"] = _sampled_tracer_provider(
                    public_key, sample_rate_val
                )

        try:
            client = Langfuse(**client_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            raise

        return cls(
            enabled=True,
            name=settings.get("name", "langfuse"),
            client=client,
            base_url=base_url,
            version=settings.get("version", "1.0"),
        )

    def __init__(
        self,
        *,
        enabled: bool = True,
        name: str = "langfuse",
        client: Langfuse,
        base_url: str,
        version: str = "1.0",
    ) -> None:
        super().__init__(enabled=enabled)
        self.name = name
        self._client = client
        self._base_url = base_url
        self._version = version

        # Per-request state lives in ContextVars so concurrent requests never
        # share handler instance attributes.
        self._active_request_id: ContextVar[str | None] = ContextVar(
            f"langfuse_active_request_id_{id(self)}", default=None
        )
        self._request_states: ContextVar[dict[str, _RequestState] | None] = ContextVar(
            f"langfuse_states_{id(self)}", default=None
        )
        self._request_trace_ids: ContextVar[dict[str, tuple[str, str]] | None] = ContextVar(
            f"langfuse_trace_ids_{id(self)}", default=None
        )

    def _set_active_request_id(self, context: EventContext) -> None:
        self._active_request_id.set(context.request_id)

    def _get_state_for_request(self, request_id: str) -> _RequestState | None:
        return (self._request_states.get() or {}).get(request_id)

    def _set_state_for_request(self, request_id: str, state: _RequestState) -> None:
        current = dict(self._request_states.get() or {})
        current[request_id] = state
        self._request_states.set(current)

    def _pop_generation_for_request(self, request_id: str) -> Any | None:
        """Remove a request's state and return its generation (``None`` if absent)."""
        current = dict(self._request_states.get() or {})
        state = current.pop(request_id, None)
        self._request_states.set(current)
        return state.generation if state else None

    def _fail_update(self, stage: str, error: Exception, generation: Any) -> None:
        """Log a failed observation update and still end the generation.

        A second failure while ending must not mask the original error, so it is
        suppressed after the log.
        """
        logger.error(f"Langfuse handler failed to {stage}: {error}", exc_info=True)
        with contextlib.suppress(Exception):
            generation.end()

    def _set_trace_ids_for_request(
        self, request_id: str, trace_id: str, observation_id: str
    ) -> None:
        current = dict(self._request_trace_ids.get() or {})
        current[request_id] = (trace_id, observation_id)
        self._request_trace_ids.set(current)

    def _discard_trace_ids_for_request(self, request_id: str) -> None:
        current = dict(self._request_trace_ids.get() or {})
        current.pop(request_id, None)
        self._request_trace_ids.set(current)

    def _get_current_trace_ids(self) -> tuple[str, str] | None:
        """Return the active request's ``(trace_id, observation_id)``, if any."""
        request_id = self._active_request_id.get()
        if request_id is None:
            return None
        return (self._request_trace_ids.get() or {}).get(request_id)

    def _get_current_trace_id(self) -> str | None:
        ids = self._get_current_trace_ids()
        return ids[0] if ids is not None else None

    def _get_current_observation_id(self) -> str | None:
        ids = self._get_current_trace_ids()
        return ids[1] if ids is not None else None

    async def on_request_start(
        self,
        request: InternalRequest,
        context: EventContext,
    ) -> None:
        if not self._enabled:
            return
        self._set_active_request_id(context)
        client = self._client
        if client is None:
            return
        try:
            span_name = self._build_span_name(request, context)
            input_data = build_request_input_data(request)
            model = getattr(request, "model", None)
            model_parameters = build_model_parameters(getattr(request, "params", None))

            # Trace-level attributes (name, user, session) are not accepted by
            # ``observation.update()`` in SDK v4 — it swallows unknown keyword
            # arguments — so they must be propagated when the span is created.
            #
            # Create a generation that persists across async lifecycle calls.
            # ``end_on_exit=False`` keeps the observation open after the context
            # manager exits; we end it explicitly in on_request_end/on_stream_end.
            with (
                propagate_attributes(
                    trace_name=span_name,
                    user_id=_sanitize_propagated_attribute(context.user_id),
                    session_id=_sanitize_propagated_attribute(context.session_id),
                ),
                client.start_as_current_observation(
                    as_type="generation",
                    name=span_name,
                    input=input_data,
                    model=model,
                    model_parameters=model_parameters,
                    version=self._version,
                    end_on_exit=False,
                    trace_context=_langfuse_trace_context(context.trace_id),
                ) as generation,
            ):
                self._set_state_for_request(context.request_id, _RequestState(generation))
                self._set_trace_ids_for_request(
                    context.request_id, generation.trace_id, generation.id
                )
        except Exception as e:
            self._pop_generation_for_request(context.request_id)
            self._discard_trace_ids_for_request(context.request_id)
            logger.error(f"Langfuse handler failed to start generation: {e}", exc_info=True)

    async def on_request_end(
        self,
        request: InternalRequest,
        response: InternalResponse,
        context: EventContext,
    ) -> None:
        self._set_active_request_id(context)
        request_id = context.request_id
        # Deliberately not gated on ``self._enabled``: a handler that was released
        # while this request was in flight must still end the generation it
        # already created, or the trace is never exported.
        generation = self._pop_generation_for_request(request_id)
        if generation is None:
            return
        try:
            output_data = build_response_output_data(response)
            usage_details = build_usage_details(context)
            cost_details = build_cost_details(context)
            metadata = build_metadata(context)

            model = getattr(response, "model", None) or getattr(request, "model", None)

            update_kwargs: dict[str, Any] = {
                "metadata": metadata,
            }
            if output_data is not None:
                update_kwargs["output"] = output_data
            if usage_details is not None:
                update_kwargs["usage_details"] = usage_details
            if cost_details is not None:
                update_kwargs["cost_details"] = cost_details
            if model:
                update_kwargs["model"] = model

            generation.update(**update_kwargs)
            self._record_tool_observations(
                generation, extract_tool_uses(getattr(response, "output", None))
            )
            generation.end()
        except Exception as e:
            self._fail_update("end generation", e, generation)

    async def on_error(
        self,
        request: InternalRequest,
        error: Exception,
        context: EventContext,
    ) -> None:
        self._set_active_request_id(context)
        request_id = context.request_id
        generation = self._pop_generation_for_request(request_id)
        if generation is None:
            return
        try:
            metadata = build_metadata(context)
            metadata["error_type"] = type(error).__name__
            generation.update(
                level="ERROR",
                status_message=str(error),
                metadata=metadata,
            )
            generation.end()
        except Exception as e:
            self._fail_update("record error", e, generation)

    async def on_stream_start(
        self,
        request: InternalRequest,
        context: EventContext,
    ) -> None:
        self._set_active_request_id(context)
        state = self._get_state_for_request(context.request_id)
        if state is None:
            return
        try:
            metadata = build_metadata(context)
            metadata["streaming"] = True
            state.generation.update(metadata=metadata)
        except Exception as e:
            logger.error(f"Langfuse handler failed to mark stream start: {e}", exc_info=True)

    async def on_stream_chunk(
        self,
        request: InternalRequest,
        chunk: str,
        context: EventContext,
    ) -> None:
        self._set_active_request_id(context)
        state = self._get_state_for_request(context.request_id)
        if state is None:
            return
        try:
            if (
                context.first_chunk_time is not None
                and context.ttft_ms is not None
                and not state.ttft_recorded
            ):
                metadata = build_metadata(context)
                metadata["ttft_ms"] = context.ttft_ms
                state.generation.update(
                    metadata=metadata,
                    completion_start_time=context.first_chunk_time,
                )
                state.ttft_recorded = True
        except Exception as e:
            logger.error(f"Langfuse handler failed to record stream chunk: {e}", exc_info=True)

    async def on_stream_end(
        self,
        request: InternalRequest,
        context: EventContext,
        error: Exception | None = None,
    ) -> None:
        self._set_active_request_id(context)
        request_id = context.request_id
        # As in ``on_request_end``: a released handler must still close the
        # generation it created so the trace reaches Langfuse.
        generation = self._pop_generation_for_request(request_id)
        if generation is None:
            return
        try:
            if error is not None:
                metadata = build_metadata(context)
                metadata["error_type"] = type(error).__name__
                metadata["streaming"] = True
                generation.update(
                    level="ERROR",
                    status_message=str(error),
                    metadata=metadata,
                )
                generation.end()
                return

            update_kwargs: dict[str, Any] = {}
            output = self._extract_stream_output(context)
            if output is not None:
                update_kwargs["output"] = output

            usage_details = build_usage_details(context)
            if usage_details is not None:
                update_kwargs["usage_details"] = usage_details

            cost_details = build_cost_details(context)
            if cost_details is not None:
                update_kwargs["cost_details"] = cost_details

            metadata = build_metadata(context)
            metadata["streaming"] = True
            update_kwargs["metadata"] = metadata

            generation.update(**update_kwargs)
            self._record_tool_observations(
                generation, extract_tool_uses(self._extract_stream_output_blocks(context))
            )
            generation.end()
        except Exception as e:
            self._fail_update("end stream generation", e, generation)

    def _build_span_name(self, request: InternalRequest, context: EventContext) -> str:
        """Build a span name from the requested endpoint path.

        Falls back to the operation name and model if the endpoint is unknown.
        """
        endpoint_name = _endpoint_display_name(context.metadata.get("endpoint"))
        if endpoint_name:
            return endpoint_name
        request_type = getattr(request, "request_type", RequestType.CHAT)
        if isinstance(request_type, RequestType):
            operation = request_type.value
        else:
            try:
                operation = RequestType(request_type).value
            except ValueError:
                operation = str(request_type)
        model = getattr(request, "model", None)
        return f"{operation} {model or 'unknown'}"

    def _extract_stream_output(self, context: EventContext) -> dict[str, Any] | None:
        """Extract accumulated output from a streaming transformer, if any."""
        output = self._extract_stream_output_blocks(context)
        if not output:
            return None
        try:
            return _format_output_for_langfuse(output)
        except TypeError, ValueError:
            return None

    def _extract_stream_output_blocks(self, context: EventContext) -> list[Any] | None:
        """Return the raw accumulated output blocks from a streaming transformer.

        Unlike ``_extract_stream_output``, this returns the unformatted block list
        so callers (e.g. tool-use extraction) can inspect block types directly.
        """
        transformer = getattr(context, "transformer", None)
        if transformer is None:
            return None
        accumulated = getattr(transformer, "get_accumulated_output", None)
        if accumulated is None or not callable(accumulated):
            return None
        try:
            output = accumulated()
        except Exception:
            return None
        return output or None

    def _record_tool_observations(
        self,
        generation: Any,
        tool_uses: list[dict[str, Any]],
    ) -> None:
        """Emit one Langfuse ``tool`` observation per tool call in the response.

        The proxy is stateless across requests, so it only observes the LLM's
        tool *invocations* (name + arguments), not the client-side execution
        results. Recording these as dedicated ``tool`` observations makes them
        appear in the Langfuse trace tree / agent graph and tool filtering
        instead of being buried inside the generation output.
        """
        if not tool_uses:
            return
        for tool_use in tool_uses:
            name = tool_use.get("name") or "tool"
            try:
                tool = generation.start_observation(
                    name=name,
                    as_type="tool",
                    input=tool_use.get("input"),
                    metadata={
                        "tool_call_id": tool_use.get("id"),
                        # Proxy does not observe tool results; they arrive in a
                        # subsequent request, so output is intentionally unset.
                        "result_observed": False,
                    },
                )
                tool.end()
            except Exception as e:
                logger.debug(f"Langfuse handler failed to record tool observation: {e}")

    def get_trace_id(self) -> str | None:
        return self._get_current_trace_id()

    def get_observation_id(self) -> str | None:
        return self._get_current_observation_id()

    def get_trace_header_name(self) -> str:
        return "x-trace-id"

    async def release(self) -> None:
        """Release the handler from a live registry without tearing down the SDK.

        Called when the owning registry is dropped (the user changed their tracing
        config, or a peer worker did). ``LangfuseResourceManager`` is a
        process-wide singleton keyed by public key, so ``client.shutdown()`` here
        would stop the media/score consumers for *every* handler sharing the
        project — and, because a client rebuilt from the same singletons reuses
        the stopped resource manager, degrade the user's tracing until restart.

        Flush instead and keep the client usable: requests already in flight hold
        this handler and must be able to end their generation. The client is
        garbage-collected once the old registry is dropped.
        """
        client = self._client
        if client is None:
            return
        try:
            await asyncio.to_thread(client.flush)
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse client for {self.name}: {e}")

    async def shutdown(self) -> None:
        client = self._client
        self._client = None
        self._enabled = False
        if client is None:
            return
        try:
            await asyncio.to_thread(client.flush)
            await asyncio.to_thread(client.shutdown)
        except Exception as e:
            logger.error(f"Failed to shutdown Langfuse client for {self.name}: {e}")
