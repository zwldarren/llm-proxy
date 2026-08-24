"""Langfuse tracing handler.

Exports traces to Langfuse using the official Langfuse Python SDK. Each request
is recorded as a ``generation`` observation with structured input, output,
usage, and metadata. Per-request state is stored in ``ContextVar`` dictionaries
keyed by ``context.request_id`` so the handler remains safe under concurrent
async load.
"""

import asyncio
import contextlib
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from langfuse import Langfuse

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
        self._request_generations: ContextVar[dict[str, Any] | None] = ContextVar(
            f"langfuse_generations_{id(self)}", default=None
        )
        self._request_trace_ids: ContextVar[dict[str, tuple[str, str]] | None] = ContextVar(
            f"langfuse_trace_ids_{id(self)}", default=None
        )
        # Track request IDs for which TTFT has already been recorded to avoid
        # redundant updates on every streaming chunk.
        self._recorded_ttft: set[str] = set()

    def _set_active_request_id(self, context: EventContext) -> None:
        self._active_request_id.set(context.request_id)

    def _get_generation_for_request(self, request_id: str) -> Any | None:
        current = self._request_generations.get()
        if current is None:
            return None
        return current.get(request_id)

    def _set_generation_for_request(self, request_id: str, generation: Any) -> None:
        current = dict(self._request_generations.get() or {})
        current[request_id] = generation
        self._request_generations.set(current)

    def _pop_generation_for_request(self, request_id: str) -> Any | None:
        current = dict(self._request_generations.get() or {})
        generation = current.pop(request_id, None)
        self._request_generations.set(current)
        return generation

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

    def _get_current_trace_id(self) -> str | None:
        request_id = self._active_request_id.get()
        if request_id is None:
            return None
        current = self._request_trace_ids.get()
        if current is None:
            return None
        ids = current.get(request_id)
        if ids is None:
            return None
        return ids[0]

    def _get_current_observation_id(self) -> str | None:
        request_id = self._active_request_id.get()
        if request_id is None:
            return None
        current = self._request_trace_ids.get()
        if current is None:
            return None
        ids = current.get(request_id)
        if ids is None:
            return None
        return ids[1]

    async def on_request_start(
        self,
        request: InternalRequest,
        context: EventContext,
    ) -> None:
        if not self._enabled:
            return
        self._set_active_request_id(context)
        try:
            c = self._client
            if c is None:
                return
            span_name = self._build_span_name(request, context)
            input_data = build_request_input_data(request)
            model = getattr(request, "model", None)
            model_parameters = build_model_parameters(getattr(request, "params", None))

            trace_name = span_name
            tags: list[str] = []

            # Create a generation that persists across async lifecycle calls.
            # ``end_on_exit=False`` keeps the observation open after the context
            # manager exits; we end it explicitly in on_request_end/on_stream_end.
            with c.start_as_current_observation(
                as_type="generation",
                name=span_name,
                input=input_data,
                model=model,
                model_parameters=model_parameters,
                version=self._version,
                end_on_exit=False,
            ) as generation:
                generation.update(
                    trace_name=trace_name,
                    tags=tags or None,
                )
                if context.user_id:
                    generation.update(trace_user_id=context.user_id)
                if context.session_id:
                    generation.update(session_id=context.session_id)

                self._set_generation_for_request(context.request_id, generation)
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
        if not self._enabled:
            return
        self._set_active_request_id(context)
        request_id = context.request_id
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
            self._recorded_ttft.discard(context.request_id)
        except Exception as e:
            logger.error(f"Langfuse handler failed to end generation: {e}", exc_info=True)
            with contextlib.suppress(Exception):
                generation.end()

    async def on_error(
        self,
        request: InternalRequest,
        error: Exception,
        context: EventContext,
    ) -> None:
        if not self._enabled:
            return
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
            self._recorded_ttft.discard(request_id)
        except Exception as e:
            logger.error(f"Langfuse handler failed to record error: {e}", exc_info=True)
            with contextlib.suppress(Exception):
                generation.end()

    async def on_stream_start(
        self,
        request: InternalRequest,
        context: EventContext,
    ) -> None:
        if not self._enabled:
            return
        self._set_active_request_id(context)
        generation = self._get_generation_for_request(context.request_id)
        if generation is None:
            return
        try:
            metadata = build_metadata(context)
            metadata["streaming"] = True
            generation.update(metadata=metadata)
        except Exception as e:
            logger.error(f"Langfuse handler failed to mark stream start: {e}", exc_info=True)

    async def on_stream_chunk(
        self,
        request: InternalRequest,
        chunk: str,
        context: EventContext,
    ) -> None:
        if not self._enabled:
            return
        self._set_active_request_id(context)
        generation = self._get_generation_for_request(context.request_id)
        if generation is None:
            return
        try:
            request_id = context.request_id
            if (
                context.first_chunk_time is not None
                and context.ttft_ms is not None
                and request_id not in self._recorded_ttft
            ):
                metadata = build_metadata(context)
                metadata["ttft_ms"] = context.ttft_ms
                generation.update(
                    metadata=metadata,
                    completion_start_time=context.first_chunk_time,
                )
                self._recorded_ttft.add(request_id)
        except Exception as e:
            logger.error(f"Langfuse handler failed to record stream chunk: {e}", exc_info=True)

    async def on_stream_end(
        self,
        request: InternalRequest,
        context: EventContext,
        error: Exception | None = None,
    ) -> None:
        if not self._enabled:
            return
        self._set_active_request_id(context)
        request_id = context.request_id
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
            self._recorded_ttft.discard(request_id)
        except Exception as e:
            logger.error(f"Langfuse handler failed to end stream generation: {e}", exc_info=True)
            with contextlib.suppress(Exception):
                generation.end()

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
