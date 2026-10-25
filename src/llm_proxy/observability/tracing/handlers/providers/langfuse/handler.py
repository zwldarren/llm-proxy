"""Langfuse tracing handler: the lifecycle adapter for the Langfuse exporter.

The handler owns only what the request lifecycle requires — creating the client
from user settings, being told when a request starts/streams/ends, and answering
the trace-id questions the response-header seam asks. Everything that reads facts
or talks to the SDK lives in
``llm_proxy.observability.sinks.langfuse.LangfuseExporter`` (ADR-0044).

The handler never reads ``EventContext`` fields itself: it snapshots the capture
through the observation collector and hands the exporter an immutable
``RequestObservation``.
"""

from typing import TYPE_CHECKING, Any, cast

lazy from langfuse import Langfuse

from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.observation.collector import capture_request_end, collect
from llm_proxy.observability.sinks.langfuse import LangfuseExporter, sampled_tracer_provider
from llm_proxy.observability.tracing.handlers.base import TracingHandler

logger = get_logger(__name__)

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.models import (
        InternalDecisionRequest,
        InternalDecisionResponse,
        InternalEmbeddingRequest,
        InternalEmbeddingResponse,
        InternalImageEditRequest,
        InternalImageRequest,
        InternalImageResponse,
        InternalRequest,
        InternalRerankRequest,
        InternalRerankResponse,
        InternalResponse,
        InternalSpeechRequest,
        InternalSpeechResponse,
        InternalSystemOneRequest,
        InternalSystemOneResponse,
        InternalTranscriptionRequest,
        InternalTranscriptionResponse,
        InternalTranslationRequest,
        InternalTranslationResponse,
    )
    from llm_proxy.observability.event_context import EventContext
    from llm_proxy.observability.observation.policy import SendContent

    #: What the tracing hooks actually receive: the pipeline hands over the
    #: strategy's own request and response objects, and only chat uses the
    #: conversation-shaped pair.
    type ObservedRequest = (
        InternalRequest
        | InternalEmbeddingRequest
        | InternalRerankRequest
        | InternalImageRequest
        | InternalImageEditRequest
        | InternalSpeechRequest
        | InternalTranscriptionRequest
        | InternalTranslationRequest
        | InternalSystemOneRequest
        | InternalDecisionRequest
    )
    type ObservedResponse = (
        InternalResponse
        | InternalEmbeddingResponse
        | InternalRerankResponse
        | InternalImageResponse
        | InternalSpeechResponse
        | InternalTranscriptionResponse
        | InternalTranslationResponse
        | InternalSystemOneResponse
        | InternalDecisionResponse
    )


class LangfuseTracingHandler(TracingHandler):
    """Tracing handler that exports request traces to Langfuse.

    Uses the official Langfuse Python SDK. A request is represented as a single
    ``generation`` observation (``embedding`` for embedding calls). Streaming
    requests accumulate output and usage on the same observation. Self-hosted
    Langfuse is supported by providing a custom ``base_url``.
    """

    provider_name = "langfuse"
    required_settings = ["public_key", "secret_key"]
    optional_settings = ["base_url", "timeout", "sample_rate", "version", "send_content"]
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
        {
            "name": "send_content",
            "type": "select",
            "required": False,
            "default": "full",
            "choices": ["inherit", "none", "full"],
            "description": (
                "Which prompt/completion content to export. 'full' (default) exports "
                "verbatim content. 'inherit' follows the logging settings: no content "
                "when log_input_output is off, masked content when mask_sensitive_data "
                "is on. 'none' never exports content."
            ),
        },
    ]

    #: What an operator may ask this backend to export (see ``SendContent``).
    _SEND_CONTENT_CHOICES: frozenset[str] = frozenset({"inherit", "none", "full"})

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
        send_content = str(settings.get("send_content") or "full").strip().lower()
        if send_content not in cls._SEND_CONTENT_CHOICES:
            raise ValueError("send_content must be one of: inherit, none, full")

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
                client_kwargs["tracer_provider"] = sampled_tracer_provider(
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
            send_content=cast("SendContent", send_content),
        )

    def __init__(
        self,
        *,
        enabled: bool = True,
        name: str = "langfuse",
        client: Langfuse,
        base_url: str,
        version: str = "1.0",
        send_content: SendContent = "full",
    ) -> None:
        super().__init__(enabled=enabled)
        self.name = name
        self._base_url = base_url
        self._version = version
        self._exporter = LangfuseExporter(
            client=client,
            base_url=base_url,
            version=version,
            send_content=send_content,
            name=name,
        )

    # -- lifecycle --------------------------------------------------------

    async def on_request_start(
        self,
        request: ObservedRequest,
        context: EventContext,
    ) -> None:
        if not self._enabled:
            return
        self._exporter.open(collect(context), request)

    async def on_request_end(
        self,
        request: ObservedRequest,
        response: ObservedResponse,
        context: EventContext,
    ) -> None:
        # Finalize the capture (usage, status, cost) so this exporter sees the
        # same facts the log row will; the intake verb repeats it idempotently.
        record = await capture_request_end(context, response=response)
        self._exporter.finish(record, request=request, response=response)

    async def on_error(
        self,
        request: ObservedRequest,
        error: Exception,
        context: EventContext,
    ) -> None:
        record = await capture_request_end(context, error=error)
        self._exporter.fail(record, error)

    async def on_stream_start(
        self,
        request: ObservedRequest,
        context: EventContext,
    ) -> None:
        self._exporter.stream_opened(collect(context))

    async def on_stream_chunk(
        self,
        request: ObservedRequest,
        chunk: str,
        context: EventContext,
    ) -> None:
        # The snapshot is taken at most once per stream: ``needs_ttft`` is false as
        # soon as the first-token update landed, so the per-chunk hook pays only a
        # dict lookup after that.
        if not self._exporter.needs_ttft(context.request_id):
            return
        self._exporter.stream_progress(collect(context))

    async def on_stream_end(
        self,
        request: ObservedRequest,
        context: EventContext,
        error: Exception | None = None,
    ) -> None:
        # Streaming folds the transformer's usage here, before the audit handler
        # runs: user handlers are registered ahead of the system ones, and the
        # export must not miss the final usage report.
        record = await capture_request_end(context, error=error, streaming=True)
        if error is not None:
            self._exporter.fail(record, error, streaming=True)
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
        await self._exporter.flush()

    async def shutdown(self) -> None:
        self._enabled = False
        await self._exporter.close()
