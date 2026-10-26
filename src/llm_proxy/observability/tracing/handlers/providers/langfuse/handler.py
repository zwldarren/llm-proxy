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

from typing import TYPE_CHECKING, Any

lazy from langfuse import Langfuse

from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.observation.policy import (
    parse_send_content,
    send_content_field_metadata,
)
from llm_proxy.observability.sinks.langfuse import LangfuseExporter, sampled_tracer_provider
from llm_proxy.observability.tracing.handlers.exporter_handler import ExporterBackedHandler

logger = get_logger(__name__)

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.observability.observation.policy import SendContent


class LangfuseTracingHandler(ExporterBackedHandler):
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
            "min": 0.0,
            "max": 1.0,
            "description": "Sampling rate (0.0-1.0) for trace export",
        },
        {
            "name": "version",
            "type": "text",
            "required": False,
            "default": "1.0",
            "description": "Version tag for traces",
        },
        send_content_field_metadata(),
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
        send_content = parse_send_content(settings)

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
            send_content=send_content,
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

    # The request lifecycle, trace-id answers, `release` and `shutdown` are the
    # shared adapter in ``tracing/handlers/exporter_handler.py``: this handler only
    # declares the backend's metadata and builds its exporter.
