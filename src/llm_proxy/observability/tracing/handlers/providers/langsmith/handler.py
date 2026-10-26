"""LangSmith tracing handler: the lifecycle adapter for its OTLP exporter.

LangSmith accepts traces over OTLP/HTTP at ``https://api.smith.langchain.com/otel/v1/traces``
(regional hosts and self-hosted deployments differ only in the host). It is
therefore the OpenTelemetry handler with a fixed transport and LangSmith's
credential headers, not a second SDK and not a second export path: the exporter
switches on :attr:`backend` to add the run-correlation attributes LangSmith's
ingest reads (see ``llm_proxy.observability.sinks.otlp``).

The API key is a credential: it never appears in an API response (``api_key`` is
masked by ``api.schemas.tracing``), and it is only sent as the ``x-api-key``
header of the export request.
"""

from typing import TYPE_CHECKING, Any

from llm_proxy.observability.observation.policy import (
    parse_send_content,
    send_content_field_metadata,
)
from llm_proxy.observability.sinks.otlp import (
    LANGSMITH_OTLP_ENDPOINT,
    LANGSMITH_PROJECT_HEADER,
    build_tracer_provider,
)
from llm_proxy.observability.tracing.handlers.providers.otel.handler import (
    OtlpTracingHandler,
)

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager

#: LangSmith's workspace header. A key scoped to an organisation rather than a
#: workspace needs it, otherwise the gateway answers 403.
_TENANT_HEADER = "X-Tenant-Id"


class LangsmithTracingHandler(OtlpTracingHandler):
    """Tracing handler that exports runs to LangSmith over OTLP/HTTP."""

    provider_name = "langsmith"
    backend = "langsmith"
    required_settings = ["api_key"]
    optional_settings = [
        "project",
        "endpoint",
        "tenant_id",
        "timeout",
        "sample_rate",
        "send_content",
    ]
    description = (
        "Export runs to LangSmith over its OTLP/HTTP gateway. Requires a LangSmith "
        "API key; the project is created on first ingest."
    )
    field_metadata = [
        {
            "name": "api_key",
            "type": "password",
            "required": True,
            "description": "LangSmith API key (lsv2_...)",
        },
        {
            "name": "project",
            "type": "text",
            "required": False,
            "default": "default",
            "description": "LangSmith project name (created on first ingest)",
        },
        {
            "name": "endpoint",
            "type": "text",
            "required": False,
            "default": LANGSMITH_OTLP_ENDPOINT,
            "description": (
                "OTLP/HTTP traces endpoint. Change the host for the EU/APAC/AWS "
                "regions or a self-hosted deployment."
            ),
        },
        {
            "name": "tenant_id",
            "type": "text",
            "required": False,
            "description": "Workspace id (X-Tenant-Id) for keys not scoped to one workspace",
        },
        {
            "name": "timeout",
            "type": "number",
            "required": False,
            "description": "Exporter request timeout in seconds",
        },
        {
            "name": "sample_rate",
            "type": "number",
            "required": False,
            "min": 0.0,
            "max": 1.0,
            "description": "Sampling rate (0.0-1.0) for run export",
        },
        send_content_field_metadata(),
    ]

    @classmethod
    def validate_config(cls, settings: dict[str, Any]) -> bool:
        api_key = settings.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            return False
        endpoint = settings.get("endpoint")
        if endpoint is not None and not isinstance(endpoint, str):
            return False
        try:
            cls.parse_headers(settings)
            cls.parse_timeout(settings)
            cls.parse_sample_rate(settings)
            parse_send_content(settings)
        except ValueError:
            return False
        return True

    @classmethod
    def _headers(cls, settings: dict[str, Any]) -> dict[str, str]:
        headers = {key: value for key, value in cls.parse_headers(settings).items()}
        headers["x-api-key"] = str(settings["api_key"])
        project = str(settings.get("project") or "default")
        headers[LANGSMITH_PROJECT_HEADER] = project
        tenant_id = settings.get("tenant_id")
        if isinstance(tenant_id, str) and tenant_id:
            headers[_TENANT_HEADER] = tenant_id
        return headers

    @classmethod
    def create_handler(
        cls,
        settings: dict[str, Any],
        config_manager: DatabaseConfigManager | None = None,
    ) -> LangsmithTracingHandler:
        """Create a LangSmith handler from configuration.

        Raises:
            ValueError: If ``api_key`` is missing or a setting is invalid
        """
        api_key = settings.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("LangSmith handler requires a non-empty string 'api_key' in settings")

        endpoint = settings.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            endpoint = LANGSMITH_OTLP_ENDPOINT

        provider = build_tracer_provider(
            endpoint=endpoint,
            protocol="http",
            headers=cls._headers(settings),
            timeout=cls.parse_timeout(settings),
            service_name="llm-proxy",
            sample_rate=cls.parse_sample_rate(settings),
        )
        return cls(
            enabled=True,
            name=settings.get("name", cls.provider_name),
            provider=provider,
            backend=cls.backend,
            send_content=parse_send_content(settings),
        )


__all__ = ["LangsmithTracingHandler"]
