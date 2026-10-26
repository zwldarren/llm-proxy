"""OpenTelemetry (OTLP) tracing handler: the lifecycle adapter for its exporter.

The handler owns only what the request lifecycle requires: parsing the operator's
settings into a tracer provider, and being told when a request starts, streams and
ends. Everything that reads facts or talks to the SDK lives in
``llm_proxy.observability.sinks.otlp.OtlpExporter`` (ADR-0044).

The same handler is the base of the LangSmith backend: LangSmith accepts OTLP
over HTTP, so its provider fixes the transport and supplies its own credentials
instead of growing a second export path.
"""

from typing import TYPE_CHECKING, Any, cast

from llm_proxy.observability.observation.policy import (
    SendContent,
    parse_send_content,
    send_content_field_metadata,
)
from llm_proxy.observability.sinks.otlp import (
    OtlpBackend,
    OtlpExporter,
    OtlpProtocol,
    build_tracer_provider,
)
from llm_proxy.observability.tracing.handlers.exporter_handler import ExporterBackedHandler

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    from llm_proxy.config.manager import DatabaseConfigManager

_PROTOCOLS: frozenset[str] = frozenset({"http", "grpc"})


class OtlpTracingHandler(ExporterBackedHandler):
    """Tracing handler that exports one span per request over OTLP.

    Works with any OpenTelemetry collector (Tempo, Jaeger, Phoenix, Grafana, …)
    over HTTP/protobuf or gRPC. Self-hosted collectors are supported by pointing
    ``endpoint`` at them.
    """

    provider_name = "otel"
    required_settings = ["endpoint"]
    optional_settings = [
        "protocol",
        "headers",
        "insecure",
        "timeout",
        "service_name",
        "sample_rate",
        "send_content",
    ]
    description = (
        "Export spans to any OTLP collector (Tempo, Jaeger, Phoenix, Grafana, …) "
        "over HTTP/protobuf or gRPC."
    )
    field_metadata = [
        {
            "name": "endpoint",
            "type": "text",
            "required": True,
            "description": (
                "OTLP traces endpoint including the scheme — HTTP: "
                "https://collector:4318/v1/traces, gRPC: https://collector:4317"
            ),
        },
        {
            "name": "protocol",
            "type": "select",
            "required": False,
            "default": "http",
            "choices": ["http", "grpc"],
            "description": "OTLP transport protocol",
        },
        {
            "name": "headers",
            "type": "headers",
            "required": False,
            "description": "Extra HTTP headers or gRPC metadata (e.g. Authorization)",
        },
        {
            "name": "insecure",
            "type": "boolean",
            "required": False,
            "default": False,
            "description": "Send gRPC without TLS (plain-text collector)",
        },
        {
            "name": "service_name",
            "type": "text",
            "required": False,
            "default": "llm-proxy",
            "description": "OpenTelemetry service.name resource attribute",
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
            "description": "Sampling rate (0.0-1.0) for span export",
        },
        send_content_field_metadata(),
    ]

    #: Which attribute vocabulary to emit (see ``OtlpBackend``).
    backend: OtlpBackend = "otel"

    # -- settings shared with the LangSmith backend -----------------------

    @staticmethod
    def parse_timeout(settings: dict[str, Any]) -> float | None:
        timeout = settings.get("timeout")
        if timeout is None or timeout == "":
            return None
        timeout_val = float(timeout)
        if timeout_val <= 0:
            raise ValueError("timeout must be a positive number")
        return timeout_val

    @staticmethod
    def parse_sample_rate(settings: dict[str, Any]) -> float | None:
        sample_rate = settings.get("sample_rate")
        if sample_rate is None or sample_rate == "":
            return None
        sample_rate_val = float(sample_rate)
        if not (0.0 <= sample_rate_val <= 1.0):
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        return sample_rate_val

    @staticmethod
    def parse_insecure(settings: dict[str, Any]) -> bool | None:
        """Whether gRPC should skip TLS; ``None`` lets the endpoint scheme decide.

        The SDK infers TLS from an ``http://`` endpoint only while this is unset,
        so an absent setting must stay ``None`` rather than becoming ``False``
        (ADR-0048: the scheme is what lets gRPC infer TLS).
        """
        value = settings.get("insecure")
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        raise ValueError("insecure must be a boolean")

    @staticmethod
    def parse_headers(settings: dict[str, Any]) -> dict[str, str]:
        headers = settings.get("headers")
        if headers is None:
            return {}
        if not isinstance(headers, dict):
            raise ValueError("headers must be an object of header name/value pairs")
        return {
            str(key): str(value)
            for key, value in headers.items()
            if str(key).strip() and value not in (None, "")
        }

    @classmethod
    def parse_protocol(cls, settings: dict[str, Any]) -> OtlpProtocol:
        protocol = str(settings.get("protocol") or "http").strip().lower()
        if protocol not in _PROTOCOLS:
            raise ValueError("protocol must be one of: http, grpc")
        return cast("OtlpProtocol", protocol)

    # -- config -----------------------------------------------------------

    @classmethod
    def validate_config(cls, settings: dict[str, Any]) -> bool:
        endpoint = settings.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            return False
        try:
            cls.parse_protocol(settings)
            cls.parse_insecure(settings)
            cls.parse_headers(settings)
            cls.parse_timeout(settings)
            cls.parse_sample_rate(settings)
            parse_send_content(settings)
        except ValueError:
            return False
        return True

    @classmethod
    def create_handler(
        cls,
        settings: dict[str, Any],
        config_manager: DatabaseConfigManager | None = None,
    ) -> OtlpTracingHandler:
        """Create an OTLP handler from configuration.

        Raises:
            ValueError: If ``endpoint`` is missing or a setting is invalid
        """
        endpoint = settings.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("OTLP handler requires a non-empty string 'endpoint' in settings")

        provider = build_tracer_provider(
            endpoint=endpoint,
            protocol=cls.parse_protocol(settings),
            headers=cls.parse_headers(settings),
            insecure=cls.parse_insecure(settings),
            timeout=cls.parse_timeout(settings),
            service_name=str(settings.get("service_name") or "llm-proxy"),
            sample_rate=cls.parse_sample_rate(settings),
        )
        return cls(
            enabled=True,
            name=settings.get("name", cls.provider_name),
            provider=provider,
            backend=cls.backend,
            send_content=parse_send_content(settings),
        )

    def __init__(
        self,
        *,
        enabled: bool = True,
        name: str = "otel",
        provider: TracerProvider,
        backend: OtlpBackend = "otel",
        send_content: SendContent = "full",
    ) -> None:
        super().__init__(enabled=enabled)
        self.name = name
        self._exporter = OtlpExporter(
            provider=provider,
            backend=backend,
            name=name,
            send_content=send_content,
        )


__all__ = ["OtlpTracingHandler"]
