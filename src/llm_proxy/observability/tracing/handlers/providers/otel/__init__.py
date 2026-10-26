"""OpenTelemetry (OTLP) tracing provider."""

from llm_proxy.observability.tracing.handlers.providers.otel.handler import (
    OtlpTracingHandler,
)

__all__ = ["OtlpTracingHandler"]
