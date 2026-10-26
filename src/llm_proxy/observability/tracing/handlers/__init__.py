"""Lifecycle tracing system for request processing."""

from llm_proxy.observability.tracing.handlers.audit_log import AuditLogHandler
from llm_proxy.observability.tracing.handlers.base import TracingHandler
from llm_proxy.observability.tracing.handlers.exporter_handler import ExporterBackedHandler
from llm_proxy.observability.tracing.handlers.logging import LoggingHandler
from llm_proxy.observability.tracing.handlers.providers.langfuse import (
    LangfuseTracingHandler,
)
from llm_proxy.observability.tracing.handlers.providers.langsmith import (
    LangsmithTracingHandler,
)
from llm_proxy.observability.tracing.handlers.providers.otel import (
    OtlpTracingHandler,
)
from llm_proxy.observability.tracing.handlers.registry import (
    TracingRegistry,
    get_handler_type,
    get_tracing_registry,
    list_handler_types,
    register_handler_type,
    register_tracing_handler,
)

# Register built-in handler types. Order is the order the API lists them in, and
# (for the global registry) the order the response-header seam asks them for a
# trace id: the first handler with one wins.
register_handler_type("langfuse", LangfuseTracingHandler)
register_handler_type("otel", OtlpTracingHandler)
register_handler_type("langsmith", LangsmithTracingHandler)
register_handler_type("audit_log", AuditLogHandler)

__all__ = [
    "AuditLogHandler",
    "ExporterBackedHandler",
    "LangfuseTracingHandler",
    "LangsmithTracingHandler",
    "LoggingHandler",
    "OtlpTracingHandler",
    "TracingHandler",
    "TracingRegistry",
    "get_handler_type",
    "get_tracing_registry",
    "list_handler_types",
    "register_handler_type",
    "register_tracing_handler",
]
