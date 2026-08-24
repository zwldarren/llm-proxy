"""Lifecycle tracing system for request processing."""

from llm_proxy.observability.tracing.handlers.audit_log import AuditLogHandler
from llm_proxy.observability.tracing.handlers.base import TracingHandler
from llm_proxy.observability.tracing.handlers.logging import LoggingHandler
from llm_proxy.observability.tracing.handlers.providers.langfuse import (
    LangfuseTracingHandler,
)
from llm_proxy.observability.tracing.handlers.registry import (
    TracingRegistry,
    get_handler_type,
    get_tracing_registry,
    list_handler_types,
    register_handler_type,
    register_tracing_handler,
)

# Register built-in handler types
register_handler_type("langfuse", LangfuseTracingHandler)
register_handler_type("audit_log", AuditLogHandler)

__all__ = [
    "AuditLogHandler",
    "LangfuseTracingHandler",
    "TracingHandler",
    "TracingRegistry",
    "get_handler_type",
    "get_tracing_registry",
    "list_handler_types",
    "LoggingHandler",
    "register_handler_type",
    "register_tracing_handler",
]
