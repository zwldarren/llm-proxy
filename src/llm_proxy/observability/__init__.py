"""Observability package - logging, tracing, and monitoring."""

from llm_proxy.observability.logger import (
    LogFormat,
    LoggingConfig,
    LoggingManager,
    LogLevel,
    StructuredLogger,
    get_logger,
    get_logging_manager,
)

__all__ = [
    "LogFormat",
    "LogLevel",
    "LoggingConfig",
    "LoggingManager",
    "StructuredLogger",
    "get_logger",
    "get_logging_manager",
    "TracingConfig",
]


def __getattr__(name: str):
    _LAZY = {
        "TracingConfig": "llm_proxy.observability.tracing_config",
    }
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
