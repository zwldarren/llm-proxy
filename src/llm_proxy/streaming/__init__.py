# src/llm_proxy/streaming/__init__.py
"""Unified streaming infrastructure for LLM Proxy."""

__all__ = [
    "SSEBuilder",
    "StreamEvent",
    "StreamingHandler",
    "StreamingResponseConfig",
    "StreamingTransformer",
    "StreamingUsage",
    "create_sse_error",
]

_SUBMODULES = {
    "SSEBuilder": ".sse_builder",
    "StreamEvent": ".events",
    "StreamingHandler": ".handler",
    "StreamingResponseConfig": ".handler",
    "StreamingTransformer": ".transformer",
    "StreamingUsage": ".transformer",
    "create_sse_error": ".sse_builder",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        import importlib

        mod = importlib.import_module(_SUBMODULES[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
