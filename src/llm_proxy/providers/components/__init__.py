"""Composable components for provider adapters.

These components extract cross-cutting concerns from BaseProvider:
- HttpTransport: HTTP client lifecycle, streaming POST, response status checks
- RetryPolicy: Exponential backoff retry for operations and generators
- ErrorTranslator: Maps transport/HTTP errors to ProviderError
"""

from llm_proxy.providers.components.error_translator import ErrorTranslator
from llm_proxy.providers.components.http_transport import HttpTransport
from llm_proxy.providers.components.retry_policy import RetryPolicy

__all__ = [
    "ErrorTranslator",
    "HttpTransport",
    "RetryPolicy",
]
