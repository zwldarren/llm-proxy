"""Unified error handling."""

from llm_proxy.core.errors.classification import (
    CONTEXT_LENGTH_FINISH_REASONS,
    RETRYABLE_STATUS_CODES,
    ErrorCategory,
    classify_error,
    error_type_from_stream_finish_reason,
    is_context_length_finish_reason,
    is_retryable_stream_finish_reason,
)
from llm_proxy.core.errors.handler import (
    ErrorHandler,
    get_error_handler,
    register_formatter_factory,
)
from llm_proxy.core.errors.protocols import ErrorFormatter, ErrorProtocol
from llm_proxy.core.errors.utils import get_error_type_for_status

__all__ = [
    "CONTEXT_LENGTH_FINISH_REASONS",
    "ErrorCategory",
    "RETRYABLE_STATUS_CODES",
    "ErrorFormatter",
    "ErrorHandler",
    "ErrorProtocol",
    "classify_error",
    "error_type_from_stream_finish_reason",
    "get_error_handler",
    "get_error_type_for_status",
    "is_context_length_finish_reason",
    "is_retryable_stream_finish_reason",
    "register_formatter_factory",
]
