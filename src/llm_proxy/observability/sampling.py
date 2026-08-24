"""Unified sampling logic for logging and tracing.

This module consolidates sampling decisions that were previously
scattered between logging middleware and tracing handlers.
"""

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request

    from llm_proxy.config.types.logging_config import LoggingConfig
    from llm_proxy.observability.types import LogType


@dataclass(frozen=True)
class SamplingDecision:
    """Result of sampling decision for a request.

    Attributes:
        should_capture_full_body: Whether to capture full request/response body
        log_type: The determined log type for this request
    """

    should_capture_full_body: bool
    log_type: LogType


def determine_log_type(path: str) -> LogType:
    """Determine log type based on request path.

    Args:
        path: The request path

    Returns:
        LogType for this request
    """
    from llm_proxy.observability.types import LogType

    if path.startswith("/v1/"):
        if path.startswith("/v1/models"):
            return LogType.AUDIT
        return LogType.ENDPOINT
    return LogType.AUDIT


def should_exclude_from_logging(path: str) -> bool:
    """Check if a path should be excluded from logging.

    Args:
        path: The request path

    Returns:
        True if path should be excluded
    """
    # Paths that should not be logged to avoid feedback loops or noise
    excluded_paths = {
        "/api/logs",
    }

    if path in excluded_paths:
        return True

    return path.startswith("/api/logs/")


def make_sampling_decision(
    config: LoggingConfig,
    request: Request,
    path: str,
    force_full_log_header: str = "x-log-full",
) -> SamplingDecision:
    """Make unified sampling decision for all handlers.

    This replaces the sampling logic that was in middleware.py.
    The decision is made once and passed to all handlers via EventContext.

    Args:
        config: Logging configuration with sampling rates
        request: The FastAPI request
        path: The request path
        force_full_log_header: Header name to force full logging

    Returns:
        SamplingDecision with capture flags and log type
    """
    log_type = determine_log_type(path)

    # Check force full log header
    force_full = request.headers.get(force_full_log_header, "").lower() in ("true", "1", "yes")

    # Get sampling rate for this log type
    sampling_rate = config.get_sampling_rate(log_type)

    # Make sampling decision
    should_capture_full_body = force_full or (random.random() < sampling_rate)

    return SamplingDecision(
        should_capture_full_body=should_capture_full_body,
        log_type=log_type,
    )
