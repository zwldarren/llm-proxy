"""Attempt tracking for provider selection.

This module provides the ProviderAttempt dataclass for recording
individual provider attempts during fallback handling.
"""

from dataclasses import dataclass


@dataclass
class ProviderAttempt:
    """Record of an attempt to use a provider."""

    provider_name: str
    provider_model_name: str | None
    priority: int
    attempt_number: int
    success: bool = False
    error: Exception | None = None
    status_code: int | None = None
    response_time_ms: int | None = None


__all__ = [
    "ProviderAttempt",
]
