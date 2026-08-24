"""Selection state for provider selection and fallback tracking.

This module extracts the mutable state from ProviderSelector into a separate
dataclass, making the state management explicit and the ownership model clear.
"""

from dataclasses import dataclass, field

from llm_proxy.core.attempt_tracker import ProviderAttempt
from llm_proxy.core.exceptions import ProviderError


@dataclass
class SelectionState:
    """Mutable state for a single provider selection session.

    This class tracks which providers have been tried, whether the selector
    is exhausted, timing information, and error history. It is designed to be
    created fresh for each request via ProviderSelector().

    IMPORTANT: This class is NOT thread-safe. Each concurrent request MUST
    have its own SelectionState instance.

    Attributes:
        used_provider_keys: Set of provider mapping keys that have been tried.
        exhausted: Whether all providers have been exhausted.
        attempts: List of recorded provider attempts.
        last_error: The most recent ProviderError encountered.
        stream_started: Whether streaming has started for the current attempt.
        role_transformed: Whether role transformation (developer→system) has been applied.
        start_time: Timestamp when timing started, or None.
        last_selected_key: The key of the most recently selected provider,
            used by the circuit breaker to attribute failures/successes.
        sticky_key: The pinned provider key for the session_sticky strategy,
            resolved once per request from Redis (if configured) before the
            first selection.
    """

    used_provider_keys: set[str] = field(default_factory=set)
    exhausted: bool = False
    attempts: list[ProviderAttempt] = field(default_factory=list)
    last_error: ProviderError | None = None
    stream_started: bool = False
    role_transformed: bool = False
    start_time: float | None = None
    last_selected_key: str | None = None
    sticky_key: str | None = None

    @property
    def attempt_count(self) -> int:
        """Number of provider attempts made so far."""
        return len(self.attempts)


__all__ = ["SelectionState"]
