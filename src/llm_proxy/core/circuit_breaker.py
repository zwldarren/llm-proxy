"""Circuit breaker for provider fallback protection.

Prevents repeated requests to failing providers by temporarily
removing them from the selection pool after consecutive failures.

Standard three-state circuit breaker:
- CLOSED: Normal operation, failures counted toward threshold
- OPEN: Provider skipped entirely, no requests allowed
- HALF_OPEN: Single probe request allowed after cooldown;
  success resets to CLOSED, failure reopens immediately
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation, failures being counted
    OPEN = auto()  # Circuit is open, provider is skipped
    HALF_OPEN = auto()  # Allowing one probe request


@dataclass
class CircuitBreakerConfig:
    """Configuration for the circuit breaker.

    Attributes:
        enabled: Whether the circuit breaker is active. When False,
            is_available() always returns True.
        failure_threshold: Number of consecutive failures before the
            circuit transitions from CLOSED to OPEN. Must be >= 1.
        cooldown_seconds: Time in seconds the circuit stays OPEN before
            transitioning to HALF_OPEN. Must be > 0.
    """

    enabled: bool = True
    failure_threshold: int = 5
    cooldown_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {self.failure_threshold}")
        if self.cooldown_seconds <= 0:
            raise ValueError(f"cooldown_seconds must be > 0, got {self.cooldown_seconds}")


@dataclass
class _Circuit:
    """Internal state for a single provider's circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = field(default_factory=time.time)
    half_open_request_allowed: bool = True


class CircuitBreakerStore:
    """Shared store for circuit breaker states across requests.

    Designed as an application-level singleton (stored on ``app.state``).
    Each provider is keyed by its unique mapping key (provider:model:index).

    Thread-safety: asyncio is cooperative single-threaded; this store
    does not need explicit locking for typical FastAPI use.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self._config = config or CircuitBreakerConfig()
        self._circuits: dict[str, _Circuit] = {}

    @property
    def config(self) -> CircuitBreakerConfig:
        return self._config

    def update_config(self, config: CircuitBreakerConfig) -> None:
        """Update the circuit breaker configuration at runtime.

        Args:
            config: New configuration to apply.
        """
        self._config = config
        logger.info(
            f"Circuit breaker config updated: enabled={config.enabled}, "
            f"failure_threshold={config.failure_threshold}, "
            f"cooldown_seconds={config.cooldown_seconds}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self, provider_key: str) -> bool:
        """Check whether *provider_key* can receive a request.

        May transition OPEN → HALF_OPEN when the cooldown has elapsed.
        """
        if not self._config.enabled:
            return True

        circuit = self._circuits.get(provider_key)
        if circuit is None:
            return True

        if circuit.state == CircuitState.CLOSED:
            return True

        if circuit.state == CircuitState.OPEN:
            elapsed = time.time() - circuit.last_state_change
            if elapsed >= self._config.cooldown_seconds:
                circuit.state = CircuitState.HALF_OPEN
                circuit.last_state_change = time.time()
                circuit.half_open_request_allowed = True
                logger.info(
                    f"Circuit breaker for '{provider_key}' transitioning "
                    f"OPEN → HALF_OPEN after {elapsed:.1f}s cooldown"
                )
                # Fall through to HALF_OPEN handling below
            else:
                return False

        # HALF_OPEN: allow exactly one probe request at a time
        if circuit.state == CircuitState.HALF_OPEN:
            if circuit.half_open_request_allowed:
                circuit.half_open_request_allowed = False
                return True
            # Probe already in-flight; if it has been running longer than
            # cooldown_seconds, assume it was lost (e.g. client disconnect)
            # and allow another probe.
            elapsed = time.time() - circuit.last_state_change
            if elapsed >= self._config.cooldown_seconds:
                circuit.half_open_request_allowed = True
                logger.info(
                    f"Circuit breaker for '{provider_key}' allowing another "
                    f"HALF_OPEN probe after {elapsed:.1f}s (previous probe lost)"
                )
                return True
            return False

        # Defensive fallback: unknown future states default to available
        return True

    def record_failure(self, provider_key: str) -> None:
        """Record a failure, potentially opening or re-opening the circuit."""
        if not self._config.enabled:
            return

        circuit = self._circuits.get(provider_key)
        if circuit is None:
            circuit = _Circuit()
            self._circuits[provider_key] = circuit

        now = time.time()

        if circuit.state == CircuitState.CLOSED:
            circuit.failure_count += 1
            circuit.last_failure_time = now
            if circuit.failure_count >= self._config.failure_threshold:
                circuit.state = CircuitState.OPEN
                circuit.last_state_change = now
                logger.warning(
                    f"Circuit breaker OPEN for '{provider_key}' after "
                    f"{circuit.failure_count} consecutive failures"
                )
        elif circuit.state == CircuitState.HALF_OPEN:
            # Probe request failed → immediately reopen
            circuit.state = CircuitState.OPEN
            circuit.last_state_change = now
            circuit.failure_count = self._config.failure_threshold
            logger.warning(f"Circuit breaker RE-OPENED for '{provider_key}' after probe failure")

    def record_success(self, provider_key: str) -> None:
        """Record a success, resetting the circuit to CLOSED."""
        if not self._config.enabled:
            return

        circuit = self._circuits.get(provider_key)
        if circuit is None:
            return

        if circuit.state != CircuitState.CLOSED:
            logger.info(f"Circuit breaker CLOSED for '{provider_key}' after successful request")

        circuit.state = CircuitState.CLOSED
        circuit.failure_count = 0
        circuit.last_failure_time = 0.0
        circuit.half_open_request_allowed = True

    def reset(self, provider_key: str | None = None) -> None:
        """Reset circuit breaker state.

        Args:
            provider_key: Specific provider to reset, or ``None`` to reset all.
        """
        if provider_key is None:
            count = len(self._circuits)
            self._circuits.clear()
            if count:
                logger.info(f"All circuit breakers reset ({count} cleared)")
        else:
            removed = self._circuits.pop(provider_key, None)
            if removed:
                logger.info(f"Circuit breaker reset for '{provider_key}'")

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------

    @property
    def circuit_count(self) -> int:
        """Return the number of tracked circuits."""
        return len(self._circuits)

    def get_state(self, provider_key: str) -> dict | None:
        """Return a snapshot of the circuit state for *provider_key*."""
        circuit = self._circuits.get(provider_key)
        if circuit is None:
            return None
        return {
            "state": circuit.state.name,
            "failure_count": circuit.failure_count,
            "last_failure_time": circuit.last_failure_time,
            "last_state_change": circuit.last_state_change,
            "cooldown_seconds": self._config.cooldown_seconds,
        }

    def get_all_states(self) -> dict[str, dict]:
        """Return snapshots for every known circuit."""
        result = {}
        for k in self._circuits:
            s = self.get_state(k)
            if s is not None:
                result[k] = s
        return result
