"""Tests for circuit breaker module."""

import time

from llm_proxy.core.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerStore,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig defaults."""

    def test_defaults(self):
        config = CircuitBreakerConfig()
        assert config.enabled is True
        assert config.failure_threshold == 5
        assert config.cooldown_seconds == 60.0

    def test_custom_values(self):
        config = CircuitBreakerConfig(
            enabled=False,
            failure_threshold=3,
            cooldown_seconds=30.0,
        )
        assert config.enabled is False
        assert config.failure_threshold == 3
        assert config.cooldown_seconds == 30.0


class TestCircuitBreakerStoreBasic:
    """Basic tests for CircuitBreakerStore."""

    def test_is_available_when_disabled(self):
        store = CircuitBreakerStore(CircuitBreakerConfig(enabled=False))
        # Record failures to open the circuit
        for _ in range(10):
            store.record_failure("provider:model:0")
        assert store.is_available("provider:model:0") is True

    def test_is_available_unknown_key(self):
        store = CircuitBreakerStore()
        assert store.is_available("unknown:model:0") is True

    def test_is_available_closed_state(self):
        store = CircuitBreakerStore()
        store.record_failure("p:m:0")
        assert store.is_available("p:m:0") is True

    def test_record_success_resets_unknown_key(self):
        """record_success on unknown key should not raise."""
        store = CircuitBreakerStore()
        store.record_success("unknown:model:0")  # Should not raise

    def test_record_failure_creates_new_circuit(self):
        store = CircuitBreakerStore()
        assert store.get_state("p:m:0") is None
        store.record_failure("p:m:0")
        state = store.get_state("p:m:0")
        assert state is not None
        assert state["state"] == "CLOSED"
        assert state["failure_count"] == 1


class TestCircuitBreakerOpenClose:
    """Tests for circuit breaker state transitions."""

    def test_circuit_opens_after_threshold(self):
        store = CircuitBreakerStore(
            CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=999.0)
        )
        key = "openai:gpt-4:0"

        # 3 failures should open the circuit
        store.record_failure(key)
        store.record_failure(key)
        state_before = store.get_state(key)
        assert state_before is not None
        assert state_before["state"] == "CLOSED"
        assert state_before["failure_count"] == 2

        store.record_failure(key)
        state_after = store.get_state(key)
        assert state_after is not None
        assert state_after["state"] == "OPEN"
        assert state_after["failure_count"] == 3

        # Should not be available
        assert store.is_available(key) is False

    def test_circuit_opens_exactly_at_threshold(self):
        store = CircuitBreakerStore(
            CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=999.0)
        )
        key = "p:m:0"

        # 1 failure should open immediately
        store.record_failure(key)
        state = store.get_state(key)
        assert state is not None
        assert state["state"] == "OPEN"
        assert store.is_available(key) is False

    def test_circuit_transitions_to_half_open_after_cooldown(self):
        store = CircuitBreakerStore(CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.1))
        key = "p:m:0"

        # Open the circuit
        store.record_failure(key)
        assert store.is_available(key) is False

        # Wait for cooldown
        time.sleep(0.15)

        # Should transition to HALF_OPEN on is_available check
        assert store.is_available(key) is True
        state = store.get_state(key)
        assert state is not None
        assert state["state"] == "HALF_OPEN"

    def test_half_open_allows_one_probe_then_blocks(self):
        store = CircuitBreakerStore(CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.1))
        key = "p:m:0"

        store.record_failure(key)  # Open circuit
        time.sleep(0.15)

        # First probe is allowed
        assert store.is_available(key) is True

        # Second probe is blocked (only one allowed in HALF_OPEN)
        assert store.is_available(key) is False

    def test_half_open_success_resets_to_closed(self):
        store = CircuitBreakerStore(CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.1))
        key = "p:m:0"

        store.record_failure(key)  # Open
        time.sleep(0.15)
        assert store.is_available(key) is True  # HALF_OPEN probe

        # Success resets
        store.record_success(key)
        state = store.get_state(key)
        assert state is not None
        assert state["state"] == "CLOSED"
        assert state["failure_count"] == 0

    def test_half_open_failure_reopens(self):
        store = CircuitBreakerStore(CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=0.1))
        key = "p:m:0"

        store.record_failure(key)  # Open
        time.sleep(0.15)
        assert store.is_available(key) is True  # HALF_OPEN

        # Probe fails → back to OPEN
        store.record_failure(key)
        state = store.get_state(key)
        assert state is not None
        assert state["state"] == "OPEN"
        assert store.is_available(key) is False

    def test_success_closes_open_circuit(self):
        """record_success should close even an OPEN circuit."""
        store = CircuitBreakerStore(
            CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=999.0)
        )
        key = "p:m:0"

        store.record_failure(key)
        store.record_failure(key)
        assert store.get_state(key)["state"] == "OPEN"

        store.record_success(key)
        state = store.get_state(key)
        assert state is not None
        assert state["state"] == "CLOSED"
        assert state["failure_count"] == 0


class TestCircuitBreakerReset:
    """Tests for reset functionality."""

    def test_reset_specific_key(self):
        store = CircuitBreakerStore(CircuitBreakerConfig(failure_threshold=1))
        key1 = "p1:m:0"
        key2 = "p2:m:0"

        store.record_failure(key1)
        store.record_failure(key2)
        assert store.get_state(key1)["state"] == "OPEN"
        assert store.get_state(key2)["state"] == "OPEN"

        store.reset(key1)
        assert store.get_state(key1) is None
        assert store.get_state(key2) is not None  # Still open

    def test_reset_all(self):
        store = CircuitBreakerStore(CircuitBreakerConfig(failure_threshold=1))
        store.record_failure("p1:m:0")
        store.record_failure("p2:m:0")

        store.reset()
        assert store.get_state("p1:m:0") is None
        assert store.get_state("p2:m:0") is None

    def test_reset_unknown_key(self):
        """reset on unknown key should not raise."""
        store = CircuitBreakerStore()
        store.reset("unknown")  # Should not raise


class TestCircuitBreakerGetAllStates:
    """Tests for observability helpers."""

    def test_get_all_states_empty(self):
        store = CircuitBreakerStore()
        assert store.get_all_states() == {}

    def test_get_all_states_with_circuits(self):
        store = CircuitBreakerStore(CircuitBreakerConfig(failure_threshold=2))
        store.record_failure("p1:m:0")
        store.record_failure("p2:m:0")

        states = store.get_all_states()
        assert len(states) == 2
        assert states["p1:m:0"]["failure_count"] == 1
        assert states["p2:m:0"]["failure_count"] == 1


class TestCircuitBreakerWithMockTime:
    """Tests using mock time for deterministic cooldown behavior."""

    def test_cooldown_not_elapsed(self):
        store = CircuitBreakerStore(
            CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=60.0)
        )
        key = "p:m:0"

        store.record_failure(key)  # Opens circuit
        # Immediately check — should still be OPEN
        assert store.is_available(key) is False

    def test_is_available_idempotent_in_open_state(self):
        """is_available should return False consistently when circuit is OPEN."""
        store = CircuitBreakerStore(
            CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=60.0)
        )
        key = "p:m:0"
        store.record_failure(key)

        for _ in range(10):
            assert store.is_available(key) is False

    def test_multiple_independent_circuits(self):
        store = CircuitBreakerStore(
            CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=999.0)
        )

        # Open circuit for p1 but not p2
        store.record_failure("p1:gpt-4:0")
        store.record_failure("p1:gpt-4:0")
        store.record_failure("p2:claude:0")

        assert store.is_available("p1:gpt-4:0") is False
        assert store.is_available("p2:claude:0") is True

    def test_failure_count_resets_on_close(self):
        store = CircuitBreakerStore(
            CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=999.0)
        )
        key = "p:m:0"

        # Build up 2 failures but don't open
        store.record_failure(key)
        store.record_failure(key)
        assert store.get_state(key)["failure_count"] == 2

        # Success resets count
        store.record_success(key)
        assert store.get_state(key)["failure_count"] == 0

        # Need 3 more failures to open again (fresh start)
        store.record_failure(key)
        store.record_failure(key)
        assert store.get_state(key)["state"] == "CLOSED"
        store.record_failure(key)
        assert store.get_state(key)["state"] == "OPEN"
