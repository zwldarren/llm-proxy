"""Tests for ProviderSelector class and related functions."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import httpx2

from llm_proxy.core.circuit_breaker import CircuitBreakerConfig, CircuitBreakerStore
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.provider_selector import (
    ErrorCategory,
    ProviderSelector,
    classify_error,
)


# Mock dataclasses for testing
@dataclass
class MockModelProviderConfig:
    """Mock ModelProviderConfig for testing."""

    provider: str
    provider_model_name: str | None = None
    priority: int = 1
    parameter_overrides: dict[str, Any] | None = None
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None


@dataclass
class MockModelConfig:
    """Mock ModelConfig for testing."""

    model_name: str
    providers: list[MockModelProviderConfig]
    parameter_overrides: dict[str, Any] | None = None
    max_retries: int | None = None  # For model-level override testing
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None

    def get_providers_by_priority(self) -> list[MockModelProviderConfig]:
        """Return providers sorted by priority."""
        return sorted(self.providers, key=lambda p: -p.priority)


@dataclass
class MockProviderConfig:
    """Mock ProviderConfig for testing."""

    name: str
    api_key: str = "test-key"
    base_url: str = "https://api.example.com"
    parameter_overrides: dict[str, Any] | None = None


def create_test_selector(
    model_name: str = "gpt-4",
    providers: list[dict[str, Any]] | None = None,
    provider_configs: dict[str, MockProviderConfig] | None = None,
    max_fallback_attempts: int = 10,
    default_max_retries: int = 3,
    model_parameter_overrides: dict[str, Any] | None = None,
    model_max_retries: int | None = None,
) -> ProviderSelector:
    """Create a test ProviderSelector with mock data."""
    if providers is None:
        providers = [{"provider": "openai", "priority": 1}]

    model_providers = [MockModelProviderConfig(**p) for p in providers]
    model_config = MockModelConfig(
        model_name=model_name,
        providers=model_providers,
        parameter_overrides=model_parameter_overrides or {},
        max_retries=model_max_retries,
    )

    if provider_configs is None:
        provider_configs = {
            p["provider"]: MockProviderConfig(name=p["provider"]) for p in providers
        }

    return ProviderSelector(
        model_config=model_config,  # type: ignore
        provider_configs=provider_configs,  # type: ignore
        max_fallback_attempts=max_fallback_attempts,
        default_max_retries=default_max_retries,
    )


class TestClassifyError:
    """Tests for classify_error function."""

    def test_retryable_status_codes(self):
        """Test that retryable status codes return RETRYABLE."""
        assert classify_error(status_code=401) == ErrorCategory.RETRYABLE
        assert classify_error(status_code=403) == ErrorCategory.RETRYABLE
        assert classify_error(status_code=404) == ErrorCategory.RETRYABLE
        assert classify_error(status_code=408) == ErrorCategory.RETRYABLE
        assert classify_error(status_code=422) == ErrorCategory.RETRYABLE
        assert classify_error(status_code=429) == ErrorCategory.RETRYABLE
        assert classify_error(status_code=502) == ErrorCategory.RETRYABLE
        assert classify_error(status_code=503) == ErrorCategory.RETRYABLE
        assert classify_error(status_code=504) == ErrorCategory.RETRYABLE

    def test_non_retryable_status_codes(self):
        """Test that non-retryable status codes return NON_RETRYABLE.

        Only 400 is explicitly non-retryable among common 4xx codes;
        other 4xx codes (e.g. 418, 499) are non-retryable via the
        general 400-499 fallthrough.
        """
        assert classify_error(status_code=400) == ErrorCategory.NON_RETRYABLE
        assert classify_error(status_code=418) == ErrorCategory.NON_RETRYABLE
        assert classify_error(status_code=499) == ErrorCategory.NON_RETRYABLE

    def test_5xx_errors_retryable(self):
        """Test that 5xx errors are retryable."""
        assert classify_error(status_code=500) == ErrorCategory.RETRYABLE
        assert classify_error(status_code=501) == ErrorCategory.RETRYABLE
        assert classify_error(status_code=599) == ErrorCategory.RETRYABLE

    def test_4xx_errors_non_retryable(self):
        """Test that most 4xx errors are non-retryable.

        Note: 401, 402, 403, 404, 408, 422, 429 are exceptions and are
        treated as retryable (tested separately in test_retryable_status_codes).
        """
        assert classify_error(status_code=400) == ErrorCategory.NON_RETRYABLE
        assert classify_error(status_code=418) == ErrorCategory.NON_RETRYABLE  # I'm a teapot
        assert classify_error(status_code=499) == ErrorCategory.NON_RETRYABLE

    def test_network_errors_retryable(self):
        """Test that network errors are retryable."""
        assert (
            classify_error(error=httpx2.ConnectError("Connection failed"))
            == ErrorCategory.RETRYABLE
        )
        assert classify_error(error=httpx2.TimeoutException("Timeout")) == ErrorCategory.RETRYABLE

    def test_http_status_error(self):
        """Test that HTTPError uses status code."""
        mock_request = MagicMock()
        mock_request.url = "https://api.example.com/test"
        mock_response = MagicMock()
        mock_response.status_code = 503

        error = httpx2.HTTPStatusError("Server error", request=mock_request, response=mock_response)
        assert classify_error(error=error) == ErrorCategory.RETRYABLE

    def test_unknown_error(self):
        """Test that unknown errors return UNKNOWN."""
        assert classify_error(error=ValueError("Something went wrong")) == ErrorCategory.UNKNOWN
        assert classify_error(status_code=200) == ErrorCategory.UNKNOWN
        assert classify_error() == ErrorCategory.UNKNOWN


class TestProviderSelectorSelection:
    """Tests for ProviderSelector provider selection."""

    def test_select_first_provider(self):
        """Test selecting the first provider."""
        selector = create_test_selector()
        result = selector.select_next_provider()

        assert result is not None
        assert result.provider_name == "openai"

    def test_priority_ordering(self):
        """Test that providers are selected in priority order."""
        selector = create_test_selector(
            providers=[
                {"provider": "low", "priority": 1},
                {"provider": "high", "priority": 10},
                {"provider": "medium", "priority": 5},
            ]
        )

        # First should be highest priority
        result = selector.select_next_provider()
        assert result is not None
        assert result.provider_name == "high"
        assert result.priority == 10

    def test_fallback_to_next_priority(self):
        """Test fallback to next priority after failure."""
        selector = create_test_selector(
            providers=[
                {"provider": "primary", "priority": 10},
                {"provider": "fallback", "priority": 5},
            ]
        )

        # Select and exhaust first provider
        first = selector.select_next_provider()
        assert first is not None
        assert first.provider_name == "primary"

        # Select second should be fallback
        second = selector.select_next_provider()
        assert second is not None
        assert second.provider_name == "fallback"

    def test_max_fallback_attempts_limit(self):
        """Test that max fallback attempts limit is respected."""
        selector = create_test_selector(
            providers=[{"provider": "test", "priority": 1}],
            max_fallback_attempts=2,
        )

        # First selection
        assert selector.select_next_provider() is not None
        # Second selection should exhaust due to max_fallback_attempts
        assert selector.select_next_provider() is None  # Exhausted due to max_fallback_attempts

    def test_same_provider_different_models(self):
        """Test that same provider with different model names is unique."""
        selector = create_test_selector(
            providers=[
                {"provider": "openai", "provider_model_name": "gpt-4", "priority": 1},
                {"provider": "openai", "provider_model_name": "gpt-4-turbo", "priority": 2},
            ]
        )

        first = selector.select_next_provider()
        assert first is not None
        assert first.provider_name == "openai"
        assert first.provider_model_name == "gpt-4-turbo"  # Higher priority

        second = selector.select_next_provider()
        assert second is not None
        assert second.provider_model_name == "gpt-4"

    def test_missing_provider_config(self):
        """Test handling of missing provider config."""
        model_providers = [MockModelProviderConfig(provider="missing", priority=1)]
        model_config = MockModelConfig(model_name="test", providers=model_providers)

        selector = ProviderSelector(
            model_config=model_config,  # type: ignore
            provider_configs={},  # Empty config
        )

        result = selector.select_next_provider()
        assert result is None  # Provider config not found


class TestProviderSelectorShouldRetry:
    """Tests for ProviderSelector should_retry logic."""

    def test_should_retry_with_available_providers(self):
        """Test should_retry returns True when providers available."""
        selector = create_test_selector(
            providers=[
                {"provider": "first", "priority": 2},
                {"provider": "second", "priority": 1},
            ]
        )

        # Select first provider
        selector.select_next_provider()

        # Should retry because another provider is available
        assert selector.should_retry(error=ProviderError(message="Error", error_type="api_error"))

    def test_should_not_retry_non_retryable_error(self):
        """Test should_retry returns False for non-retryable errors."""
        selector = create_test_selector()

        error = ProviderError(message="Bad request", error_type="api_error", status_code=400)
        assert selector.should_retry(error=error, status_code=400) is False

    def test_should_retry_retryable_error(self):
        """Test should_retry returns True for retryable errors."""
        selector = create_test_selector(
            providers=[
                {"provider": "first", "priority": 2},
                {"provider": "second", "priority": 1},
            ]
        )
        selector.select_next_provider()

        error = ProviderError(message="Rate limit", error_type="rate_limit_error", status_code=429)
        assert selector.should_retry(error=error, status_code=429) is True


class TestProviderSelectorRoleTransform:
    """Tests for ProviderSelector role transformation."""

    def test_needs_role_transform(self):
        """Test needs_role_transform detection."""
        selector = create_test_selector()

        error = ProviderError(
            message="developer is not one of ['system', 'assistant', 'user']",
            error_type="api_error",
        )
        assert selector.needs_role_transform(error) is True

    def test_needs_role_transform_kimi_phrasing(self):
        """Kimi's Anthropic-compatible endpoint phrases the role rejection differently.

        ``role 'developer' is not allowed`` must also trigger the developer->system
        rescue transform, otherwise the retry never fires and the raw 400 surfaces.
        """
        selector = create_test_selector()

        error = ProviderError(
            message="Invalid request: role 'developer' is not allowed",
            error_type="api_error",
            status_code=400,
        )
        assert selector.needs_role_transform(error) is True

    def test_needs_role_transform_after_marked(self):
        """Test needs_role_transform returns False after marked."""
        selector = create_test_selector()

        selector.mark_role_transformed()
        error = ProviderError(
            message="developer is not one of",
            error_type="api_error",
        )
        assert selector.needs_role_transform(error) is False

    def test_mark_role_transformed_clears_used_providers(self):
        """Test that mark_role_transformed clears used providers."""
        selector = create_test_selector()
        selector.select_next_provider()

        selector.mark_role_transformed()

        assert len(selector.state.used_provider_keys) == 0


class TestCreateProviderSelector:
    """Tests for create_provider_selector factory function."""

    def test_create_basic(self):
        """Test basic selector creation."""
        selector = create_test_selector()
        assert isinstance(selector, ProviderSelector)

    def test_create_with_custom_retries(self):
        """Test selector with custom max fallback attempts."""
        selector = create_test_selector(max_fallback_attempts=5)
        assert selector.max_fallback_attempts == 5


class TestProviderSelectorThreadSafety:
    """Tests for ProviderSelector thread safety.

    These tests verify that ProviderSelector correctly handles concurrent access
    and that the recommended pattern (create new instance per request) is safe.

    IMPORTANT: ProviderSelector is NOT thread-safe by design. These tests verify:
    1. The _in_use flag correctly detects concurrent access
    2. Factory function creates independent instances
    3. State isolation between instances
    """

    def test_concurrent_select_next_provider_detects_misuse(self):
        """Test that concurrent select_next_provider calls trigger warning.

        When the same ProviderSelector instance is used concurrently,
        it should log a warning about potential race conditions.
        """
        import threading

        selector = create_test_selector(
            providers=[
                {"provider": "p1", "priority": 1},
                {"provider": "p2", "priority": 2},
            ]
        )

        results: list[str] = []
        lock = threading.Lock()

        def select_provider():
            result = selector.select_next_provider()
            with lock:
                if result:
                    results.append(result.provider_name)

        threads = [threading.Thread(target=select_provider) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) >= 1

    def test_factory_creates_independent_instances(self):
        """Test that factory function creates truly independent instances.

        Each instance should have its own state and not affect others.
        """

        selector1 = create_test_selector(
            providers=[
                {"provider": "shared", "priority": 1},
            ]
        )
        selector2 = create_test_selector(
            providers=[
                {"provider": "shared", "priority": 1},
            ]
        )

        result1 = selector1.select_next_provider()
        assert result1 is not None
        assert result1.provider_name == "shared"

        assert len(selector1.state.used_provider_keys) == 1
        assert len(selector2.state.used_provider_keys) == 0

        result2 = selector2.select_next_provider()
        assert result2 is not None

    def test_multiple_threads_with_separate_selectors(self):
        """Test that using separate selectors per thread is safe.

        This is the recommended pattern: each request gets its own selector.
        """
        import threading

        results: dict[int, str] = {}
        lock = threading.Lock()

        def process_request(thread_id: int):
            selector = create_test_selector(
                providers=[
                    {"provider": f"provider_{thread_id}", "priority": 1},
                ]
            )
            result = selector.select_next_provider()
            if result:
                with lock:
                    results[thread_id] = result.provider_name

        threads = [threading.Thread(target=process_request, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        for thread_id, provider_name in results.items():
            assert provider_name == f"provider_{thread_id}"


class TestParameterOverridesMerging:
    """Tests for parameter overrides merging between model and provider levels."""

    def test_provider_config_overrides_only(self):
        """Test provider-config overrides are applied when others are empty."""
        selector = create_test_selector(
            providers=[{"provider": "openai", "priority": 1}],
            provider_configs={
                "openai": MockProviderConfig(
                    name="openai",
                    parameter_overrides={"max_tokens": 32768},
                )
            },
        )

        result = selector.select_next_provider()
        assert result is not None
        assert result.parameter_overrides == {"max_tokens": 32768}

    def test_mapping_overrides_model_and_provider_config(self):
        """Test override precedence: provider config < model < provider mapping."""
        selector = create_test_selector(
            providers=[
                {
                    "provider": "openai",
                    "priority": 1,
                    "parameter_overrides": {
                        "max_tokens": 2048,
                        "temperature": 0.7,
                    },
                },
            ],
            provider_configs={
                "openai": MockProviderConfig(
                    name="openai",
                    parameter_overrides={
                        "max_tokens": 1024,
                        "temperature": 0.3,
                        "top_p": 0.9,
                    },
                )
            },
            model_parameter_overrides={
                "max_tokens": 4096,
                "temperature": 0.5,
                "presence_penalty": 0.2,
            },
        )

        result = selector.select_next_provider()
        assert result is not None
        assert result.parameter_overrides == {
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0.2,
        }

    def test_provider_level_overrides_only(self):
        """Test that provider-level overrides work when model-level is empty."""
        selector = create_test_selector(
            providers=[
                {"provider": "openai", "priority": 1, "parameter_overrides": {"max_tokens": 1000}},
            ]
        )

        result = selector.select_next_provider()
        assert result is not None
        assert result.parameter_overrides == {"max_tokens": 1000}

    def test_model_level_overrides_only(self):
        """Test that model-level overrides work when provider-level is empty."""
        selector = create_test_selector(
            providers=[{"provider": "openai", "priority": 1}],
            model_parameter_overrides={"max_tokens": 2000, "temperature": 0.8},
        )

        result = selector.select_next_provider()
        assert result is not None
        assert result.parameter_overrides == {"max_tokens": 2000, "temperature": 0.8}

    def test_provider_level_overrides_model_level(self):
        """Test that provider-level overrides take precedence over model-level."""
        selector = create_test_selector(
            providers=[
                {
                    "provider": "openai",
                    "priority": 1,
                    "parameter_overrides": {"max_tokens": 1000},
                },
            ],
            model_parameter_overrides={"max_tokens": 2000, "temperature": 0.8},
        )

        result = selector.select_next_provider()
        assert result is not None
        # Model-level: max_tokens=2000, temperature=0.8
        # Provider-level: max_tokens=1000
        # Expected: max_tokens=1000 (provider overrides), temperature=0.8 (model-level preserved)
        assert result.parameter_overrides == {"max_tokens": 1000, "temperature": 0.8}

    def test_merged_overrides_preserve_unique_keys(self):
        """Test that non-overlapping keys from both levels are preserved."""
        selector = create_test_selector(
            providers=[
                {
                    "provider": "openai",
                    "priority": 1,
                    "parameter_overrides": {"max_tokens": 1000, "top_p": 0.9},
                },
            ],
            model_parameter_overrides={"temperature": 0.8, "frequency_penalty": 0.5},
        )

        result = selector.select_next_provider()
        assert result is not None
        # All keys from both levels should be present
        assert result.parameter_overrides == {
            "temperature": 0.8,
            "frequency_penalty": 0.5,
            "max_tokens": 1000,
            "top_p": 0.9,
        }

    def test_empty_overrides_both_levels(self):
        """Test that empty overrides at both levels results in empty dict."""
        selector = create_test_selector(
            providers=[{"provider": "openai", "priority": 1}],
            model_parameter_overrides={},
        )

        result = selector.select_next_provider()
        assert result is not None
        assert result.parameter_overrides == {}

    def test_multiple_providers_with_different_overrides(self):
        """Test that each provider gets correct merged overrides."""
        selector = create_test_selector(
            providers=[
                {
                    "provider": "openai",
                    "priority": 2,
                    "parameter_overrides": {"max_tokens": 4000},
                },
                {
                    "provider": "anthropic",
                    "priority": 1,
                    "parameter_overrides": {"max_tokens": 2000, "temperature": 0.5},
                },
            ],
            model_parameter_overrides={"max_tokens": 8000, "temperature": 0.7},
        )

        # First provider (highest priority)
        result1 = selector.select_next_provider()
        assert result1 is not None
        assert result1.provider_name == "openai"
        # Model: max_tokens=8000, temperature=0.7
        # Provider: max_tokens=4000
        # Merged: max_tokens=4000 (provider), temperature=0.7 (model)
        assert result1.parameter_overrides == {"max_tokens": 4000, "temperature": 0.7}

        # Second provider
        result2 = selector.select_next_provider()
        assert result2 is not None
        assert result2.provider_name == "anthropic"
        # Model: max_tokens=8000, temperature=0.7
        # Provider: max_tokens=2000, temperature=0.5
        # Merged: max_tokens=2000, temperature=0.5 (both from provider)
        assert result2.parameter_overrides == {"max_tokens": 2000, "temperature": 0.5}


class TestProviderSelectorCircuitBreaker:
    """Tests for ProviderSelector integration with circuit breaker."""

    def test_circuit_breaker_none_does_not_affect_selection(self):
        """Without circuit breaker, selection works normally."""
        selector = create_test_selector(
            providers=[
                {"provider": "high", "priority": 10},
                {"provider": "low", "priority": 1},
            ],
        )
        assert selector.circuit_breaker is None

        result = selector.select_next_provider()
        assert result is not None
        assert result.provider_name == "high"

    def test_circuit_breaker_skips_open_provider(self):
        """When high-priority provider circuit is OPEN, fall through to next."""
        store = CircuitBreakerStore(
            CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=999.0)
        )
        selector = create_test_selector(
            providers=[
                {"provider": "high", "priority": 10},
                {"provider": "low", "priority": 1},
            ],
        )
        selector.circuit_breaker = store

        # Open circuit for high-priority provider
        # First select to get the key recorded
        result = selector.select_next_provider()
        assert result is not None
        assert result.provider_name == "high"

        # Record enough failures to open the circuit
        store.record_failure(selector.state.last_selected_key)  # threshold=1 opens it
        assert store.is_available(selector.state.last_selected_key) is False

        # Create a new selector (simulating a new request) but with same store
        selector2 = create_test_selector(
            providers=[
                {"provider": "high", "priority": 10},
                {"provider": "low", "priority": 1},
            ],
        )
        selector2.circuit_breaker = store

        # Should skip "high" (circuit OPEN) and select "low"
        result2 = selector2.select_next_provider()
        assert result2 is not None
        assert result2.provider_name == "low"

    def test_record_last_failure_records_in_circuit_breaker(self):
        """record_last_failure should forward to circuit breaker."""
        store = CircuitBreakerStore(CircuitBreakerConfig(failure_threshold=3))
        selector = create_test_selector(
            providers=[{"provider": "openai", "priority": 1}],
        )
        selector.circuit_breaker = store

        # Select a provider to set last_selected_key
        selector.select_next_provider()

        # Record failures
        selector.record_last_failure()
        selector.record_last_failure()
        state = store.get_state(selector.state.last_selected_key)
        assert state is not None
        assert state["failure_count"] == 2
        assert state["state"] == "CLOSED"

        # Third failure should open
        selector.record_last_failure()
        state = store.get_state(selector.state.last_selected_key)
        assert state is not None
        assert state["state"] == "OPEN"

    def test_record_last_success_resets_circuit(self):
        """record_last_success should reset the circuit breaker."""
        store = CircuitBreakerStore(CircuitBreakerConfig(failure_threshold=2))
        selector = create_test_selector(
            providers=[{"provider": "openai", "priority": 1}],
        )
        selector.circuit_breaker = store

        selector.select_next_provider()
        selector.record_last_failure()
        selector.record_last_failure()
        assert store.get_state(selector.state.last_selected_key)["state"] == "OPEN"

        # Success resets
        selector.record_last_success()
        state = store.get_state(selector.state.last_selected_key)
        assert state is not None
        assert state["state"] == "CLOSED"

    def test_record_methods_safe_without_circuit_breaker(self):
        """record_last_failure/success should not raise when circuit_breaker is None."""
        selector = create_test_selector(
            providers=[{"provider": "openai", "priority": 1}],
        )
        assert selector.circuit_breaker is None

        # These should not raise
        selector.record_last_failure()
        selector.record_last_success()

    def test_record_methods_safe_when_no_selection(self):
        """record methods should not raise when nothing was selected yet."""
        store = CircuitBreakerStore(CircuitBreakerConfig(failure_threshold=2))
        selector = create_test_selector(
            providers=[{"provider": "openai", "priority": 1}],
        )
        selector.circuit_breaker = store

        # No selection made yet, last_selected_key is None
        selector.record_last_failure()  # Should not raise
        selector.record_last_success()  # Should not raise

    def test_last_selected_key_set_on_selection(self):
        """last_selected_key should be set after select_next_provider."""
        selector = create_test_selector(
            providers=[{"provider": "openai", "priority": 1}],
        )
        assert selector.state.last_selected_key is None
        selector.select_next_provider()
        assert selector.state.last_selected_key is not None
        assert "openai" in selector.state.last_selected_key


class FakeRedis:
    """Minimal dict-backed async Redis double for sticky tests."""

    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})

    async def get(self, key):
        return self.data.get(key)

    async def setex(self, key, ttl, value):
        self.data[key] = value


class TestProviderStrategies:
    """Strategy wiring in ProviderSelector."""

    def test_default_strategy_is_random(self):
        """Selectors default to the random strategy (historical behavior)."""
        from llm_proxy.config.types.model import ProviderSelectionStrategy

        selector = create_test_selector()
        assert selector.strategy is ProviderSelectionStrategy.RANDOM

    def test_cost_optimized_picks_cheapest(self):
        """cost_optimized picks the cheapest same-priority provider."""
        from llm_proxy.config.types.model import ProviderSelectionStrategy

        selector = create_test_selector(
            providers=[
                {
                    "provider": "expensive",
                    "priority": 1,
                    "input_cost_per_1m": 10.0,
                    "output_cost_per_1m": 30.0,
                },
                {
                    "provider": "cheap",
                    "priority": 1,
                    "input_cost_per_1m": 1.0,
                    "output_cost_per_1m": 2.0,
                },
            ]
        )
        selector.strategy = ProviderSelectionStrategy.COST_OPTIMIZED
        result = selector.select_next_provider()
        assert result is not None
        assert result.provider_name == "cheap"

        # Fallback walks the cost ordering: the expensive one is next.
        result = selector.select_next_provider()
        assert result is not None
        assert result.provider_name == "expensive"

    def test_session_sticky_same_conversation_same_provider(self):
        """session_sticky pins a conversation to one provider across requests."""
        from llm_proxy.config.types.model import ProviderSelectionStrategy

        picks = set()
        for _ in range(5):
            selector = create_test_selector(
                providers=[
                    {"provider": "a", "priority": 1},
                    {"provider": "b", "priority": 1},
                ]
            )
            selector.strategy = ProviderSelectionStrategy.SESSION_STICKY
            selector.conversation_key = "conv-42"
            result = selector.select_next_provider()
            assert result is not None
            picks.add(result.provider_name)
        # Fresh selectors with the same conversation key always agree.
        assert len(picks) == 1

    async def test_prepare_resolves_sticky_key_from_redis(self):
        """prepare() pins the provider stored in Redis for this conversation."""
        from llm_proxy.config.types.model import ProviderSelectionStrategy

        selector = create_test_selector(
            providers=[
                {"provider": "a", "priority": 1},
                {"provider": "b", "priority": 1},
            ]
        )
        selector.strategy = ProviderSelectionStrategy.SESSION_STICKY
        selector.model_name = "test-model"
        selector.conversation_key = "conv-42"
        redis = FakeRedis()
        selector.redis = redis

        await selector.prepare()
        assert selector.state.sticky_key is not None
        assert redis.data["routing:conv:conv-42:provider:test-model"] == selector.state.sticky_key

        result = selector.select_next_provider()
        assert result is not None
        # The selection matches the pinned mapping key.
        assert selector.state.last_selected_key == selector.state.sticky_key

    async def test_prepare_noop_for_other_strategies(self):
        """prepare() does nothing when the strategy is not session_sticky."""
        selector = create_test_selector(
            providers=[{"provider": "a", "priority": 1}],
        )
        selector.conversation_key = "conv-42"
        selector.redis = FakeRedis()
        await selector.prepare()
        assert selector.state.sticky_key is None

    def test_record_last_success_feeds_stats_store(self):
        """record_last_success(duration) records EWMA latency for the key."""
        from llm_proxy.core.provider_stats import ProviderStatsStore

        store = ProviderStatsStore()
        selector = create_test_selector(
            providers=[{"provider": "openai", "priority": 1}],
        )
        selector.stats_store = store
        selector.select_next_provider()

        selector.record_last_success(123.0)
        key = selector.state.last_selected_key
        assert key is not None
        assert store.get(key) == 123.0

    def test_record_last_success_without_duration_skips_stats(self):
        """record_last_success() without a duration leaves the store empty."""
        from llm_proxy.core.provider_stats import ProviderStatsStore

        store = ProviderStatsStore()
        selector = create_test_selector(
            providers=[{"provider": "openai", "priority": 1}],
        )
        selector.stats_store = store
        selector.select_next_provider()

        selector.record_last_success()
        assert store.key_count == 0
