"""Provider selector for priority-based provider selection and fallback."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_proxy.config.types.model import (
    DEFAULT_PROVIDER_SELECTION_STRATEGY,
    ModelConfig,
    ModelProviderConfig,
    ProviderSelectionStrategy,
)
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.attempt_tracker import ProviderAttempt
from llm_proxy.core.constants import DEFAULT_MAX_FALLBACK_ATTEMPTS, DEFAULT_MAX_RETRIES
from llm_proxy.core.errors import (
    ErrorCategory,
    classify_error,
)
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.provider_stats import DEFAULT_LATENCY_KIND, LatencyKind
from llm_proxy.core.provider_strategy import (
    StrategyContext,
    pick_provider,
    resolve_sticky_key,
)
from llm_proxy.core.selection_state import SelectionState
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.core.circuit_breaker import CircuitBreakerStore
    from llm_proxy.core.provider_stats import ProviderStatsStore

logger = get_logger(__name__)


@dataclass
class ProviderSelectionResult:
    """Result of provider selection."""

    provider_name: str
    provider_config: ProviderConfig
    provider_model_name: str | None
    priority: int
    parameter_overrides: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3  # Resolved effective retry count for this provider


def _get_provider_mapping_key(provider: ModelProviderConfig, index: int) -> str:
    """Generate a unique key for a provider mapping.

    This allows the same provider to be used multiple times with different
    model names. The key combines provider name, model name, and index
    to ensure uniqueness.

    The index is a position in the *canonical* enumeration built by
    :meth:`ProviderSelector._get_providers_by_priority_with_keys` (priority
    descending, then content order), never in whatever order the provider list
    happened to arrive in — see :func:`_mapping_order_key` for why. The format
    itself is read by ``api/routers/config/circuit_breaker.py``
    (``_parse_provider_key``) and rendered in the console, so it is a contract:
    keep the three colon-separated parts.

    Args:
        provider: The provider configuration
        index: Position in the canonical enumeration (for uniqueness)

    Returns:
        A unique string key for this provider mapping
    """
    return f"{provider.provider}:{provider.provider_model_name or ''}:{index}"


def _overrides_fingerprint(overrides: dict[str, Any] | None) -> str:
    """Short, order-independent digest of a mapping's parameter overrides.

    Two mappings of one provider and upstream model can differ only by their
    overrides, so the overrides are part of a mapping's identity. JSON with
    sorted keys makes the digest independent of dict insertion order.
    """
    if not overrides:
        return ""
    payload = json.dumps(overrides, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _mapping_order_key(provider: ModelProviderConfig) -> tuple[str, str, str]:
    """Canonical ordering key for the mappings of one priority group.

    Provider mapping keys carry a positional index, and that index is what the
    circuit breaker, the latency store and the Redis sticky mapping are keyed
    by. Enumeration therefore must depend on *which* mappings exist, never on
    the order rows came back from the database or a config was submitted in:
    a mapping that shifts position would otherwise inherit another mapping's
    breaker state and sticky pins, and every conversation would be re-hashed
    onto a new provider after an unrelated edit.

    Mappings that compare equal here are indistinguishable upstream (same
    provider, upstream model and overrides), so the stable sort keeps their
    relative order and swapping them is harmless.
    """
    return (
        provider.provider,
        provider.provider_model_name or "",
        _overrides_fingerprint(provider.parameter_overrides),
    )


@dataclass
class ProviderSelector:
    """Handles priority-based provider selection with fallback support.

    This class manages the selection of providers for a model based on their
    configured priorities. The same provider can be configured multiple times
    with different model names.

    State management is delegated to SelectionState, which holds per-request
    mutable state (attempt tracking, exhaustion, timing, etc.). Each request
    should use a fresh SelectionState created via create_state().

    Attributes:
        model_config: The model configuration containing provider priorities
        provider_configs: Dictionary mapping provider names to their configurations
        max_fallback_attempts: Maximum number of fallback provider switches.
            Configurable via ServerParams.max_fallback_attempts. Default: 10
        default_max_retries: Default retry count per provider (from global config);
            can be overridden per-model via ModelConfig.max_retries.
        state: The SelectionState tracking per-request mutable state.
        circuit_breaker: Optional shared circuit breaker store.
        strategy: Selection strategy applied within the highest available
            priority group (default: the global default,
            :data:`DEFAULT_PROVIDER_SELECTION_STRATEGY`).
        model_name: Proxy-facing model name (namespaces the sticky mapping).
        conversation_key: Stable conversation identifier for session_sticky.
        redis: Optional async Redis client for the sticky provider mapping.
        stats_store: Optional EWMA latency stats for the balanced strategy.
        latency_kind: Which latency bucket this request's measurement (and the
            balanced strategy's comparison) belongs to — time-to-first-token for
            a streaming request, total duration otherwise.
    """

    model_config: ModelConfig
    provider_configs: dict[str, ProviderConfig]
    max_fallback_attempts: int = DEFAULT_MAX_FALLBACK_ATTEMPTS
    default_max_retries: int = DEFAULT_MAX_RETRIES
    state: SelectionState = field(default_factory=SelectionState)
    circuit_breaker: CircuitBreakerStore | None = None
    strategy: ProviderSelectionStrategy = DEFAULT_PROVIDER_SELECTION_STRATEGY
    model_name: str = ""
    conversation_key: str | None = None
    redis: Any | None = None
    stats_store: ProviderStatsStore | None = None
    latency_kind: LatencyKind = DEFAULT_LATENCY_KIND

    @property
    def attempt_count(self) -> int:
        """Number of provider attempts made so far."""
        return self.state.attempt_count

    def _get_providers_by_priority_with_keys(
        self,
    ) -> list[list[tuple[ModelProviderConfig, str]]]:
        """Group providers by priority level with their unique keys.

        The enumeration is canonical (see :func:`_mapping_order_key`): priority
        groups descending, and inside a group the mappings in content order.
        That is what makes a mapping key's positional index reproducible, so
        breaker state, latency samples and sticky pins stay attached to the
        same mapping across reloads and workers.

        Returns:
            List of provider groups, sorted by priority (highest first).
            Each group contains tuples of (provider_config, unique_key).
        """
        grouped: dict[int, list[ModelProviderConfig]] = {}
        for prov in self.model_config.get_providers_by_priority():
            grouped.setdefault(prov.priority, []).append(prov)

        # Sort by priority descending, then by mapping content inside a group,
        # and hand out indices in that single canonical order.
        ordered: list[ModelProviderConfig] = []
        for priority in sorted(grouped, reverse=True):
            ordered.extend(sorted(grouped[priority], key=_mapping_order_key))

        priority_groups: dict[int, list[tuple[ModelProviderConfig, str]]] = {}
        for idx, prov in enumerate(ordered):
            key = _get_provider_mapping_key(prov, idx)
            priority_groups.setdefault(prov.priority, []).append((prov, key))

        return [priority_groups[p] for p in sorted(priority_groups, reverse=True)]

    def _strategy_context(self) -> StrategyContext:
        """Assemble the per-request strategy inputs."""
        return StrategyContext(
            model_config=self.model_config,
            model_name=self.model_name,
            conversation_key=self.conversation_key,
            redis=self.redis,
            stats_store=self.stats_store,
            latency_kind=self.latency_kind,
        )

    def _first_available_group(self) -> list[tuple[ModelProviderConfig, str]]:
        """Return the available candidates of the highest-priority non-empty group."""
        for group in self._get_providers_by_priority_with_keys():
            available = [
                (p, k)
                for p, k in group
                if k not in self.state.used_provider_keys
                and (self.circuit_breaker is None or self.circuit_breaker.is_available(k))
            ]
            if available:
                return available
        return []

    async def prepare(self) -> None:
        """Resolve per-request async selection inputs before the first pick.

        For the ``session_sticky`` strategy with Redis configured, resolves
        and pins this conversation's provider mapping key. No-op for other
        strategies, when sticky resolution is unavailable (the ordering then
        degrades to stateless rendezvous hashing at pick time), and when the
        highest priority group holds a single candidate — there is nothing to
        choose, so a Redis round trip would only add latency.
        """
        if self.strategy is not ProviderSelectionStrategy.SESSION_STICKY:
            return
        if not self.conversation_key or self.redis is None:
            return
        available = self._first_available_group()
        if len(available) < 2:
            return
        self.state.sticky_key = await resolve_sticky_key(available, self._strategy_context())

    def select_next_provider(self) -> ProviderSelectionResult | None:
        """Select the next provider to try.

        Returns:
            ProviderSelectionResult if a provider is available, None if exhausted
        """
        if self.state.exhausted:
            return None

        if self.state.attempt_count >= self.max_fallback_attempts:
            logger.warning(
                f"Max fallback attempts ({self.max_fallback_attempts}) reached for model"
            )
            self.state.exhausted = True
            return None

        priority_groups = self._get_providers_by_priority_with_keys()
        for group in priority_groups:
            available = [
                (p, k)
                for p, k in group
                if k not in self.state.used_provider_keys
                and (self.circuit_breaker is None or self.circuit_breaker.is_available(k))
            ]
            if available:
                picked = pick_provider(
                    available,
                    self.strategy,
                    self._strategy_context(),
                    sticky_key=self.state.sticky_key,
                )
                if picked is None:
                    continue
                selected, selected_key = picked
                self.state.used_provider_keys.add(selected_key)
                self.state.last_selected_key = selected_key
                provider_config = self.provider_configs.get(selected.provider)
                if not provider_config:
                    logger.warning(f"Provider config not found for '{selected.provider}'")
                    continue
                provider_model_name = selected.provider_model_name or self.model_config.model_name

                # Merge overrides from least-specific to most-specific:
                # provider config -> model config -> per-model provider mapping
                provider_overrides = getattr(provider_config, "parameter_overrides", {}) or {}
                merged_overrides = {
                    **provider_overrides,
                    **(self.model_config.parameter_overrides or {}),
                    **(selected.parameter_overrides or {}),
                }

                # Resolve effective retry count: model-level overrides global default
                effective_max_retries = (
                    self.model_config.max_retries
                    if self.model_config.max_retries is not None
                    else self.default_max_retries
                )

                return ProviderSelectionResult(
                    provider_name=selected.provider,
                    provider_config=provider_config,
                    provider_model_name=provider_model_name,
                    priority=selected.priority,
                    parameter_overrides=merged_overrides,
                    max_retries=effective_max_retries,
                )
        self.state.exhausted = True
        return None

    def should_retry(
        self,
        error: Exception | None = None,
        status_code: int | None = None,
    ) -> bool:
        """Determine if we should retry with another provider.

        Any error is eligible for fallback as long as another provider mapping
        remains. Error classification is only used to detect the
        developer->system role transform and to enrich the log message -- it
        never blocks a fallback. Upstreams routinely return a generic status
        (e.g. a context-window rejection as a plain 400, or an error type we do
        not recognize), so trusting the classification would surface errors that
        a lower-priority provider could have served.

        Args:
            error: The exception that occurred
            status_code: The HTTP status code

        Returns:
            True if we should try another provider, False otherwise
        """
        if self.state.exhausted:
            return False

        error_category = classify_error(error, status_code)

        # A role rejection is the one case worth handling on the *same*
        # provider first: normalize ``developer`` -> ``system`` and retry before
        # spending a fallback. Once transformed, role errors fall through to
        # normal fallback handling like any other error.
        if error_category == ErrorCategory.ROLE_ERROR and not self.state.role_transformed:
            logger.info("Unsupported role detected, will retry with role transformation")
            return True

        if self._has_available_candidate():
            logger.info(
                f"Provider error ({error_category.value}, status={status_code}), "
                "attempting fallback to next provider"
            )
            return True

        logger.info(
            f"Provider error ({error_category.value}, status={status_code}) and no "
            "further provider available, not attempting fallback"
        )
        return False

    def _has_available_candidate(self) -> bool:
        """Whether another provider mapping can still be tried.

        Mirrors the availability filter used by ``select_next_provider`` so the
        retry decision agrees with the selection that follows it.
        """
        for group in self._get_providers_by_priority_with_keys():
            for _provider, key in group:
                if key in self.state.used_provider_keys:
                    continue
                if self.circuit_breaker is not None and not self.circuit_breaker.is_available(key):
                    continue
                return True
        return False

    def record_last_failure(self) -> None:
        """Record a failure for the last selected provider in the circuit breaker."""
        if self.circuit_breaker is not None and self.state.last_selected_key is not None:
            self.circuit_breaker.record_failure(self.state.last_selected_key)

    def record_last_success(
        self,
        duration_ms: float | None = None,
        kind: LatencyKind = DEFAULT_LATENCY_KIND,
    ) -> None:
        """Record a success for the last selected provider.

        Resets the circuit breaker and, when a latency measurement is
        provided, feeds the EWMA stats store used by the ``balanced``
        strategy. The measurement goes into the bucket named by *kind*
        (streaming requests pass TTFT, non-streaming pass total response
        time) so the two are never averaged together.
        """
        key = self.state.last_selected_key
        if key is None:
            return
        if self.circuit_breaker is not None:
            self.circuit_breaker.record_success(key)
        if duration_ms is not None and self.stats_store is not None:
            self.stats_store.observe(key, duration_ms, kind)

    def needs_role_transform(self, error: Exception | None = None) -> bool:
        """Check if the request needs role transformation (developer to system).

        Args:
            error: The exception that occurred

        Returns:
            True if role transformation is needed
        """
        if self.state.role_transformed:
            return False
        return classify_error(error) == ErrorCategory.ROLE_ERROR

    def mark_role_transformed(self) -> None:
        """Mark that role transformation has been performed."""
        self.state.role_transformed = True
        self.state.used_provider_keys.clear()
        logger.info("Role transformation (developer -> system) applied, retrying request")

    @property
    def exhausted(self) -> bool:
        """Check if the selector is exhausted."""
        return self.state.exhausted

    @property
    def last_error(self) -> ProviderError | None:
        """Get the last recorded error."""
        return self.state.last_error

    def reset_stream_state(self) -> None:
        """Reset stream state for a new provider attempt."""
        self.state.stream_started = False


def create_provider_selector(
    model_config: ModelConfig,
    provider_configs: dict[str, ProviderConfig],
    max_fallback_attempts: int = DEFAULT_MAX_FALLBACK_ATTEMPTS,
    default_max_retries: int = DEFAULT_MAX_RETRIES,
    circuit_breaker: CircuitBreakerStore | None = None,
    strategy: ProviderSelectionStrategy = DEFAULT_PROVIDER_SELECTION_STRATEGY,
    model_name: str = "",
    conversation_key: str | None = None,
    redis: Any | None = None,
    stats_store: ProviderStatsStore | None = None,
    latency_kind: LatencyKind = DEFAULT_LATENCY_KIND,
) -> ProviderSelector:
    """Create a ProviderSelector for a model.

    Args:
        model_config: The model configuration
        provider_configs: Dictionary of provider configurations
        max_fallback_attempts: Maximum number of fallback provider switches
        default_max_retries: Default retry count per provider
        circuit_breaker: Optional shared circuit breaker store
        strategy: Selection strategy within same-priority groups
            (defaults to the global default, ``session_sticky``)
        model_name: Proxy-facing model name (sticky mapping namespace)
        conversation_key: Stable conversation id for session_sticky
        redis: Optional async Redis client for the sticky mapping
        stats_store: Optional EWMA latency stats for the balanced strategy
        latency_kind: Latency bucket this request is measured in (streaming:
            time-to-first-token, otherwise total duration)

    Returns:
        A configured ProviderSelector instance
    """
    return ProviderSelector(
        model_config=model_config,
        provider_configs=provider_configs,
        max_fallback_attempts=max_fallback_attempts,
        default_max_retries=default_max_retries,
        circuit_breaker=circuit_breaker,
        strategy=strategy,
        model_name=model_name,
        conversation_key=conversation_key,
        redis=redis,
        stats_store=stats_store,
        latency_kind=latency_kind,
    )


__all__ = [
    "DEFAULT_MAX_FALLBACK_ATTEMPTS",
    "DEFAULT_MAX_RETRIES",
    "ErrorCategory",
    "ProviderAttempt",
    "ProviderSelectionResult",
    "ProviderSelector",
    "SelectionState",
    "classify_error",
    "create_provider_selector",
]
