"""Request-free adapter construction and internal out-of-band provider calls.

An internal caller — the routing judge today (ADR-0018), anything else later —
has a ``ProxyConfig``, a model name and an ``app_state``, but no FastAPI
``Request``. Everything needed to reach a provider already lives on
``app_state`` (the shared HTTP client manager, the circuit breaker, the provider
stats store); the only ``Request``-bound code is a handful of ``app.state``
lookups in ``llm_proxy.api.dependencies``. The request-free half lives here so
``llm_proxy.routing`` can construct and call an adapter while keeping the
``routing -> core`` rule (never ``routing -> api``).

An internal call is invisible to the pipeline by construction:
``http.upstream_timing.current_upstream_timer()`` returns ``None`` off the
request path, so it is not traced, not billed and not logged. Callers that need
telemetry record it themselves, on the request the call belongs to.

Two behavioural rules, both consequences of sharing infrastructure with client
traffic:

* **The deadline belongs to the caller.** The provider transport's defaults
  (connect 10 s, read 600 s, retry backoff capped at 30 s) are sized for a client
  request, not for an out-of-band call made while a client waits, so internal
  calls here attempt exactly once and run inside :func:`call_systemone`'s timeout.
  Note the transport's spelling: ``max_retries`` is an **attempt count**, so ``1``
  is one attempt and ``0`` means the operation is never attempted at all (the
  retry policy falls out of its loop and raises ``RuntimeError``).
* **The circuit breaker is not the caller's to trip.** A deadline expiring while
  the provider is healthy — a cold model load, say — must not mark a provider
  down for real traffic, so these helpers never record a failure. An internal
  caller that needs circuit breaking owns a breaker of its own.
"""

import asyncio
import time
from dataclasses import dataclass

from llm_proxy.config.types.model import ModelConfig, ProviderSelectionStrategy
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.adapter import BaseAdapter, get_adapter, list_providers
from llm_proxy.core.circuit_breaker import CircuitBreakerStore
from llm_proxy.core.exceptions import (
    AdapterNotFoundError,
    ConfigurationError,
    NotFoundError,
    ValidationError,
)
from llm_proxy.core.provider_selector import ProviderSelectionResult, create_provider_selector
from llm_proxy.core.provider_stats import ProviderStatsStore
from llm_proxy.http.client import AsyncSession, ProviderHTTPClientManager
from llm_proxy.models.systemone import InternalSystemOneRequest, InternalSystemOneResponse
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

#: Default deadline for an internal call. Short by construction: the caller is
#: holding a client request open, so a slow internal call is a failure, not a wait.
DEFAULT_INTERNAL_DEADLINE_S = 3.0

#: Attempts for an internal call, i.e. no retries. ``RetryPolicy`` counts attempts,
#: so ``1`` is one try and ``0`` never tries at all.
INTERNAL_CALL_ATTEMPTS = 1

#: Grace period for closing a client this call opened. The call's deadline has
#: already been spent by the time ``finally`` runs, so the close gets a small
#: budget of its own; a close that hangs must not stall the waiting client.
INTERNAL_CLOSE_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class InternalCallOutcome:
    """What an internal call returned, or how it failed.

    Failure is data, not an exception: an internal caller's fallback (for the
    routing judge, "the ensemble decides") is a decision, and every failure mode
    — no provider, transport error, deadline — lands here identically.
    """

    response: InternalSystemOneResponse | None = None
    error: str | None = None
    latency_ms: float = 0.0
    provider_name: str | None = None
    provider_model_name: str | None = None

    @property
    def ok(self) -> bool:
        """True when a response came back."""
        return self.response is not None


def build_provider_adapter(
    provider_name: str,
    provider_config: ProviderConfig,
    http_client: AsyncSession | None = None,
    unknown_fields_policy: str | None = None,
    unsupported_block_policy: str | None = None,
    http_client_manager: ProviderHTTPClientManager | None = None,
    max_retries: int = 3,
) -> BaseAdapter:
    """Build a provider adapter from configuration alone, with no ``Request``.

    The API layer's ``create_adapter_for_provider`` is this function plus the
    process-lifetime services it reads (:mod:`llm_proxy.services`); nothing here
    reads request state. With neither ``http_client`` nor ``http_client_manager`` the adapter
    lazily opens its own client, which is correct but unpooled — pass the shared
    manager when one is available.
    """
    if not provider_config.type or not provider_config.type.strip():
        raise ValidationError(
            message=(
                f"Provider '{provider_name}' has no type configured. "
                f"Please ensure provider_config.type is set (e.g., 'ollama', 'openai')."
            ),
        )

    try:
        return get_adapter(
            provider_config.type,
            provider_name=provider_name,
            api_key=provider_config.get_api_key(),
            base_url=provider_config.base_url,
            timeout=provider_config.timeout,
            max_retries=max_retries,
            custom_headers=provider_config.custom_headers,
            parameter_overrides=provider_config.parameter_overrides,
            endpoint_base_urls=provider_config.endpoint_base_urls,
            # Kill switch for providers with native-protocol endpoints
            # (e.g. DeepSeek's Anthropic/Responses passthrough); forwarded
            # opaquely through AdapterConfig.extra — only adapters that read
            # it are affected.
            native_passthrough=provider_config.metadata.get("native_passthrough", True),
            # Upstream API dialect for Gemini ("generate_content" default |
            # "interactions") — forwarded opaquely through AdapterConfig.extra;
            # only the Gemini adapter reads it.
            api_variant=provider_config.metadata.get("api_variant", "generate_content"),
            # App-attribution identity (OpenRouter's HTTP-Referer /
            # X-OpenRouter-Title). Forwarded opaquely; only adapters for
            # upstreams that track client apps read it.
            app_attribution=provider_config.app_attribution.model_dump(),
            unknown_fields_policy=unknown_fields_policy,
            unsupported_block_policy=unsupported_block_policy,
            http_client=http_client,
            http_client_manager=http_client_manager,
        )
    except AdapterNotFoundError as e:
        raise NotFoundError(
            message=f"Provider '{provider_name}' not found. Available: {list_providers()}",
        ) from e
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(
            message=f"Failed to create provider adapter: {e}",
        ) from e


def select_internal_provider(
    model_config: ModelConfig,
    provider_configs: dict[str, ProviderConfig],
    *,
    circuit_breaker: CircuitBreakerStore | None = None,
    stats_store: ProviderStatsStore | None = None,
) -> ProviderSelectionResult | None:
    """Pick the first available provider for a model, without HTTP or Redis.

    Deliberately the plain ``RANDOM`` strategy: an internal call has no session
    affinity to honour and no fallback walk to run — if the first choice is
    unavailable the caller falls back to its own default, not to a second
    provider. Returns ``None`` when nothing is available.
    """
    selector = create_provider_selector(
        model_config=model_config,
        provider_configs=provider_configs,
        circuit_breaker=circuit_breaker,
        strategy=ProviderSelectionStrategy.RANDOM,
        stats_store=stats_store,
        default_max_retries=INTERNAL_CALL_ATTEMPTS,
    )
    return selector.select_next_provider()


async def call_systemone(
    request: InternalSystemOneRequest,
    *,
    model_config: ModelConfig | None,
    provider_configs: dict[str, ProviderConfig],
    http_client: AsyncSession | None = None,
    http_client_manager: ProviderHTTPClientManager | None = None,
    circuit_breaker: CircuitBreakerStore | None = None,
    stats_store: ProviderStatsStore | None = None,
    unknown_fields_policy: str | None = None,
    unsupported_block_policy: str | None = None,
    deadline_s: float = DEFAULT_INTERNAL_DEADLINE_S,
) -> InternalCallOutcome:
    """Evaluate a System One request inside a hard deadline, never raising.

    Zero retries, one deadline, and every failure returned as data so the caller
    can fall back without exception plumbing.
    """
    started = time.perf_counter()
    if model_config is None:
        return InternalCallOutcome(error="unknown model", latency_ms=0.0)

    selection = select_internal_provider(
        model_config,
        provider_configs,
        circuit_breaker=circuit_breaker,
        stats_store=stats_store,
    )
    if selection is None:
        return InternalCallOutcome(
            error="no available provider",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # An adapter built without a concrete client opens one of its own (with a logged
    # warning); this call owns that client and must close it. When a manager is
    # available the pool owns the session instead: resolve it here, because the
    # adapter reads only ``http_client`` and would otherwise open — and leak — its own.
    owns_client = http_client is None
    adapter: BaseAdapter | None = None
    try:
        if http_client is None and http_client_manager is not None:
            http_client = await http_client_manager.get_client(selection.provider_name)
            owns_client = False
        adapter = build_provider_adapter(
            selection.provider_name,
            selection.provider_config,
            http_client=http_client,
            unknown_fields_policy=unknown_fields_policy,
            unsupported_block_policy=unsupported_block_policy,
            http_client_manager=http_client_manager,
            # One attempt, never a retry: a retry inside a client's request is a
            # longer stall, not a second chance (see the module docstring).
            max_retries=INTERNAL_CALL_ATTEMPTS,
        )
        async with asyncio.timeout(deadline_s):
            response = await adapter.systemone(request)
    except TimeoutError:
        # The caller's own deadline, not an upstream read timeout: the transport's
        # read timeout is measured in minutes, and this one is measured in the
        # milliseconds a waiting client can spare.
        return InternalCallOutcome(
            error=f"deadline exceeded ({deadline_s:g}s)",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider_name=selection.provider_name,
            provider_model_name=selection.provider_model_name,
        )
    except Exception as exc:  # noqa: BLE001 - an internal failure is an abstention
        detail = str(exc).strip()
        return InternalCallOutcome(
            error=f"{exc.__class__.__name__}: {detail}" if detail else exc.__class__.__name__,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider_name=selection.provider_name,
            provider_model_name=selection.provider_model_name,
        )
    finally:
        if owns_client and adapter is not None:
            # ``close`` is transport I/O that runs after the call's deadline has
            # expired. A failure here must not turn a successful call into an
            # exception (the never-raising contract covers cleanup too), and a
            # slow close must not hold the waiting client past its own budget.
            try:
                await asyncio.wait_for(adapter.close(), timeout=INTERNAL_CLOSE_TIMEOUT_S)
            except Exception:  # noqa: BLE001 - cleanup failure is not the caller's problem
                logger.debug("internal call client close failed", exc_info=True)

    return InternalCallOutcome(
        response=response,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        provider_name=selection.provider_name,
        provider_model_name=selection.provider_model_name,
    )


__all__ = [
    "DEFAULT_INTERNAL_DEADLINE_S",
    "INTERNAL_CALL_ATTEMPTS",
    "INTERNAL_CLOSE_TIMEOUT_S",
    "InternalCallOutcome",
    "build_provider_adapter",
    "call_systemone",
    "select_internal_provider",
]
