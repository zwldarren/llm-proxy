"""Shared fallback plumbing for provider retry logic.

Used by both the non-streaming execution path (``stages.request_execution`` /
RetryExecutor) and the streaming path (``stages.fallback_handler`` /
FallbackHandler) so that retry classification, role-transform detection,
failure recording, and provider swapping behave identically across both.

Lives inside the ``stages`` package so it can import the stage composition
(``stages.composition``) at module level without an import cycle — the
former ``core/processing/fallback.py`` was imported BY ``stages/*`` and
therefore had to lazily import the stages it needed to re-run.
"""

from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from fastapi import Request

from llm_proxy.core.exceptions import LLMProxyError, ProviderError
from llm_proxy.core.processing.stages.composition import rerun_per_provider_stages
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.core.adapter import BaseAdapter
    from llm_proxy.core.processing.base import RequestContext
    from llm_proxy.core.processing.stages.parameter_override import ParameterOverrideService
    from llm_proxy.core.provider_selector import ProviderSelector
    from llm_proxy.models import InternalRequest

logger = get_logger(__name__)


class FallbackAction(StrEnum):
    """Unified fallback decision shared by the streaming and non-streaming paths."""

    ABORT = "abort"
    RETRY_ROLE_TRANSFORM = "retry_role_transform"
    RETRY_NEXT_PROVIDER = "retry_next_provider"


@dataclass
class FallbackDecision:
    """Result of :func:`plan_fallback`."""

    action: FallbackAction
    error: ProviderError
    provider_name: str


def plan_fallback(
    error: Exception,
    orchestrator: ProviderSelector,
    provider_name: str,
) -> FallbackDecision:
    """Classify an error and decide the unified fallback action.

    Wraps raw (non-``ProviderError``) exceptions into a 500 ``ProviderError`` so
    both execution paths reason about a single error type. Delegates the retry
    classification and role-transform detection to the shared ``ProviderSelector``
    so streaming and non-streaming behave identically.

    Returns:
        A ``FallbackDecision`` with one of three actions:
        - ``ABORT``: do not retry (terminal error or selector exhausted).
        - ``RETRY_ROLE_TRANSFORM``: retry the *same* provider with the
          ``developer`` role normalized to ``system``.
        - ``RETRY_NEXT_PROVIDER``: fall back to the next priority provider.
    """
    if isinstance(error, ProviderError):
        wrapped = error
    else:
        wrapped = ProviderError(
            message=f"Unexpected error from provider: {type(error).__name__}: {error}",
            error_type="api_error",
            status_code=500,
            provider_name=provider_name,
        )

    if not orchestrator.should_retry(error=wrapped, status_code=wrapped.status_code):
        return FallbackDecision(FallbackAction.ABORT, wrapped, provider_name)
    if orchestrator.needs_role_transform(wrapped):
        return FallbackDecision(FallbackAction.RETRY_ROLE_TRANSFORM, wrapped, provider_name)
    return FallbackDecision(FallbackAction.RETRY_NEXT_PROVIDER, wrapped, provider_name)


def record_fallback(
    decision: FallbackDecision,
    orchestrator: ProviderSelector,
    event_context: EventContext | None,
    *,
    provider_type: str | None,
) -> None:
    """Record a fallback to a *different* provider.

    Updates the circuit breaker for the failed provider and appends a fallback
    attempt record to the event context. Must only be called for
    ``RETRY_NEXT_PROVIDER`` -- a same-provider role-transform retry is not a
    fallback and is intentionally not recorded here.
    """
    orchestrator.record_last_failure()
    record_fallback_attempt(
        event_context,
        decision.provider_name,
        decision.error,
        status_code=decision.error.status_code,
        provider_type=provider_type,
    )


async def swap_adapters(
    exit_stack: AsyncExitStack,
    old_adapter: BaseAdapter,
    new_adapter: BaseAdapter,
) -> None:
    """Close the old adapter and register the new one on the exit stack."""
    await old_adapter.close()
    old = exit_stack.pop_all()
    await old.aclose()
    await exit_stack.enter_async_context(new_adapter)


async def execute_fallback(
    decision: FallbackDecision,
    orchestrator: ProviderSelector,
    *,
    event_context: EventContext | None,
    provider_type: str | None,
    exit_stack: AsyncExitStack,
    current_adapter: BaseAdapter,
    req: Request,
    unified_request: InternalRequest,
    raw_request_data: dict[str, Any],
    context: RequestContext,
    param_override_service: ParameterOverrideService,
    process_request: Any = None,
) -> tuple[BaseAdapter, InternalRequest] | None:
    """Record the failure and swap in the next priority provider.

    ``raw_request_data`` must be the pristine client body
    (``PipelineState.original_raw_data``), never the override-modified copy.

    Returns the new ``(adapter, request)`` pair, or ``None`` when no further
    provider is available.
    """
    record_fallback(
        decision,
        orchestrator,
        event_context,
        provider_type=provider_type,
    )
    selection = orchestrator.select_next_provider()
    if selection is None:
        return None
    result = await setup_fallback_provider(
        selection,
        req,
        unified_request,
        raw_request_data,
        context,
        param_override_service,
    )
    if result is None:
        return None
    new_adapter, new_request = result
    await swap_adapters(exit_stack, current_adapter, new_adapter)
    if process_request is not None:
        new_request = await process_request(new_request, new_adapter)
    return new_adapter, new_request


def record_fallback_attempt(
    event_context: EventContext | None,
    provider_name: str,
    error: Exception,
    status_code: int | None = None,
    provider_type: str | None = None,
) -> None:
    """Record a failed provider attempt for fallback tracking.

    Args:
        event_context: Event context to record to
        provider_name: User's configured provider name
        error: The error that occurred
        status_code: HTTP status code if available
        provider_type: Provider type (e.g., "openai", "anthropic")
    """
    if event_context is None:
        return

    error_info: dict[str, Any] = {
        "provider": provider_name,
        "error_message": str(error),
        "status_code": status_code,
    }

    if provider_type is not None:
        error_info["provider_type"] = provider_type

    if isinstance(error, ProviderError):
        error_info["error_type"] = error.error_type
        error_info["provider_error_name"] = error.provider_name

    event_context.fallback_attempts.append(error_info)
    logger.info(
        f"Recorded fallback attempt: provider={provider_name}, "
        f"type={provider_type}, "
        f"error={error_info.get('error_type', type(error).__name__)}"
    )


async def _create_fallback_adapter(
    selection: Any, req: Any, context: RequestContext
) -> BaseAdapter:
    """Create the adapter for a fallback selection and bind it to the context.

    Surfaces same-provider retries from the new adapter's RetryPolicy into
    the EventContext so they appear in log_metadata / the frontend logs page,
    and records the selected provider on the request and context.
    """
    adapter = await context.adapter_factory(req, selection)
    event_context = context.event_context
    if event_context is not None:
        adapter.set_retry_recorder(event_context.retry_attempts.append)
        event_context.provider = selection.provider_name
        if selection.provider_model_name:
            event_context.provider_model_name = selection.provider_model_name
    req.state.provider = selection.provider_name
    return adapter


def _rebuild_fallback_request(
    selection: Any,
    req: Any,
    unified_request: InternalRequest,
    raw_request_data: dict[str, Any],
    context: RequestContext,
    param_override_service: ParameterOverrideService,
) -> tuple[dict[str, Any], InternalRequest]:
    """Re-parse the pristine body with this provider's parameter overrides.

    Always re-parses — even with no overrides — so the failed provider's
    overridden values are shed from the request object.
    """
    new_raw_data, new_request = param_override_service.apply(
        raw_data=raw_request_data,
        unified_request=unified_request,
        parameter_overrides=selection.parameter_overrides or {},
        provider_model_name=selection.provider_model_name,
        request_id=getattr(req.state, "request_id", None),
    )
    if context.event_context is not None:
        context.event_context.request_body = new_raw_data
    return new_raw_data, new_request


async def setup_fallback_provider(
    selection: Any,
    req: Any,
    unified_request: InternalRequest,
    raw_request_data: dict[str, Any],
    context: RequestContext,
    param_override_service: ParameterOverrideService,
) -> tuple[BaseAdapter, InternalRequest] | None:
    """Set up a fallback provider after the primary provider fails.

    The request is rebuilt from ``raw_request_data``, which MUST be the
    pristine client body (``PipelineState.original_raw_data``): each provider
    attempt applies its own parameter overrides to the original body so the
    failed provider's overrides never leak into the next attempt, then
    re-runs the per-provider request stages (previous-response resolution,
    web search) on the fresh parse via
    :func:`rerun_per_provider_stages` — the same composition owner the main
    pipeline consumes.

    Providers whose stage re-run rejects the request (e.g. a proxy-local
    ``previous_response_id`` the provider cannot resolve) are skipped and the
    next priority provider is tried.

    Args:
        selection: The ProviderSelectionResult from select_next_provider()
        req: The FastAPI request
        unified_request: The current unified request
        raw_request_data: Pristine client request body for parameter overrides
        context: Request context containing orchestrator and dependencies
        param_override_service: Service to apply parameter overrides

    Returns:
        Tuple of (new_adapter, updated_unified_request), or None when no
        viable provider remains.
    """
    # Guard against a selector that re-offers the same provider mapping
    # (the contract is that select_next_provider eventually exhausts; a
    # violation would otherwise loop forever when stage re-run keeps
    # rejecting the request).
    seen: set[tuple[Any, Any]] = set()
    while selection is not None:
        selection_key = (
            selection.provider_name,
            getattr(selection, "provider_model_name", None),
        )
        if selection_key in seen:
            logger.error(
                f"Provider selector re-offered {selection_key} during fallback "
                "setup; aborting fallback"
            )
            return None
        seen.add(selection_key)
        adapter = await _create_fallback_adapter(selection, req, context)
        new_raw_data, new_request = _rebuild_fallback_request(
            selection,
            req,
            unified_request,
            raw_request_data,
            context,
            param_override_service,
        )

        try:
            await rerun_per_provider_stages(
                selection, adapter, req, new_request, new_raw_data, context
            )
        except LLMProxyError as e:
            # This provider cannot serve the rebuilt request (e.g. an
            # unresolvable proxy-local previous_response_id on a non-native
            # upstream). Record the attempt and try the next provider instead
            # of aborting the fallback chain.
            logger.info(
                f"Fallback provider {selection.provider_name} rejected the request "
                f"during stage re-run: {e}; trying next provider"
            )
            record_fallback_attempt(
                context.event_context,
                selection.provider_name,
                e,
                status_code=getattr(e, "status_code", None),
                provider_type=adapter.provider_name,
            )
            with suppress(Exception):
                await adapter.close()
            selection = context.orchestrator.select_next_provider()
            continue

        return adapter, new_request

    return None
