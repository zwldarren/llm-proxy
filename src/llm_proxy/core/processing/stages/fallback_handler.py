"""Fallback handling for streaming requests.

Extracted from StreamingProcessor to consolidate all provider fallback
logic into a single, testable class. Retry classification, role-transform
detection, failure recording, and provider swapping are delegated to the
shared helpers in ``llm_proxy.core.processing.fallback`` so the streaming and
non-streaming paths behave identically.
"""

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import Response

from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.errors.handler import ErrorHandler
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.fallback import (
    FallbackAction,
    execute_fallback,
    plan_fallback,
    record_fallback,
    setup_fallback_provider,
    swap_adapters,
)
from llm_proxy.core.processing.stages.parameter_override import ParameterOverrideService
from llm_proxy.core.processing.stages.role_normalization import normalize_developer_roles
from llm_proxy.models import InternalRequest
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class FallbackHandler:
    """Handles provider fallback logic for streaming requests.

    Consolidates error classification, fallback attempt recording, and
    next-provider selection into a single service that reuses the same
    :mod:`llm_proxy.core.processing.fallback` helpers as the non-streaming
    execution path.
    """

    def __init__(
        self,
        error_handler: ErrorHandler,
        param_override_service: ParameterOverrideService,
    ):
        self._error_handler = error_handler
        self._param_override_service = param_override_service

    @staticmethod
    async def switch_adapter(
        exit_stack: AsyncExitStack,
        old_adapter: BaseAdapter,
        new_adapter: BaseAdapter,
    ) -> None:
        """Close old_adapter and register new_adapter for cleanup."""
        await swap_adapters(exit_stack, old_adapter, new_adapter)

    async def _next_provider_selection(
        self,
        error: ProviderError,
        current_adapter: BaseAdapter,
        context: RequestContext,
        unified_request: InternalRequest,
        raw_request_data: dict[str, Any],
        req: Request,
        log_message: str,
    ) -> tuple[BaseAdapter, InternalRequest] | None:
        """Record a fallback attempt and select the next priority provider.

        Used by the pre-stream error paths (context exceeded, retryable finish
        reason, empty stream). The caller performs the adapter swap because it
        owns the streaming exit stack for those cases.
        """
        decision = plan_fallback(error, context.orchestrator, current_adapter.provider_name)
        if decision.action is not FallbackAction.RETRY_NEXT_PROVIDER:
            return None
        logger.warning(log_message)
        record_fallback(
            decision,
            context.orchestrator,
            context.event_context,
            provider_type=current_adapter.provider_name,
        )
        selection = context.orchestrator.select_next_provider()
        if selection is None:
            return None
        return await setup_fallback_provider(
            selection,
            req,
            unified_request,
            raw_request_data,
            context,
            self._param_override_service,
        )

    async def handle_context_exceeded(
        self,
        context_exceeded_reason: str,
        current_adapter: BaseAdapter,
        context: RequestContext,
        unified_request: InternalRequest,
        raw_request_data: dict[str, Any],
        req: Request,
    ) -> tuple[BaseAdapter, InternalRequest] | None:
        """Handle context-length-exceeded finish reason."""
        error = self._error_handler.create_context_length_error(
            current_adapter.provider_name, context_exceeded_reason
        )
        return await self._next_provider_selection(
            error,
            current_adapter,
            context,
            unified_request,
            raw_request_data,
            req,
            f"Context length exceeded on {current_adapter.provider_name} "
            f"(finish_reason={context_exceeded_reason}), retrying with next provider",
        )

    async def handle_retryable_finish_reason(
        self,
        retryable_stream_finish_reason: str,
        current_adapter: BaseAdapter,
        context: RequestContext,
        unified_request: InternalRequest,
        raw_request_data: dict[str, Any],
        req: Request,
    ) -> tuple[BaseAdapter, InternalRequest] | None:
        """Handle retryable finish reason."""
        error = self._error_handler.create_retryable_stream_error(
            current_adapter.provider_name, retryable_stream_finish_reason
        )
        return await self._next_provider_selection(
            error,
            current_adapter,
            context,
            unified_request,
            raw_request_data,
            req,
            f"Retryable finish_reason from {current_adapter.provider_name} "
            f"({retryable_stream_finish_reason}), retrying with next provider",
        )

    async def handle_empty_stream(
        self,
        current_adapter: BaseAdapter,
        context: RequestContext,
        unified_request: InternalRequest,
        raw_request_data: dict[str, Any],
        req: Request,
    ) -> tuple[BaseAdapter, InternalRequest] | None:
        """Handle empty stream."""
        error = self._error_handler.create_empty_stream_error(current_adapter.provider_name)
        return await self._next_provider_selection(
            error,
            current_adapter,
            context,
            unified_request,
            raw_request_data,
            req,
            f"Empty stream from {current_adapter.provider_name}, retrying with next provider",
        )

    async def handle_stream_error(
        self,
        e: Exception,
        current_adapter: BaseAdapter,
        context: RequestContext,
        unified_request: InternalRequest,
        raw_request_data: dict[str, Any],
        req: Request,
        event_context: EventContext | None,
        exit_stack: AsyncExitStack,
    ) -> tuple[BaseAdapter, InternalRequest] | Response | None:
        """Handle any error during stream setup with unified fallback logic.

        Returns:
            tuple: New (adapter, request) to retry the while loop.
            Response: Terminal error response -- the caller should break.
            None: The error is retryable but no more providers are available.
        """
        decision = plan_fallback(e, context.orchestrator, current_adapter.provider_name)

        if decision.action is FallbackAction.ABORT:
            return self._error_handler.format_response(decision.error)

        if decision.action is FallbackAction.RETRY_ROLE_TRANSFORM:
            logger.warning(
                f"Role error from {current_adapter.provider_name}, "
                "applying developer -> system role transformation and retrying"
            )
            normalize_developer_roles(unified_request)
            context.orchestrator.mark_role_transformed()
            return current_adapter, unified_request

        swapped = await execute_fallback(
            decision,
            context.orchestrator,
            event_context=event_context,
            provider_type=current_adapter.provider_name,
            exit_stack=exit_stack,
            current_adapter=current_adapter,
            req=req,
            unified_request=unified_request,
            raw_request_data=raw_request_data,
            context=context,
            param_override_service=self._param_override_service,
        )
        if swapped is None:
            return None
        context.orchestrator.reset_stream_state()
        return swapped


__all__ = ["FallbackHandler"]
