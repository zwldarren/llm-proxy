"""Provider selection stage."""

from typing import Any

from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.stages.base import PipelineStage, PipelineState


class ProviderSelectionStage(PipelineStage):
    """Select provider and create adapter. Also handles fallback selection during retry."""

    async def process(self, state: PipelineState, context: RequestContext) -> None:
        selection = context.orchestrator.select_next_provider()
        if selection is None:
            raise ConfigurationError(
                f"No providers available for model '{state.unified_request.model}'"
            )

        adapter = await context.adapter_factory(state.req, selection)
        await state.exit_stack.enter_async_context(adapter)
        # Surface same-provider retries from this adapter's RetryPolicy into
        # the EventContext so they appear in log_metadata / the frontend logs
        # page. No-op for adapters without a RetryPolicy (BaseAdapter default).
        if state.event_context is not None:
            adapter.set_retry_recorder(state.event_context.retry_attempts.append)

        state.unified_request.request_id = getattr(state.req.state, "request_id", None)
        internal_model = state.unified_request.model
        state.req.state.model = state.unified_request.model
        state.req.state.provider = selection.provider_name

        state.event_context.provider = selection.provider_name

        if context.routing_decision is not None:
            internal_model = context.routing_decision.model
            # Keep the original virtual model name (e.g. fast) in the log model column;
            # usage/cost tracking uses internal_model (the resolved real model) below.
            state.event_context.model = context.requested_model
            state.req.state.model = context.requested_model
            state.event_context.provider_model_name = internal_model
        else:
            state.event_context.model = internal_model
        state.event_context.internal_model = internal_model

        # Single source of truth for the client-visible model name: the alias
        # the client requested, before the provider_model_name override below.
        # Every response echo point reads this (see InternalRequest.echo_model).
        state.unified_request.user_facing_model = state.event_context.model

        # Capture routing analytics in event context for logging
        if context.routing_decision is not None:
            decision = context.routing_decision
            routing_meta: dict[str, Any] = {
                "complexity": float(getattr(decision, "complexity", 0.0) or 0.0),
                "confidence": float(getattr(decision, "confidence", 0.0) or 0.0),
                "reasoning": getattr(decision, "reasoning", None),
                "cost_estimate": float(getattr(decision, "cost_estimate", 0.0) or 0.0),
                "savings": float(getattr(decision, "savings", 0.0) or 0.0),
                "tier": str(getattr(decision, "tier", "unknown")),
                "requested_model": context.requested_model,
                "resolved_model": internal_model,
            }
            if context.verbose_routing_logs:
                routing_meta.update(
                    {
                        "candidate_scorecards": list(getattr(decision, "candidate_scorecards", [])),
                        "weights_used": dict(getattr(decision, "weights_used", {})),
                        "guardrail_notes": list(getattr(decision, "guardrail_notes", [])),
                        "signal_votes": dict(getattr(decision, "signal_votes", {})),
                    }
                )
            state.event_context.metadata["routing"] = routing_meta

        if selection.provider_model_name:
            state.unified_request.model = selection.provider_model_name
            if context.routing_decision is None:
                state.event_context.provider_model_name = selection.provider_model_name

        state.selection = selection
        state.adapter = adapter

        # Call process_request middleware if configured
        if context.process_request is not None:
            state.unified_request = await context.process_request(
                state.unified_request, state.adapter
            )
