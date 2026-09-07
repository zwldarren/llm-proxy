"""Single composition owner for the per-provider stage pipeline.

The per-provider request-mutating stage order (previous-response resolution,
web search) is defined exactly once, here, and consumed by BOTH execution
paths that run per-provider stages:

- ``UnifiedProcessor._stages`` builds its main pipeline from
  :func:`create_per_provider_stages` (between ParameterOverride and
  RequestExecution).
- The fallback re-parse path re-runs the SAME list via
  :func:`rerun_per_provider_stages` (see ADR-0008).

Adding a stage to :func:`create_per_provider_stages` therefore makes both the
main pipeline and the fallback re-run see it — a new stage can no longer be
silently skipped by the fallback re-parse.

This module deliberately lives inside the ``stages`` package and imports
stages freely; the fallback plumbing in ``stages.fallback`` imports *this*
module (not the other way around), which is what dissolves the former
``stages/* <-> core/processing/fallback.py`` import cycle.
"""

from typing import TYPE_CHECKING, Any

from fastapi import Request

from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.stages.base import PipelineStage, PipelineState
from llm_proxy.core.processing.stages.previous_response import (
    PreviousResponseResolutionStage,
)
from llm_proxy.core.processing.stages.role_normalization import normalize_developer_roles
from llm_proxy.core.processing.stages.web_search import WebSearchStage
from llm_proxy.observability.event_context import EventContext

if TYPE_CHECKING:
    from llm_proxy.core.adapter import BaseAdapter
    from llm_proxy.models import InternalRequest


def create_per_provider_stages() -> list[PipelineStage]:
    """The ordered per-provider request-mutating stages.

    This is the single source of truth for their order. ``UnifiedProcessor``
    splices this list into its pipeline, and the fallback re-parse re-runs it
    (:func:`rerun_per_provider_stages`). ProviderSelection and ParameterOverride
    are intentionally NOT part of this list: the fallback path handles each
    through its specialized path (``select_next_provider`` /
    ``_rebuild_fallback_request``).
    """
    return [
        PreviousResponseResolutionStage(),
        WebSearchStage(),
    ]


async def rerun_per_provider_stages(
    selection: Any,
    adapter: BaseAdapter,
    req: Request,
    new_request: InternalRequest,
    raw_data: dict[str, Any],
    context: RequestContext,
) -> None:
    """Re-run the per-provider request-mutating stages on a freshly parsed request.

    PreviousResponseResolutionStage and WebSearchStage mutate the parsed
    request (materializing stored conversations, converting web-search tools)
    and set request flags (``previous_response_materialized``,
    ``native_request_disabled``) based on the SELECTED provider — e.g. the
    web-search interception decision depends on the provider's
    ``native_web_search`` flag. A fallback re-parse starts from the pristine
    client body, so these decisions must be re-evaluated for the new provider
    instead of inheriting the failed provider's mutated request.

    The stage list comes from :func:`create_per_provider_stages` — the same
    composition the main pipeline consumes.

    Raises:
        LLMProxyError: when the request is not viable for this provider (e.g.
            an unresolvable proxy-local ``previous_response_id`` on a
            non-native upstream). The caller skips to the next provider.
    """
    event_context = context.event_context
    if event_context is None:
        # Direct-adapter/test paths may lack an EventContext; the stages
        # re-run here never read it, but PipelineState requires the field.
        event_context = EventContext(
            request_id=getattr(req.state, "request_id", "") or "",
            trace_id="",
            model=None,
        )

    stage_state = PipelineState(
        raw_data=raw_data,
        unified_request=new_request,
        req=req,
        strategy=None,
        trace_id=event_context.trace_id,
        event_context=event_context,
        selection=selection,
        adapter=adapter,
    )
    # Role transformation is sticky across providers: mark_role_transformed
    # clears used_provider_keys so every provider is retried with transformed
    # roles, but the fallback re-parse starts from the pristine client body.
    # Re-apply the transform AFTER the composed stages have run:
    # PreviousResponseResolutionStage materializes stored conversations whose
    # items round-trip their original roles (including ``developer``), so the
    # transform must see the post-materialization request — matching the
    # former interleaving (PreviousResponse → normalize → WebSearch;
    # WebSearchStage does not inspect roles, so the trailing position is
    # equivalent for it).
    # The interception decision is per-provider, so reset the shared context
    # flag and let WebSearchStage recompute it for THIS provider.
    context.proxy_web_search_active = False
    for stage in create_per_provider_stages():
        await stage.process(stage_state, context)
    if context.orchestrator.state.role_transformed:
        normalize_developer_roles(new_request)
