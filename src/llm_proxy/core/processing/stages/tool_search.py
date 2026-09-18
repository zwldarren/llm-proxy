"""Tool-search stage: make tools discovered via ``tool_search_output`` callable.

Codex's tool-search flow works in two steps: the model calls the hosted
``tool_search`` tool, the upstream returns a ``tool_search_output`` item listing
the matching tool definitions, and the model then calls one of those tools.

Only a native OpenAI Responses upstream understands ``tool_search`` server-side
and keeps the discovered definitions in its own context — for it, injecting the
definitions as regular top-level tools would change lazy-loading semantics (and
namespace-typed specs are rejected at the top level). Every other provider only
ever sees the bridged ``tool_search`` function tool, so the discovered
definitions must be appended to the request's tool list or the model can never
call them.

The protocol serializer parses the definitions into ``_discovered_tools`` and
leaves them out of ``tools``; this stage decides whether to append them, based
on the selected provider.
"""

from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.stages.base import (
    PipelineStage,
    PipelineState,
    is_native_responses_upstream,
)
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


class ToolSearchStage(PipelineStage):
    """Append ``tool_search``-discovered tools for non-native Responses providers.

    Runs after PreviousResponseResolutionStage (which may attach additional
    discovered tools while replaying a stored turn) and is a no-op when the
    request declares no discovered tools — the common case, since a
    ``tool_search`` result usually re-lists already-declared tools.
    """

    async def process(self, state: PipelineState, context: RequestContext) -> None:
        request = state.unified_request
        if getattr(request, "_discovered_tools_materialized", False):
            return
        request._discovered_tools_materialized = True

        discovered = getattr(request, "_discovered_tools", None)
        if not discovered:
            return

        if is_native_responses_upstream(state.adapter):
            # OpenAI handles tool_search server-side; the definitions stay in
            # its own context. Sending them as top-level tools would defeat
            # deferred loading and namespace tools would be rejected.
            logger.debug(
                f"Skipping tool_search injection for native Responses upstream "
                f"({len(discovered)} discovered tools)"
            )
            return

        request.tools = [*(request.tools or []), *discovered]
        logger.debug(
            f"Injected {len(discovered)} tool_search-discovered tool(s) for provider "
            f"{getattr(state.selection, 'provider_name', None)!r}"
        )
