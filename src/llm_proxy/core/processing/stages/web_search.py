"""Web search stage."""

from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.stages.base import PipelineStage, PipelineState
from llm_proxy.models.tools.openai_builtin import WebSearchTool


class WebSearchStage(PipelineStage):
    """Intercept web_search tool and inject function tool into the request.

    Only intercepts when the selected provider has native_web_search=False.
    Otherwise, web_search tools pass through to the upstream provider for
    native handling.

    Also converts ``web_search_options`` (OpenAI Chat Completions standard
    field) into a ``WebSearchTool`` when the selected provider lacks native
    web search — matching the LiteLLM pattern of treating
    ``web_search_options`` as the canonical trigger for web search in
    Chat Completions requests.
    """

    async def process(self, state: PipelineState, context: RequestContext) -> None:
        interceptor = context.web_search_interceptor
        if interceptor is None:
            return

        # Per-provider guard: only passthrough if this provider opted into
        # native web search. Default (False) means proxy intercepts.
        if state.selection is None:
            return
        provider_config = state.selection.provider_config
        native_ws = getattr(provider_config, "native_web_search", False)

        if (
            not native_ws
            and state.unified_request.params
            and state.unified_request.params.openai
            and state.unified_request.params.openai.web_search_options
            and not interceptor.has_web_search_tool(state.unified_request.tools)
        ):
            wso = state.unified_request.params.openai.web_search_options
            if isinstance(wso, bool):
                # OpenAI accepts bare `true` to enable web search with defaults
                ws_tool = WebSearchTool(
                    name="web_search",
                    type="web_search",
                )
            else:
                ws_tool = WebSearchTool(
                    name="web_search",
                    type="web_search",
                    search_context_size=wso.get("search_context_size"),
                    user_location=wso.get("user_location"),
                )
            if state.unified_request.tools is None:
                state.unified_request.tools = [ws_tool]
            else:
                state.unified_request.tools.append(ws_tool)

        if not interceptor.has_web_search_tool(state.unified_request.tools):
            return

        if native_ws:
            return  # Passthrough: let upstream provider handle natively

        context.proxy_web_search_active = True
        # Proxy-side web search interception rewrites tools and injects results
        # into the parsed response; the verbatim request/response passthrough
        # path cannot do that, so it must be disabled for this request.
        state.unified_request.native_request_disabled = True
        state.unified_request.tools = interceptor.filter_web_search_tools(
            state.unified_request.tools
        )
        function_tool = interceptor.convert_web_search_to_function(context.web_search_tool_config)
        if state.unified_request.tools is None:
            state.unified_request.tools = [function_tool]
        else:
            state.unified_request.tools.append(function_tool)
