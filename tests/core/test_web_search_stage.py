"""Tests for WebSearchStage per-provider native_web_search logic."""

from unittest.mock import MagicMock

import pytest

from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.processing.base import RequestContext, ServiceDependencies
from llm_proxy.core.processing.stages import PipelineState, WebSearchStage
from llm_proxy.core.provider_selector import ProviderSelectionResult
from llm_proxy.models import ConversationContext, InternalRequest, Message
from llm_proxy.models.content_blocks import TextBlock
from llm_proxy.models.tools import FunctionTool, OpenAIWebSearchTool
from llm_proxy.web_search.interceptor import WebSearchInterceptor
from llm_proxy.web_search.provider import WebSearchProvider, WebSearchResponse


class _FakeWebSearchProvider(WebSearchProvider):
    async def search(self, query, config=None, **kwargs):
        return WebSearchResponse(results=[], search_id="fake")

    async def close(self) -> None:
        pass


def _build_state(with_selection=None):
    """Build a PipelineState with web_search tools."""
    return PipelineState(
        raw_data={},
        unified_request=InternalRequest(
            model="test-model",
            stream=True,
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            tools=[OpenAIWebSearchTool(name="web_search", type="web_search")],
        ),
        req=MagicMock(),
        strategy=MagicMock(),
        trace_id="t",
        event_context=MagicMock(),
        selection=with_selection,
    )


def _build_context():
    interceptor = WebSearchInterceptor(_FakeWebSearchProvider())
    return RequestContext(
        orchestrator=MagicMock(),
        services=ServiceDependencies(
            adapter_factory=MagicMock(), web_search_interceptor=interceptor
        ),
    )


@pytest.mark.asyncio
async def test_passthrough_when_native_enabled():
    """WebSearchStage should NOT intercept when provider has native_web_search=True."""
    provider_config = ProviderConfig(type="anthropic", native_web_search=True)
    selection = ProviderSelectionResult(
        provider_name="test-provider",
        provider_config=provider_config,
        provider_model_name="claude-sonnet",
        priority=0,
    )
    state = _build_state(with_selection=selection)
    context = _build_context()

    await WebSearchStage().process(state, context)

    # Tools should remain unchanged — web_search tool still present
    assert state.unified_request.tools is not None
    assert len(state.unified_request.tools) == 1
    assert isinstance(state.unified_request.tools[0], OpenAIWebSearchTool)


@pytest.mark.asyncio
async def test_intercept_by_default():
    """WebSearchStage should intercept by default when native_web_search is False."""
    provider_config = ProviderConfig(type="anthropic", native_web_search=False)
    selection = ProviderSelectionResult(
        provider_name="test-provider",
        provider_config=provider_config,
        provider_model_name="claude-sonnet",
        priority=0,
    )
    state = _build_state(with_selection=selection)
    context = _build_context()

    await WebSearchStage().process(state, context)

    # Web search tool should be replaced by function tool (default = intercept)
    assert state.unified_request.tools is not None
    assert len(state.unified_request.tools) == 1
    assert isinstance(state.unified_request.tools[0], FunctionTool)
    assert state.unified_request.tools[0].name == "web_search"


@pytest.mark.asyncio
async def test_passthrough_when_no_selection():
    """Should not intercept when state.selection is None."""
    state = _build_state(with_selection=None)
    context = _build_context()

    await WebSearchStage().process(state, context)

    # No provider selection means the stage cannot determine whether the
    # provider opted in, so it must leave tools untouched.
    assert state.unified_request.tools is not None
    assert len(state.unified_request.tools) == 1
    assert isinstance(state.unified_request.tools[0], OpenAIWebSearchTool)
