"""Regression tests for the per-provider stage composition owner.

``stages.composition.create_per_provider_stages`` is the single source of
truth for the per-provider stage order; both ``UnifiedProcessor._stages`` and
the fallback re-parse (``rerun_per_provider_stages``) must consume it, so a
stage added to the composition runs on BOTH paths (no silent fallback skip).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.processing.base import RequestContext, ServiceDependencies
from llm_proxy.core.processing.stages import PipelineStage, PipelineState
from llm_proxy.core.processing.stages import composition as composition_module
from llm_proxy.core.processing.stages.composition import create_per_provider_stages
from llm_proxy.models import Message
from llm_proxy.models.content_blocks import TextBlock
from llm_proxy.core.processing.stages.fallback import setup_fallback_provider
from llm_proxy.core.processing.stages.parameter_override import ParameterOverrideService
from llm_proxy.core.processing.unified import UnifiedProcessor
from llm_proxy.core.provider_selector import ProviderSelectionResult
from llm_proxy.models import InternalRequest
from llm_proxy.protocols.openai.handler import openai_protocol
from llm_proxy.protocols.registry import get_protocol_serializer


class _RecordingStage(PipelineStage):
    """Stage wrapper that records its label, then delegates to a real stage."""

    def __init__(self, stage: PipelineStage | None, label: str, ran: list[str]):
        self._stage = stage
        self.label = label
        self._ran = ran

    async def process(self, state: PipelineState, context: RequestContext) -> None:
        self._ran.append(self.label)
        if self._stage is not None:
            await self._stage.process(state, context)


def _build_selection(provider_name: str = "openai") -> ProviderSelectionResult:
    return ProviderSelectionResult(
        provider_name=provider_name,
        provider_config=ProviderConfig(type="openai", api_key="test-key"),
        provider_model_name="gpt-4o-mini",
        priority=1,
        parameter_overrides=None,
    )


def _build_context(role_transformed: bool | None = None) -> RequestContext:
    """RequestContext with an explicit role-transformed flag.

    ``None`` keeps the flag falsy — an incidental ``MagicMock()`` would make
    ``state.role_transformed`` truthy and silently enable the transform, which
    is exactly what masked the normalize-order regression (see the sticky-role
    test below)."""
    orchestrator = MagicMock()
    if role_transformed is not None:
        orchestrator.state.role_transformed = role_transformed
    return RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=MagicMock())),
    )


def _build_mock_request() -> MagicMock:
    req = MagicMock()
    req.state = MagicMock()
    req.state.request_id = "req-1"
    return req


def _openai_override_service() -> ParameterOverrideService:
    return ParameterOverrideService(get_protocol_serializer("openai"))


@pytest.mark.asyncio
async def test_fallback_rerun_executes_stages_added_to_composition(monkeypatch) -> None:
    """A stage appended to the composition owner must run on the fallback
    re-parse path too — the regression this guards against is the former
    hand-maintained re-run list silently skipping newly added stages."""
    ran: list[str] = []
    original = create_per_provider_stages()
    sentinel = _RecordingStage(None, "sentinel", ran)

    monkeypatch.setattr(
        composition_module,
        "create_per_provider_stages",
        lambda: [*original, sentinel],
    )

    service = _openai_override_service()
    context = _build_context(role_transformed=False)
    req = _build_mock_request()
    raw = {
        "model": "fast",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    unified_request = get_protocol_serializer("openai").parse_request(raw)

    result = await setup_fallback_provider(
        _build_selection(), req, unified_request, raw, context, service
    )
    assert result is not None
    _adapter, new_request = result
    # The sentinel ran exactly once, on the freshly rebuilt fallback request.
    assert ran == ["sentinel"]
    assert isinstance(new_request, InternalRequest)


@pytest.mark.asyncio
async def test_fallback_rerun_order_follows_composition(monkeypatch) -> None:
    """The re-run path executes the composed stages in composition order,
    with appended stages running after them."""
    ran: list[str] = []
    real = create_per_provider_stages()
    wrapped = [
        _RecordingStage(real[0], "previous-response", ran),
        _RecordingStage(real[1], "web-search", ran),
        _RecordingStage(None, "appended", ran),
    ]
    monkeypatch.setattr(
        composition_module,
        "create_per_provider_stages",
        lambda: wrapped,
    )

    service = _openai_override_service()
    context = _build_context(role_transformed=False)
    req = _build_mock_request()
    raw = {
        "model": "fast",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    result = await setup_fallback_provider(
        _build_selection(),
        req,
        get_protocol_serializer("openai").parse_request(raw),
        raw,
        context,
        service,
    )

    assert result is not None
    # The fallback re-run follows the composition owner's order — both real
    # stages ran (delegated through), then the appended one.
    assert ran == ["previous-response", "web-search", "appended"]


def test_unified_processor_pipeline_consumes_composition(monkeypatch) -> None:
    """UnifiedProcessor splices the composition owner's stage list into its
    pipeline, so a stage added there appears in the main pipeline too."""
    from llm_proxy.core.processing.stages import (
        ParameterOverrideStage,
        ProviderSelectionStage,
        RequestExecutionStage,
    )

    ran: list[str] = []
    sentinel = _RecordingStage(None, "sentinel", ran)

    monkeypatch.setattr(
        "llm_proxy.core.processing.unified.create_per_provider_stages",
        lambda: [sentinel],
    )

    processor = UnifiedProcessor(protocol_endpoint=openai_protocol)

    stages = processor._stages
    assert isinstance(stages[0], ProviderSelectionStage)
    assert isinstance(stages[1], ParameterOverrideStage)
    # The composed block sits between ParameterOverride and RequestExecution,
    # exactly where the hand-written list used to live.
    assert stages[2] is sentinel
    assert isinstance(stages[3], RequestExecutionStage)
    assert len(stages) == 4


def test_composition_owner_list_matches_documented_pipeline() -> None:
    """The composition owner produces the documented per-provider order:
    PreviousResponseResolution -> WebSearch."""
    stages = create_per_provider_stages()
    assert [type(s).__name__ for s in stages] == [
        "PreviousResponseResolutionStage",
        "WebSearchStage",
    ]
    # Fresh instances per call — stages must not share mutable state across
    # the main pipeline and the fallback re-run.
    assert all(a is not b for a, b in zip(stages, create_per_provider_stages(), strict=True))


@pytest.mark.asyncio
async def test_fallback_rerun_normalizes_roles_after_materialization(monkeypatch) -> None:
    """Sticky role transformation must be re-applied AFTER the composed stages.

    PreviousResponseResolutionStage materializes stored conversations whose
    items round-trip their original roles — including ``developer``
    (serializer.py rehydrates unknown roles verbatim). The former fallback
    order was PreviousResponse -> normalize -> WebSearch; applying the
    transform before the composition loop would leak materialized developer
    roles to the next provider. This pins the transform to the
    post-composition position: a stage that injects a developer-role message
    (standing in for stored-turn materialization) must come out normalized.
    """

    class _MaterializingStage(PipelineStage):
        """Simulates stored-turn materialization injecting developer roles."""

        async def process(self, state: PipelineState, context: RequestContext) -> None:
            state.unified_request.conversation.messages.append(
                Message(role="developer", content=[TextBlock(text="stored assistant turn")])
            )

    monkeypatch.setattr(
        composition_module,
        "create_per_provider_stages",
        lambda: [_MaterializingStage()],
    )

    service = _openai_override_service()
    context = _build_context(role_transformed=True)
    req = _build_mock_request()
    raw = {
        "model": "fast",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    result = await setup_fallback_provider(
        _build_selection(),
        req,
        get_protocol_serializer("openai").parse_request(raw),
        raw,
        context,
        service,
    )

    assert result is not None
    _adapter, new_request = result
    # The materialized developer message must not survive the re-parse.
    assert [m.role for m in new_request.conversation.messages if m.role == "developer"] == []
    # ...and the transform's side effects must hold (rebuild path forced).
    assert new_request.native_request_disabled is True
