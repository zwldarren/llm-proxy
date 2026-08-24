"""Tests for UnifiedProcessor parameter override behavior."""

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from llm_proxy.config.types import LoggingConfig
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.processing.base import RequestContext, ServiceDependencies
from llm_proxy.core.processing.unified import UnifiedProcessor
from llm_proxy.core.provider_selector import ProviderSelectionResult
from llm_proxy.models import InternalResponse, TextBlock
from llm_proxy.protocols.openai.handler import openai_protocol
from llm_proxy.protocols.openai.schemas import ChatCompletionRequest


@pytest.mark.asyncio
async def test_event_context_request_body_reflects_applied_overrides(monkeypatch) -> None:
    """EventContext should capture the request body after parameter overrides."""
    selection = ProviderSelectionResult(
        provider_name="openai",
        provider_config=ProviderConfig(type="openai", api_key="test-key"),
        provider_model_name="minimax-m2.5",
        priority=1,
        parameter_overrides={"max_tokens": 32768},
    )

    orchestrator = MagicMock()
    orchestrator.select_next_provider.return_value = selection

    adapter = MagicMock()
    adapter.provider_name = "openai"

    captured_requests = []

    async def _chat_completion(req):
        captured_requests.append(req)
        return InternalResponse(
            id="resp-1",
            model=req.model,
            output=[TextBlock(text="ok")],
            request_id=req.request_id,
        )

    adapter.chat_completion = AsyncMock(side_effect=_chat_completion)

    tracing_registry = MagicMock()
    tracing_registry.on_request_start = AsyncMock()
    tracing_registry.on_request_end = AsyncMock()
    tracing_registry.on_error = AsyncMock()
    tracing_registry.get_observation_id.return_value = None

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(
            adapter_factory=AsyncMock(return_value=adapter),
            tracing_registry=tracing_registry,
        ),
    )

    processor = UnifiedProcessor(protocol_endpoint=openai_protocol)

    monkeypatch.setattr(
        "llm_proxy.core.processing.unified.load_logging_config",
        lambda: LoggingConfig(enable_database_logging=True),
    )

    request_model = ChatCompletionRequest(
        model="proxy-model",
        messages=[{"role": "user", "content": "hello", "id": "user-1"}],
        temperature=0.7,
        stream=False,
    )

    req = MagicMock()
    req.state = MagicMock()
    req.state.request_id = "req-123"
    req.state.model = None
    req.state.internal_model = None
    req.state.provider = None
    req.state.db_session = None
    req.url = MagicMock(path="/v1/chat/completions")
    req.method = "POST"
    req.headers = {}
    req.client = MagicMock(host="127.0.0.1")

    await processor.process(request_model, req, context)

    assert len(captured_requests) == 1
    outbound_request = captured_requests[0]
    assert outbound_request.params.max_tokens == 32768
    assert outbound_request.params.temperature == 0.7

    assert context.event_context is not None
    assert isinstance(context.event_context.request_body, dict)
    assert context.event_context.request_body.get("max_tokens") == 32768


def _build_pipeline_mocks(selection, adapter):
    """Shared pipeline scaffolding: orchestrator, tracing registry, request."""
    orchestrator = MagicMock()
    orchestrator.select_next_provider.return_value = selection

    tracing_registry = MagicMock()
    tracing_registry.on_request_start = AsyncMock()
    tracing_registry.on_request_end = AsyncMock()
    tracing_registry.on_error = AsyncMock()
    tracing_registry.get_observation_id.return_value = None

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(
            adapter_factory=AsyncMock(return_value=adapter),
            tracing_registry=tracing_registry,
        ),
    )

    req = MagicMock()
    req.state = MagicMock()
    req.state.request_id = "req-123"
    req.state.model = None
    req.state.internal_model = None
    req.state.provider = None
    req.state.db_session = None
    req.url = MagicMock(path="/v1/chat/completions")
    req.method = "POST"
    req.headers = {}
    req.client = MagicMock(host="127.0.0.1")
    return context, req


async def _run_processor(selection, adapter, request_model, monkeypatch):
    """Run the pipeline end-to-end; returns (response, context)."""
    context, req = _build_pipeline_mocks(selection, adapter)
    processor = UnifiedProcessor(protocol_endpoint=openai_protocol)

    monkeypatch.setattr(
        "llm_proxy.core.processing.unified.load_logging_config",
        lambda: LoggingConfig(enable_database_logging=True),
    )

    response = await processor.process(request_model, req, context)
    return response, context


@pytest.mark.asyncio
async def test_static_model_override_wins_over_provider_model_name(monkeypatch) -> None:
    """A static ``model`` key in parameter_overrides must win over the
    resolved provider model name (the mapping must not clobber it)."""
    selection = ProviderSelectionResult(
        provider_name="openai",
        provider_config=ProviderConfig(type="openai", api_key="test-key"),
        provider_model_name="minimax-m2.5",
        priority=1,
        parameter_overrides={"model": "gpt-4o"},
    )

    adapter = MagicMock()
    adapter.provider_name = "openai"

    captured_requests = []

    async def _chat_completion(req):
        captured_requests.append(req)
        return InternalResponse(
            id="resp-1",
            model=req.model,
            output=[TextBlock(text="ok")],
            request_id=req.request_id,
        )

    adapter.chat_completion = AsyncMock(side_effect=_chat_completion)

    request_model = ChatCompletionRequest(
        model="proxy-model",
        messages=[{"role": "user", "content": "hello", "id": "user-1"}],
        stream=False,
    )

    response, _ = await _run_processor(selection, adapter, request_model, monkeypatch)

    assert len(captured_requests) == 1
    assert captured_requests[0].model == "gpt-4o"


@pytest.mark.asyncio
async def test_non_stream_response_echoes_user_facing_alias(monkeypatch) -> None:
    """The non-streaming response body must echo the client-requested alias
    even when the upstream reports a different (provider) model name."""
    selection = ProviderSelectionResult(
        provider_name="openai",
        provider_config=ProviderConfig(type="openai", api_key="test-key"),
        provider_model_name="gpt-4o-mini",
        priority=1,
    )

    adapter = MagicMock()
    adapter.provider_name = "openai"

    async def _chat_completion(req):
        # Upstream reports a different serving model (OpenRouter-style).
        return InternalResponse(
            id="resp-1",
            model="gpt-4o-2024-11-20",
            output=[TextBlock(text="ok")],
            request_id=req.request_id,
        )

    adapter.chat_completion = AsyncMock(side_effect=_chat_completion)

    request_model = ChatCompletionRequest(
        model="fast",
        messages=[{"role": "user", "content": "hello", "id": "user-1"}],
        stream=False,
    )

    response, context = await _run_processor(selection, adapter, request_model, monkeypatch)

    body = orjson.loads(response.body)
    assert body["model"] == "fast"
    assert context.event_context is not None
    # The resolved provider model name stays in provider_model_name for
    # logging/billing; the client-visible model is the alias.
    assert context.event_context.provider_model_name == "gpt-4o-mini"
