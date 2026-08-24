"""Regression tests for streaming image edit routing in StreamingProcessor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.core.errors import get_error_handler
from llm_proxy.core.processing.base import RequestContext, ServiceDependencies
from llm_proxy.core.processing.stages import (
    ParameterOverrideService,
    RequestExecutionStage,
)
from llm_proxy.core.processing.strategies import StreamingResponseMarker
from llm_proxy.core.processing.streaming_processor import StreamingProcessor
from llm_proxy.models.image import (
    ImageEditSource,
    InternalImageEditRequest,
    InternalImageRequest,
)
from llm_proxy.protocols.openai.images_edits_handler import image_edits_protocol
from llm_proxy.protocols.openai.images_handler import image_generations_protocol
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.streaming.handler import StreamingHandler


def _create_execution_stage(protocol):
    serializer = get_protocol_serializer(protocol.name)
    streaming_handler = StreamingHandler()
    param_override_service = ParameterOverrideService(serializer)
    streaming_processor = StreamingProcessor(
        protocol_endpoint=protocol,
        streaming_handler=streaming_handler,
        error_handler=get_error_handler(),
        param_override_service=param_override_service,
    )
    return RequestExecutionStage(
        protocol_name=protocol.name,
        protocol_endpoint=protocol,
        serializer=serializer,
        streaming_processor=streaming_processor,
        param_override_service=param_override_service,
    )


def _build_mock_request():
    req = MagicMock()
    req.state = MagicMock()
    req.state.request_id = "req-123"
    req.state.provider = "openai"
    return req


@pytest.mark.asyncio
async def test_streaming_image_edit_routes_to_stream_image_edit():
    """Regression: streaming image edit must call adapter.stream_image_edit.

    Previously _process_image_streaming routed every image stream to
    adapter.stream_image_generation, which hit the generations endpoint and
    accessed InternalImageEditRequest.response_format, raising AttributeError.
    """

    async def _mock_edit_stream():
        yield 'data: {"type":"image_generation.progress"}\n\n'
        yield "data: [DONE]\n\n"

    adapter = MagicMock()
    adapter.provider_name = "openai"
    adapter.stream_image_edit = AsyncMock(return_value=_mock_edit_stream())
    adapter.stream_image_generation = AsyncMock()

    context = RequestContext(
        orchestrator=MagicMock(),
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    stage = _create_execution_stage(image_edits_protocol)
    request = InternalImageEditRequest(
        model="gpt-image-1",
        prompt="add a hat",
        images=[ImageEditSource(file_id="file_123")],
        stream=True,
    )
    streaming_marker = StreamingResponseMarker(request, adapter)

    response = await stage._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": request.model, "stream": True},
        req=_build_mock_request(),
        context=context,
        trace_id="trace-1",
    )

    adapter.stream_image_edit.assert_awaited_once()
    adapter.stream_image_generation.assert_not_awaited()
    assert response is not None


@pytest.mark.asyncio
async def test_streaming_image_generation_routes_to_stream_image_generation():
    """Streaming image generation must still call adapter.stream_image_generation."""

    async def _mock_gen_stream():
        yield 'data: {"type":"image_generation.progress"}\n\n'
        yield "data: [DONE]\n\n"

    adapter = MagicMock()
    adapter.provider_name = "openai"
    adapter.stream_image_generation = AsyncMock(return_value=_mock_gen_stream())
    adapter.stream_image_edit = AsyncMock()

    context = RequestContext(
        orchestrator=MagicMock(),
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    stage = _create_execution_stage(image_generations_protocol)
    request = InternalImageRequest(
        model="dall-e-3",
        prompt="a cat",
        stream=True,
    )
    streaming_marker = StreamingResponseMarker(request, adapter)

    response = await stage._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": request.model, "stream": True},
        req=_build_mock_request(),
        context=context,
        trace_id="trace-1",
    )

    adapter.stream_image_generation.assert_awaited_once()
    adapter.stream_image_edit.assert_not_awaited()
    assert response is not None
