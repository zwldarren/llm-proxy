"""Tests for RequestExecutionStage audio streaming behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.core.processing.base import RequestContext, ServiceDependencies
from llm_proxy.core.processing.stages import (
    ParameterOverrideService,
    PipelineState,
    RequestExecutionStage,
)
from llm_proxy.core.processing.strategies import StreamingResponseMarker
from llm_proxy.core.processing.streaming_processor import StreamingProcessor
from llm_proxy.models import (
    InternalSpeechRequest,
    InternalTranscriptionRequest,
)
from llm_proxy.protocols.openai.audio_speech_handler import speech_protocol
from llm_proxy.protocols.openai.audio_transcription_handler import (
    transcription_protocol,
)
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.streaming.handler import StreamingHandler


def _build_mock_request():
    req = MagicMock()
    req.state = MagicMock()
    req.state.request_id = "req-123"
    req.state.provider = "openai"
    return req


def _create_execution_stage(protocol):
    serializer = get_protocol_serializer(protocol.name)
    streaming_handler = StreamingHandler()
    from llm_proxy.core.errors import get_error_handler

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


def _make_state(request_obj, req, context):
    return PipelineState(
        raw_data={"model": request_obj.model},
        unified_request=request_obj,
        req=req,
        strategy=MagicMock(),
        trace_id="trace-1",
        event_context=MagicMock(),
    )


@pytest.mark.asyncio
async def test_speech_streaming_response_media_type():
    """Speech streaming must return the correct audio MIME type."""

    async def _mock_stream():
        yield b"audio-chunk-1"
        yield b"audio-chunk-2"

    adapter = MagicMock()
    adapter.provider_name = "openai"
    adapter.stream_speech = AsyncMock(return_value=_mock_stream())

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    stage = _create_execution_stage(speech_protocol)
    request = InternalSpeechRequest(
        model="tts-1",
        input="Hello",
        voice="alloy",
        response_format="mp3",
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

    assert response.media_type == "audio/mpeg"

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    assert chunks == [b"audio-chunk-1", b"audio-chunk-2"]


@pytest.mark.asyncio
async def test_speech_streaming_response_media_type_opus():
    """Speech streaming must return audio/opus for opus format."""

    async def _mock_stream():
        yield b"audio-chunk"

    adapter = MagicMock()
    adapter.provider_name = "openai"
    adapter.stream_speech = AsyncMock(return_value=_mock_stream())

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    stage = _create_execution_stage(speech_protocol)
    request = InternalSpeechRequest(
        model="tts-1",
        input="Hello",
        voice="alloy",
        response_format="opus",
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

    assert response.media_type == "audio/opus"


@pytest.mark.asyncio
async def test_transcription_streaming_response_media_type():
    """Transcription streaming may return text/event-stream (SSE lines)."""

    async def _mock_stream():
        yield '{"text":"Hello"}\n'

    adapter = MagicMock()
    adapter.provider_name = "openai"
    adapter.stream_transcription = AsyncMock(return_value=_mock_stream())

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    stage = _create_execution_stage(transcription_protocol)
    request = InternalTranscriptionRequest(
        model="whisper-1",
        file=b"audio",
        filename="audio.mp3",
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

    assert response.media_type == "text/event-stream"

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    assert chunks == ['{"text":"Hello"}\n']
