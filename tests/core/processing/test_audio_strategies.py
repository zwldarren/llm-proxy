"""Tests for audio processing strategies."""

import pytest
from fastapi import Response

from llm_proxy.core.processing.strategies import (
    SpeechStrategy,
    StreamingResponseMarker,
    TranscriptionStrategy,
    TranslationStrategy,
)
from llm_proxy.models import (
    InternalSpeechRequest,
    InternalSpeechResponse,
    InternalTranscriptionRequest,
    InternalTranscriptionResponse,
    InternalTranslationRequest,
    InternalTranslationResponse,
)
from llm_proxy.protocols.registry import get_protocol_serializer


class MockAdapter:
    """Mock adapter for strategy tests."""

    def __init__(self):
        self.provider_name = "openai"

    async def speech(self, request):
        return InternalSpeechResponse(
            content=b"audio-data",
            content_type="audio/mpeg",
            request_id=request.request_id,
        )

    async def stream_speech(self, request):
        async def _gen():
            yield b"chunk1"
            yield b"chunk2"

        return _gen()

    async def transcription(self, request):
        return InternalTranscriptionResponse(
            text="Hello world",
            _response_format=request.response_format,
        )

    async def stream_transcription(self, request):
        async def _gen():
            yield '{"text":"Hello"}\n'

        return _gen()

    async def translation(self, request):
        return InternalTranslationResponse(
            text="Hello world",
            _response_format=request.response_format,
        )


@pytest.fixture
def mock_adapter():
    return MockAdapter()


@pytest.fixture
def mock_context():
    from unittest.mock import MagicMock

    return MagicMock()


class TestSpeechStrategy:
    """Tests for SpeechStrategy."""

    @pytest.mark.asyncio
    async def test_execute_non_streaming(self, mock_adapter, mock_context):
        """Test non-streaming speech execution."""
        strategy = SpeechStrategy()
        request = InternalSpeechRequest(
            model="tts-1",
            input="Hello",
            voice="alloy",
            stream=False,
        )
        result = await strategy.execute(request, mock_adapter, mock_context)

        assert isinstance(result, InternalSpeechResponse)
        assert result.content == b"audio-data"
        assert result.content_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_execute_streaming(self, mock_adapter, mock_context):
        """Test streaming speech execution returns marker."""
        strategy = SpeechStrategy()
        request = InternalSpeechRequest(
            model="tts-1",
            input="Hello",
            voice="alloy",
            stream=True,
        )
        result = await strategy.execute(request, mock_adapter, mock_context)

        assert isinstance(result, StreamingResponseMarker)
        assert result.request == request
        assert result.adapter == mock_adapter

    @pytest.mark.asyncio
    async def test_format_response(self, mock_adapter, mock_context):
        """Test speech response formatting."""
        strategy = SpeechStrategy()
        response = InternalSpeechResponse(
            content=b"audio-data",
            content_type="audio/mpeg",
        )
        serializer = get_protocol_serializer("speech")
        result = await strategy.format_response(response, serializer, "speech")

        assert isinstance(result, Response)
        assert result.body == b"audio-data"
        assert result.media_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_format_response_mp3(self, mock_adapter, mock_context):
        """Test speech response formatting for MP3."""
        strategy = SpeechStrategy()
        response = InternalSpeechResponse(
            content=b"audio-data",
            content_type="audio/mpeg",
        )
        serializer = get_protocol_serializer("speech")
        result = await strategy.format_response(response, serializer, "speech")

        assert result.media_type == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_format_response_opus(self, mock_adapter, mock_context):
        """Test speech response formatting for OPUS."""
        strategy = SpeechStrategy()
        response = InternalSpeechResponse(
            content=b"audio-data",
            content_type="audio/opus",
        )
        serializer = get_protocol_serializer("speech")
        result = await strategy.format_response(response, serializer, "speech")

        assert result.media_type == "audio/opus"


class TestTranscriptionStrategy:
    """Tests for TranscriptionStrategy."""

    @pytest.mark.asyncio
    async def test_execute_non_streaming(self, mock_adapter, mock_context):
        """Test non-streaming transcription execution."""
        strategy = TranscriptionStrategy()
        request = InternalTranscriptionRequest(
            model="whisper-1",
            file=b"audio",
            filename="audio.mp3",
            stream=False,
        )
        result = await strategy.execute(request, mock_adapter, mock_context)

        assert isinstance(result, InternalTranscriptionResponse)
        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_execute_streaming(self, mock_adapter, mock_context):
        """Test streaming transcription execution returns marker."""
        strategy = TranscriptionStrategy()
        request = InternalTranscriptionRequest(
            model="whisper-1",
            file=b"audio",
            filename="audio.mp3",
            stream=True,
        )
        result = await strategy.execute(request, mock_adapter, mock_context)

        assert isinstance(result, StreamingResponseMarker)

    @pytest.mark.asyncio
    async def test_format_response_json(self, mock_adapter, mock_context):
        """Test transcription JSON response formatting."""
        strategy = TranscriptionStrategy()
        response = InternalTranscriptionResponse(
            text="Hello world",
            _response_format="json",
        )
        serializer = get_protocol_serializer("transcription")
        result = await strategy.format_response(response, serializer, "transcription")

        assert isinstance(result, Response)
        assert result.media_type == "application/json"
        body = result.body
        assert b"Hello world" in body

    @pytest.mark.asyncio
    async def test_format_response_text(self, mock_adapter, mock_context):
        """Test transcription text response formatting."""
        strategy = TranscriptionStrategy()
        response = InternalTranscriptionResponse(
            text="Hello world",
            _response_format="text",
        )
        serializer = get_protocol_serializer("transcription")
        result = await strategy.format_response(response, serializer, "transcription")

        assert isinstance(result, Response)
        assert result.media_type == "text/plain"
        assert result.body == b"Hello world"

    @pytest.mark.asyncio
    async def test_format_response_srt(self, mock_adapter, mock_context):
        """Test transcription SRT response formatting."""
        strategy = TranscriptionStrategy()
        response = InternalTranscriptionResponse(
            text="1\n00:00:00,000 --> ...",
            _response_format="srt",
        )
        serializer = get_protocol_serializer("transcription")
        result = await strategy.format_response(response, serializer, "transcription")

        assert isinstance(result, Response)
        assert result.media_type == "text/plain"
        assert result.body == b"1\n00:00:00,000 --> ..."

    @pytest.mark.asyncio
    async def test_format_response_vtt(self, mock_adapter, mock_context):
        """Test transcription VTT response formatting."""
        strategy = TranscriptionStrategy()
        response = InternalTranscriptionResponse(
            text="WEBVTT\n\n...",
            _response_format="vtt",
        )
        serializer = get_protocol_serializer("transcription")
        result = await strategy.format_response(response, serializer, "transcription")

        assert isinstance(result, Response)
        assert result.media_type == "text/plain"
        assert result.body == b"WEBVTT\n\n..."

    @pytest.mark.asyncio
    async def test_format_response_verbose_json(self, mock_adapter, mock_context):
        """Test transcription verbose_json response formatting."""
        strategy = TranscriptionStrategy()
        response = InternalTranscriptionResponse(
            text="Hello world",
            language="en",
            duration=5.0,
            _response_format="verbose_json",
        )
        serializer = get_protocol_serializer("transcription")
        result = await strategy.format_response(response, serializer, "transcription")

        assert isinstance(result, Response)
        assert result.media_type == "application/json"
        body = result.body
        assert b"Hello world" in body
        assert b"en" in body
        assert b"5.0" in body


class TestTranslationStrategy:
    """Tests for TranslationStrategy."""

    @pytest.mark.asyncio
    async def test_execute_non_streaming(self, mock_adapter, mock_context):
        """Test non-streaming translation execution."""
        strategy = TranslationStrategy()
        request = InternalTranslationRequest(
            model="whisper-1",
            file=b"audio",
            filename="audio.mp3",
        )
        result = await strategy.execute(request, mock_adapter, mock_context)

        assert isinstance(result, InternalTranslationResponse)
        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_format_response_json(self, mock_adapter, mock_context):
        """Test translation JSON response formatting."""
        strategy = TranslationStrategy()
        response = InternalTranslationResponse(
            text="Hello world",
            _response_format="json",
        )
        serializer = get_protocol_serializer("translation")
        result = await strategy.format_response(response, serializer, "translation")

        assert isinstance(result, Response)
        assert result.media_type == "application/json"
        body = result.body
        assert b"Hello world" in body

    @pytest.mark.asyncio
    async def test_format_response_text(self, mock_adapter, mock_context):
        """Test translation text response formatting.

        This is a regression test for the bug where text format was
        JSON-encoded instead of returned as plain text.
        """
        strategy = TranslationStrategy()
        response = InternalTranslationResponse(
            text="Hello world",
            _response_format="text",
        )
        serializer = get_protocol_serializer("translation")
        result = await strategy.format_response(response, serializer, "translation")

        assert isinstance(result, Response)
        assert result.media_type == "text/plain"
        assert result.body == b"Hello world"

    @pytest.mark.asyncio
    async def test_format_response_srt(self, mock_adapter, mock_context):
        """Test translation SRT response formatting."""
        strategy = TranslationStrategy()
        response = InternalTranslationResponse(
            text="1\n00:00:00,000 --> ...",
            _response_format="srt",
        )
        serializer = get_protocol_serializer("translation")
        result = await strategy.format_response(response, serializer, "translation")

        assert isinstance(result, Response)
        assert result.media_type == "text/plain"
        assert result.body == b"1\n00:00:00,000 --> ..."

    @pytest.mark.asyncio
    async def test_format_response_vtt(self, mock_adapter, mock_context):
        """Test translation VTT response formatting."""
        strategy = TranslationStrategy()
        response = InternalTranslationResponse(
            text="WEBVTT\n\n...",
            _response_format="vtt",
        )
        serializer = get_protocol_serializer("translation")
        result = await strategy.format_response(response, serializer, "translation")

        assert isinstance(result, Response)
        assert result.media_type == "text/plain"
        assert result.body == b"WEBVTT\n\n..."

    @pytest.mark.asyncio
    async def test_format_response_verbose_json(self, mock_adapter, mock_context):
        """Test translation verbose_json response formatting."""
        strategy = TranslationStrategy()
        response = InternalTranslationResponse(
            text="Hello world",
            language="english",
            duration=5.0,
            _response_format="verbose_json",
        )
        serializer = get_protocol_serializer("translation")
        result = await strategy.format_response(response, serializer, "translation")

        assert isinstance(result, Response)
        assert result.media_type == "application/json"
        body = result.body
        assert b"Hello world" in body
        assert b"english" in body
        assert b"5.0" in body

    @pytest.mark.asyncio
    async def test_format_response_text_not_json_encoded(self, mock_adapter, mock_context):
        """Test that text format is NOT JSON-encoded.

        Regression test: previously text format fell through to JSON
        serialization and returned '"Hello world"' (with literal quotes).
        """
        strategy = TranslationStrategy()
        response = InternalTranslationResponse(
            text="Hello world",
            _response_format="text",
        )
        serializer = get_protocol_serializer("translation")
        result = await strategy.format_response(response, serializer, "translation")

        # Must be plain bytes, not JSON-encoded string
        assert result.body == b"Hello world"
        assert b'"' not in result.body
