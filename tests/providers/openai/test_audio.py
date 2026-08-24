"""Tests for OpenAI adapter audio endpoints.

These endpoints are shared with the standard OpenAI adapter, so the
implementation is inherited from BaseProvider. Tests verify the endpoints
are wired correctly for the openai provider.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.models import (
    InternalSpeechRequest,
    InternalSpeechResponse,
    InternalTranscriptionRequest,
    InternalTranscriptionResponse,
    InternalTranslationRequest,
    InternalTranslationResponse,
)
from llm_proxy.models.types import PromptTokensDetails, Usage
from llm_proxy.providers.openai.adapter import OpenAIAdapter


class MockResponse:
    """Mock HTTP response for non-streaming audio requests."""

    def __init__(
        self,
        status_code: int,
        content: bytes = b"",
        text: str = "",
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self.content = content
        self.text = text
        self.headers = headers or {}


class MockStreamResponse:
    """Mock streaming HTTP response that acts as an async context manager."""

    def __init__(self, status_code: int, raw_chunks: list[bytes], lines: list[str]):
        self.status_code = status_code
        self._raw_chunks = raw_chunks
        self._lines = lines
        self._closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, _exc_val, _exc_tb):
        self._closed = True

    async def aclose(self):
        self._closed = True

    def iter_raw(self):
        class _Iter:
            def __init__(inner_self, chunks):
                inner_self._chunks = chunks
                inner_self._idx = 0

            def __aiter__(inner_self):
                return inner_self

            async def __anext__(inner_self):
                if inner_self._idx >= len(inner_self._chunks):
                    raise StopAsyncIteration
                chunk = inner_self._chunks[inner_self._idx]
                inner_self._idx += 1
                return chunk

        return _Iter(self._raw_chunks)

    def iter_lines(self):
        class _Iter:
            def __init__(inner_self, lines):
                inner_self._lines = lines
                inner_self._idx = 0

            def __aiter__(inner_self):
                return inner_self

            async def __anext__(inner_self):
                if inner_self._idx >= len(inner_self._lines):
                    raise StopAsyncIteration
                line = inner_self._lines[inner_self._idx]
                inner_self._idx += 1
                return line

        return _Iter(self._lines)


@pytest.fixture
def responses_adapter():
    """Create an OpenAI adapter for testing."""
    return OpenAIAdapter(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
    )


@pytest.mark.asyncio
async def test_speech(responses_adapter):
    """Test speech generation via OpenAI adapter."""
    mock_response = MockResponse(
        200,
        content=b"audio-content-here",
        headers={"content-type": "audio/mpeg"},
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    responses_adapter._http_client = mock_client

    request = InternalSpeechRequest(
        model="tts-1",
        input="Hello world",
        voice="alloy",
        response_format="mp3",
    )
    response = await responses_adapter.speech(request)

    assert isinstance(response, InternalSpeechResponse)
    assert response.content == b"audio-content-here"
    assert response.content_type == "audio/mpeg"

    mock_client.post.assert_awaited_once()
    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "tts-1"
    assert call_kwargs["json"]["input"] == "Hello world"
    assert call_kwargs["json"]["voice"] == "alloy"
    assert "stream" not in call_kwargs["json"]

    assert mock_client.post.call_args.args[0] == "https://api.openai.com/v1/audio/speech"


@pytest.mark.asyncio
async def test_stream_speech(responses_adapter):
    """Test streaming speech generation via OpenAI adapter."""
    mock_response = MockStreamResponse(
        200,
        raw_chunks=[b"chunk1", b"chunk2", b"chunk3"],
        lines=[],
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    responses_adapter._http_client = mock_client

    request = InternalSpeechRequest(
        model="tts-1",
        input="Hello world",
        voice="alloy",
        response_format="mp3",
        stream=True,
    )
    stream = await responses_adapter.stream_speech(request)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == [b"chunk1", b"chunk2", b"chunk3"]
    assert mock_response._closed is True


@pytest.mark.asyncio
async def test_transcription_json(responses_adapter):
    """Test transcription via OpenAI adapter with JSON response."""
    mock_response = MockResponse(
        200,
        text='{"text": "Hello world"}',
        headers={"content-type": "application/json"},
    )
    mock_response.json = lambda: {"text": "Hello world"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    responses_adapter._http_client = mock_client

    request = InternalTranscriptionRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        response_format="json",
    )
    response = await responses_adapter.transcription(request)

    assert isinstance(response, InternalTranscriptionResponse)
    assert response.text == "Hello world"

    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["data"]["model"] == "whisper-1"
    assert call_kwargs["data"]["response_format"] == "json"


@pytest.mark.asyncio
async def test_stream_transcription(responses_adapter):
    """Test streaming transcription via OpenAI adapter."""
    mock_response = MockStreamResponse(
        200,
        raw_chunks=[],
        lines=['{"text": "Hello"}', '{"text": " world"}'],
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    responses_adapter._http_client = mock_client

    request = InternalTranscriptionRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        response_format="json",
    )
    stream = await responses_adapter.stream_transcription(request)

    lines = []
    async for line in stream:
        lines.append(line)

    assert lines == ['{"text": "Hello"}\n', '{"text": " world"}\n']
    assert mock_response._closed is True


@pytest.mark.asyncio
async def test_translation_json(responses_adapter):
    """Test translation via OpenAI adapter with JSON response."""
    mock_response = MockResponse(
        200,
        text='{"text": "Hello world"}',
        headers={"content-type": "application/json"},
    )
    mock_response.json = lambda: {"text": "Hello world"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    responses_adapter._http_client = mock_client

    request = InternalTranslationRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        response_format="json",
    )
    response = await responses_adapter.translation(request)

    assert isinstance(response, InternalTranslationResponse)
    assert response.text == "Hello world"

    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["data"]["model"] == "whisper-1"
    assert call_kwargs["data"]["response_format"] == "json"


@pytest.mark.asyncio
async def test_transcription_with_usage(responses_adapter):
    """Test transcription response with usage data."""
    mock_response = MockResponse(
        200,
        text='{"text": "Hello", "usage": {"input_tokens": 10, "output_tokens": 5}}',
        headers={"content-type": "application/json"},
    )
    mock_response.json = lambda: {
        "text": "Hello",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    responses_adapter._http_client = mock_client

    request = InternalTranscriptionRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        response_format="json",
    )
    response = await responses_adapter.transcription(request)

    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_translation_with_usage(responses_adapter):
    """Test translation response with usage data."""
    mock_response = MockResponse(
        200,
        text='{"text": "Hello", "usage": {"input_tokens": 10, "output_tokens": 5}}',
        headers={"content-type": "application/json"},
    )
    mock_response.json = lambda: {
        "text": "Hello",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    responses_adapter._http_client = mock_client

    request = InternalTranslationRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        response_format="json",
    )
    response = await responses_adapter.translation(request)

    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5


class TestAudioUsageFormatting:
    """Regression tests for STT usage preservation and formatting."""

    def test_parse_usage_preserves_duration_and_input_details(self, responses_adapter):
        usage = responses_adapter._parse_usage(
            {
                "type": "duration",
                "seconds": 120.5,
                "input_token_details": {"audio_tokens": 100, "text_tokens": 10},
            }
        )
        assert usage is not None
        assert usage.audio_duration_seconds == 120.5
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.audio_tokens == 100
        assert usage.prompt_tokens_details.text_tokens == 10

    def test_format_transcription_includes_duration_and_details(self):
        from llm_proxy.protocols.openai.audio_serializer import OpenAIAudioSerializer

        serializer = OpenAIAudioSerializer()
        response = InternalTranscriptionResponse(
            text="Hello world",
            task="transcribe",
            language="en",
            duration=10.0,
            usage=Usage(
                input_tokens=0,
                output_tokens=0,
                audio_duration_seconds=120.5,
                prompt_tokens_details=PromptTokensDetails(audio_tokens=100, text_tokens=10),
            ),
        )
        response._response_format = "verbose_json"
        result = serializer.format_transcription_response(response, "verbose_json")
        assert result["usage"]["audio_duration_seconds"] == 120.5
        assert result["usage"]["type"] == "duration"
        assert result["usage"]["input_token_details"] == {
            "audio_tokens": 100,
            "text_tokens": 10,
        }

    def test_format_transcription_token_usage_has_type_tokens(self):
        from llm_proxy.protocols.openai.audio_serializer import OpenAIAudioSerializer

        serializer = OpenAIAudioSerializer()
        response = InternalTranscriptionResponse(
            text="Hello world",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        response._response_format = "json"
        result = serializer.format_transcription_response(response, "json")
        assert result["usage"]["type"] == "tokens"
        assert result["usage"]["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_transcription_verbose_json_with_duration_usage(self, responses_adapter):
        mock_response = MockResponse(
            200,
            text='{"text": "Hello", "usage": {"type": "duration", "seconds": 60}}',
            headers={"content-type": "application/json"},
        )
        mock_response.json = lambda: {
            "text": "Hello",
            "usage": {"type": "duration", "seconds": 60},
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        responses_adapter._http_client = mock_client

        request = InternalTranscriptionRequest(
            model="whisper-1",
            file=b"fake-audio",
            filename="audio.mp3",
            response_format="verbose_json",
        )
        response = await responses_adapter.transcription(request)

        assert response.usage is not None
        assert response.usage.audio_duration_seconds == 60

        from llm_proxy.protocols.openai.audio_serializer import OpenAIAudioSerializer

        serializer = OpenAIAudioSerializer()
        response._response_format = "verbose_json"
        result = serializer.format_transcription_response(response, "verbose_json")
        assert result["usage"]["type"] == "duration"
        assert result["usage"]["audio_duration_seconds"] == 60
