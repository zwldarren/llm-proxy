"""Tests for OpenAI provider adapter audio endpoints."""

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
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase


class MockResponse:
    """Mock HTTP response for non-streaming audio requests."""

    def __init__(
        self,
        status_code: int,
        content: bytes = b"",
        text: str = "",
        headers: dict[str, str] | None = None,
        json_data: dict | None = None,
    ):
        self.status_code = status_code
        self.content = content
        self.text = text
        self.headers = headers or {}
        self._json_data = json_data

    def json(self) -> dict:
        """Return JSON data for responses with application/json content type."""
        if self._json_data is not None:
            return self._json_data
        return {}


class MockStreamResponse:
    """Mock streaming HTTP response that acts as an async context manager."""

    def __init__(self, status_code: int, raw_chunks: list[bytes], lines: list[str | bytes]):
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
def openai_adapter():
    """Create an OpenAI adapter for testing."""
    return OpenAICompatibleBase(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
    )


@pytest.mark.asyncio
async def test_speech(openai_adapter):
    """Test OpenAI speech generation request."""
    mock_response = MockResponse(
        200,
        content=b"audio-content-here",
        headers={"content-type": "audio/mpeg"},
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    openai_adapter._http_client = mock_client

    request = InternalSpeechRequest(
        model="tts-1",
        input="Hello world",
        voice="alloy",
        response_format="mp3",
    )
    response = await openai_adapter.speech(request)

    assert isinstance(response, InternalSpeechResponse)
    assert response.content == b"audio-content-here"
    assert response.content_type == "audio/mpeg"

    mock_client.post.assert_awaited_once()
    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "tts-1"
    assert call_kwargs["json"]["input"] == "Hello world"
    assert call_kwargs["json"]["voice"] == "alloy"
    assert "stream" not in call_kwargs["json"]


@pytest.mark.asyncio
async def test_stream_speech(openai_adapter):
    """Test OpenAI streaming speech generation request."""
    mock_response = MockStreamResponse(
        200,
        raw_chunks=[b"chunk1", b"chunk2", b"chunk3"],
        lines=[],
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    openai_adapter._http_client = mock_client

    request = InternalSpeechRequest(
        model="tts-1",
        input="Hello world",
        voice="alloy",
        response_format="mp3",
        stream=True,
    )
    stream = await openai_adapter.stream_speech(request)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == [b"chunk1", b"chunk2", b"chunk3"]
    assert mock_response._closed is True


@pytest.mark.asyncio
async def test_transcription_json(openai_adapter):
    """Test OpenAI transcription request with JSON response."""
    mock_response = MockResponse(
        200,
        text='{"text": "Hello world"}',
        headers={"content-type": "application/json"},
        json_data={"text": "Hello world"},
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    openai_adapter._http_client = mock_client

    request = InternalTranscriptionRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        response_format="json",
    )
    response = await openai_adapter.transcription(request)

    assert isinstance(response, InternalTranscriptionResponse)
    assert response.text == "Hello world"
    assert response._response_format == "json"


@pytest.mark.asyncio
async def test_transcription_text(openai_adapter):
    """Test OpenAI transcription request with plain text response."""
    mock_response = MockResponse(
        200,
        text="Hello world",
        headers={"content-type": "text/plain"},
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    openai_adapter._http_client = mock_client

    request = InternalTranscriptionRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        response_format="text",
    )
    response = await openai_adapter.transcription(request)

    assert isinstance(response, InternalTranscriptionResponse)
    assert response.text == "Hello world"
    assert response._response_format == "text"


@pytest.mark.asyncio
async def test_stream_transcription(openai_adapter):
    """Test OpenAI streaming transcription request."""
    mock_response = MockStreamResponse(
        200,
        raw_chunks=[],
        lines=[b'{"text":"Hello"}', b'{"text":" world"}'],
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    openai_adapter._http_client = mock_client

    request = InternalTranscriptionRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        response_format="json",
        stream=True,
    )
    stream = await openai_adapter.stream_transcription(request)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == ['{"text":"Hello"}\n', '{"text":" world"}\n']
    assert mock_response._closed is True


@pytest.mark.asyncio
async def test_translation_json(openai_adapter):
    """Test OpenAI translation request with JSON response."""
    mock_response = MockResponse(
        200,
        text='{"text": "Hello world"}',
        headers={"content-type": "application/json"},
        json_data={"text": "Hello world"},
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    openai_adapter._http_client = mock_client

    request = InternalTranslationRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        response_format="json",
    )
    response = await openai_adapter.translation(request)

    assert isinstance(response, InternalTranslationResponse)
    assert response.text == "Hello world"
    assert response._response_format == "json"


@pytest.mark.asyncio
async def test_translation_text(openai_adapter):
    """Test OpenAI translation request with plain text response."""
    mock_response = MockResponse(
        200,
        text="Hello world",
        headers={"content-type": "text/plain"},
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    openai_adapter._http_client = mock_client

    request = InternalTranslationRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        response_format="text",
    )
    response = await openai_adapter.translation(request)

    assert isinstance(response, InternalTranslationResponse)
    assert response.text == "Hello world"
    assert response._response_format == "text"


@pytest.mark.asyncio
async def test_speech_with_instructions(openai_adapter):
    """Test OpenAI speech generation with instructions."""
    mock_response = MockResponse(
        200,
        content=b"audio-content",
        headers={"content-type": "audio/mpeg"},
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    openai_adapter._http_client = mock_client

    request = InternalSpeechRequest(
        model="gpt-4o-mini-tts",
        input="Hello",
        voice="alloy",
        instructions="Speak cheerfully",
    )
    _ = await openai_adapter.speech(request)

    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["json"]["instructions"] == "Speak cheerfully"


@pytest.mark.asyncio
async def test_transcription_with_optional_params(openai_adapter):
    """Test OpenAI transcription with optional parameters."""
    mock_response = MockResponse(
        200,
        text='{"text": "Hello"}',
        headers={"content-type": "application/json"},
        json_data={"text": "Hello"},
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    openai_adapter._http_client = mock_client

    request = InternalTranscriptionRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        language="en",
        prompt="Greetings",
        temperature=0.5,
        timestamp_granularities=["word", "segment"],
        include=["logprobs"],
    )
    _ = await openai_adapter.transcription(request)

    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["data"]["language"] == "en"
    assert call_kwargs["data"]["prompt"] == "Greetings"
    assert call_kwargs["data"]["temperature"] == "0.5"
    assert call_kwargs["data"]["timestamp_granularities[]"] == ["word", "segment"]
    assert call_kwargs["data"]["include[]"] == ["logprobs"]


@pytest.mark.asyncio
async def test_translation_with_optional_params(openai_adapter):
    """Test OpenAI translation with optional parameters."""
    mock_response = MockResponse(
        200,
        text='{"text": "Hello"}',
        headers={"content-type": "application/json"},
        json_data={"text": "Hello"},
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    openai_adapter._http_client = mock_client

    request = InternalTranslationRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
        prompt="Greetings",
        temperature=0.5,
    )
    _ = await openai_adapter.translation(request)

    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["data"]["prompt"] == "Greetings"
    assert call_kwargs["data"]["temperature"] == "0.5"


@pytest.mark.asyncio
async def test_transcription_usage(openai_adapter):
    """Test OpenAI transcription response with usage data."""
    mock_response = MockResponse(
        200,
        text='{"text": "Hello", "usage": {"input_tokens": 10, "output_tokens": 5}}',
        headers={"content-type": "application/json"},
        json_data={
            "text": "Hello",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    openai_adapter._http_client = mock_client

    request = InternalTranscriptionRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
    )
    response = await openai_adapter.transcription(request)

    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_speech_captures_rate_limit_headers(openai_adapter):
    """Upstream rate-limit headers are captured in provider_info."""
    mock_response = MockResponse(
        200,
        content=b"fake-audio",
        headers={
            "content-type": "audio/mp3",
            "x-ratelimit-limit": "100",
            "x-ratelimit-remaining": "99",
            "retry-after": "2",
            "RateLimit-Reset": "1234567890",
        },
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    openai_adapter._http_client = mock_client

    request = InternalSpeechRequest(
        model="tts-1",
        input="hello",
        voice="alloy",
        response_format="mp3",
    )
    response = await openai_adapter.speech(request)

    assert response.provider_info.get("_rate_limit_headers") == {
        "x-ratelimit-limit": "100",
        "x-ratelimit-remaining": "99",
        "retry-after": "2",
        "RateLimit-Reset": "1234567890",
    }


@pytest.mark.asyncio
async def test_transcription_captures_rate_limit_headers(openai_adapter):
    """Upstream rate-limit headers are captured in transcription provider_info."""
    mock_response = MockResponse(
        200,
        text='{"text": "Hello"}',
        headers={
            "content-type": "application/json",
            "x-ratelimit-limit": "100",
            "x-ratelimit-remaining": "99",
        },
        json_data={"text": "Hello"},
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    openai_adapter._http_client = mock_client

    request = InternalTranscriptionRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
    )
    response = await openai_adapter.transcription(request)

    assert response.provider_info.get("_rate_limit_headers") == {
        "x-ratelimit-limit": "100",
        "x-ratelimit-remaining": "99",
    }


@pytest.mark.asyncio
async def test_translation_captures_rate_limit_headers(openai_adapter):
    """Upstream rate-limit headers are captured in translation provider_info."""
    mock_response = MockResponse(
        200,
        text='{"text": "Hola"}',
        headers={
            "content-type": "application/json",
            "retry-after": "5",
        },
        json_data={"text": "Hola"},
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    openai_adapter._http_client = mock_client

    request = InternalTranslationRequest(
        model="whisper-1",
        file=b"fake-audio",
        filename="audio.mp3",
    )
    response = await openai_adapter.translation(request)

    assert response.provider_info.get("_rate_limit_headers") == {
        "retry-after": "5",
    }


def test_stream_transcription_includes_usage_include(openai_adapter):
    """Streaming transcription auto-injects include[]=usage (gpt-4o-transcribe)."""
    from llm_proxy.models import InternalTranscriptionRequest

    request = InternalTranscriptionRequest(
        model="gpt-4o-transcribe",
        file=b"fake-audio",
        filename="audio.mp3",
        include=["logprobs"],
    )
    data, _ = openai_adapter._build_transcription_data(request, stream=True)

    assert data.get("stream") == "true"
    includes = data.get("include[]", [])
    assert "usage" in includes
    assert "logprobs" in includes
