"""Ollama speech compatibility (POST /v1/audio/speech, Ollama v0.40.8+)."""

import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import InternalSpeechRequest
from llm_proxy.providers.ollama.adapter import OllamaAdapter


def _request(**overrides: Any) -> InternalSpeechRequest:
    fields: dict[str, Any] = {
        "model": "phonon",
        "input": "Hello from Ollama.",
        "voice": "default",
    }
    fields.update(overrides)
    return InternalSpeechRequest(**fields)


@pytest.fixture
def provider() -> OllamaAdapter:
    return OllamaAdapter()


@pytest.mark.asyncio
async def test_speech_uses_ollama_compat_endpoint_and_defaults_to_wav(
    provider: OllamaAdapter, mock_response_cls, make_mock_client
) -> None:
    response = mock_response_cls(
        content_data=b"RIFF\x00\x00\x00\x00WAVEfmt ",
        headers={"content-type": "audio/wav"},
    )
    client = make_mock_client(response)

    with patch.object(provider, "_get_client", return_value=client):
        result = await provider.speech(_request())

    assert client.post.call_args.args[0] == "http://localhost:11434/v1/audio/speech"
    assert client.post.call_args.kwargs["json"] == {
        "model": "phonon",
        "input": "Hello from Ollama.",
        "voice": "default",
        "response_format": "wav",
        "speed": 1.0,
    }
    assert result.content == b"RIFF\x00\x00\x00\x00WAVEfmt "
    assert result.content_type == "audio/wav"


@pytest.mark.asyncio
async def test_speech_forwards_pcm_response_format(
    provider: OllamaAdapter, mock_response_cls, make_mock_client
) -> None:
    response = mock_response_cls(
        content_data=b"\x00\x01\x02\x03",
        headers={"content-type": "audio/pcm"},
    )
    client = make_mock_client(response)

    with patch.object(provider, "_get_client", return_value=client):
        result = await provider.speech(_request(response_format="pcm"))

    assert client.post.call_args.kwargs["json"]["response_format"] == "pcm"
    assert result.content_type == "audio/pcm"


@pytest.mark.asyncio
async def test_speech_falls_back_to_wav_for_unsupported_format(
    provider: OllamaAdapter, mock_response_cls, make_mock_client, caplog
) -> None:
    caplog.set_level(logging.INFO, logger="llm_proxy.providers.ollama.adapter")
    response = mock_response_cls(
        content_data=b"RIFF",
        headers={"content-type": "audio/wav"},
    )
    client = make_mock_client(response)

    with patch.object(provider, "_get_client", return_value=client):
        await provider.speech(_request(response_format="mp3"))

    assert client.post.call_args.kwargs["json"]["response_format"] == "wav"
    assert any("supports WAV/PCM only" in record.message for record in caplog.records)


def test_stream_media_type_matches_effective_format(provider: OllamaAdapter) -> None:
    assert provider.speech_stream_media_type(_request()) == "audio/wav"
    assert provider.speech_stream_media_type(_request(response_format="wav")) == "audio/wav"
    assert provider.speech_stream_media_type(_request(response_format="pcm")) == "audio/pcm"


@pytest.mark.asyncio
async def test_speech_sends_bearer_api_key(mock_response_cls, make_mock_client) -> None:
    provider = OllamaAdapter(api_key="ollama-key")
    client = make_mock_client(
        mock_response_cls(content_data=b"RIFF", headers={"content-type": "audio/wav"})
    )

    with patch.object(provider, "_get_client", return_value=client):
        await provider.speech(_request())

    assert client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer ollama-key"


@pytest.mark.asyncio
async def test_speech_maps_upstream_400(
    provider: OllamaAdapter, mock_response_cls, make_mock_client
) -> None:
    response = mock_response_cls(
        status_code=400,
        text_data=(
            '{"error": {"message": "only speed 1 is supported", "type": "invalid_request_error"}}'
        ),
    )
    client = make_mock_client(response)

    with (
        patch.object(provider, "_get_client", return_value=client),
        pytest.raises(ProviderError) as exc_info,
    ):
        await provider.speech(_request(speed=2.0))

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_type == "invalid_request_error"
    assert "only speed 1 is supported" in exc_info.value.message


class _ChunkIterator:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class _StreamResponse:
    status_code = 200
    headers = {"content-type": "audio/wav"}

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aenter__(self) -> _StreamResponse:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True

    def iter_raw(self) -> AsyncIterator[bytes]:
        return _ChunkIterator(self._chunks)


@pytest.mark.asyncio
async def test_stream_speech_relays_raw_audio(provider: OllamaAdapter, make_mock_client) -> None:
    response = _StreamResponse([b"chunk1", b"chunk2"])
    client = make_mock_client(response)

    with patch.object(provider, "_get_client", return_value=client):
        stream = await provider.stream_speech(_request(stream=True))
        chunks = [chunk async for chunk in stream]

    assert chunks == [b"chunk1", b"chunk2"]
    assert client.post.call_args.args[0] == "http://localhost:11434/v1/audio/speech"
    assert client.post.call_args.kwargs["json"]["response_format"] == "wav"
    assert response.closed is True
