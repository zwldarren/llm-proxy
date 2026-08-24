"""Tests for Gemini adapter speech (TTS) support via generateContent.

Covers the /v1/audio/speech path: request body building, response audio
extraction, format negotiation (WAV/PCM), and streaming.
"""

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import GeminiSpecificParams, GenerationParams, InternalSpeechRequest
from llm_proxy.providers.gemini import GeminiAdapter
from llm_proxy.serialization.gemini.speech import DEFAULT_SAMPLE_RATE

TTS_MODEL = "gemini-3.1-flash-tts-preview"
PCM = b"\x10\x00" * 50  # 100 bytes of fake PCM
PCM_B64 = base64.b64encode(PCM).decode()


def _request(**overrides) -> InternalSpeechRequest:
    kwargs = {"model": TTS_MODEL, "input": "Hello world", "voice": "Kore"}
    kwargs.update(overrides)
    return InternalSpeechRequest(**kwargs)


def _gemini_tts_payload(data: str = PCM_B64) -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/L16;codec=pcm;rate=24000",
                                "data": data,
                            }
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10},
    }


def _patch_http(adapter, monkeypatch, response):
    """Patch the HTTP layer so adapter methods talk to a mock response."""
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    async def _get_client():
        return mock_client

    async def _check_response_status(_resp):
        pass

    monkeypatch.setattr(adapter, "_get_client", _get_client)
    monkeypatch.setattr(adapter, "_check_response_status", _check_response_status)
    monkeypatch.setattr(adapter._retry, "execute", lambda op, *a, **kw: op())
    return mock_client


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


class TestSpeechRequestBuilding:
    def test_basic_body_shape(self):
        adapter = GeminiAdapter(api_key="k")
        body = adapter._build_speech_raw(_request())
        assert body["contents"] == [{"parts": [{"text": "Hello world"}]}]
        config = body["generationConfig"]
        assert config["responseModalities"] == ["AUDIO"]
        assert config["speechConfig"] == {
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
        }

    def test_openai_voice_mapped_to_gemini(self):
        adapter = GeminiAdapter(api_key="k")
        body = adapter._build_speech_raw(_request(voice="nova"))
        voice = body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"][
            "voiceName"
        ]
        assert voice == "Sulafat"

    def test_instructions_prepended_as_directorial_prompt(self):
        adapter = GeminiAdapter(api_key="k")
        body = adapter._build_speech_raw(
            _request(instructions="Say in a spooky whisper", input="Boo!")
        )
        text = body["contents"][0]["parts"][0]["text"]
        assert text == "Say in a spooky whisper\n\nBoo!"

    def test_extra_speech_config_consumed_not_leaked(self):
        """speech_config in extra overrides the voice-derived config and is
        popped so the dispatch's extra merge can't leak it top-level."""
        adapter = GeminiAdapter(api_key="k")
        multi = {
            "multiSpeakerVoiceConfig": {
                "speakerVoiceConfigs": [
                    {
                        "speaker": "Joe",
                        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}},
                    }
                ]
            }
        }
        request = _request(extra={"speech_config": multi, "language_code": "en-US"})
        body = adapter._build_speech_raw(request)
        assert body["generationConfig"]["speechConfig"] == multi
        assert "speech_config" not in request.extra
        assert "language_code" not in request.extra
        assert "speech_config" not in body

    def test_params_gemini_speech_config_wins(self):
        adapter = GeminiAdapter(api_key="k")
        explicit = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Charon"}}}
        request = _request(
            voice="nova",
            params=GenerationParams(gemini=GeminiSpecificParams(speech_config=explicit)),
        )
        body = adapter._build_speech_raw(request)
        assert body["generationConfig"]["speechConfig"] == explicit

    def test_language_code_from_extra(self):
        adapter = GeminiAdapter(api_key="k")
        request = _request(extra={"language_code": "cmn-CN"})
        body = adapter._build_speech_raw(request)
        assert body["generationConfig"]["speechConfig"]["languageCode"] == "cmn-CN"

    def test_speech_url(self):
        adapter = GeminiAdapter(api_key="k")
        url = adapter._speech_url(_request())
        assert url.endswith(f"/models/{TTS_MODEL}:generateContent")
        stream_url = adapter._speech_url(_request(stream=True))
        assert stream_url.endswith(f"/models/{TTS_MODEL}:streamGenerateContent?alt=sse")


# ---------------------------------------------------------------------------
# Format negotiation
# ---------------------------------------------------------------------------


class TestFormatNegotiation:
    def test_wav_requested_returns_wav(self):
        adapter = GeminiAdapter(api_key="k")
        content, content_type = adapter._encode_speech_output(PCM, DEFAULT_SAMPLE_RATE, "wav")
        assert content[:4] == b"RIFF"
        assert content_type == "audio/wav"

    def test_pcm_requested_returns_raw_pcm(self):
        adapter = GeminiAdapter(api_key="k")
        content, content_type = adapter._encode_speech_output(PCM, DEFAULT_SAMPLE_RATE, "pcm")
        assert content == PCM
        assert content_type == "audio/L16"

    def test_mp3_request_falls_back_to_wav_honestly(self):
        """Gemini cannot produce mp3; we return WAV and say so."""
        adapter = GeminiAdapter(api_key="k")
        content, content_type = adapter._encode_speech_output(PCM, DEFAULT_SAMPLE_RATE, "mp3")
        assert content[:4] == b"RIFF"
        assert content_type == "audio/wav"

    def test_stream_media_type(self):
        adapter = GeminiAdapter(api_key="k")
        assert adapter.speech_stream_media_type(_request(response_format="mp3")) == "audio/wav"
        assert adapter.speech_stream_media_type(_request(response_format="pcm")) == "audio/L16"


# ---------------------------------------------------------------------------
# Non-streaming speech()
# ---------------------------------------------------------------------------


class TestSpeechEndpoint:
    @pytest.mark.asyncio
    async def test_speech_success_wav(self, monkeypatch):
        adapter = GeminiAdapter(api_key="k")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = _gemini_tts_payload()
        mock_client = _patch_http(adapter, monkeypatch, response)

        result = await adapter.speech(_request())

        url = mock_client.post.call_args.args[0]
        assert url.endswith(f"/models/{TTS_MODEL}:generateContent")
        assert result.content[:4] == b"RIFF"
        assert result.content[44:] == PCM
        assert result.content_type == "audio/wav"

    @pytest.mark.asyncio
    async def test_speech_success_pcm(self, monkeypatch):
        adapter = GeminiAdapter(api_key="k")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = _gemini_tts_payload()
        _patch_http(adapter, monkeypatch, response)

        result = await adapter.speech(_request(response_format="pcm"))
        assert result.content == PCM
        assert result.content_type == "audio/L16"

    @pytest.mark.asyncio
    async def test_speech_multiple_audio_parts_concatenated(self, monkeypatch):
        adapter = GeminiAdapter(api_key="k")
        payload = _gemini_tts_payload()
        payload["candidates"][0]["content"]["parts"].append(
            {"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": PCM_B64}}
        )
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = payload
        _patch_http(adapter, monkeypatch, response)

        result = await adapter.speech(_request(response_format="pcm"))
        assert result.content == PCM + PCM

    @pytest.mark.asyncio
    async def test_speech_no_audio_raises(self, monkeypatch):
        adapter = GeminiAdapter(api_key="k")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "no audio"}]}}]
        }
        _patch_http(adapter, monkeypatch, response)

        with pytest.raises(ProviderError, match="no audio"):
            await adapter.speech(_request())


# ---------------------------------------------------------------------------
# Streaming stream_speech()
# ---------------------------------------------------------------------------


class _AsyncCM:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        return False


class _MockStreamResponse:
    def __init__(self, lines: list[bytes], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines

    async def iter_lines(self):
        for line in self._lines:
            yield line


class TestStreamSpeech:
    def _patch_stream(self, adapter, monkeypatch, lines):
        stream_response = _MockStreamResponse(lines)

        async def _get_client():
            return MagicMock()

        monkeypatch.setattr(adapter, "_get_client", _get_client)
        monkeypatch.setattr(adapter, "_streaming_post", lambda *a, **kw: _AsyncCM(stream_response))
        monkeypatch.setattr(adapter._retry, "execute_generator", lambda factory, **kw: factory())

    @pytest.mark.asyncio
    async def test_stream_emits_wav_header_then_pcm(self, monkeypatch):
        adapter = GeminiAdapter(api_key="k")
        chunk = _gemini_tts_payload()
        lines = [f"data: {__import__('orjson').dumps(chunk).decode()}".encode()]
        self._patch_stream(adapter, monkeypatch, lines)

        gen = await adapter.stream_speech(_request(response_format="wav"))
        chunks = [chunk async for chunk in gen]

        assert len(chunks) == 2
        assert chunks[0][:4] == b"RIFF"
        # Streaming WAV advertises unknown length.
        assert int.from_bytes(chunks[0][40:44], "little") == 0xFFFFFFFF
        assert int.from_bytes(chunks[0][24:28], "little") == DEFAULT_SAMPLE_RATE
        assert chunks[1] == PCM

    @pytest.mark.asyncio
    async def test_stream_pcm_no_header(self, monkeypatch):
        adapter = GeminiAdapter(api_key="k")
        chunk = _gemini_tts_payload()
        lines = [f"data: {__import__('orjson').dumps(chunk).decode()}".encode()]
        self._patch_stream(adapter, monkeypatch, lines)

        gen = await adapter.stream_speech(_request(response_format="pcm"))
        chunks = [chunk async for chunk in gen]
        assert chunks == [PCM]

    @pytest.mark.asyncio
    async def test_stream_uses_sse_endpoint(self, monkeypatch):
        adapter = GeminiAdapter(api_key="k")
        captured = {}

        def _streaming_post(client, url, **kwargs):
            captured["url"] = url
            return _AsyncCM(_MockStreamResponse([]))

        async def _get_client():
            return MagicMock()

        monkeypatch.setattr(adapter, "_get_client", _get_client)
        monkeypatch.setattr(adapter, "_streaming_post", _streaming_post)
        monkeypatch.setattr(adapter._retry, "execute_generator", lambda factory, **kw: factory())

        gen = await adapter.stream_speech(_request())
        _ = [chunk async for chunk in gen]
        assert captured["url"].endswith(f"/models/{TTS_MODEL}:streamGenerateContent?alt=sse")
