"""Tests for Gemini TTS support: shared speech helpers, request building,
response parsing, and streaming transformation.

References:
- https://ai.google.dev/gemini-api/docs/speech-generation
- https://ai.google.dev/api/generate-content#SpeechConfig
"""

import base64

import orjson
import pytest

from llm_proxy.models import (
    ConversationContext,
    GeminiSpecificParams,
    GenerationParams,
    InternalRequest,
    Message,
    OpenAISpecificParams,
    TextBlock,
)
from llm_proxy.models.content_blocks import AudioBlock, ImageBlock, VideoBlock
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.gemini.request_builder import GeminiRequestBuilderMixin
from llm_proxy.serialization.gemini.response_parser import GeminiResponseParserMixin
from llm_proxy.serialization.gemini.speech import (
    DEFAULT_GEMINI_VOICE,
    DEFAULT_SAMPLE_RATE,
    build_speech_config,
    is_gemini_tts_model,
    parse_audio_mime,
    pcm_to_wav,
    resolve_voice,
    wav_header,
)
from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer

# ---------------------------------------------------------------------------
# Shared helpers: voice resolution
# ---------------------------------------------------------------------------


class TestResolveVoice:
    def test_gemini_voice_passthrough_canonical_case(self):
        assert resolve_voice("Puck") == "Puck"

    def test_gemini_voice_case_insensitive(self):
        assert resolve_voice("kore") == "Kore"
        assert resolve_voice("SULAFAT") == "Sulafat"

    def test_openai_voice_mapped(self):
        assert resolve_voice("alloy") == "Kore"
        assert resolve_voice("nova") == "Sulafat"
        assert resolve_voice("shimmer") == "Vindemiatrix"

    def test_unknown_voice_falls_back_to_default(self):
        assert resolve_voice("not-a-voice") == DEFAULT_GEMINI_VOICE

    def test_none_and_empty_fall_back_to_default(self):
        assert resolve_voice(None) == DEFAULT_GEMINI_VOICE
        assert resolve_voice("") == DEFAULT_GEMINI_VOICE


class TestBuildSpeechConfig:
    def test_single_speaker_schema(self):
        config = build_speech_config("Puck")
        assert config == {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}}

    def test_language_code_included(self):
        config = build_speech_config("Kore", language_code="cmn-CN")
        assert config["languageCode"] == "cmn-CN"
        assert config["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"


class TestIsGeminiTtsModel:
    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3.1-flash-tts-preview",
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts",
        ],
    )
    def test_tts_models(self, model):
        assert is_gemini_tts_model(model)

    @pytest.mark.parametrize(
        "model",
        ["gemini-3.0-pro", "gemini-2.0-flash", "gemini-3.1-flash-image", "", None],
    )
    def test_non_tts_models(self, model):
        assert not is_gemini_tts_model(model)


# ---------------------------------------------------------------------------
# Shared helpers: audio encoding
# ---------------------------------------------------------------------------


class TestParseAudioMime:
    def test_full_l16_mime(self):
        base, rate = parse_audio_mime("audio/L16;codec=pcm;rate=24000")
        assert base == "audio/l16"
        assert rate == 24000

    def test_custom_rate(self):
        _, rate = parse_audio_mime("audio/L16;codec=pcm;rate=16000")
        assert rate == 16000

    def test_missing_rate_falls_back(self):
        base, rate = parse_audio_mime("audio/l16")
        assert base == "audio/l16"
        assert rate == DEFAULT_SAMPLE_RATE

    def test_invalid_rate_falls_back(self):
        _, rate = parse_audio_mime("audio/L16;rate=abc")
        assert rate == DEFAULT_SAMPLE_RATE

    def test_none_mime(self):
        base, rate = parse_audio_mime(None)
        assert base == "audio/l16"
        assert rate == DEFAULT_SAMPLE_RATE


class TestWavEncoding:
    def test_pcm_to_wav_structure(self):
        pcm = b"\x01\x02" * 100  # 200 bytes
        wav = pcm_to_wav(pcm, sample_rate=24000)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        # RIFF chunk size = data size + 36
        assert int.from_bytes(wav[4:8], "little") == len(pcm) + 36
        # data chunk size
        assert wav[36:40] == b"data"
        assert int.from_bytes(wav[40:44], "little") == len(pcm)
        assert wav[44:] == pcm

    def test_wav_header_sample_rate(self):
        header = wav_header(0, sample_rate=16000)
        assert int.from_bytes(header[24:28], "little") == 16000

    def test_streaming_header_unknown_length(self):
        header = wav_header(None)
        assert int.from_bytes(header[40:44], "little") == 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Request builder: speechConfig / responseModalities
# ---------------------------------------------------------------------------


class _ConcreteGeminiBuilder(GeminiRequestBuilderMixin):
    """Concrete implementation of the Gemini builder mixin for testing."""

    def _convert_conversation_to_gemini(self, conversation, context):
        contents = []
        for msg in conversation.messages:
            text = "".join(b.text for b in msg.content if hasattr(b, "text"))
            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": text}] if text else []})
        return contents, None


@pytest.fixture
def builder():
    return _ConcreteGeminiBuilder()


def _request(model: str, params: GenerationParams | None = None) -> InternalRequest:
    return InternalRequest(
        model=model,
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="Hello")])]
        ),
        params=params or GenerationParams(),
    )


def _build(builder, request: InternalRequest) -> dict:
    return builder._build_provider_request(
        request, BuildContext.from_request(request, base_url="https://test.example.com")
    )


class TestSpeechConfigBuilding:
    def test_explicit_speech_config_mapped(self, builder):
        """params.gemini.speech_config maps to generationConfig.speechConfig."""
        speech_config = {
            "multiSpeakerVoiceConfig": {
                "speakerVoiceConfigs": [
                    {
                        "speaker": "Joe",
                        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}},
                    }
                ]
            }
        }
        request = _request(
            "gemini-3.1-flash-tts-preview",
            GenerationParams(gemini=GeminiSpecificParams(speech_config=speech_config)),
        )
        body = _build(builder, request)
        assert body["generationConfig"]["speechConfig"] == speech_config

    def test_tts_model_auto_injects_audio_modality_and_voice(self, builder):
        """TTS models get responseModalities ["AUDIO"] and a default voice."""
        body = _build(builder, _request("gemini-3.1-flash-tts-preview"))
        config = body["generationConfig"]
        assert config["responseModalities"] == ["AUDIO"]
        assert config["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == (
            DEFAULT_GEMINI_VOICE
        )

    def test_explicit_response_modalities_not_overwritten(self, builder):
        request = _request(
            "gemini-3.1-flash-tts-preview",
            GenerationParams(gemini=GeminiSpecificParams(response_modalities=["AUDIO"])),
        )
        body = _build(builder, request)
        assert body["generationConfig"]["responseModalities"] == ["AUDIO"]

    def test_openai_audio_modalities_translated(self, builder):
        """OpenAI modalities:["audio"] + audio.voice are translated for Gemini."""
        request = _request(
            "gemini-3.1-flash-tts-preview",
            GenerationParams(
                openai=OpenAISpecificParams(
                    modalities=["text", "audio"], audio={"voice": "nova", "format": "wav"}
                )
            ),
        )
        body = _build(builder, request)
        config = body["generationConfig"]
        # TTS model only supports AUDIO output.
        assert config["responseModalities"] == ["AUDIO"]
        assert config["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == (
            "Sulafat"
        )

    def test_openai_audio_modalities_non_tts_model(self, builder):
        """Non-TTS models keep TEXT alongside AUDIO for audio chat requests."""
        request = _request(
            "gemini-3.0-pro",
            GenerationParams(
                openai=OpenAISpecificParams(modalities=["audio"], audio={"voice": "Puck"})
            ),
        )
        body = _build(builder, request)
        config = body["generationConfig"]
        assert config["responseModalities"] == ["TEXT", "AUDIO"]
        assert config["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Puck"

    def test_openai_voice_does_not_override_explicit_speech_config(self, builder):
        explicit = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Charon"}}}
        request = _request(
            "gemini-3.1-flash-tts-preview",
            GenerationParams(
                gemini=GeminiSpecificParams(speech_config=explicit),
                openai=OpenAISpecificParams(modalities=["audio"], audio={"voice": "nova"}),
            ),
        )
        body = _build(builder, request)
        assert body["generationConfig"]["speechConfig"] == explicit

    def test_plain_chat_model_untouched(self, builder):
        """Regular text models get no speechConfig or responseModalities."""
        body = _build(builder, _request("gemini-3.0-pro"))
        assert "generationConfig" not in body or "speechConfig" not in body.get(
            "generationConfig", {}
        )


# ---------------------------------------------------------------------------
# Response parser: audio inlineData -> AudioBlock
# ---------------------------------------------------------------------------


class _ConcreteGeminiParser(GeminiResponseParserMixin):
    pass


@pytest.fixture
def parser():
    return _ConcreteGeminiParser()


def _gemini_payload(parts: list[dict]) -> dict:
    return {
        "candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
    }


class TestAudioResponseParsing:
    def test_audio_inline_data_camel_case(self, parser):
        """Gemini REST emits camelCase mimeType."""
        payload = _gemini_payload(
            [{"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": "QUJD"}}]
        )
        result = parser.parse_provider_response(payload, model="gemini-3.1-flash-tts-preview")
        audio = [b for b in result.output if isinstance(b, AudioBlock)]
        assert len(audio) == 1
        assert audio[0].source.type == "base64"
        assert audio[0].source.data == "QUJD"
        assert audio[0].source.media_type == "audio/L16;codec=pcm;rate=24000"

    def test_audio_inline_data_snake_case(self, parser):
        payload = _gemini_payload([{"inlineData": {"mime_type": "audio/wav", "data": "QUJD"}}])
        result = parser.parse_provider_response(payload, model="gemini-3.1-flash-tts-preview")
        assert any(isinstance(b, AudioBlock) for b in result.output)

    def test_image_and_video_still_classified(self, parser):
        payload = _gemini_payload(
            [
                {"inlineData": {"mime_type": "image/png", "data": "aW1n"}},
                {"inlineData": {"mime_type": "video/mp4", "data": "dmlk"}},
                {"inlineData": {"mimeType": "audio/L16;rate=24000", "data": "YXVk"}},
            ]
        )
        result = parser.parse_provider_response(payload, model="m")
        assert any(isinstance(b, ImageBlock) for b in result.output)
        assert any(isinstance(b, VideoBlock) for b in result.output)
        assert any(isinstance(b, AudioBlock) for b in result.output)

    def test_get_audio_extraction(self, parser):
        payload = _gemini_payload(
            [{"inlineData": {"mimeType": "audio/L16;rate=24000", "data": "QUJD"}}]
        )
        result = parser.parse_provider_response(payload, model="m")
        audio = result.get_audio()
        assert audio is not None
        assert audio["data"] == "QUJD"


# ---------------------------------------------------------------------------
# Streaming transformer: audio chunks -> OpenAI audio deltas
# ---------------------------------------------------------------------------


class TestAudioStreaming:
    def test_audio_chunk_becomes_audio_delta(self):
        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/L16;codec=pcm;rate=24000",
                                    "data": base64.b64encode(b"pcm-data").decode(),
                                }
                            }
                        ]
                    }
                }
            ]
        }
        transformer = GeminiStreamingTransformer(model="gemini-3.1-flash-tts-preview")
        result_str = transformer.transform_chunk(chunk)
        assert result_str is not None
        result = orjson.loads(result_str.removeprefix("data: ").strip())
        audio = result["choices"][0]["delta"]["audio"]
        assert audio["data"] == base64.b64encode(b"pcm-data").decode()
        # Audio must NOT be degraded to markdown image text.
        assert "content" not in result["choices"][0]["delta"]

    def test_audio_accumulated_into_output_blocks(self):
        transformer = GeminiStreamingTransformer(model="m")
        b64 = base64.b64encode(b"pcm").decode()
        transformer.convert_chunk(
            {
                "candidates": [
                    {"content": {"parts": [{"inlineData": {"mimeType": "audio/l16", "data": b64}}]}}
                ]
            }
        )
        transformer.convert_chunk(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"inlineData": {"mimeType": "audio/l16", "data": b64}}]
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        )
        audio_blocks = [b for b in transformer._accumulated_output if isinstance(b, AudioBlock)]
        assert len(audio_blocks) == 1
        assert audio_blocks[0].source.data == b64 + b64
        assert audio_blocks[0].source.media_type == "audio/l16"

    def test_image_chunk_still_markdown(self):
        chunk = {
            "candidates": [
                {"content": {"parts": [{"inlineData": {"mime_type": "image/png", "data": "aW1n"}}]}}
            ]
        }
        transformer = GeminiStreamingTransformer(model="m")
        result_str = transformer.transform_chunk(chunk)
        assert result_str is not None
        result = orjson.loads(result_str.removeprefix("data: ").strip())
        assert "![image](data:image/png;base64,aW1n)" in result["choices"][0]["delta"]["content"]
