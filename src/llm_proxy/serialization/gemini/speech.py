"""Shared helpers for Gemini text-to-speech (TTS) support.

Gemini TTS models (e.g. ``gemini-3.1-flash-tts-preview``) generate audio via
the regular ``generateContent`` endpoint with
``generationConfig.responseModalities = ["AUDIO"]`` and a ``speechConfig``
object. The response audio is returned as base64-encoded raw PCM
(``audio/L16;codec=pcm;rate=24000``) in ``inlineData`` parts.

This module is the single source of truth for:
- voice name resolution (OpenAI voice names -> Gemini prebuilt voices),
- ``speechConfig`` construction (single- and multi-speaker),
- parsing the ``audio/L16;...;rate=NNNN`` MIME type,
- wrapping raw PCM in a WAV container (Gemini never returns WAV directly).

Both the chat path (``request_builder`` translating OpenAI ``modalities:
["audio"]`` requests) and the speech path (``GeminiAdapter.speech`` for the
``/v1/audio/speech`` endpoint) build on these helpers so the two paths can
never drift apart.

References:
- https://ai.google.dev/gemini-api/docs/speech-generation
- https://ai.google.dev/api/generate-content#SpeechConfig
"""

import struct
from contextlib import suppress
from typing import Any

# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

# The 30 prebuilt voices supported by Gemini TTS models.
# https://ai.google.dev/gemini-api/docs/speech-generation#voice-options
GEMINI_TTS_VOICES: tuple[str, ...] = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
)

DEFAULT_GEMINI_VOICE = "Kore"

# Mapping from OpenAI voice names to Gemini prebuilt voices with a similar
# character. Gemini voice names pass through untouched (case-insensitive);
# anything unknown falls back to DEFAULT_GEMINI_VOICE.
_OPENAI_TO_GEMINI_VOICE: dict[str, str] = {
    "alloy": "Kore",  # neutral, firm
    "ash": "Charon",  # informative
    "ballad": "Leda",  # youthful
    "coral": "Aoede",  # breezy
    "echo": "Orus",  # firm
    "fable": "Puck",  # upbeat
    "nova": "Sulafat",  # warm
    "onyx": "Algenib",  # gravelly, deep
    "sage": "Despina",  # smooth
    "shimmer": "Vindemiatrix",  # gentle
    "verse": "Schedar",  # even
}


def resolve_voice(voice: str | None) -> str:
    """Resolve a client-provided voice name to a Gemini prebuilt voice.

    Gemini voice names are matched case-insensitively and returned in their
    canonical casing; known OpenAI voice names are mapped to a Gemini
    equivalent; anything else falls back to ``DEFAULT_GEMINI_VOICE``.
    """
    if not voice:
        return DEFAULT_GEMINI_VOICE
    lowered = voice.strip().lower()
    for gemini_voice in GEMINI_TTS_VOICES:
        if gemini_voice.lower() == lowered:
            return gemini_voice
    return _OPENAI_TO_GEMINI_VOICE.get(lowered, DEFAULT_GEMINI_VOICE)


def build_speech_config(
    voice: str | None = None,
    *,
    language_code: str | None = None,
) -> dict[str, Any]:
    """Build a single-speaker Gemini ``speechConfig`` object.

    Follows the official schema:
    ``{"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": ...}}}``.
    """
    config: dict[str, Any] = {
        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": resolve_voice(voice)}}
    }
    if language_code:
        config["languageCode"] = language_code
    return config


def is_gemini_tts_model(model: str | None) -> bool:
    """Check if the model name indicates a Gemini TTS model.

    Matches names like ``gemini-2.5-flash-preview-tts``,
    ``gemini-2.5-pro-preview-tts`` and ``gemini-3.1-flash-tts-preview``.
    """
    if not model:
        return False
    return "-tts" in model.lower()


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 24000


def parse_audio_mime(mime_type: str | None) -> tuple[str, int]:
    """Parse a Gemini audio MIME type into (base type, sample rate).

    Gemini returns types like ``audio/L16;codec=pcm;rate=24000``.
    Returns (``audio/l16``, 24000); unknown/missing rates fall back to
    ``DEFAULT_SAMPLE_RATE``.
    """
    if not mime_type:
        return "audio/l16", DEFAULT_SAMPLE_RATE
    parts = [p.strip() for p in mime_type.split(";")]
    base = parts[0].lower()
    rate = DEFAULT_SAMPLE_RATE
    for param in parts[1:]:
        key, _, value = param.partition("=")
        if key.strip().lower() == "rate":
            with suppress(ValueError):
                rate = int(value.strip())
    return base, rate


def wav_header(
    data_size: int | None,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Build a 44-byte RIFF/WAV header for 16-bit PCM data.

    ``data_size=None`` produces a streaming header whose size fields are
    ``0xFFFFFFFF`` (unknown length), which common players accept for
    chunked/streamed WAV playback.
    """
    # Streaming: unknown length. Both size fields must carry the "unknown"
    # marker 0xFFFFFFFF — computing RIFF size + 36 here would wrap around to
    # 35 (an invalid header some players reject).
    riff_size = 0xFFFFFFFF if data_size is None else (data_size + 36) & 0xFFFFFFFF
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, 8 * sample_width
        )
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF if data_size is None else data_size)
    )


def pcm_to_wav(
    pcm: bytes,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw PCM bytes in a complete WAV container."""
    return (
        wav_header(len(pcm), sample_rate=sample_rate, channels=channels, sample_width=sample_width)
        + pcm
    )


__all__ = [
    "DEFAULT_GEMINI_VOICE",
    "DEFAULT_SAMPLE_RATE",
    "GEMINI_TTS_VOICES",
    "build_speech_config",
    "is_gemini_tts_model",
    "parse_audio_mime",
    "pcm_to_wav",
    "resolve_voice",
    "wav_header",
]
