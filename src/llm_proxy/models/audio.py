"""Unified audio request and response models."""

from dataclasses import dataclass, field
from typing import Any

from llm_proxy.models.internal import RequestMetadata
from llm_proxy.models.params import GenerationParams
from llm_proxy.models.tools import ToolDefinition
from llm_proxy.models.types import Usage


@dataclass
class InternalSpeechRequest:
    """Unified text-to-speech request.

    Attributes:
        request_type: The type of request - always "speech".
        model: The TTS model to use (e.g., "tts-1", "gpt-4o-mini-tts").
        input: The text to generate audio for. Maximum 4096 characters.
        voice: The voice to use when generating audio.
        instructions: Additional instructions for voice control.
        response_format: The audio format ("mp3", "opus", "aac", "flac", "wav", "pcm").
        speed: The speed of generated audio (0.25 to 4.0).
        stream_format: Streaming format ("sse" or "audio").
        stream: Whether to stream the response.
        request_id: Optional request identifier for tracking.
        extra: Additional provider-specific parameters.
    """

    model: str
    input: str
    voice: str
    instructions: str | None = None
    response_format: str = "mp3"
    speed: float = 1.0
    stream_format: str | None = None
    stream: bool = False
    request_type: str = field(default="speech", init=False)
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    tools: list[ToolDefinition] | None = None
    params: GenerationParams = field(default_factory=GenerationParams)
    metadata: RequestMetadata = field(default_factory=RequestMetadata)
    _raw_protocol_data: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class InternalSpeechResponse:
    """Unified text-to-speech response.

    Attributes:
        content: The binary audio content.
        content_type: MIME type of the audio content.
        request_id: Optional request identifier for correlation.
        provider_info: Additional provider-specific information.
    """

    content: bytes
    content_type: str = "audio/mpeg"
    request_id: str | None = None
    provider_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalTranscriptionRequest:
    """Unified audio transcription request.

    Attributes:
        request_type: The type of request - always "transcription".
        model: The model to use for transcription (e.g., "whisper-1", "gpt-4o-transcribe").
        file: The raw audio file bytes.
        filename: The original filename of the audio file.
        language: The language of the input audio in ISO-639-1 format.
        prompt: An optional text to guide the model's style.
        response_format: The format of the transcript output.
        temperature: Sampling temperature (0 to 1).
        timestamp_granularities: Timestamp granularity options.
        include: Additional data to include in the response.
        stream: Whether to stream the response.
        request_id: Optional request identifier for tracking.
        extra: Additional provider-specific parameters.
    """

    model: str
    file: bytes
    filename: str
    language: str | None = None
    prompt: str | None = None
    response_format: str = "json"
    temperature: float = 0.0
    timestamp_granularities: list[str] | None = None
    include: list[str] | None = None
    stream: bool = False
    request_type: str = field(default="transcription", init=False)
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    tools: list[ToolDefinition] | None = None
    params: GenerationParams = field(default_factory=GenerationParams)
    metadata: RequestMetadata = field(default_factory=RequestMetadata)
    _raw_protocol_data: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class InternalTranscriptionResponse:
    """Unified audio transcription response.

    Attributes:
        text: The transcribed text.
        task: The task type (e.g., "transcribe").
        language: The detected language of the input audio.
        duration: The duration of the input audio in seconds.
        segments: Segments of the transcribed text with timestamps.
        words: Extracted words with timestamps.
        logprobs: Log probabilities of the tokens.
        usage: Token or duration usage statistics.
        request_id: Optional request identifier for correlation.
        provider_info: Additional provider-specific information.
    """

    text: str
    task: str | None = None
    language: str | None = None
    duration: float | None = None
    segments: list[dict[str, Any]] | None = None
    words: list[dict[str, Any]] | None = None
    logprobs: list[dict[str, Any]] | None = None
    usage: Usage | None = None
    request_id: str | None = None
    provider_info: dict[str, Any] = field(default_factory=dict)
    _response_format: str = field(default="json", repr=False)


@dataclass
class InternalTranslationRequest:
    """Unified audio translation request.

    Attributes:
        request_type: The type of request - always "translation".
        model: The model to use for translation (e.g., "whisper-1").
        file: The raw audio file bytes.
        filename: The original filename of the audio file.
        prompt: An optional text to guide the model's style.
        response_format: The format of the translation output.
        temperature: Sampling temperature (0 to 1).
        request_id: Optional request identifier for tracking.
        extra: Additional provider-specific parameters.
    """

    model: str
    file: bytes
    filename: str
    prompt: str | None = None
    response_format: str = "json"
    temperature: float = 0.0
    request_type: str = field(default="translation", init=False)
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    tools: list[ToolDefinition] | None = None
    params: GenerationParams = field(default_factory=GenerationParams)
    metadata: RequestMetadata = field(default_factory=RequestMetadata)
    stream: bool = False
    _raw_protocol_data: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class InternalTranslationResponse:
    """Unified audio translation response.

    Attributes:
        text: The translated text.
        language: The language of the output (always "english").
        duration: The duration of the input audio in seconds.
        segments: Segments of the translated text with timestamps.
        usage: Token or duration usage statistics.
        request_id: Optional request identifier for correlation.
        provider_info: Additional provider-specific information.
    """

    text: str
    language: str = "english"
    duration: float | None = None
    segments: list[dict[str, Any]] | None = None
    usage: Usage | None = None
    request_id: str | None = None
    provider_info: dict[str, Any] = field(default_factory=dict)
    _response_format: str = field(default="json", repr=False)
