"""Audio processing strategies: speech, transcription, and translation."""

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from llm_proxy.protocols.openai.audio_serializer import BaseOpenAIAudioSerializer
    from llm_proxy.protocols.serializer_base import ProtocolSerializer

import orjson
from fastapi import Response

from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.strategies.base import (
    ProcessingStrategy,
    StreamingResponseMarker,
)
from llm_proxy.core.request_type import RequestType
from llm_proxy.models import (
    InternalSpeechResponse,
    InternalTranscriptionResponse,
    InternalTranslationResponse,
)


class SpeechStrategy(ProcessingStrategy):
    """Strategy for text-to-speech requests."""

    request_type = RequestType.SPEECH
    trace_name = "llm-proxy-speech-request"

    async def execute(
        self, unified_request: Any, adapter: BaseAdapter, context: RequestContext
    ) -> Any:
        if unified_request.stream:
            return StreamingResponseMarker(unified_request, adapter)
        return await adapter.speech(unified_request)

    async def format_response(
        self, response: Any, serializer: ProtocolSerializer, protocol_name: str
    ) -> Response:
        if isinstance(response, StreamingResponseMarker):
            raise RuntimeError("StreamingResponseMarker should not reach format_response")
        speech_response = cast(InternalSpeechResponse, response)
        return Response(
            content=speech_response.content,
            media_type=speech_response.content_type,
        )


class TranscriptionStrategy(ProcessingStrategy):
    """Strategy for audio transcription requests."""

    request_type = RequestType.TRANSCRIPTION
    trace_name = "llm-proxy-transcription-request"

    async def execute(
        self, unified_request: Any, adapter: BaseAdapter, context: RequestContext
    ) -> Any:
        if unified_request.stream:
            return StreamingResponseMarker(unified_request, adapter)
        return await adapter.transcription(unified_request)

    async def format_response(
        self, response: Any, serializer: ProtocolSerializer, protocol_name: str
    ) -> Response:
        if isinstance(response, StreamingResponseMarker):
            raise RuntimeError("StreamingResponseMarker should not reach format_response")
        transcription_response = cast(InternalTranscriptionResponse, response)
        format_type = getattr(transcription_response, "_response_format", "json")

        if format_type in ("text", "srt", "vtt"):
            return Response(
                content=transcription_response.text.encode("utf-8"),
                media_type="text/plain",
            )

        # Use serializer's transcription-specific method
        audio_serializer = cast("BaseOpenAIAudioSerializer", serializer)
        result = audio_serializer.format_transcription_response(transcription_response, format_type)
        if isinstance(result, str):
            return Response(content=orjson.dumps({"text": result}), media_type="application/json")
        return Response(content=orjson.dumps(result), media_type="application/json")


class TranslationStrategy(ProcessingStrategy):
    """Strategy for audio translation requests."""

    request_type = RequestType.TRANSLATION
    trace_name = "llm-proxy-translation-request"

    async def execute(
        self, unified_request: Any, adapter: BaseAdapter, context: RequestContext
    ) -> Any:
        return await adapter.translation(unified_request)

    async def format_response(
        self, response: Any, serializer: ProtocolSerializer, protocol_name: str
    ) -> Response:
        translation_response = cast(InternalTranslationResponse, response)
        format_type = getattr(translation_response, "_response_format", "json")

        if format_type in ("text", "srt", "vtt"):
            return Response(
                content=translation_response.text.encode("utf-8"),
                media_type="text/plain",
            )

        # Use serializer's translation-specific method
        audio_serializer = cast("BaseOpenAIAudioSerializer", serializer)
        result = audio_serializer.format_translation_response(translation_response, format_type)
        if isinstance(result, str):
            return Response(content=orjson.dumps({"text": result}), media_type="application/json")
        return Response(content=orjson.dumps(result), media_type="application/json")
