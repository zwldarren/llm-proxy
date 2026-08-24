"""OpenAI Audio protocol serializer.

Converts between OpenAI Audio wire format and InternalSpeechRequest/
InternalTranscriptionRequest/InternalTranslationRequest models.
"""

from typing import Any, cast

from llm_proxy.models import (
    InternalSpeechRequest,
    InternalSpeechResponse,
    InternalTranscriptionRequest,
    InternalTranscriptionResponse,
    InternalTranslationRequest,
    InternalTranslationResponse,
)
from llm_proxy.protocols.registry import register_protocol_serializer
from llm_proxy.protocols.serializer_base import ProtocolSerializer


class BaseOpenAIAudioSerializer(ProtocolSerializer):
    """Shared logic for OpenAI Audio protocol serializers."""

    @property
    def protocol_name(self) -> str:
        return "audio"

    def parse_request(self, data: dict[str, Any]) -> Any:
        """Dispatch to the appropriate parse method based on request fields.

        Speech requests have 'input' and 'voice' fields.
        Transcription/translation requests may include 'file' and 'filename'.
        """
        if "input" in data and "voice" in data:
            return self.parse_speech_request(data)

        # For transcription/translation, check if file bytes are present
        # (from wrapper's model_dump during fallback reparse)
        file = data.get("file")
        filename = data.get("filename", "audio.mp3")
        if file is not None and isinstance(file, (bytes, bytearray)):
            # Determine which type based on context clues
            if "timestamp_granularities" in data or "language" in data or "include" in data:
                return self.parse_transcription_request(data, file, filename)
            return self.parse_translation_request(data, file, filename)

        msg = (
            "parse_request requires file bytes for transcription/translation. "
            "Use parse_transcription_request(data, file, filename) instead."
        )
        raise NotImplementedError(msg)

    def format_response(self, response: object, context=None) -> dict[str, Any]:
        """Format audio response by dispatching to the appropriate method.

        Determines the response type and delegates to the specific
        format method (format_speech_response, format_transcription_response,
        or format_translation_response).
        """
        from llm_proxy.models import (
            InternalSpeechResponse,
            InternalTranscriptionResponse,
            InternalTranslationResponse,
        )

        if isinstance(response, InternalSpeechResponse):
            return self.format_speech_response(response)
        if isinstance(response, InternalTranscriptionResponse):
            format_type = getattr(response, "_response_format", "json")
            result = self.format_transcription_response(response, format_type)
            if isinstance(result, str):
                return {"text": result}
            return result
        if isinstance(response, InternalTranslationResponse):
            format_type = getattr(response, "_response_format", "json")
            result = self.format_translation_response(response, format_type)
            if isinstance(result, str):
                return {"text": result}
            return result
        if isinstance(response, dict):
            return cast(dict[str, Any], response)
        return {"error": "Invalid response type"}

    def parse_speech_request(self, data: dict[str, Any]) -> InternalSpeechRequest:
        """Parse speech request from wire format dict."""
        return InternalSpeechRequest(
            model=data["model"],
            input=data["input"],
            voice=data["voice"],
            instructions=data.get("instructions"),
            response_format=data.get("response_format", "mp3"),
            speed=data.get("speed", 1.0),
            stream_format=data.get("stream_format"),
            stream=data.get("stream", False),
            extra={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "model",
                    "input",
                    "voice",
                    "instructions",
                    "response_format",
                    "speed",
                    "stream_format",
                    "stream",
                }
            },
        )

    def format_speech_response(self, response: InternalSpeechResponse) -> dict[str, Any]:
        """Format speech response to wire format dict."""
        return {
            "content": response.content,
            "content_type": response.content_type,
        }

    @staticmethod
    def _build_audio_usage(usage) -> dict[str, Any]:
        """Build the OpenAI Audio usage dict from a Usage object.

        Preserves token counts, input token details, and duration-based billing
        metadata (``audio_duration_seconds`` / ``type``) for STT responses.
        """
        result: dict[str, Any] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
        if usage.audio_duration_seconds is not None:
            result["audio_duration_seconds"] = usage.audio_duration_seconds
            result["type"] = "duration"
        else:
            result["type"] = "tokens"

        if usage.prompt_tokens_details is not None:
            ptd = usage.prompt_tokens_details
            details: dict[str, Any] = {}
            for key in (
                "audio_tokens",
                "cached_tokens",
                "text_tokens",
                "cache_write_tokens",
                "image_tokens",
                "video_tokens",
            ):
                value = getattr(ptd, key, None)
                if value is not None:
                    details[key] = value
            if details:
                result["input_token_details"] = details

        if usage.completion_tokens_details is not None:
            ctd = usage.completion_tokens_details
            details = {}
            for key in (
                "accepted_prediction_tokens",
                "audio_tokens",
                "reasoning_tokens",
                "rejected_prediction_tokens",
                "image_tokens",
            ):
                value = getattr(ctd, key, None)
                if value is not None:
                    details[key] = value
            if details:
                result["output_token_details"] = details

        return result

    def parse_transcription_request(
        self,
        data: dict[str, Any],
        file: bytes,
        filename: str,
    ) -> InternalTranscriptionRequest:
        """Parse transcription request from wire format dict and file."""
        return InternalTranscriptionRequest(
            model=data["model"],
            file=file,
            filename=filename,
            language=data.get("language"),
            prompt=data.get("prompt"),
            response_format=data.get("response_format", "json"),
            temperature=data.get("temperature", 0.0),
            timestamp_granularities=data.get("timestamp_granularities"),
            include=data.get("include"),
            stream=data.get("stream", False),
            extra={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "model",
                    "file",
                    "language",
                    "prompt",
                    "response_format",
                    "temperature",
                    "timestamp_granularities",
                    "include",
                    "stream",
                }
            },
        )

    def format_transcription_response(
        self,
        response: InternalTranscriptionResponse,
        format_type: str = "json",
    ) -> dict[str, Any] | str:
        """Format transcription response to wire format.

        Supports json, verbose_json, text, srt, vtt, and diarized_json formats.
        """
        if format_type == "text":
            return response.text

        result: dict[str, Any] = {"text": response.text}

        if format_type == "verbose_json":
            if response.task is not None:
                result["task"] = response.task
            if response.language is not None:
                result["language"] = response.language
            if response.duration is not None:
                result["duration"] = response.duration
            if response.segments is not None:
                result["segments"] = response.segments
            if response.words is not None:
                result["words"] = response.words

        if format_type == "diarized_json":
            if response.duration is not None:
                result["duration"] = response.duration
            if response.segments is not None:
                result["segments"] = response.segments
            result["task"] = response.task or "transcribe"

        if response.logprobs is not None:
            result["logprobs"] = response.logprobs

        if response.usage is not None:
            result["usage"] = self._build_audio_usage(response.usage)

        return result

    def parse_translation_request(
        self,
        data: dict[str, Any],
        file: bytes,
        filename: str,
    ) -> InternalTranslationRequest:
        """Parse translation request from wire format dict and file."""
        return InternalTranslationRequest(
            model=data["model"],
            file=file,
            filename=filename,
            prompt=data.get("prompt"),
            response_format=data.get("response_format", "json"),
            temperature=data.get("temperature", 0.0),
            extra={
                k: v
                for k, v in data.items()
                if k not in {"model", "file", "prompt", "response_format", "temperature"}
            },
        )

    def format_translation_response(
        self,
        response: InternalTranslationResponse,
        format_type: str = "json",
    ) -> dict[str, Any] | str:
        """Format translation response to wire format.

        Supports json, verbose_json, text, srt, and vtt formats.
        """
        if format_type == "text":
            return response.text

        if format_type in ("srt", "vtt"):
            return response.text

        result: dict[str, Any] = {"text": response.text}

        if format_type == "verbose_json":
            result["language"] = response.language
            if response.duration is not None:
                result["duration"] = response.duration
            if response.segments is not None:
                result["segments"] = response.segments

        if response.usage is not None:
            result["usage"] = self._build_audio_usage(response.usage)

        return result


@register_protocol_serializer("speech")
@register_protocol_serializer("transcription")
@register_protocol_serializer("translation")
class OpenAIAudioSerializer(BaseOpenAIAudioSerializer):
    """Protocol serializer for OpenAI Audio endpoints."""


__all__ = ["OpenAIAudioSerializer"]
