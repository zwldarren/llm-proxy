"""Audio capability mixin — speech, transcription, and translation.

Owns the OpenAI-shaped wire format for the audio endpoints: raw body
builders (``_build_speech_raw`` / ``_build_transcription_raw`` /
``_build_translation_raw`` plus the multipart data builders), endpoint URL
resolution, and the HTTP execution methods (``speech`` / ``stream_speech`` /
``transcription`` / ``stream_transcription`` / ``translation``).

Adapters include this mixin when they speak the OpenAI audio wire format
(OpenAI, Gemini, the openai-compatible family). Providers with a different
audio dialect override the execution methods (e.g. Gemini's
``_build_speech_raw`` / ``speech``).
"""

from collections.abc import AsyncIterator
from typing import Any, cast

from llm_proxy.models import (
    InternalSpeechRequest,
    InternalSpeechResponse,
    InternalTranscriptionRequest,
    InternalTranscriptionResponse,
    InternalTranslationRequest,
    InternalTranslationResponse,
)
from llm_proxy.providers.base import _extract_rate_limit_headers
from llm_proxy.providers.capabilities.host import AudioSelf


class AudioCapabilityMixin:
    """Mixin for provider adapters that support speech, transcription, and translation.

    Endpoint constants are overridden by subclasses (e.g. OpenAI's
    ``SPEECH_ENDPOINT = "/audio/speech"``).
    """

    SPEECH_ENDPOINT: str = ""
    TRANSCRIPTION_ENDPOINT: str = ""
    TRANSLATION_ENDPOINT: str = ""

    def _build_speech_raw(self: AudioSelf, request: InternalSpeechRequest) -> dict[str, Any]:
        """Build raw speech body without extra merge or field policy.

        The dispatch's _finalize_body handles merging request.extra and
        applying the configured field policy.
        """
        body: dict[str, Any] = {
            "model": request.model,
            "input": request.input,
            "voice": request.voice,
            "response_format": request.response_format,
            "speed": request.speed,
        }
        if request.instructions is not None:
            body["instructions"] = request.instructions
        if request.stream_format is not None:
            body["stream_format"] = request.stream_format
        if request.stream:
            body["stream"] = request.stream
        return body

    def _build_transcription_raw(
        self: AudioSelf, request: Any, stream: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build raw transcription data + files."""
        return self._build_transcription_data(request, stream=stream)

    def _build_translation_raw(
        self: AudioSelf, request: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build raw translation data + files."""
        return self._build_translation_data(request)

    def _speech_url(self: AudioSelf, request: InternalSpeechRequest | None = None) -> str:
        if self.SPEECH_ENDPOINT:
            model = request.model if request else None
            return self._resolve_endpoint_url("speech", self.SPEECH_ENDPOINT, model=model)
        raise NotImplementedError("Subclasses must define SPEECH_ENDPOINT or override _speech_url")

    def _transcription_url(
        self: AudioSelf, request: InternalTranscriptionRequest | None = None
    ) -> str:
        if self.TRANSCRIPTION_ENDPOINT:
            model = request.model if request else None
            return self._resolve_endpoint_url(
                "transcription", self.TRANSCRIPTION_ENDPOINT, model=model
            )
        raise NotImplementedError(
            "Subclasses must define TRANSCRIPTION_ENDPOINT or override _transcription_url"
        )

    def _translation_url(self: AudioSelf, request: InternalTranslationRequest | None = None) -> str:
        if self.TRANSLATION_ENDPOINT:
            model = request.model if request else None
            return self._resolve_endpoint_url("translation", self.TRANSLATION_ENDPOINT, model=model)
        raise NotImplementedError(
            "Subclasses must define TRANSLATION_ENDPOINT or override _translation_url"
        )

    def _build_transcription_data(
        self: AudioSelf, request: InternalTranscriptionRequest, stream: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        data: dict[str, Any] = {
            "model": request.model,
            "response_format": request.response_format,
        }
        if request.language is not None:
            data["language"] = request.language
        if request.prompt is not None:
            data["prompt"] = request.prompt
        if request.temperature != 0.0:
            data["temperature"] = str(request.temperature)
        if request.timestamp_granularities is not None:
            for g in request.timestamp_granularities:
                data.setdefault("timestamp_granularities[]", []).append(g)
        if request.include is not None:
            for i in request.include:
                data.setdefault("include[]", []).append(i)
        if stream:
            data["stream"] = "true"
            # Ensure usage tracking in the SSE stream. OpenAI's streaming
            # transcription only includes usage when `include[]=usage` is
            # present (gpt-4o-transcribe family). Harmless for providers
            # that ignore unknown include values.
            includes: list[str] = data.setdefault("include[]", [])
            if "usage" not in includes:
                includes.append("usage")
        files = {"file": (request.filename, request.file)}
        return data, files

    def _build_translation_data(
        self: AudioSelf, request: InternalTranslationRequest
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        data: dict[str, Any] = {
            "model": request.model,
            "response_format": request.response_format,
        }
        if request.prompt is not None:
            data["prompt"] = request.prompt
        if request.temperature != 0.0:
            data["temperature"] = str(request.temperature)
        files = {"file": (request.filename, request.file)}
        return data, files

    def _parse_usage(self: AudioSelf, raw_usage: Any) -> Any:
        """Parse usage dict from provider response into a Usage object.

        Handles token-based usage (chat, gpt-4o-transcribe) and duration-based
        usage (whisper transcription/translation: {"type": "duration", "seconds": N}).
        """
        from llm_proxy.models.types import CompletionTokensDetails, PromptTokensDetails, Usage

        if not raw_usage or not isinstance(raw_usage, dict):
            return None

        def _prompt_details_from_input(input_details: dict[str, Any]) -> PromptTokensDetails | None:
            audio_in = input_details.get("audio_tokens")
            cached = input_details.get("cached_tokens")
            text_tokens = input_details.get("text_tokens")
            if audio_in is not None or cached is not None or text_tokens is not None:
                return PromptTokensDetails(
                    audio_tokens=audio_in,
                    cached_tokens=cached,
                    text_tokens=text_tokens,
                )
            return None

        # Duration-based usage (whisper transcription/translation)
        if raw_usage.get("type") == "duration":
            seconds = raw_usage.get("seconds")
            usage = Usage(
                audio_duration_seconds=seconds if isinstance(seconds, int | float) else None
            )
            # Duration-based responses may still carry input token details
            # (e.g. gpt-4o-transcribe with audio/text token breakdown).
            input_details = raw_usage.get("input_token_details") or raw_usage.get(
                "prompt_tokens_details"
            )
            if isinstance(input_details, dict):
                usage.prompt_tokens_details = _prompt_details_from_input(input_details)
            return usage

        # Token-based usage; audio token details may be nested under
        # input_token_details (gpt-4o-transcribe) or prompt_tokens_details (chat)
        prompt_details = None
        input_details = raw_usage.get("input_token_details")
        if input_details is None:
            input_details = raw_usage.get("prompt_tokens_details")
        if isinstance(input_details, dict):
            prompt_details = _prompt_details_from_input(input_details)

        completion_details = None
        output_details = raw_usage.get("completion_tokens_details")
        if isinstance(output_details, dict):
            audio_out = output_details.get("audio_tokens")
            if audio_out is not None:
                completion_details = CompletionTokensDetails(audio_tokens=audio_out)

        return Usage(
            input_tokens=raw_usage.get("input_tokens", 0),
            output_tokens=raw_usage.get("output_tokens", 0),
            total_tokens=raw_usage.get("total_tokens"),
            prompt_tokens_details=prompt_details,
            completion_tokens_details=completion_details,
        )

    def _parse_transcription_response(
        self: AudioSelf, data: dict[str, Any], format_type: str
    ) -> InternalTranscriptionResponse:
        response = InternalTranscriptionResponse(
            text=data.get("text", ""),
            task=data.get("task"),
            language=data.get("language"),
            duration=data.get("duration"),
            segments=data.get("segments"),
            words=data.get("words"),
            logprobs=data.get("logprobs"),
            usage=self._parse_usage(data.get("usage")),
        )
        response._response_format = format_type
        return response

    def _parse_translation_response(
        self: AudioSelf, data: dict[str, Any], format_type: str
    ) -> InternalTranslationResponse:
        response = InternalTranslationResponse(
            text=data.get("text", ""),
            language=data.get("language", "english"),
            duration=data.get("duration"),
            segments=data.get("segments"),
            usage=self._parse_usage(data.get("usage")),
        )
        response._response_format = format_type
        return response

    async def speech(
        self: AudioSelf, request: InternalSpeechRequest, **_kwargs: Any
    ) -> InternalSpeechResponse:
        url = self._speech_url(request)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="speech")
        if outbound.json_body is None:
            raise ValueError("Expected json_body for speech request, got None")
        body = outbound.json_body

        async def _make_request():
            client = await self._get_client()
            response = await client.post(url, headers=headers, json=body)
            await self._check_response_status(response)
            return response

        response = await self._with_retry(_make_request)
        return InternalSpeechResponse(
            content=response.content,
            content_type=response.headers.get("content-type", f"audio/{request.response_format}"),
            request_id=request.request_id,
            provider_info={
                "_rate_limit_headers": _extract_rate_limit_headers(
                    getattr(response, "headers", None)
                )
            },
        )

    async def stream_speech(
        self: AudioSelf, request: InternalSpeechRequest, **_kwargs: Any
    ) -> AsyncIterator[bytes]:
        url = self._speech_url(request)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="speech")
        if outbound.json_body is None:
            raise ValueError("Expected json_body for speech request, got None")
        body = outbound.json_body
        body["stream"] = True
        stream_timeout = self._get_stream_timeout()

        async def _generator():
            client = await self._get_client()
            async with self._streaming_post(
                client,
                url,
                headers=headers,
                json=body,
                timeout=stream_timeout,
            ) as response:
                await self._check_response_status(response)
                async for chunk in response.iter_raw():
                    yield chunk

        return cast(AsyncIterator[bytes], self._with_retry_generator(_generator))

    async def _post_audio_form(
        self: AudioSelf,
        url: str,
        request: Any,
        *,
        request_type: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Shared multipart POST for transcription/translation.

        Returns the decoded response body (parsed JSON, or ``{"text": ...}``
        for plain-text responses) plus the extracted rate-limit headers.
        """
        headers = self._build_headers()
        headers.pop("Content-Type", None)

        outbound = self._build_outbound_body(request, request_type=request_type)
        data, files = outbound.form_data, outbound.files

        async def _make_request():
            client = await self._get_client()
            response = await client.post(url, headers=headers, data=data, files=files)
            await self._check_response_status(response)
            return response

        response = await self._with_retry(_make_request)
        if "application/json" in response.headers.get("content-type", ""):
            response_data = response.json()
        else:
            response_data = {"text": response.text}
        return response_data, _extract_rate_limit_headers(getattr(response, "headers", None))

    async def transcription(
        self: AudioSelf, request: InternalTranscriptionRequest, **_kwargs: Any
    ) -> InternalTranscriptionResponse:
        response_data, rate_limit_headers = await self._post_audio_form(
            self._transcription_url(request), request, request_type="transcription"
        )
        result = self._parse_transcription_response(response_data, request.response_format)
        result.provider_info["_rate_limit_headers"] = rate_limit_headers
        return result

    async def stream_transcription(
        self: AudioSelf, request: InternalTranscriptionRequest, **_kwargs: Any
    ) -> AsyncIterator[str]:
        url = self._transcription_url(request)
        headers = self._build_headers()
        headers.pop("Content-Type", None)

        outbound = self._build_outbound_body(request, request_type="transcription", stream=True)
        if outbound.form_data is None:
            raise ValueError("Expected form_data for transcription request, got None")
        data, files = outbound.form_data, outbound.files
        # stream flag is already set by _build_outbound_body with stream=True
        stream_timeout = self._get_stream_timeout()

        async def _generator():
            client = await self._get_client()
            async with self._streaming_post(
                client,
                url,
                headers=headers,
                data=data,
                files=files,
                timeout=stream_timeout,
            ) as response:
                await self._check_response_status(response)
                async for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    yield f"{line_str}\n"

        return self._with_retry_generator(_generator)

    async def translation(
        self: AudioSelf, request: InternalTranslationRequest, **_kwargs: Any
    ) -> InternalTranslationResponse:
        response_data, rate_limit_headers = await self._post_audio_form(
            self._translation_url(request), request, request_type="translation"
        )
        result = self._parse_translation_response(response_data, request.response_format)
        result.provider_info["_rate_limit_headers"] = rate_limit_headers
        return result


__all__ = ["AudioCapabilityMixin"]
