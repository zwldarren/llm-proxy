"""OpenAI audio transcription protocol endpoint."""

from typing import Any

from fastapi import Request
from starlette.datastructures import UploadFile

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openai.audio_serializer import (  # noqa: F401
    OpenAIAudioSerializer,
)


class _TranscriptionRequestWrapper:
    """Wrapper for multipart transcription requests.

    Quacks like a Pydantic BaseModel so UnifiedProcessor can handle it.
    Stores file bytes so fallback reparse can reconstruct the request.
    """

    def __init__(self, data: dict[str, Any], file: bytes, filename: str) -> None:
        self._data = data
        self.file = file
        self.filename = filename
        self.model = data.get("model", "")

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:  # noqa
        # Include file/filename in raw_request_data so fallback reparse works
        return {
            **self._data,
            "file": self.file,
            "filename": self.filename,
        }


async def parse_transcription_request(fastapi_request: Request) -> Any:
    """Parse multipart form data into a request wrapper."""
    form = await fastapi_request.form()
    file_field = form.get("file")
    if isinstance(file_field, UploadFile):
        file_bytes = await file_field.read()
        filename = file_field.filename or "audio.mp3"
    else:
        file_bytes = b""
        filename = "audio.mp3"

    def _get_str(name: str, default: str = "") -> str:
        val = form.get(name, default)
        return val if isinstance(val, str) else default

    model = _get_str("model")
    if not model:
        raise ValidationError(message="Missing required field 'model'")

    if not file_bytes:
        raise ValidationError(message="Missing required field 'file'")

    # Parse temperature safely
    temperature = 0.0
    try:
        temperature = float(_get_str("temperature", "0.0"))
    except ValueError:
        raise ValidationError(message="Invalid temperature value. Must be a number.") from None

    data: dict[str, Any] = {
        "model": model,
        "language": _get_str("language") or None,
        "prompt": _get_str("prompt") or None,
        "response_format": _get_str("response_format", "json"),
        "temperature": temperature,
        "stream": _get_str("stream", "false").lower() == "true",
    }

    # Handle multi-value fields
    timestamp_granularities: list[str] = []
    include: list[str] = []
    for key, val in form.multi_items():
        if key == "timestamp_granularities[]" and isinstance(val, str):
            timestamp_granularities.append(val)
        elif key == "include[]" and isinstance(val, str):
            include.append(val)

    if timestamp_granularities:
        data["timestamp_granularities"] = timestamp_granularities
    if include:
        data["include"] = include

    # Capture unknown form fields flat into data so the serializer
    # extracts them into InternalTranscriptionRequest.extra.
    known_fields = {
        "model",
        "file",
        "language",
        "prompt",
        "response_format",
        "temperature",
        "stream",
        "timestamp_granularities[]",
        "include[]",
    }
    for key, val in form.multi_items():
        if key not in known_fields and key not in data and isinstance(val, str):
            data[key] = val

    return _TranscriptionRequestWrapper(
        data={k: v for k, v in data.items() if v is not None},
        file=file_bytes,
        filename=filename,
    )


transcription_protocol = ProtocolEndpoint(
    name="transcription",
    paths=["/v1/audio/transcriptions"],
    request_model=None,
    parse_http_request=parse_transcription_request,
    tags=["audio"],
)


__all__ = ["transcription_protocol"]
