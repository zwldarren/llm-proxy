"""OpenAI audio translation protocol endpoint."""

from typing import Any

from fastapi import Request
from starlette.datastructures import UploadFile

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openai.audio_serializer import (  # noqa: F401
    OpenAIAudioSerializer,
)


class _TranslationRequestWrapper:
    """Wrapper for multipart translation requests.

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


async def parse_translation_request(fastapi_request: Request) -> Any:
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
        "prompt": _get_str("prompt") or None,
        "response_format": _get_str("response_format", "json"),
        "temperature": temperature,
    }

    # Capture unknown form fields flat into data so the serializer
    # extracts them into InternalTranslationRequest.extra.
    known_fields = {"model", "file", "prompt", "response_format", "temperature"}
    for key, val in form.multi_items():
        if key not in known_fields and isinstance(val, str):
            data[key] = val

    return _TranslationRequestWrapper(
        data={k: v for k, v in data.items() if v is not None},
        file=file_bytes,
        filename=filename,
    )


translation_protocol = ProtocolEndpoint(
    name="translation",
    paths=["/v1/audio/translations"],
    request_model=None,
    parse_http_request=parse_translation_request,
    tags=["audio"],
)


__all__ = ["translation_protocol"]
