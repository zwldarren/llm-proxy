"""OpenAI images edits protocol endpoint."""

from typing import Any

from fastapi import Request
from starlette.datastructures import UploadFile

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openai.images_handler import _validate_image_schema
from llm_proxy.protocols.openai.images_serializer import (  # noqa: F401
    ImageEditsSerializer,
)
from llm_proxy.protocols.openai.schemas import ImageEditRequestSchema


async def _read_image_upload(upload: UploadFile) -> dict[str, Any]:
    """Read an uploaded image into the request data retained by the pipeline."""
    content = await upload.read()
    if not content:
        raise ValidationError(
            message=f"Uploaded image '{upload.filename or 'image'}' is empty",
            code="invalid_request_error",
            status_code=400,
        )
    return {
        "file": content,
        "filename": upload.filename or "image.png",
        "content_type": upload.content_type or "image/png",
    }


async def parse_image_edit_request(
    fastapi_request: Request,
) -> ImageEditRequestSchema:
    """Parse JSON edits and the official multipart ``image[]`` form."""
    content_type = fastapi_request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type != "multipart/form-data":
        try:
            data = await fastapi_request.json()
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                message="Invalid JSON request body",
                code="invalid_request_error",
                status_code=400,
            ) from exc
        if not isinstance(data, dict):
            raise ValidationError(
                message="Request body must be a JSON object",
                code="invalid_request_error",
                status_code=400,
            )
        return _validate_image_schema(ImageEditRequestSchema, data)

    form = await fastapi_request.form()
    data: dict[str, Any] = {}
    images: list[dict[str, Any]] = []
    for key, value in form.multi_items():
        if key in {"image", "image[]"}:
            # Only file uploads are meaningful for these fields; the JSON
            # body form uses the ``images`` array instead.
            if isinstance(value, UploadFile):
                images.append(await _read_image_upload(value))
            continue
        if key == "mask" and isinstance(value, UploadFile):
            data[key] = await _read_image_upload(value)
            continue
        data[key] = value

    data["images"] = images
    return _validate_image_schema(ImageEditRequestSchema, data)


image_edits_protocol = ProtocolEndpoint(
    name="image_edits",
    paths=["/v1/images/edits"],
    request_model=None,
    parse_http_request=parse_image_edit_request,
    tags=["images"],
)


__all__ = ["image_edits_protocol", "parse_image_edit_request"]
