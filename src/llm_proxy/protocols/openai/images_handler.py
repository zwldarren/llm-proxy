"""OpenAI images protocol endpoint."""

from typing import Any

from fastapi import Request
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openai.images_serializer import (  # noqa: F401
    ImageGenerationsSerializer,
)
from llm_proxy.protocols.openai.schemas import ImageGenerationRequestSchema


def _validate_image_schema[T: BaseModel](schema: type[T], data: Any) -> T:
    """Validate an image request and expose OpenAI-style HTTP 400 errors."""
    try:
        return schema.model_validate(data)
    except PydanticValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ())) or "request"
        raise ValidationError(
            message=f"Invalid value for {location}: {first.get('msg', 'validation failed')}",
            code="invalid_request_error",
            status_code=400,
        ) from None


async def parse_image_generation_request(fastapi_request: Request) -> ImageGenerationRequestSchema:
    """Parse and validate the JSON generations request."""
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
    return _validate_image_schema(ImageGenerationRequestSchema, data)


image_generations_protocol = ProtocolEndpoint(
    name="image_generations",
    paths=["/v1/images/generations"],
    request_model=None,
    parse_http_request=parse_image_generation_request,
    tags=["images"],
)


__all__ = ["image_generations_protocol", "parse_image_generation_request"]
