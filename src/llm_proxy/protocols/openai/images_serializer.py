"""OpenAI Images protocol serializer.

Converts between OpenAI Images wire format and InternalImageRequest/
InternalImageResponse/InternalImageEditRequest models.

Registered as both "image_generations" and "image_edits" to resolve the
serializer lookup crash that occurred when two endpoints shared one serializer.
"""

from typing import Any, cast

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models.image import (
    ImageEditSource,
    ImageSize,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalImageResponse,
)
from llm_proxy.protocols.registry import register_protocol_serializer
from llm_proxy.protocols.serializer_base import ProtocolSerializer


class BaseOpenAIImagesSerializer(ProtocolSerializer):
    """Shared logic for OpenAI Images protocol serializers."""

    _IMAGE_SIZES = frozenset(
        {
            "256x256",
            "512x512",
            "1024x1024",
            "1024x1536",
            "1536x1024",
            "1792x1024",
            "1024x1792",
            "auto",
        }
    )

    # Image edits accepts a narrower set per the official SDK: no 1792x1024
    # or 1024x1792 (those are dall-e-3 generation sizes).
    _EDIT_SIZES = frozenset(
        {
            "256x256",
            "512x512",
            "1024x1024",
            "1024x1536",
            "1536x1024",
            "auto",
        }
    )

    @staticmethod
    def _validate_choice(value: Any, field: str, choices: frozenset[str]) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or value not in choices:
            allowed = ", ".join(sorted(choices))
            raise ValidationError(
                message=f"Invalid {field} '{value}'. Expected one of: {allowed}",
                code="invalid_request_error",
                status_code=400,
            )
        return value

    @staticmethod
    def _parse_int_range(
        value: Any,
        *,
        field: str,
        default: int | None,
        minimum: int,
        maximum: int,
    ) -> int | None:
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                message=f"{field} must be an integer between {minimum} and {maximum}",
                code="invalid_request_error",
                status_code=400,
            ) from exc
        if not minimum <= parsed <= maximum:
            raise ValidationError(
                message=f"{field} must be between {minimum} and {maximum}",
                code="invalid_request_error",
                status_code=400,
            )
        return parsed

    def _parse_size(
        self, size_str: str | None, *, allowed: frozenset[str] | None = None
    ) -> ImageSize | None:
        if size_str is None:
            return None
        if (
            not isinstance(size_str, str)
            or not size_str
            or size_str not in (allowed or self._IMAGE_SIZES)
        ):
            raise ValidationError(
                message=f"Invalid size '{size_str}'",
                code="invalid_request_error",
                status_code=400,
            )
        if size_str == "auto":
            return None
        try:
            return ImageSize.parse(size_str)
        except ValueError as exc:
            raise ValidationError(
                message=str(exc), code="invalid_request_error", status_code=400
            ) from None

    def _format_image_data(self, response: InternalImageResponse) -> dict[str, Any]:
        data = [
            {
                k: v
                for k, v in [
                    ("url", img.url),
                    ("b64_json", img.b64_json),
                    ("revised_prompt", img.revised_prompt),
                ]
                if v is not None
            }
            for img in response.data
        ]
        result: dict[str, Any] = {
            "created": response.created,
            "data": data,
        }
        if response.background:
            result["background"] = response.background
        if response.output_format:
            result["output_format"] = response.output_format
        if response.quality:
            result["quality"] = response.quality
        if response.size:
            result["size"] = response.size
        if response.usage:
            details = response.usage.prompt_tokens_details
            if isinstance(details, dict):
                text_tokens = details.get("text_tokens")
                image_tokens = details.get("image_tokens")
            else:
                text_tokens = getattr(details, "text_tokens", None)
                image_tokens = getattr(details, "image_tokens", None)
            input_tokens_details = {
                "text_tokens": text_tokens or 0,
                "image_tokens": image_tokens or 0,
            }
            result["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
                "input_tokens_details": input_tokens_details,
            }
        return result

    def _parse_images(self, raw_images: list[dict] | None) -> list[ImageEditSource]:
        if not raw_images:
            return []
        if not all(isinstance(img, dict) for img in raw_images):
            raise ValidationError(
                message="images must contain image objects",
                code="invalid_request_error",
                status_code=400,
            )
        return [
            ImageEditSource(
                file_id=img.get("file_id"),
                image_url=img.get("image_url"),
                file=img.get("file"),
                filename=img.get("filename"),
                content_type=img.get("content_type"),
            )
            for img in raw_images
        ]

    def _parse_mask(self, raw_mask: dict | None) -> ImageEditSource | None:
        if not raw_mask:
            return None
        if not isinstance(raw_mask, dict):
            raise ValidationError(
                message="mask must be an object",
                code="invalid_request_error",
                status_code=400,
            )
        return ImageEditSource(
            file_id=raw_mask.get("file_id"),
            image_url=raw_mask.get("image_url"),
            file=raw_mask.get("file"),
            filename=raw_mask.get("filename"),
            content_type=raw_mask.get("content_type"),
        )


@register_protocol_serializer("image_generations")
class ImageGenerationsSerializer(BaseOpenAIImagesSerializer):
    """Protocol serializer for /v1/images/generations."""

    @property
    def protocol_name(self) -> str:
        return "image_generations"

    def parse_request(self, data: dict[str, Any]) -> InternalImageRequest:
        size_raw = data.get("size")
        size = self._parse_size(size_raw)
        quality = self._validate_choice(
            data.get("quality"),
            "quality",
            frozenset({"standard", "hd", "low", "medium", "high", "auto"}),
        )
        style = self._validate_choice(data.get("style"), "style", frozenset({"vivid", "natural"}))
        response_format = self._validate_choice(
            data.get("response_format"), "response_format", frozenset({"url", "b64_json"})
        )
        background = self._validate_choice(
            data.get("background"), "background", frozenset({"transparent", "opaque", "auto"})
        )
        moderation = self._validate_choice(
            data.get("moderation"), "moderation", frozenset({"low", "auto"})
        )
        output_format = self._validate_choice(
            data.get("output_format"), "output_format", frozenset({"png", "jpeg", "webp"})
        )
        return InternalImageRequest(
            model=data.get("model", ""),
            prompt=data.get("prompt", ""),
            stream=data.get("stream", False),
            n=self._parse_int_range(data.get("n"), field="n", default=1, minimum=1, maximum=10)
            or 1,
            size=size,
            size_auto=size_raw == "auto",
            quality=quality,
            style=style,
            response_format=response_format or "url",
            user=data.get("user"),
            background=background,
            moderation=moderation,
            output_compression=self._parse_int_range(
                data.get("output_compression"),
                field="output_compression",
                default=None,
                minimum=0,
                maximum=100,
            ),
            output_format=output_format,
            partial_images=self._parse_int_range(
                data.get("partial_images"),
                field="partial_images",
                default=None,
                minimum=0,
                maximum=3,
            ),
        )

    def format_response(self, response: object, context=None) -> dict[str, Any]:
        if isinstance(response, InternalImageResponse):
            return self._format_image_data(response)
        if isinstance(response, dict):
            return cast("dict[str, Any]", response)
        return {"error": "Invalid response type"}


@register_protocol_serializer("image_edits")
class ImageEditsSerializer(BaseOpenAIImagesSerializer):
    """Protocol serializer for /v1/images/edits."""

    @property
    def protocol_name(self) -> str:
        return "image_edits"

    def parse_request(self, data: dict[str, Any]) -> InternalImageEditRequest:
        size = self._parse_size(data.get("size"), allowed=self._EDIT_SIZES)
        size_raw = data.get("size")

        n = self._parse_int_range(data.get("n"), field="n", default=1, minimum=1, maximum=10) or 1
        quality = self._validate_choice(
            data.get("quality"), "quality", frozenset({"standard", "low", "medium", "high", "auto"})
        )
        background = self._validate_choice(
            data.get("background"), "background", frozenset({"transparent", "opaque", "auto"})
        )
        input_fidelity = self._validate_choice(
            data.get("input_fidelity"), "input_fidelity", frozenset({"low", "high"})
        )
        moderation = self._validate_choice(
            data.get("moderation"), "moderation", frozenset({"low", "auto"})
        )
        output_format = self._validate_choice(
            data.get("output_format"), "output_format", frozenset({"png", "jpeg", "webp"})
        )
        response_format = self._validate_choice(
            data.get("response_format"), "response_format", frozenset({"url", "b64_json"})
        )
        partial_images = self._parse_int_range(
            data.get("partial_images"),
            field="partial_images",
            default=None,
            minimum=0,
            maximum=3,
        )

        # The JSON body uses the ``images`` array form; multipart uploads use
        # the ``image``/``image[]`` file fields (handled by the endpoint).
        images = self._parse_images(data.get("images"))

        return InternalImageEditRequest(
            model=data.get("model", ""),
            prompt=data.get("prompt", ""),
            images=images,
            mask=self._parse_mask(data.get("mask")),
            stream=data.get("stream", False),
            background=background,
            input_fidelity=input_fidelity,
            moderation=moderation,
            n=n,
            response_format=response_format,
            output_compression=self._parse_int_range(
                data.get("output_compression"),
                field="output_compression",
                default=None,
                minimum=0,
                maximum=100,
            ),
            output_format=output_format,
            partial_images=partial_images,
            quality=quality,
            size=size,
            size_auto=size_raw == "auto",
            user=data.get("user"),
            extra={},
        )

    def format_response(self, response: object, context=None) -> dict[str, Any]:
        if isinstance(response, InternalImageResponse):
            return self._format_image_data(response)
        if isinstance(response, dict):
            return cast("dict[str, Any]", response)
        return {"error": "Invalid response type"}


__all__ = [
    "ImageGenerationsSerializer",
    "ImageEditsSerializer",
]
