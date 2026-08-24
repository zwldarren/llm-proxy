"""Tests for OpenAI Images protocol serializer.

TDD: Tests written before implementation. Covers the creation of
ProtocolSerializer instances for "image_generations" and "image_edits"
protocol names, which currently do not exist and cause a crash when
_apply_overrides_and_reparse() calls get_protocol_serializer().
"""

from llm_proxy.models.image import (
    ImageData,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalImageResponse,
)
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.protocols.serializer_base import ProtocolSerializer


class TestImageGenerationsProtocolSerializer:
    """ProtocolSerializer for image_generations handles parse and format."""

    def test_serializer_is_registered(self):
        """get_protocol_serializer('image_generations') returns a valid instance."""
        serializer = get_protocol_serializer("image_generations")
        assert serializer is not None
        assert isinstance(serializer, ProtocolSerializer)

    def test_protocol_name(self):
        """protocol_name property returns a non-empty string."""
        serializer = get_protocol_serializer("image_generations")
        assert serializer.protocol_name
        assert isinstance(serializer.protocol_name, str)

    def test_parse_request_basic(self):
        """parse_request creates InternalImageRequest from wire data."""
        serializer = get_protocol_serializer("image_generations")
        data = {
            "model": "dall-e-3",
            "prompt": "A cute cat",
            "n": 1,
            "size": "1024x1024",
            "quality": "hd",
            "style": "vivid",
            "response_format": "url",
        }
        result = serializer.parse_request(data)
        assert isinstance(result, InternalImageRequest)
        assert result.model == "dall-e-3"
        assert result.prompt == "A cute cat"
        assert result.n == 1
        assert result.size is not None
        assert result.size.width == 1024
        assert result.size.height == 1024
        assert result.quality == "hd"
        assert result.style == "vivid"
        assert result.response_format == "url"

    def test_parse_request_minimal(self):
        """parse_request handles minimal data with only prompt."""
        serializer = get_protocol_serializer("image_generations")
        data = {"prompt": "A dog"}
        result = serializer.parse_request(data)
        assert isinstance(result, InternalImageRequest)
        assert result.prompt == "A dog"
        assert result.model == ""
        assert result.n == 1
        assert result.size is None
        assert result.response_format == "url"

    def test_parse_request_extra_fields(self):
        """Extra fields are passed through via extra dict."""
        serializer = get_protocol_serializer("image_generations")
        data = {
            "prompt": "Test",
            "model": "gpt-image-1",
            "background": "transparent",
            "output_format": "png",
            "moderation": "low",
        }
        result = serializer.parse_request(data)
        assert result.background == "transparent"
        assert result.output_format == "png"
        assert result.moderation == "low"

    def test_format_response_basic(self):
        """format_response converts InternalImageResponse to wire dict."""
        serializer = get_protocol_serializer("image_generations")
        response = InternalImageResponse(
            created=1234567890,
            data=[
                ImageData(url="https://example.com/img.png", revised_prompt="A revised prompt"),
            ],
        )
        result = serializer.format_response(response)
        assert result["created"] == 1234567890
        assert len(result["data"]) == 1
        assert result["data"][0]["url"] == "https://example.com/img.png"
        assert result["data"][0]["revised_prompt"] == "A revised prompt"

    def test_format_response_with_optional_fields(self):
        """format_response includes optional fields when present."""
        serializer = get_protocol_serializer("image_generations")
        response = InternalImageResponse(
            created=1234567890,
            data=[ImageData(b64_json="base64data")],
            background="transparent",
            output_format="png",
            quality="hd",
            size="1024x1024",
        )
        result = serializer.format_response(response)
        assert result["background"] == "transparent"
        assert result["output_format"] == "png"
        assert result["quality"] == "hd"
        assert result["size"] == "1024x1024"

    def test_format_response_skips_none_fields(self):
        """format_response omits optional fields that are None."""
        serializer = get_protocol_serializer("image_generations")
        response = InternalImageResponse(
            created=1234567890,
            data=[ImageData(url="https://example.com/img.png")],
        )
        result = serializer.format_response(response)
        assert "background" not in result
        assert "output_format" not in result
        assert "quality" not in result
        assert "size" not in result


class TestImageEditsProtocolSerializer:
    """ProtocolSerializer for image_edits handles parse and format."""

    def test_serializer_is_registered(self):
        """get_protocol_serializer('image_edits') returns a valid instance."""
        serializer = get_protocol_serializer("image_edits")
        assert serializer is not None
        assert isinstance(serializer, ProtocolSerializer)

    def test_parse_request_basic(self):
        """parse_request creates InternalImageEditRequest from wire data."""
        serializer = get_protocol_serializer("image_edits")
        data = {
            "model": "dall-e-2",
            "prompt": "Edit this image",
            "images": [{"file_id": "file_abc123"}],
            "n": 2,
            "size": "512x512",
        }
        result = serializer.parse_request(data)
        assert isinstance(result, InternalImageEditRequest)
        assert result.model == "dall-e-2"
        assert result.prompt == "Edit this image"
        assert result.n == 2
        assert result.size is not None
        assert result.size.width == 512
        assert result.size.height == 512

    def test_parse_request_uses_images_array_only(self):
        """The JSON body uses the ``images`` array; a singular ``image``
        object is not part of the OpenAI contract and is not merged."""
        serializer = get_protocol_serializer("image_edits")
        data = {
            "model": "gpt-image-1",
            "prompt": "Make it blue",
            "image": {"image_url": "https://example.com/base.png"},
            "images": [{"image_url": "https://example.com/second.png"}],
        }
        result = serializer.parse_request(data)
        assert isinstance(result, InternalImageEditRequest)
        assert [img.image_url for img in result.images] == ["https://example.com/second.png"]

    def test_parse_request_with_mask(self):
        """parse_request handles mask field."""
        serializer = get_protocol_serializer("image_edits")
        data = {
            "prompt": "Edit",
            "mask": {"file_id": "mask_file_123"},
        }
        result = serializer.parse_request(data)
        assert result.mask is not None
        assert result.mask.file_id == "mask_file_123"

    def test_format_response(self):
        """format_response for edits returns same format as generations."""
        serializer = get_protocol_serializer("image_edits")
        response = InternalImageResponse(
            created=1234567890,
            data=[ImageData(url="https://example.com/edit.png")],
        )
        result = serializer.format_response(response)
        assert result["created"] == 1234567890
        assert result["data"][0]["url"] == "https://example.com/edit.png"

    def test_format_response_with_usage(self):
        """format_response includes usage when present."""
        from llm_proxy.models.types import Usage

        serializer = get_protocol_serializer("image_edits")
        response = InternalImageResponse(
            created=1234567890,
            data=[ImageData(url="https://example.com/img.png")],
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        )
        result = serializer.format_response(response)
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30
