"""Tests for InternalImageRequest and InternalImageResponse."""

import pytest

from llm_proxy.models.image import (
    ImageData,
    ImageEditSource,
    ImageSize,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalImageResponse,
)
from llm_proxy.models.types import Usage


class TestImageSize:
    """Tests for ImageSize model."""

    def test_create_image_size(self):
        """Test creating ImageSize."""
        size = ImageSize(width=1024, height=1024)
        assert size.width == 1024
        assert size.height == 1024

    def test_parse_size_string(self):
        """Test parsing size string like '1024x1024'."""
        size = ImageSize.parse("1024x1024")
        assert size.width == 1024
        assert size.height == 1024

    def test_parse_size_string_different_dimensions(self):
        """Test parsing different dimensions."""
        size = ImageSize.parse("512x768")
        assert size.width == 512
        assert size.height == 768

    def test_parse_size_string_invalid_format(self):
        """Test parsing invalid size format."""
        with pytest.raises(ValueError, match="Invalid size format"):
            ImageSize.parse("invalid")

    def test_parse_size_string_invalid_dimensions(self):
        """Test parsing size with non-integer dimensions."""
        with pytest.raises(ValueError, match="Invalid size dimensions"):
            ImageSize.parse("abcxdef")


class TestInternalImageRequest:
    """Tests for InternalImageRequest model."""

    def test_create_minimal_request(self):
        """Test creating minimal request."""
        request = InternalImageRequest(
            model="dall-e-3",
            prompt="A sunset over mountains",
        )
        assert request.model == "dall-e-3"
        assert request.prompt == "A sunset over mountains"
        assert request.n == 1
        assert request.response_format == "url"

    def test_create_with_size(self):
        """Test creating request with size."""
        request = InternalImageRequest(
            model="dall-e-3",
            prompt="A cat",
            size=ImageSize(width=1024, height=1024),
        )
        assert request.size is not None
        assert request.size.width == 1024
        assert request.size.height == 1024

    def test_create_with_all_parameters(self):
        """Test creating request with all parameters."""
        request = InternalImageRequest(
            model="dall-e-3",
            prompt="A dog",
            n=2,
            size=ImageSize.parse("1024x1024"),
            quality="hd",
            style="vivid",
            response_format="b64_json",
            user="user-123",
            background="transparent",
            output_format="png",
            request_id="req-456",
            extra={"custom": "value"},
        )
        assert request.n == 2
        assert request.quality == "hd"
        assert request.style == "vivid"
        assert request.response_format == "b64_json"
        assert request.background == "transparent"
        assert request.output_format == "png"

    def test_create_with_gpt_image_parameters(self):
        """Test creating request with GPT image model parameters."""
        request = InternalImageRequest(
            model="gpt-image-1",
            prompt="A dog",
            moderation="auto",
            output_compression=80,
            output_format="webp",
            partial_images=2,
            stream=True,
        )
        assert request.moderation == "auto"
        assert request.output_compression == 80
        assert request.output_format == "webp"
        assert request.partial_images == 2
        assert request.stream is True


class TestImageData:
    """Tests for ImageData model."""

    def test_create_with_url(self):
        """Test creating with URL."""
        data = ImageData(url="https://example.com/image.png")
        assert data.url == "https://example.com/image.png"
        assert data.b64_json is None
        assert data.revised_prompt is None

    def test_create_with_b64_json(self):
        """Test creating with base64 data."""
        data = ImageData(b64_json="base64imagedata")
        assert data.b64_json == "base64imagedata"
        assert data.url is None

    def test_create_with_revised_prompt(self):
        """Test creating with revised prompt."""
        data = ImageData(
            url="https://example.com/image.png",
            revised_prompt="A more detailed sunset",
        )
        assert data.revised_prompt == "A more detailed sunset"


class TestInternalImageResponse:
    """Tests for InternalImageResponse model."""

    def test_create_minimal_response(self):
        """Test creating minimal response."""
        response = InternalImageResponse(
            created=1234567890,
            data=[ImageData(url="https://example.com/image.png")],
        )
        assert response.created == 1234567890
        assert len(response.data) == 1
        assert response.usage is None

    def test_create_with_usage(self):
        """Test creating response with usage."""
        response = InternalImageResponse(
            created=1234567890,
            data=[ImageData(url="https://example.com/image.png")],
            usage=Usage(input_tokens=10, output_tokens=0, total_tokens=10),
        )
        assert response.usage is not None
        assert response.usage.input_tokens == 10

    def test_create_with_metadata_fields(self):
        """Test creating response with background, output_format, quality, size."""
        response = InternalImageResponse(
            created=1234567890,
            data=[ImageData(b64_json="abc")],
            background="transparent",
            output_format="png",
            quality="high",
            size="1024x1024",
        )
        assert response.background == "transparent"
        assert response.output_format == "png"
        assert response.quality == "high"
        assert response.size == "1024x1024"


class TestInternalImageEditRequest:
    """Tests for InternalImageEditRequest model."""

    def test_create_minimal_request(self):
        """Test creating minimal edit request."""
        request = InternalImageEditRequest(
            model="gpt-image-1",
            prompt="Add a sun",
            images=[ImageEditSource(image_url="https://example.com/photo.png")],
        )
        assert request.model == "gpt-image-1"
        assert request.prompt == "Add a sun"
        assert len(request.images) == 1
        assert request.images[0].image_url == "https://example.com/photo.png"
        assert request.n == 1
        assert request.stream is False
        assert request.mask is None
        assert request.input_fidelity is None

    def test_create_with_file_id(self):
        """Test creating request with file_id source."""
        request = InternalImageEditRequest(
            model="gpt-image-1",
            prompt="Edit",
            images=[ImageEditSource(file_id="file-abc123")],
        )
        assert request.images[0].file_id == "file-abc123"
        assert request.images[0].image_url is None

    def test_create_with_multiple_images(self):
        """Test creating request with multiple image sources."""
        request = InternalImageEditRequest(
            model="gpt-image-1",
            prompt="Combine these",
            images=[
                ImageEditSource(image_url="https://example.com/a.png"),
                ImageEditSource(image_url="https://example.com/b.png"),
            ],
        )
        assert len(request.images) == 2

    def test_create_with_size(self):
        """Test creating request with size."""
        request = InternalImageEditRequest(
            model="gpt-image-1",
            prompt="Edit",
            images=[ImageEditSource(image_url="https://example.com/photo.png")],
            size=ImageSize(width=512, height=512),
        )
        assert request.size is not None
        assert request.size.width == 512
        assert request.size.height == 512

    def test_create_with_all_params(self):
        """Test creating request with all parameters."""
        request = InternalImageEditRequest(
            model="gpt-image-1",
            prompt="Add sun",
            images=[ImageEditSource(file_id="file-1", image_url="https://example.com/photo.png")],
            mask=ImageEditSource(file_id="mask-file", image_url="https://example.com/mask.png"),
            background="transparent",
            input_fidelity="high",
            moderation="auto",
            n=2,
            output_compression=80,
            output_format="png",
            partial_images=2,
            quality="high",
            size=ImageSize.parse("1024x1024"),
            user="user-1",
            stream=True,
            request_id="req-1",
            extra={"custom": True},
        )
        assert request.n == 2
        assert request.mask is not None
        assert request.mask.file_id == "mask-file"
        assert request.mask.image_url == "https://example.com/mask.png"
        assert request.background == "transparent"
        assert request.input_fidelity == "high"
        assert request.moderation == "auto"
        assert request.output_compression == 80
        assert request.output_format == "png"
        assert request.partial_images == 2
        assert request.quality == "high"
        assert request.stream is True
        assert request.user == "user-1"
        assert request.request_id == "req-1"
        assert request.extra["custom"] is True
