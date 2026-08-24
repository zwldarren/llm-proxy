"""Unified image generation request and response models."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_proxy.models.conversation import ConversationContext
from llm_proxy.models.internal import RequestMetadata
from llm_proxy.models.params import GenerationParams
from llm_proxy.models.tools import ToolDefinition
from llm_proxy.models.types import Usage

if TYPE_CHECKING:
    pass


@dataclass
class ImageSize:
    """Image dimensions.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
    """

    width: int
    height: int

    @classmethod
    def parse(cls, size_str: str) -> ImageSize:
        """Parse size string like '1024x1024'.

        Args:
            size_str: Size string in 'WIDTHxHEIGHT' format.

        Returns:
            ImageSize instance.

        Raises:
            ValueError: If size string is invalid.
        """
        parts = size_str.split("x")
        if len(parts) != 2:
            msg = f"Invalid size format: {size_str}. Expected 'WIDTHxHEIGHT'"
            raise ValueError(msg)
        try:
            return cls(width=int(parts[0]), height=int(parts[1]))
        except ValueError as err:
            msg = f"Invalid size dimensions: {size_str}. Width and height must be integers"
            raise ValueError(msg) from err


@dataclass
class ImageData:
    """Single image generation result.

    Attributes:
        url: URL of the generated image (if response_format is 'url').
        b64_json: Base64-encoded JSON data (if response_format is 'b64_json').
        revised_prompt: The revised prompt used by the model (for DALL-E 3).
    """

    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


@dataclass
class InternalImageRequest:
    """Unified image generation request.

    This is the protocol-agnostic request format for image generation.
    All protocol handlers parse their specific request formats into InternalImageRequest.

    Attributes:
        request_type: The type of request - always "image".
            Used by UnifiedProcessor to route to the image generation handler.
        model: The model to use for image generation (e.g., "dall-e-3", "stable-diffusion-xl").
        prompt: The text description of the image to generate.
        n: Number of images to generate.
        size: Image dimensions.
        quality: Image quality ("standard" or "hd" for DALL-E).
        style: Image style ("vivid" or "natural" for DALL-E 3).
        response_format: Format for returned images ("url" or "b64_json").
        user: A unique identifier representing the end-user.
        background: Background type for images ("transparent" or "opaque").
        output_format: Output format ("png", "webp", "jpg").
        request_id: Optional request identifier for tracking.
        extra: Additional provider-specific parameters.
    """

    request_type: str = field(default="image_generation", init=False)
    model: str
    prompt: str
    stream: bool = False
    tools: list[ToolDefinition] | None = None
    conversation: ConversationContext | None = None
    params: GenerationParams = field(default_factory=GenerationParams)
    metadata: RequestMetadata = field(default_factory=RequestMetadata)
    n: int = 1
    size: ImageSize | None = None
    size_auto: bool = False
    quality: str | None = None
    style: str | None = None
    response_format: str = "url"
    user: str | None = None
    background: str | None = None
    moderation: str | None = None
    output_compression: int | None = None
    output_format: str | None = None
    partial_images: int | None = None
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalImageResponse:
    """Unified image generation and edit response.

    This is the protocol-agnostic response format for image generation and editing.
    All provider adapters convert their responses to InternalImageResponse.

    Attributes:
        created: Unix timestamp of when the images were created.
        data: List of generated image data.
        model: The model used for image generation.
        usage: Optional token usage information.
        request_id: Optional request identifier for correlation.
        provider_info: Additional provider-specific information.
    """

    created: int
    data: list[ImageData]
    model: str = ""
    background: str | None = None
    output_format: str | None = None
    quality: str | None = None
    size: str | None = None
    usage: Usage | None = None
    request_id: str | None = None
    provider_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageEditSource:
    """Image source reference for edit requests.

    Attributes:
        file_id: The File API ID of an uploaded image.
        image_url: A fully qualified URL or base64-encoded data URL.
    """

    file_id: str | None = None
    image_url: str | None = None
    file: bytes | bytearray | None = None
    filename: str | None = None
    content_type: str | None = None


@dataclass
class InternalImageEditRequest:
    """Unified image edit request.

    Attributes:
        request_type: The type of request - always "image_edit".
        model: The model to use for image editing.
        prompt: The text description of the edit to apply.
        images: Array of image references to edit (up to 16 for GPT image models).
        background: Background type ("transparent", "opaque", or "auto").
        moderation: Moderation setting ("low" or "auto").
        n: Number of images to generate.
        output_compression: Output compression level (integer).
        output_format: Output format ("png", "jpeg", or "webp").
        partial_images: Number of partial images to return (0-3).
        quality: Image quality ("low", "medium", "high", or "auto").
        size: Image dimensions.
        user: A unique identifier representing the end-user.
        extra: Additional provider-specific parameters.
    """

    request_type: str = field(default="image_edit", init=False)
    model: str
    prompt: str
    images: list[ImageEditSource] = field(default_factory=list)
    mask: ImageEditSource | None = None
    stream: bool = False
    tools: list[ToolDefinition] | None = None
    conversation: ConversationContext | None = None
    params: GenerationParams = field(default_factory=GenerationParams)
    metadata: RequestMetadata = field(default_factory=RequestMetadata)
    background: str | None = None
    input_fidelity: str | None = None
    moderation: str | None = None
    n: int = 1
    response_format: str | None = None
    output_compression: int | None = None
    output_format: str | None = None
    partial_images: int | None = None
    quality: str | None = None
    size: ImageSize | None = None
    size_auto: bool = False
    user: str | None = None
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
