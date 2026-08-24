# src/llm_proxy/protocols/openai/schemas.py
"""Pydantic schemas for OpenAI protocol."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion request."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    stop: list[str] | str | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[int | str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    user: str | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    thinking: dict[str, Any] | bool | None = None
    audio: dict[str, Any] | None = None
    modalities: list[str] | None = None
    reasoning_effort: str | None = None
    prediction: dict[str, Any] | None = None
    web_search_options: dict[str, Any] | None = None
    service_tier: str | None = None
    verbosity: str | None = None
    store: bool | None = None
    metadata: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    safety_identifier: str | None = None


class EmbeddingRequestSchema(BaseModel):
    """Request model for OpenAI-compatible embeddings."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Model to use for embeddings")
    input: str | list[str] | list[int] | list[list[int]] = Field(
        ..., description="Text to embed or token ids (single array or array of arrays)"
    )
    encoding_format: str | None = Field("float", description="Output format: float or base64")
    dimensions: int | None = Field(None, description="Output dimensions (for shortening)")
    user: str | None = Field(None, description="User identifier")


class ImageGenerationRequestSchema(BaseModel):
    """Request model for OpenAI-compatible image generation."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., description="Text description of the desired image(s)")
    model: str | None = Field(
        None,
        description="Model to use for image generation (defaults to dall-e-2)",
    )
    n: int | None = Field(None, ge=1, le=10, description="Number of images to generate (1-10)")
    quality: Literal["standard", "hd", "low", "medium", "high", "auto"] | None = Field(
        None,
        description="Quality of the generated image (e.g., standard, hd, low, medium, high, auto)",
    )
    response_format: Literal["url", "b64_json"] | None = Field(
        None, description="Format for returned images (url or b64_json)"
    )
    size: (
        Literal[
            "256x256",
            "512x512",
            "1024x1024",
            "1024x1536",
            "1536x1024",
            "1792x1024",
            "1024x1792",
            "auto",
        ]
        | None
    ) = Field(
        None,
        description="Size of the generated images (e.g., 1024x1024, 1536x1024, 1024x1536, auto)",
    )
    style: Literal["vivid", "natural"] | None = Field(
        None, description="Style of the generated images (e.g., vivid, natural)"
    )
    user: str | None = Field(None, description="User identifier for abuse monitoring")
    background: Literal["transparent", "opaque", "auto"] | None = Field(
        None, description="Background transparency (e.g., transparent, opaque, auto)"
    )
    moderation: Literal["low", "auto"] | None = Field(
        None, description="Content moderation level (e.g., low, auto)"
    )
    output_format: Literal["png", "jpeg", "webp"] | None = Field(
        None, description="Output format (e.g., png, jpeg, webp)"
    )
    output_compression: int | None = Field(
        None, ge=0, le=100, description="Compression level 0-100%"
    )
    partial_images: int | None = Field(
        None, ge=0, le=3, description="Number of partial images for streaming"
    )
    stream: bool | None = Field(None, description="Enable streaming mode")


class ImageEditRequestSchema(BaseModel):
    """Request model for OpenAI-compatible image edits."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., description="Text description of the edit to apply")
    model: str | None = Field(None, description="Model to use for image editing")
    images: list[dict[str, Any]] = Field(
        ..., min_length=1, max_length=16, description="Images to edit (up to 16)"
    )
    mask: dict[str, Any] | None = Field(None, description="Optional mask image defining edit areas")
    background: Literal["transparent", "opaque", "auto"] | None = Field(
        None, description="Background type (e.g., transparent, opaque, auto)"
    )
    input_fidelity: Literal["low", "high"] | None = Field(None, description="Input fidelity level")
    moderation: Literal["low", "auto"] | None = Field(
        None, description="Content moderation level (e.g., low, auto)"
    )
    n: int | None = Field(None, ge=1, le=10, description="Number of images to generate (1-10)")
    output_compression: int | None = Field(
        None, ge=0, le=100, description="Compression level 0-100%"
    )
    output_format: Literal["png", "jpeg", "webp"] | None = Field(
        None, description="Output format (e.g., png, jpeg, webp)"
    )
    partial_images: int | None = Field(
        None, ge=0, le=3, description="Number of partial images for streaming"
    )
    quality: Literal["standard", "low", "medium", "high", "auto"] | None = Field(
        None, description="Image quality (e.g., standard, low, medium, high, auto)"
    )
    size: (
        Literal[
            "256x256",
            "512x512",
            "1024x1024",
            "1024x1536",
            "1536x1024",
            "auto",
        ]
        | None
    ) = Field(None, description="Image size (e.g., 1024x1024, auto)")
    response_format: Literal["url", "b64_json"] | None = Field(
        None, description="Format for returned images (url or b64_json)"
    )
    stream: bool | None = Field(None, description="Enable streaming mode")
    user: str | None = Field(None, description="User identifier for abuse monitoring")


class SpeechRequestSchema(BaseModel):
    """Request model for OpenAI-compatible audio speech generation."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="TTS model to use (e.g., tts-1, gpt-4o-mini-tts)")
    input: str = Field(..., description="Text to generate audio for (max 4096 chars)")
    voice: str = Field(..., description="Voice to use when generating audio")
    instructions: str | None = Field(None, description="Additional voice instructions")
    response_format: str | None = Field(
        "mp3", description="Audio format (mp3, opus, aac, flac, wav, pcm)"
    )
    speed: float | None = Field(1.0, description="Audio speed (0.25 to 4.0)")
    stream_format: str | None = Field(None, description="Stream format (sse or audio)")
    stream: bool = False
