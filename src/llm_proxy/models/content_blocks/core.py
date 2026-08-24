"""Universal ContentBlock types understood by all providers."""

from dataclasses import dataclass, field
from typing import Any

from llm_proxy.models.types import AudioSource, DocumentSource, ImageSource, VideoSource


@dataclass
class ContentBlock:
    """Base class for all content blocks."""


@dataclass
class TextBlock(ContentBlock):
    """Text content block with optional raw citation dicts."""

    text: str
    citations: list[dict[str, Any]] | None = None
    cache_control: Any | None = None


@dataclass
class ImageBlock(ContentBlock):
    """Image content block with source and optional detail level."""

    source: ImageSource
    detail: str | None = None
    cache_control: Any | None = None


@dataclass
class AudioBlock(ContentBlock):
    """Audio content block with source."""

    source: AudioSource
    cache_control: Any | None = None


@dataclass
class VideoBlock(ContentBlock):
    """Video content block with source."""

    source: VideoSource
    cache_control: Any | None = None


@dataclass
class DocumentBlock(ContentBlock):
    """Document content block with source and optional title, context, and citations."""

    source: DocumentSource
    title: str | None = None
    context: str | None = None
    citations: dict[str, Any] | None = None
    cache_control: Any | None = None


@dataclass
class ToolUseBlock(ContentBlock):
    """Tool use request block."""

    id: str
    name: str
    input: dict[str, Any]
    cache_control: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultBlock(ContentBlock):
    """Tool result block with content from tool execution."""

    tool_use_id: str
    content: str | list[ContentBlock]
    is_error: bool = False
    name: str | None = None
    cache_control: Any | None = None
