"""Universal ContentBlock types understood by all providers."""

from dataclasses import dataclass, field
from typing import Any, Literal

from llm_proxy.models.types import AudioSource, DocumentSource, ImageSource, VideoSource

#: Providers whose thought signatures may be replayed to the same provider.
#: ``openai`` marks a genuine Responses ``encrypted_content`` blob (which is
#: not an Anthropic verification payload and must not be replayed as one).
#: Declared here, next to the universal blocks that carry it, so every block or
#: lane mentioning a signature origin reads one alias (extended re-exports it).
SignatureOrigin = Literal["gemini", "openai"]


@dataclass
class ContentBlock:
    """Base class for all content blocks."""


@dataclass
class TextBlock(ContentBlock):
    """Text content block with optional raw citation dicts.

    ``signature`` carries a provider signature issued *on a text part* — Gemini
    3/4 may put a ``thoughtSignature`` on the last part of a response, including
    a text one, and the documented advice is to return every signature exactly
    where it arrived. ``signature_origin`` records which provider issued it, so
    only Gemini-issued signatures are replayed to Gemini.
    """

    text: str
    citations: list[dict[str, Any]] | None = None
    cache_control: Any | None = None
    signature: str | None = None
    signature_origin: SignatureOrigin | None = None
    # OpenAI explicit prompt-cache boundary (``{"mode": "explicit"}``) carried
    # on this content part. Preserved verbatim (separately from
    # ``cache_control``) so a native OpenAI upstream receives the client's exact
    # breakpoint instead of the Anthropic-style mirror.
    prompt_cache_breakpoint: dict[str, Any] | None = None


@dataclass
class ImageBlock(ContentBlock):
    """Image content block with source and optional detail level."""

    source: ImageSource
    detail: str | None = None
    cache_control: Any | None = None
    # Anthropic image transformations config (e.g. oversized_image: error).
    transformations: dict[str, Any] | None = None
    prompt_cache_breakpoint: dict[str, Any] | None = None


@dataclass
class AudioBlock(ContentBlock):
    """Audio content block with source."""

    source: AudioSource
    cache_control: Any | None = None
    prompt_cache_breakpoint: dict[str, Any] | None = None


@dataclass
class VideoBlock(ContentBlock):
    """Video content block with source.

    ``extra`` carries the source-level knobs a wire dialect declares but the
    shared model has no field for (MiniMax's ``fps`` and
    ``max_long_side_pixel``). They ride along so a dialect that understands
    them can replay them verbatim instead of losing the tuning on a
    round-trip.
    """

    source: VideoSource
    cache_control: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


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
    toolset_name: str | None = None
    cache_control: Any | None = None
    # Raw tool definitions carried by an OpenResponses ``tool_search_output`` item.
    # ``content`` holds their JSON rendering (the tool result text other providers
    # receive); this is the structured copy, so the materialize round-trip
    # (``conversation_to_input_items``) never has to re-parse the text to re-emit
    # the ``tool_search_output`` item. None for every other tool result.
    discovered_tools: list[dict[str, Any]] | None = None
