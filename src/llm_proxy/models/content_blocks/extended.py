"""Extended ContentBlock types supported by multiple providers but not universal."""

from dataclasses import dataclass, field
from typing import Any, Literal

from llm_proxy.models.content_blocks.core import ContentBlock

#: Providers whose thought signatures may be replayed to the same provider.
#: ``openai`` marks a genuine Responses ``encrypted_content`` blob (which is
#: not an Anthropic verification payload and must not be replayed as one).
SignatureOrigin = Literal["gemini", "openai"]


@dataclass
class ThinkingBlock(ContentBlock):
    """Thinking / reasoning content. Supported by Anthropic and OpenAI o-series.

    ``signature`` is provider-specific (Anthropic signature, OpenAI reasoning
    signature, Gemini thoughtSignature). ``signature_origin`` records which
    provider produced the signature so target serializers can decide whether
    replaying it is valid (e.g. only Gemini-issued thoughtSignatures may be
    sent back to Gemini).

    ``extra`` carries opaque provider payloads that must round-trip unchanged.
    OpenRouter's ``reasoning_details`` array (``reasoning.text``,
    ``reasoning.encrypted``, ``reasoning.summary`` entries) lives here: models
    that emit encrypted or summarized reasoning require it to be echoed back
    verbatim, and its entries cannot be represented by ``thinking`` alone.
    """

    thinking: str
    signature: str | None = None
    signature_origin: SignatureOrigin | None = None
    encrypted_content: str | None = None
    cache_control: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RedactedThinkingBlock(ContentBlock):
    """Redacted thinking content block.

    ``extra`` mirrors ``ThinkingBlock.extra`` so an opaque provider payload that
    arrives alongside redacted reasoning is not dropped on the round trip.
    """

    data: str
    cache_control: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefusalBlock(ContentBlock):
    """Safety refusal content block."""

    refusal: str
    cache_control: Any | None = None


@dataclass
class FileBlock(ContentBlock):
    """File content block with file data, a URL, or a file_id.

    ``file_data`` is base64/data-URI content, ``file_url`` a remote URL and
    ``file_id`` a Files-API reference: OpenAI's Responses ``input_file``
    distinguishes all three. ``detail`` (``auto``/``low``/``high``) controls
    the rendering detail for file inputs.
    """

    file_data: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None
    detail: str | None = None
    cache_control: Any | None = None


@dataclass
class RawBlock(ContentBlock):
    """Opaque passthrough for any provider-specific block.

    serializer internal code recognizes the provider_type and handles
    serialization. This is the escape hatch: new provider features
    never require changes to the unified model.

    Example provider_type values:
        "anthropic:container_upload"
        "anthropic:mid_conv_system"
        "gemini:function_call"
    """

    provider_type: str
    data: dict[str, Any]
    cache_control: Any | None = None


@dataclass
class CustomToolUseBlock(ContentBlock):
    """Custom tool use request block."""

    id: str
    name: str
    input: str
    cache_control: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerToolUseBlock(ContentBlock):
    """Server-side tool use request block."""

    id: str
    name: str
    input: dict[str, Any]
    type: str = "server_tool_use"
    cache_control: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)
