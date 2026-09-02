"""Extended ContentBlock types supported by multiple providers but not universal."""

from dataclasses import dataclass, field
from typing import Any, Literal

from llm_proxy.models.content_blocks.core import ContentBlock

#: Providers whose thought signatures may be replayed to the same provider.
SignatureOrigin = Literal["gemini"]


@dataclass
class ThinkingBlock(ContentBlock):
    """Thinking / reasoning content. Supported by Anthropic and OpenAI o-series.

    ``signature`` is provider-specific (Anthropic signature, OpenAI reasoning
    signature, Gemini thoughtSignature). ``signature_origin`` records which
    provider produced the signature so target serializers can decide whether
    replaying it is valid (e.g. only Gemini-issued thoughtSignatures may be
    sent back to Gemini).
    """

    thinking: str
    signature: str | None = None
    signature_origin: SignatureOrigin | None = None
    encrypted_content: str | None = None
    cache_control: Any | None = None


@dataclass
class RedactedThinkingBlock(ContentBlock):
    """Redacted thinking content block."""

    data: str
    cache_control: Any | None = None


@dataclass
class RefusalBlock(ContentBlock):
    """Safety refusal content block."""

    refusal: str
    cache_control: Any | None = None


@dataclass
class FileBlock(ContentBlock):
    """File content block with file data or file_id."""

    file_data: str | None = None
    file_id: str | None = None
    filename: str | None = None
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
