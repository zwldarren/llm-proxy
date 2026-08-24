"""Anthropic-specific ContentBlock types.

These types are NOT exported from the default namespace. They are used
internally by the Anthropic serializer/deserializer to represent
Anthropic-specific content blocks that have no equivalent in other providers.
"""

from dataclasses import dataclass
from typing import Any, Literal

from llm_proxy.models.content_blocks.core import ContentBlock, ToolResultBlock


@dataclass
class CacheControl:
    """Cache control for content blocks."""

    type: Literal["ephemeral"] = "ephemeral"
    ttl: Literal["5m", "1h"] | None = None


@dataclass
class Caller:
    """Tool caller information."""

    type: Literal["direct", "code_execution_20250825", "code_execution_20260120"]
    tool_id: str | None = None


@dataclass
class Citation:
    """Base citation type."""

    cited_text: str
    document_index: int | None = None
    document_title: str | None = None


@dataclass
class CitationCharLocation(Citation):
    """Character-based citation."""

    type: Literal["char_location"] = "char_location"
    start_char_index: int = 0
    end_char_index: int = 0


@dataclass
class CitationPageLocation(Citation):
    """Page-based citation."""

    type: Literal["page_location"] = "page_location"
    start_page_number: int = 0
    end_page_number: int = 0


@dataclass
class CitationContentBlockLocation(Citation):
    """Content block-based citation."""

    type: Literal["content_block_location"] = "content_block_location"
    start_block_index: int = 0
    end_block_index: int = 0


@dataclass
class CitationWebSearchResultLocation:
    """Web search result citation."""

    cited_text: str
    encrypted_index: str
    title: str
    url: str
    type: Literal["web_search_result_location"] = "web_search_result_location"


@dataclass
class CitationSearchResultLocation:
    """Search result location citation."""

    cited_text: str
    search_result_index: int
    source: str
    start_block_index: int
    end_block_index: int
    title: str
    type: Literal["search_result_location"] = "search_result_location"


@dataclass
class ContainerUploadBlock(ContentBlock):
    """Container upload block for file uploads to code execution containers."""

    file_id: str | None = None
    filename: str | None = None
    content: str | None = None
    media_type: str | None = None


@dataclass
class MidConversationSystemBlock(ContentBlock):
    """System instructions that appear mid-conversation.

    Maps to Anthropic's ``mid_conv_system`` content block.
    """

    content: list[ContentBlock]
    cache_control: Any | None = None


@dataclass
class SearchResultBlock(ContentBlock):
    """Search result content block for tool results.

    Maps to Anthropic's search_result content block where ``content`` is
    an array of text blocks. We store the parsed blocks directly to preserve
    citations metadata.
    """

    source: str | None = None
    file_id: str | None = None
    title: str | None = None
    content: list[ContentBlock] | None = None
    metadata: dict[str, Any] | None = None
    cache_control: Any | None = None


@dataclass
class ToolReferenceBlock(ContentBlock):
    """Reference to a tool definition."""

    tool_id: str
    tool_name: str | None = None
    tool_type: str | None = None


@dataclass
class WebSearchToolResultBlock(ContentBlock):
    """Result from web_search built-in tool."""

    tool_use_id: str
    content: str | list[ContentBlock] | list[dict[str, Any]]
    is_error: bool = False
    caller: Caller | None = None


@dataclass
class WebFetchToolResultBlock(ContentBlock):
    """Result from web_fetch built-in tool."""

    tool_use_id: str
    content: str | list[ContentBlock]
    is_error: bool = False
    caller: Caller | None = None


@dataclass
class WebSearchResultContentBlock(ContentBlock):
    """Individual web search result item.

    Used within WebSearchToolResultBlock.content.
    """

    url: str
    title: str
    encoded_content: str
    page_age: str | None = None
    type: str = "web_search_result"


@dataclass
class CodeExecutionToolResultBlock(ToolResultBlock):
    """Result from code_execution built-in tool."""

    caller: Caller | None = None


@dataclass
class BashCodeExecutionToolResultBlock(ToolResultBlock):
    """Result from bash built-in tool."""

    caller: Caller | None = None


@dataclass
class TextEditorCodeExecutionToolResultBlock(ToolResultBlock):
    """Result from text_editor built-in tool."""

    caller: Caller | None = None


@dataclass
class ToolSearchToolResultBlock(ToolResultBlock):
    """Result from tool_search built-in tool."""

    caller: Caller | None = None
