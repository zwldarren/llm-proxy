"""ContentBlock types for unified protocol format.

Core types are universal. Extended types are multi-provider.
Anthropic builtin types are Anthropic-specific.
"""

from llm_proxy.models.content_blocks.anthropic_builtin import (
    BashCodeExecutionToolResultBlock,
    CacheControl,
    Caller,
    Citation,
    CitationCharLocation,
    CitationContentBlockLocation,
    CitationPageLocation,
    CitationSearchResultLocation,
    CitationWebSearchResultLocation,
    CodeExecutionToolResultBlock,
    ContainerUploadBlock,
    MidConversationSystemBlock,
    SearchResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    ToolReferenceBlock,
    ToolSearchToolResultBlock,
    WebFetchToolResultBlock,
    WebSearchResultContentBlock,
    WebSearchToolResultBlock,
)
from llm_proxy.models.content_blocks.core import (
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
)
from llm_proxy.models.content_blocks.extended import (
    CustomToolUseBlock,
    FileBlock,
    RawBlock,
    RedactedThinkingBlock,
    RefusalBlock,
    ServerToolUseBlock,
    ThinkingBlock,
)

__all__ = [
    # core
    "AudioBlock",
    "ContentBlock",
    "DocumentBlock",
    "ImageBlock",
    "TextBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "VideoBlock",
    # extended
    "CustomToolUseBlock",
    "FileBlock",
    "RawBlock",
    "RedactedThinkingBlock",
    "RefusalBlock",
    "ServerToolUseBlock",
    "ThinkingBlock",
    # anthropic_builtin
    "BashCodeExecutionToolResultBlock",
    "CacheControl",
    "Caller",
    "Citation",
    "CitationCharLocation",
    "CitationContentBlockLocation",
    "CitationPageLocation",
    "CitationSearchResultLocation",
    "CitationWebSearchResultLocation",
    "CodeExecutionToolResultBlock",
    "ContainerUploadBlock",
    "MidConversationSystemBlock",
    "SearchResultBlock",
    "TextEditorCodeExecutionToolResultBlock",
    "ToolReferenceBlock",
    "ToolSearchToolResultBlock",
    "WebFetchToolResultBlock",
    "WebSearchResultContentBlock",
    "WebSearchToolResultBlock",
]
