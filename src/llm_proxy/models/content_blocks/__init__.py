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

#: Block types that carry a tool-call id. Used wherever an "is this a tool
#: call?" check must cover the custom-tool (``custom_tool_call``) and
#: server-side web-search shapes, not just the plain ``tool_use`` block — for
#: example pairing cached reasoning with the tool call it precedes.
TOOL_CALL_BLOCK_TYPES = (ToolUseBlock, CustomToolUseBlock, ServerToolUseBlock)

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
    # tool-call shapes
    "TOOL_CALL_BLOCK_TYPES",
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
