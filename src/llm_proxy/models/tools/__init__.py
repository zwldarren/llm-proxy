"""Tool definition types for unified protocol format."""

from llm_proxy.models.tools.anthropic_builtin import (
    BashTool,
    CodeExecutionTool,
    MemoryTool,
    TextEditorTool,
    ToolSearchTool,
    UserLocation,
    WebFetchTool,
    WebSearchTool,
)
from llm_proxy.models.tools.core import (
    AllowedToolsConfig,
    CustomTool,
    FunctionTool,
    ToolChoice,
    ToolChoiceAllowedTools,
    ToolChoiceCustom,
    ToolChoiceFunction,
    ToolChoiceNamed,
    ToolChoiceSpec,
    ToolDefinition,
    is_web_search_tool_name,
)
from llm_proxy.models.tools.openai_builtin import (
    CodeInterpreterTool,
    FileSearchTool,
    OpenAIToolSearchTool,
)
from llm_proxy.models.tools.openai_builtin import (
    WebSearchTool as OpenAIWebSearchTool,
)

AnthropicWebSearchTool = WebSearchTool

__all__ = [
    # core
    "AllowedToolsConfig",
    "CustomTool",
    "FunctionTool",
    "ToolChoice",
    "ToolChoiceAllowedTools",
    "ToolChoiceCustom",
    "ToolChoiceFunction",
    "ToolChoiceNamed",
    "ToolChoiceSpec",
    "ToolDefinition",
    "is_web_search_tool_name",
    # anthropic_builtin
    "BashTool",
    "CodeExecutionTool",
    "MemoryTool",
    "TextEditorTool",
    "ToolSearchTool",
    "UserLocation",
    "WebFetchTool",
    "WebSearchTool",
    "AnthropicWebSearchTool",
    "OpenAIWebSearchTool",
    "CodeInterpreterTool",
    "FileSearchTool",
    "OpenAIToolSearchTool",
]
