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
    custom_tool_bridge_description,
    declares_web_search_tool,
    is_web_search_tool_name,
    is_web_search_tool_type,
)
from llm_proxy.models.tools.hosted import (
    DEPRECATED_HOSTED_TOOL_TYPES,
    PROVIDER_HOSTED_TOOL_TYPES,
    HostedTool,
    is_provider_hosted_tool_type,
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
    "custom_tool_bridge_description",
    "is_web_search_tool_name",
    "is_web_search_tool_type",
    "declares_web_search_tool",
    # hosted
    "HostedTool",
    "PROVIDER_HOSTED_TOOL_TYPES",
    "DEPRECATED_HOSTED_TOOL_TYPES",
    "is_provider_hosted_tool_type",
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
