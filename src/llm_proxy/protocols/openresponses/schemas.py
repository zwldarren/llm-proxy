"""Pydantic models for OpenResponses API request and response schemas."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# Content Types (Input)
# =============================================================================


class InputTextContent(BaseModel):
    """Text input content."""

    type: Literal["input_text"] = "input_text"
    text: str = Field(..., description="The text input")


class InputImageContent(BaseModel):
    """Image input content."""

    type: Literal["input_image"] = "input_image"
    image_url: str = Field(..., description="URL or base64 data URL of the image")
    detail: Literal["low", "high", "auto"] = Field("auto", description="Image detail level")


class InputFileContent(BaseModel):
    """File input content."""

    type: Literal["input_file"] = "input_file"
    file_url: str | None = Field(None, description="URL of the file")
    file_data: str | None = Field(None, description="Base64 encoded file data")
    file_id: str | None = Field(None, description="File ID from Files API")
    filename: str | None = Field(None, description="Name of the file")


class InputVideoContent(BaseModel):
    """Video input content."""

    type: Literal["input_video"] = "input_video"
    video_url: str = Field(..., description="URL of the video")


class InputAudioContent(BaseModel):
    """Audio input content."""

    type: Literal["input_audio"] = "input_audio"
    audio_data: str | None = Field(None, description="Base64 encoded audio data")
    audio_url: str | None = Field(None, description="URL or base64 data URL of the audio")
    format: Literal["wav", "mp3", "ogg", "flac", "webm", "mp4"] = Field(
        "wav", description="Audio format"
    )


class EncryptedContentItem(BaseModel):
    """Encrypted content item.

    Used inside ``function_call_output.output`` arrays (and reasoning content)
    to carry opaque, round-trippable encrypted blobs (e.g. Codex reasoning
    continuity with ``store=false``).
    """

    model_config = ConfigDict(extra="allow")
    type: Literal["encrypted_content"] = "encrypted_content"
    encrypted_content: str = Field(..., description="Opaque encrypted content blob")


InputContent = (
    InputTextContent | InputImageContent | InputFileContent | InputVideoContent | InputAudioContent
)


# =============================================================================
# Content Types (Output)
# =============================================================================


class UrlCitation(BaseModel):
    """URL citation annotation."""

    type: Literal["url_citation"] = "url_citation"
    url: str = Field(..., description="URL of the cited resource")
    start_index: int = Field(..., description="Start index in the text")
    end_index: int = Field(..., description="End index in the text")
    title: str = Field(..., description="Title of the cited resource")


class OutputTextContent(BaseModel):
    """Text output content."""

    type: Literal["output_text"] = "output_text"
    text: str = Field(..., description="The text content")
    annotations: list[UrlCitation] = Field(default_factory=list, description="URL citations")
    logprobs: list[LogprobsContent] | None = Field(None, description="Log probabilities")


class LogprobsToken(BaseModel):
    """Log probability for a token."""

    token: str = Field(..., description="The token")
    logprob: float = Field(..., description="Log probability")
    bytes: list[int] | None = Field(None, description="Token bytes as integer array")


class LogprobsContent(LogprobsToken):
    """Log probability content for a token position."""

    top_logprobs: list[LogprobsToken] = Field(
        default_factory=list, description="Most likely tokens and their log probabilities"
    )


class Logprobs(BaseModel):
    """Log probability information."""

    content: list[LogprobsContent] | None = Field(
        None, description="Log probability content for each token"
    )


class RefusalContent(BaseModel):
    """Refusal content from the model."""

    type: Literal["refusal"] = "refusal"
    refusal: str = Field(..., description="The refusal message")


class SummaryTextContent(BaseModel):
    """Summary text content."""

    type: Literal["summary_text"] = "summary_text"
    text: str = Field(..., description="The summary text")


class ReasoningTextContent(BaseModel):
    """Reasoning text content. Per OpenResponses spec, uses output_text type."""

    type: Literal["output_text"] = "output_text"
    text: str = Field(..., description="The reasoning text")


OutputContent = OutputTextContent | RefusalContent


# =============================================================================
# Item Types (Input/Request)
# =============================================================================


class UserMessageItemParam(BaseModel):
    """User message item for request."""

    type: Literal["message"] = "message"
    role: Literal["user"] = "user"
    content: str | list[InputContent] = Field(..., description="Message content")
    id: str | None = Field(None, description="Unique item ID")


class SystemMessageItemParam(BaseModel):
    """System message item for request."""

    type: Literal["message"] = "message"
    role: Literal["system"] = "system"
    content: str | list[InputTextContent] = Field(..., description="Message content")
    id: str | None = Field(None, description="Unique item ID")


class DeveloperMessageItemParam(BaseModel):
    """Developer message item for request."""

    type: Literal["message"] = "message"
    role: Literal["developer"] = "developer"
    content: str | list[InputTextContent] = Field(..., description="Message content")
    id: str | None = Field(None, description="Unique item ID")


class AssistantMessageItemParam(BaseModel):
    """Assistant message item for request."""

    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: str | list[OutputTextContent | RefusalContent] = Field(
        ..., description="Message content"
    )
    id: str | None = Field(None, description="Unique item ID")
    status: str | None = Field(None, description="Item status")
    phase: Literal["commentary", "final_answer"] | None = Field(
        None, description="Assistant message phase (commentary or final_answer)"
    )


class FunctionCallItemParam(BaseModel):
    """Function call item for request."""

    type: Literal["function_call"] = "function_call"
    call_id: str = Field(..., description="Unique call ID")
    name: str = Field(..., description="Function name")
    arguments: str = Field(..., description="JSON-encoded arguments")
    id: str | None = Field(None, description="Unique item ID")
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        None, description="Item status"
    )
    thought_signature: str | None = Field(
        None,
        description="Gemini thought signature for multi-turn reasoning/tool-call context",
    )


FunctionCallOutputContent = (
    InputTextContent | InputImageContent | EncryptedContentItem | dict[str, Any]
)
"""Content item within a ``function_call_output.output`` array.

A plain ``str`` output is handled separately; this alias covers the array form
where each element is a typed content item or (permissively) an arbitrary dict.
"""


class FunctionCallOutputItemParam(BaseModel):
    """Function call output item for request.

    ``output`` may be a plain string (the common case) OR an array of
    structured content items (``input_text`` / ``input_image`` /
    ``encrypted_content``). The array form is what Codex sends for tool results
    packaged as content items; rejecting it caused 422 ``Input should be a
    valid string`` errors at the /v1/responses endpoint. See OpenAI Responses
    API ``function_call_output.output`` and Codex ``FunctionCallOutputBody``.
    """

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str = Field(..., description="ID of the function call being responded to")
    output: str | list[FunctionCallOutputContent] = Field(
        ..., description="Function output as a plain string or an array of content items"
    )
    id: str | None = Field(None, description="Unique item ID")
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        None, description="Item status"
    )


class ReasoningItemParam(BaseModel):
    """Reasoning item for request."""

    type: Literal["reasoning"] = "reasoning"
    summary: list[SummaryTextContent] = Field(
        default_factory=list, description="Reasoning summaries"
    )
    content: Any | None = Field(None, description="Reasoning content")
    encrypted_content: str | None = Field(None, description="Encrypted reasoning content")
    id: str | None = Field(None, description="Unique item ID")


class ItemReferenceParam(BaseModel):
    """Reference to a previous item in the conversation."""

    type: Literal["item_reference"] = "item_reference"
    id: str = Field(..., description="ID of the referenced item")


class LocalShellCallItemParam(BaseModel):
    """Hosted Responses API local-shell tool call item (Codex native shell tool)."""

    type: Literal["local_shell_call"] = "local_shell_call"
    id: str | None = Field(None, description="Unique item ID")
    call_id: str | None = Field(None, description="Unique call ID")
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        None, description="Item status"
    )
    action: dict[str, Any] = Field(
        ...,
        description="Shell action, e.g. {type: exec, command: [...], working_directory, env}",
    )


class CustomToolCallItemParam(BaseModel):
    """Custom (freeform) tool call item.

    Analog of ``function_call`` but carries ``input`` (a JSON string) instead of
    ``arguments``. Codex uses this for freeform/custom tools such as apply_patch.
    """

    type: Literal["custom_tool_call"] = "custom_tool_call"
    id: str | None = Field(None, description="Unique item ID")
    status: str | None = Field(None, description="Item status")
    call_id: str = Field(..., description="Unique call ID")
    name: str = Field(..., description="Tool name")
    namespace: str | None = Field(None, description="Tool namespace")
    input: str = Field(..., description="JSON-encoded tool input")


class CustomToolCallOutputItemParam(BaseModel):
    """Custom (freeform) tool call output item.

    ``output`` uses the same wire encoding as ``function_call_output.output``
    (plain string or array of content items).
    """

    type: Literal["custom_tool_call_output"] = "custom_tool_call_output"
    id: str | None = Field(None, description="Unique item ID")
    call_id: str = Field(..., description="ID of the tool call being responded to")
    name: str | None = Field(None, description="Tool name")
    output: str | list[FunctionCallOutputContent] = Field(
        ..., description="Tool output as a plain string or an array of content items"
    )


class LocalShellCallOutputItemParam(BaseModel):
    """Hosted Responses API local-shell tool output item (Codex native shell tool).

    Companion to :class:`LocalShellCallItemParam`. ``output`` uses the same
    wire encoding as ``function_call_output.output`` (plain string or array of
    content items) — Codex emits it as a JSON string carrying the shell
    execution result (stdout/stderr/exit_code).
    """

    type: Literal["local_shell_call_output"] = "local_shell_call_output"
    id: str | None = Field(None, description="Unique item ID")
    call_id: str = Field(..., description="ID of the local_shell_call being responded to")
    output: str | list[FunctionCallOutputContent] = Field(
        ..., description="Shell output as a plain string or an array of content items"
    )


class ToolSearchCallItemParam(BaseModel):
    """Hosted tool-search call item."""

    type: Literal["tool_search_call"] = "tool_search_call"
    id: str | None = Field(None, description="Unique item ID")
    call_id: str | None = Field(None, description="Unique call ID")
    status: str | None = Field(None, description="Item status")
    execution: str = Field(..., description="Execution strategy")
    arguments: Any | None = Field(None, description="Tool search arguments")


class ToolSearchOutputItemParam(BaseModel):
    """Hosted tool-search output item (result of a tool_search_call)."""

    type: Literal["tool_search_output"] = "tool_search_output"
    id: str | None = Field(None, description="Unique item ID")
    call_id: str | None = Field(None, description="Unique call ID")
    status: str = Field(..., description="Item status")
    execution: str = Field(..., description="Execution strategy")
    tools: list[dict[str, Any]] = Field(default_factory=list, description="Discovered tools")


class WebSearchCallItemParam(BaseModel):
    """Hosted web-search call item (Codex Responses API web_search)."""

    type: Literal["web_search_call"] = "web_search_call"
    id: str | None = Field(None, description="Unique item ID")
    status: str | None = Field(None, description="Item status")
    action: dict[str, Any] | None = Field(None, description="Web search action")


class ImageGenerationCallItemParam(BaseModel):
    """Hosted image-generation call item."""

    type: Literal["image_generation_call"] = "image_generation_call"
    id: str | None = Field(None, description="Unique item ID")
    status: str = Field(..., description="Item status")
    revised_prompt: str | None = Field(None, description="Revised prompt")
    result: str = Field(..., description="Generated image result")


class AgentMessageItemParam(BaseModel):
    """Agent-to-agent message item (Codex multi-agent flows)."""

    type: Literal["agent_message"] = "agent_message"
    id: str | None = Field(None, description="Unique item ID")
    author: str = Field(..., description="Authoring agent name")
    recipient: str = Field(..., description="Recipient agent name")
    content: list[dict[str, Any]] = Field(..., description="Message content items")


class AdditionalToolsItemParam(BaseModel):
    """Additional tools item (Codex dynamic tool injection)."""

    type: Literal["additional_tools"] = "additional_tools"
    id: str | None = Field(None, description="Unique item ID")
    role: str = Field(..., description="Role")
    tools: list[dict[str, Any]] = Field(default_factory=list, description="Tool definitions")


class CompactionItemParam(BaseModel):
    """Context compaction summary item. Carries encrypted reasoning content.

    Codex serializes this with ``type: "compaction"`` and accepts
    ``"compaction_summary"`` as a deserialization alias.
    """

    type: Literal["compaction", "compaction_summary"] = "compaction"
    id: str | None = Field(None, description="Unique item ID")
    encrypted_content: str = Field(..., description="Opaque encrypted compaction content")
    created_by: str | None = Field(
        None, description="Identifier of the actor that created the item"
    )


class CompactionTriggerItemParam(BaseModel):
    """Compaction trigger control item (not a durable response item)."""

    type: Literal["compaction_trigger"] = "compaction_trigger"


class ContextCompactionItemParam(BaseModel):
    """Context compaction item with optional encrypted content."""

    type: Literal["context_compaction"] = "context_compaction"
    id: str | None = Field(None, description="Unique item ID")
    encrypted_content: str | None = Field(None, description="Opaque encrypted content")


ItemParam = (
    UserMessageItemParam
    | SystemMessageItemParam
    | DeveloperMessageItemParam
    | AssistantMessageItemParam
    | FunctionCallItemParam
    | FunctionCallOutputItemParam
    | ReasoningItemParam
    | ItemReferenceParam
    | LocalShellCallItemParam
    | LocalShellCallOutputItemParam
    | CustomToolCallItemParam
    | CustomToolCallOutputItemParam
    | ToolSearchCallItemParam
    | ToolSearchOutputItemParam
    | WebSearchCallItemParam
    | ImageGenerationCallItemParam
    | AgentMessageItemParam
    | AdditionalToolsItemParam
    | CompactionItemParam
    | CompactionTriggerItemParam
    | ContextCompactionItemParam
    # Forward-compatible catch-all: accept unknown/future item types as raw
    # dicts so the proxy never 422s on a valid Codex item it doesn't model yet
    # (mirrors Codex's own #[serde(other)] Other variant). Unknown items are
    # skipped by the parser.
    | dict[str, Any]
)


# =============================================================================
# Item Types (Output/Response)
# =============================================================================


class Message(BaseModel):
    """Message item in response."""

    type: Literal["message"] = "message"
    id: str = Field(..., description="Unique item ID")
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        ..., description="Item status"
    )
    role: Literal["user", "assistant", "system", "developer"] = Field(
        ..., description="Message role"
    )
    content: list[OutputContent] = Field(..., description="Message content parts")
    phase: Literal["commentary", "final_answer"] | None = Field(
        None, description="Assistant message phase (commentary or final_answer)"
    )


class FunctionCall(BaseModel):
    """Function call item in response."""

    type: Literal["function_call"] = "function_call"
    id: str = Field(..., description="Unique item ID")
    call_id: str = Field(..., description="Unique call ID")
    name: str = Field(..., description="Function name")
    arguments: str = Field(..., description="JSON-encoded arguments")
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        ..., description="Item status"
    )
    thought_signature: str | None = Field(
        None,
        description="Gemini thought signature for multi-turn reasoning/tool-call context",
    )


class FunctionCallOutput(BaseModel):
    """Function call output item in response."""

    type: Literal["function_call_output"] = "function_call_output"
    id: str = Field(..., description="Unique item ID")
    call_id: str = Field(..., description="ID of the function call")
    output: str | list[FunctionCallOutputContent] = Field(
        ..., description="Function output as a plain string or an array of content items"
    )
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        ..., description="Item status"
    )


class ReasoningBody(BaseModel):
    """Reasoning item in response."""

    type: Literal["reasoning"] = "reasoning"
    id: str = Field(..., description="Unique item ID")
    # Per OpenResponses spec, reasoning content uses output_text type
    content: list[InputTextContent | OutputTextContent] = Field(
        default_factory=list, description="Reasoning content"
    )
    summary: list[SummaryTextContent] = Field(
        default_factory=list, description="Reasoning summaries"
    )
    encrypted_content: str | None = Field(None, description="Encrypted reasoning content")


class CompactionBody(BaseModel):
    """Compaction item in response (produced by /v1/responses/compact)."""

    type: Literal["compaction"] = "compaction"
    id: str = Field(..., description="Unique item ID")
    encrypted_content: str = Field(..., description="Encrypted content produced by compaction")
    created_by: str | None = Field(
        None, description="Identifier of the actor that created the item"
    )


ItemField = (
    Message
    | FunctionCall
    | FunctionCallOutput
    | ReasoningBody
    | CompactionBody
    # Forward-compatible catch-all: stored responses may carry output items the
    # proxy itself emitted but does not model here (custom_tool_call,
    # web_search_call, tool_search_call/output, local_shell_call, ...). Without
    # this, GET /v1/responses/{id} 500s on ResponsesResponse(**stored) for any
    # such response (ItemParam already has the same catch-all on the input side).
    | dict[str, Any]
)


# =============================================================================
# Tool Types
# =============================================================================


class FunctionToolParam(BaseModel):
    """Function tool definition."""

    type: Literal["function"] = "function"
    name: str = Field(..., description="Function name")
    description: str | None = Field(None, description="Function description")
    parameters: dict[str, Any] | None = Field(None, description="JSON Schema for parameters")
    strict: bool = Field(False, description="Enable strict parameter validation")


class WebSearchToolParam(BaseModel):
    """Built-in web search tool with all OpenAI Responses API parameters."""

    model_config = ConfigDict(extra="allow")

    type: Literal["web_search", "web_search_preview"] = "web_search"
    search_context_size: Literal["low", "medium", "high"] | None = Field(
        None, description="Controls how much context from web search results is made available"
    )
    filters: dict[str, Any] | None = Field(
        None,
        description=(
            "Domain filtering configuration with allowed_domains and blocked_domains. "
            "Example: {'allowed_domains': ['example.com'], 'blocked_domains': ['bad.com']}"
        ),
    )
    external_web_access: bool | None = Field(
        None,
        description=(
            "Whether to fetch live content (true) or use only cached/indexed results (false)"
        ),
    )
    return_token_budget: Literal["default", "unlimited"] | None = Field(
        None,
        description=(
            "Controls how much web search result content the tool can return. "
            "'default' uses standard budget, 'unlimited' removes the cap"
        ),
    )
    search_content_types: list[Literal["text", "image"]] | None = Field(
        None,
        description=("Types of content to include in search results. Example: ['text', 'image']"),
    )
    image_settings: dict[str, Any] | None = Field(
        None,
        description=(
            "Settings for image search results. Example: {'max_results': 3, 'caption': True}"
        ),
    )
    user_location: dict[str, Any] | None = Field(
        None,
        description=(
            "Approximate user location for localized results. "
            "Example: {'type': 'approximate', 'country': 'US', 'city': 'New York'}"
        ),
    )


class CodeInterpreterToolParam(BaseModel):
    """Built-in code interpreter tool."""

    model_config = ConfigDict(extra="allow")

    type: Literal["code_interpreter"] = "code_interpreter"


class FileSearchToolParam(BaseModel):
    """Built-in file search tool."""

    model_config = ConfigDict(extra="allow")

    type: Literal["file_search"] = "file_search"


ToolParam = (
    FunctionToolParam
    | WebSearchToolParam
    | CodeInterpreterToolParam
    | FileSearchToolParam
    | dict[str, Any]
)


class FunctionToolChoice(BaseModel):
    """Specific function tool choice."""

    type: Literal["function"] = "function"
    name: str = Field(..., description="Function name to call")


class AllowedToolChoice(BaseModel):
    """Allowed tools configuration."""

    type: Literal["allowed_tools"] = "allowed_tools"
    tools: list[FunctionToolChoice] = Field(..., description="Allowed tools")
    # Optional per the request-side spec (AllowedToolsParam requires only
    # type + tools; mode defaults to "auto" server-side).
    mode: Literal["auto", "required", "none"] | None = Field(
        None, description="Tool selection mode"
    )


ToolChoiceParam = Literal["auto", "required", "none"] | FunctionToolChoice | AllowedToolChoice


# =============================================================================
# Configuration Parameters
# =============================================================================


class ReasoningParam(BaseModel):
    """Reasoning configuration.

    Effort values (model-dependent): none, minimal, low, medium, high, xhigh, max.
    Mode values: standard (default), pro (GPT-5.6).
    Context values: auto, current_turn, all_turns.
    Summary values: auto, concise, detailed.
    """

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = Field(
        None, description="Reasoning effort level"
    )
    mode: Literal["standard", "pro"] | None = Field(
        None, description="Reasoning mode: standard (default) or pro (GPT-5.6)"
    )
    context: Literal["auto", "current_turn", "all_turns"] | None = Field(
        None, description="Reasoning context persistence across turns"
    )
    summary: Literal["auto", "concise", "detailed"] | None = Field(
        None, description="Reasoning summary style"
    )


class TextResponseFormat(BaseModel):
    """Text response format."""

    type: Literal["text"] = "text"


class JsonObjectResponseFormat(BaseModel):
    """JSON object response format."""

    type: Literal["json_object"] = "json_object"


class JsonSchemaResponseFormat(BaseModel):
    """JSON schema response format."""

    type: Literal["json_schema"] = "json_schema"
    name: str = Field(..., description="Schema name")
    description: str | None = Field(None, description="Schema description")
    schema_: dict[str, Any] = Field(..., alias="schema", description="JSON Schema")
    strict: bool = Field(True, description="Enable strict schema validation")


TextFormatParam = TextResponseFormat | JsonObjectResponseFormat | JsonSchemaResponseFormat


class StreamOptionsParam(BaseModel):
    """Stream options."""

    include_obfuscation: bool = Field(True, description="Obfuscate sensitive data in stream")


class TextParam(BaseModel):
    """Text output configuration (format + verbosity)."""

    format: TextFormatParam | None = Field(None, description="Text format configuration")
    verbosity: Literal["low", "medium", "high"] | None = Field(
        None, description="Text verbosity level"
    )


class IncompleteDetails(BaseModel):
    """Details about incomplete response."""

    reason: str = Field(..., description="Reason for incomplete response")


class InputTokensDetails(BaseModel):
    """Input token usage details."""

    cached_tokens: int = Field(..., description="Number of cached tokens")


class OutputTokensDetails(BaseModel):
    """Output token usage details."""

    reasoning_tokens: int = Field(..., description="Number of reasoning tokens")


class Usage(BaseModel):
    """Token usage statistics."""

    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")
    total_tokens: int = Field(..., description="Total tokens used")
    input_tokens_details: InputTokensDetails | None = Field(None, description="Input details")
    output_tokens_details: OutputTokensDetails | None = Field(None, description="Output details")


class Error(BaseModel):
    """Error response."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    type: str | None = Field(None, description="Error type")
    param: str | None = Field(None, description="Parameter that caused error")


# =============================================================================
# Main Request/Response Models
# =============================================================================


class ResponsesRequest(BaseModel):
    """Request model for OpenResponses API (/v1/responses)."""

    model_config = ConfigDict(extra="allow")

    # Required fields
    model: str = Field(..., description="Model to use")
    input: str | list[ItemParam] = Field(..., description="Input text or array of items")

    # Optional fields
    previous_response_id: str | None = Field(
        None, description="ID of previous response to continue"
    )
    conversation: str | dict[str, Any] | None = Field(
        None,
        description=(
            "Conversation to continue (ID string or object). Server-side state the "
            "proxy cannot resolve locally; forwarded verbatim to native Responses "
            "providers and dropped when translating to Chat Completions."
        ),
    )
    prompt: dict[str, Any] | None = Field(
        None,
        description=(
            "Prompt template reference (id, version, variables). Forwarded verbatim "
            "to native Responses providers; dropped for Chat Completions providers."
        ),
    )
    include: list[str] | None = Field(
        None,
        description=(
            "Additional fields to include. Kept as plain strings (not a Literal "
            "enum) so future include values added by OpenAI/Codex pass through "
            "instead of being rejected with 422 — unknown values are forwarded "
            "verbatim to native Responses providers and ignored elsewhere."
        ),
    )
    tools: list[ToolParam] | None = Field(None, description="Tools available to the model")
    tool_choice: ToolChoiceParam | None = Field(None, description="Tool choice configuration")
    metadata: dict[str, str] | None = Field(None, description="Metadata key-value pairs (max 16)")
    text: TextFormatParam | TextParam | None = Field(None, description="Text output configuration")

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        """Enforce the spec's metadata limits: 16 pairs, keys <= 64, values <= 512."""
        if v is None:
            return v
        if len(v) > 16:
            raise ValueError("metadata must contain at most 16 key-value pairs")
        for key, value in v.items():
            if len(key) > 64:
                raise ValueError("metadata keys must be at most 64 characters")
            if len(value) > 512:
                raise ValueError("metadata values must be at most 512 characters")
        return v

    temperature: float | None = Field(None, ge=0, le=2, description="Sampling temperature")
    top_p: float | None = Field(None, ge=0, le=1, description="Nucleus sampling")
    presence_penalty: float | None = Field(None, ge=-2, le=2, description="Presence penalty")
    frequency_penalty: float | None = Field(None, ge=-2, le=2, description="Frequency penalty")
    parallel_tool_calls: bool | None = Field(None, description="Allow parallel tool calls")
    stream: bool = Field(False, description="Enable streaming")
    stream_options: StreamOptionsParam | None = Field(None, description="Stream options")
    background: bool | None = Field(None, description="Run in background")
    max_output_tokens: int | None = Field(None, ge=1, description="Maximum output tokens")
    max_tool_calls: int | None = Field(None, ge=1, description="Maximum tool calls")
    reasoning: ReasoningParam | None = Field(None, description="Reasoning configuration")
    safety_identifier: str | None = Field(None, description="Safety monitoring identifier")
    prompt_cache_key: str | None = Field(None, description="Prompt cache key")
    truncation: Literal["auto", "disabled"] | None = Field(None, description="Truncation mode")
    instructions: str | None = Field(None, description="Additional instructions")
    store: bool | None = Field(None, description="Store response")
    service_tier: Literal["auto", "default", "flex", "priority"] | None = Field(
        None, description="Service tier"
    )
    top_logprobs: int | None = Field(None, ge=0, le=20, description="Number of top logprobs")


class ResponsesResponse(BaseModel):
    """Response model for OpenResponses API."""

    model_config = ConfigDict(extra="allow")

    # Required fields
    id: str = Field(..., description="Unique response ID")
    object: Literal["response"] = "response"
    created_at: int = Field(..., description="Unix timestamp of creation")
    status: Literal["queued", "in_progress", "completed", "failed", "incomplete"] = Field(
        ..., description="Response status"
    )
    model: str = Field(..., description="Model used")

    # Optional fields
    completed_at: int | None = Field(None, description="Unix timestamp of completion")
    incomplete_details: IncompleteDetails | None = Field(None, description="Incomplete details")
    output: list[ItemField] = Field(default_factory=list, description="Output items")
    error: Error | None = Field(None, description="Error if failed")
    tools: list[FunctionToolParam] | None = Field(None, description="Tools used")
    tool_choice: ToolChoiceParam | None = Field(None, description="Tool choice used")
    truncation: Literal["auto", "disabled"] | None = Field(None, description="Truncation mode")
    parallel_tool_calls: bool | None = Field(None, description="Parallel tool calls allowed")
    text: TextFormatParam | TextParam | None = Field(None, description="Text format used")
    top_p: float | None = Field(None, description="Top-p used")
    presence_penalty: float | None = Field(None, description="Presence penalty used")
    frequency_penalty: float | None = Field(None, description="Frequency penalty used")
    top_logprobs: int | None = Field(None, description="Top logprobs used")
    temperature: float | None = Field(None, description="Temperature used")
    reasoning: ReasoningParam | None = Field(None, description="Reasoning config used")
    usage: Usage | None = Field(None, description="Token usage")
    logprobs: Logprobs | None = Field(None, description="Log probabilities")
    max_output_tokens: int | None = Field(None, description="Max output tokens")
    max_tool_calls: int | None = Field(None, description="Max tool calls")
    store: bool | None = Field(None, description="Store response")
    background: bool | None = Field(None, description="Background mode")
    service_tier: Literal["auto", "default", "flex", "priority"] | None = Field(
        None, description="Service tier used"
    )
    metadata: dict[str, str] | None = Field(None, description="Metadata")
    safety_identifier: str | None = Field(None, description="Safety identifier")
    prompt_cache_key: str | None = Field(None, description="Cache key")
    previous_response_id: str | None = Field(None, description="Previous response ID")
    instructions: str | None = Field(None, description="Instructions used")


__all__ = [
    "InputTextContent",
    "InputImageContent",
    "InputFileContent",
    "InputVideoContent",
    "EncryptedContentItem",
    "InputContent",
    "UrlCitation",
    "OutputTextContent",
    "RefusalContent",
    "SummaryTextContent",
    "ReasoningTextContent",
    "OutputContent",
    "UserMessageItemParam",
    "SystemMessageItemParam",
    "DeveloperMessageItemParam",
    "AssistantMessageItemParam",
    "FunctionCallItemParam",
    "FunctionCallOutputItemParam",
    "ReasoningItemParam",
    "ItemReferenceParam",
    "LocalShellCallItemParam",
    "LocalShellCallOutputItemParam",
    "CustomToolCallItemParam",
    "CustomToolCallOutputItemParam",
    "ToolSearchCallItemParam",
    "ToolSearchOutputItemParam",
    "WebSearchCallItemParam",
    "ImageGenerationCallItemParam",
    "AgentMessageItemParam",
    "AdditionalToolsItemParam",
    "CompactionItemParam",
    "CompactionTriggerItemParam",
    "ContextCompactionItemParam",
    "ItemParam",
    "Message",
    "FunctionCall",
    "FunctionCallOutput",
    "ReasoningBody",
    "ItemField",
    "FunctionToolParam",
    "FunctionToolChoice",
    "AllowedToolChoice",
    "ToolChoiceParam",
    "ReasoningParam",
    "TextResponseFormat",
    "JsonObjectResponseFormat",
    "JsonSchemaResponseFormat",
    "TextFormatParam",
    "TextParam",
    "CompactionBody",
    "StreamOptionsParam",
    "IncompleteDetails",
    "InputTokensDetails",
    "OutputTokensDetails",
    "Usage",
    "LogprobsToken",
    "LogprobsContent",
    "Logprobs",
    "Error",
    "ResponsesRequest",
    "ResponsesResponse",
]
