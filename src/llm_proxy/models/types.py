# src/llm_proxy/models/types.py
"""Supporting types for unified protocol format."""

from dataclasses import dataclass
from typing import Any, Literal

# Type aliases
ResponseStatus = Literal["completed", "incomplete", "error"]


@dataclass
class ImageSource:
    """Image source with type, data, and optional media type."""

    type: Literal["base64", "url", "file_id"]
    data: str
    media_type: str | None = None


@dataclass
class AudioSource:
    """Audio source with type, data, and optional media type."""

    type: Literal["base64", "url", "file_id"]
    data: str
    media_type: str | None
    # Audio response fields
    id: str | None = None
    expires_at: int | None = None
    transcript: str | None = None


@dataclass
class VideoSource:
    """Video source with type, data, and optional media type."""

    type: Literal["base64", "url", "file_id"]
    data: str
    media_type: str | None = None


@dataclass
class DocumentSource:
    """Document source with type, data, and optional media type."""

    type: Literal["base64", "url", "file_id", "text", "content"]
    data: Any
    media_type: str | None = None


@dataclass
class UrlCitation:
    """URL citation for annotations."""

    url: str
    title: str
    start_index: int
    end_index: int


@dataclass
class Annotation:
    """Annotation with citation information."""

    type: Literal["url_citation", "file_citation"]
    url_citation: UrlCitation | None = None
    file_citation: dict | None = None  # file_citation structure may vary


@dataclass
class CompletionTokensDetails:
    """Breakdown of tokens used in a completion."""

    accepted_prediction_tokens: int | None = None
    audio_tokens: int | None = None
    reasoning_tokens: int | None = None
    rejected_prediction_tokens: int | None = None
    image_tokens: int | None = None


@dataclass
class PromptTokensDetails:
    """Breakdown of tokens used in the prompt."""

    audio_tokens: int | None = None
    cached_tokens: int | None = None
    image_tokens: int | None = None
    text_tokens: int | None = None
    cache_write_tokens: int | None = None
    video_tokens: int | None = None


@dataclass
class Usage:
    """Token usage information — the canonical in-proxy usage record.

    Canonical contract: each billable fact is expressed in exactly ONE field,
    normalized at parse time by the provider serializer:

    - cache read  → ``cache_read_input_tokens`` (never a Gemini-style
      ``cached_content_tokens`` alias; the OpenAI-dialect expression
      ``prompt_tokens_details.cached_tokens`` is tolerated for OpenAI-family
      wire compatibility but must not coexist with the flat field);
    - cache write → ``cache_creation_input_tokens``;
    - thinking    → ``reasoning_tokens``.

    Downstream consumers (EventContext, billing) rely on this single
    expression per fact; expressing the same fact in two fields double-charges
    it in the cost calculation.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None
    # STT: duration of the transcribed audio in seconds (duration-based billing)
    audio_duration_seconds: float | None = None
    # Web search request count (server_tool_use.web_search_requests)
    web_search_requests: int | None = None

    def __post_init__(self) -> None:
        """Compute total_tokens from input and output tokens."""
        if self.total_tokens is None:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class ResponseFormat:
    """Response format specification.

    For json_schema type, the json_schema field should contain:
    {"name": str, "description": str, "schema": dict, "strict": bool}
    """

    type: Literal["text", "json_object", "json_schema"]
    # For json_schema type: {"name": str, "description": str, "schema": dict, "strict": bool}
    json_schema: dict[str, Any] | None = None


def unwrap_json_schema_wrapper(json_schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract the plain JSON Schema from a ``ResponseFormat.json_schema`` value.

    Protocol serializers (OpenAI chat completions, OpenResponses) store the
    full wrapper ``{"name", "description", "schema", "strict"}`` in
    ``ResponseFormat.json_schema``; provider serializers need the inner
    ``schema`` dict. A bare schema dict (no wrapper) is returned unchanged.
    """
    if isinstance(json_schema, dict) and isinstance(json_schema.get("schema"), dict):
        return json_schema["schema"]
    return json_schema


@dataclass
class TokenLogprob:
    """Log probability information for a single token."""

    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TokenLogprob] | None = None


@dataclass
class ChoiceLogprobs:
    """Log probability information for a choice."""

    content: list[TokenLogprob] | None = None
    refusal: list[TokenLogprob] | None = None


@dataclass
class ChoiceMetadata:
    """Metadata for a single choice in multi-choice responses (n > 1).

    Stores per-choice finish_reason, logprobs, and annotations.
    """

    finish_reason: str | None = None
    logprobs: ChoiceLogprobs | None = None
    annotations: list[dict[str, Any]] | None = None


@dataclass
class StreamOptions:
    """Streaming options for responses.

    ``include_usage`` defaults to False to match the OpenAI API's own default;
    callers that need the terminal usage chunk must request it explicitly.
    """

    include_usage: bool = False
    chunk_type: str | None = None  # Merged from StreamOptionsExtended
    include_obfuscation: bool | None = None


@dataclass
class ThinkingConfig:
    """Thinking/thinking_budget configuration for models that support it."""

    type: Literal["enabled", "disabled", "adaptive"] | None = None
    budget_tokens: int | None = None
    effort: str | None = None
    display: str | None = None


@dataclass
class CitationsConfig:
    """Citation configuration for documents."""

    enabled: bool


@dataclass
class Container:
    """Container information for code execution."""

    id: str
    expires_at: str | None = None


@dataclass
class CacheCreation:
    """Cache creation token breakdown."""

    ephemeral_1h_input_tokens: int = 0
    ephemeral_5m_input_tokens: int = 0


@dataclass
class ServerToolUsage:
    """Server tool usage information."""

    web_fetch_requests: int = 0
    web_search_requests: int = 0
