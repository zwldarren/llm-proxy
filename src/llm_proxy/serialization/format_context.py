"""FormatContext for serializer response formatting.

Replaces the loose **kwargs pattern on format_response() with a structured
dataclass carrying original request fields needed to construct the response body.

Protocol-specific field usage:
    openresponses: instructions, previous_response_id, store, metadata,
        temperature, top_p, presence_penalty, frequency_penalty, truncation,
        parallel_tool_calls, max_output_tokens, max_tool_calls, reasoning,
        service_tier, text, top_logprobs, background, safety_identifier,
        prompt_cache_key, include
    openai: (not currently used)
    anthropic: (not currently used)
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FormatContext:
    """Context for formatting protocol responses.

    Carries original request fields needed to construct the response body,
    avoiding the loose **kwargs pattern on format_response().

    See module docstring for per-protocol field usage.
    """

    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    metadata: dict[str, str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    truncation: str | None = None
    parallel_tool_calls: bool | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    reasoning: dict[str, Any] | None = None
    service_tier: str | None = None
    text: Any | None = None
    top_logprobs: int | None = None
    background: bool | None = None
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    include: list[str] | None = None
    tool_choice: Any | None = None
    # Namespace mapping for Responses→Chat tool name restoration.
    # Serialized NamespaceMapping: {"flat_name": ["namespace", "original_name"], ...}
    namespace_map: dict[str, list[str]] | None = None
    n: int | None = None
    tools: list[Any] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
