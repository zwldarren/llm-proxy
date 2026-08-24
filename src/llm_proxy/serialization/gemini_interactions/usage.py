"""Shared Interactions-API usage → billable token-count mapping.

The Interactions API reports usage with a fresh field vocabulary (unlike the
legacy ``usageMetadata``):

- ``total_input_tokens`` — prompt/context tokens.
- ``total_tool_use_tokens`` — tool-use prompt tokens, reported SEPARATELY.
- ``total_output_tokens`` — generated response tokens (excludes thinking).
- ``total_thought_tokens`` — thinking tokens.
- ``total_cached_tokens`` — cached-content tokens.
- ``total_tokens`` — prompt + responses + other internal tokens (per the
  reference, tool-use tokens are NOT part of the sum).
- ``grounding_tool_count`` — per-tool counts for server-side grounding tools
  (``google_search`` etc.).

Billing conventions (mirroring the legacy heuristic in
``serialization/gemini/usage.py``):

- Tool-use prompt tokens are billed at the input rate, EXCEPT when they came
  from Google Search grounding (Google bills those via the per-request search
  fee instead), so search-grounded calls exclude them.
- Thinking tokens are billed at the output rate, so they fold into output.

Both the non-streaming response parser and the streaming converter use this
so streaming and non-streaming billing agree.
"""

from typing import Any


def interactions_normalize_usage(usage: dict[str, Any]) -> dict[str, Any]:
    """Normalize OpenAI-style usage aliases to the Interactions vocabulary.

    The migration guide's streaming examples show OpenAI-style field names
    (``prompt_tokens`` / ``completion_tokens``) on ``interaction.completed``
    events while the API reference documents the new vocabulary
    (``total_input_tokens`` / ``total_output_tokens`` / ...). Accept both
    shapes; the new vocabulary wins when both are present.
    """
    normalized = dict(usage)
    if "total_input_tokens" not in normalized and "prompt_tokens" in normalized:
        normalized["total_input_tokens"] = normalized["prompt_tokens"]
    if "total_output_tokens" not in normalized and "completion_tokens" in normalized:
        normalized["total_output_tokens"] = normalized["completion_tokens"]
    return normalized


def interactions_input_tokens_by_modality(usage: dict[str, Any]) -> dict[str, int]:
    """Return ``{modality: tokens}`` from ``input_tokens_by_modality``.

    The usage resource reports a per-modality breakdown of input tokens
    (``text`` / ``image`` / ``audio`` / ...); used to fill the OpenAI Images
    ``input_tokens_details`` shape with real numbers instead of zeros.
    """
    result: dict[str, int] = {}
    for item in usage.get("input_tokens_by_modality") or []:
        if isinstance(item, dict) and item.get("modality") and item.get("tokens") is not None:
            result[str(item["modality"])] = int(item["tokens"])
    return result


def interactions_web_search_requests(usage: dict[str, Any] | None) -> int:
    """Return the number of Google Search grounding requests in *usage*.

    ``grounding_tool_count`` lists the server-side grounding tools that ran,
    with per-tool counts (Google Search may issue multiple searches).
    """
    if not isinstance(usage, dict):
        return 0
    usage = interactions_normalize_usage(usage)
    count = 0
    for item in usage.get("grounding_tool_count") or []:
        if isinstance(item, dict) and item.get("type") == "google_search":
            count += item.get("count", 0) or 0
    return count


def interactions_billable_token_counts(
    usage: dict[str, Any], *, has_search_grounding: bool
) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from an Interactions usage dict.

    Args:
        usage: The ``usage`` dict from an Interaction resource / stream event.
        has_search_grounding: Whether the response used Google Search
            grounding. Search-grounded tool-use tokens are excluded from input
            billing (Google charges the search fee instead). Note: the API
            reports tool-use tokens as one aggregate, so when search
            grounding is combined with function calling, all tool-use tokens
            are excluded and input is undercounted — the same documented
            limitation as the legacy usageMetadata mapping.

    Returns:
        Tuple of (input_tokens, output_tokens).
    """
    usage = interactions_normalize_usage(usage)
    input_tokens = usage.get("total_input_tokens", 0) or 0
    tool_use = usage.get("total_tool_use_tokens", 0) or 0
    output_tokens = usage.get("total_output_tokens", 0) or 0
    thoughts = usage.get("total_thought_tokens", 0) or 0

    if not has_search_grounding:
        input_tokens += tool_use

    return input_tokens, output_tokens + thoughts


__all__ = [
    "interactions_billable_token_counts",
    "interactions_input_tokens_by_modality",
    "interactions_normalize_usage",
    "interactions_web_search_requests",
]
