"""Shared Gemini usageMetadata → billable token-count mapping.

Gemini's ``usageMetadata`` carries more dimensions than the OpenAI-shaped
usage the proxy bills on:

- ``toolUsePromptTokenCount`` — tool-use prompt tokens, reported SEPARATELY
  from ``promptTokenCount``. They are billed at the input rate, EXCEPT when
  they came from Google Search grounding (Google bills those via the
  per-request search fee instead), so search-grounded calls exclude them.
  The API only exposes the aggregate across ALL tools, so when a request
  mixes search grounding with function calling, the function-call tool
  tokens are excluded too and input is undercounted — a documented
  limitation (no per-tool breakdown is available).
- ``thoughtsTokenCount`` — thinking tokens. Depending on model/version,
  ``candidatesTokenCount`` may or may not include them. Detected via the
  total: ``prompt + candidates + toolUse + thoughts == total`` means
  candidates exclude thoughts, so they are added to the output side.
  Requiring the exact match (rather than "sum != total") keeps the
  heuristic from misfiring when ``totalTokenCount`` carries additional
  dimensions. ``cachedContentTokenCount`` is NOT one of them: per the
  Gemini docs, ``promptTokenCount`` already includes the cached content
  tokens, so they are inside the sum above.

Both the non-streaming response parser and the streaming converter use this
so streaming and non-streaming billing agree.
"""

from typing import Any


def billable_token_counts(meta: dict[str, Any], *, has_search_grounding: bool) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from a Gemini usageMetadata dict.

    Args:
        meta: The ``usageMetadata`` dict from a Gemini response/chunk.
        has_search_grounding: Whether the response used Google Search
            grounding (``groundingMetadata.webSearchQueries`` non-empty).
            Search-grounded tool-use prompt tokens are excluded from input
            billing (Google charges the search fee instead). Note: the API
            reports tool-use tokens as one aggregate, so when search
            grounding is combined with function calling, all tool-use tokens
            are excluded and input is undercounted.

    Returns:
        Tuple of (input_tokens, output_tokens).
    """
    prompt = meta.get("promptTokenCount", 0) or 0
    candidates = meta.get("candidatesTokenCount", 0) or 0
    tool_use = meta.get("toolUsePromptTokenCount", 0) or 0
    thoughts = meta.get("thoughtsTokenCount", 0) or 0
    total = meta.get("totalTokenCount", 0) or 0

    input_tokens = prompt if has_search_grounding else prompt + tool_use

    # When candidatesTokenCount excludes thinking tokens, totalTokenCount is
    # the sum of prompt + candidates + toolUse + thoughts. Requiring the exact
    # match (rather than "sum != total") keeps the heuristic from misfiring
    # when totalTokenCount carries additional dimensions. cachedContentTokenCount
    # is NOT one of them: per the Gemini docs, promptTokenCount already
    # includes the cached content tokens, so they are inside the sum above.
    output_tokens = candidates
    if thoughts and total and prompt + candidates + tool_use + thoughts == total:
        output_tokens = candidates + thoughts

    return input_tokens, output_tokens


__all__ = ["billable_token_counts"]
