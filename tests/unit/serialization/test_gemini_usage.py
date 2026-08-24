"""Unit tests for the shared Gemini usageMetadata → billable token mapping."""

from llm_proxy.serialization.gemini.usage import billable_token_counts


def test_cached_content_does_not_break_thoughts_detection():
    """cachedContentTokenCount is a subset of promptTokenCount (per the Gemini
    docs, promptTokenCount includes the cached content tokens), so it must
    not affect the thoughts-detection heuristic.

    Gemini API style: candidatesTokenCount already includes thinking tokens,
    so total = prompt + candidates + toolUse with cached inside prompt.
    """
    meta = {
        "promptTokenCount": 10,  # includes the 5 cached tokens
        "candidatesTokenCount": 5,  # includes the 2 thinking tokens
        "cachedContentTokenCount": 5,
        "thoughtsTokenCount": 2,
        "totalTokenCount": 15,
    }
    input_tokens, output_tokens = billable_token_counts(meta, has_search_grounding=False)
    assert input_tokens == 10
    # Thoughts are already inside candidatesTokenCount: adding them again
    # would overcount output.
    assert output_tokens == 5


def test_cached_content_with_separate_thoughts_still_adds_them():
    """Vertex AI style: candidatesTokenCount excludes thinking tokens, so
    total = prompt + candidates + toolUse + thoughts even with cached content."""
    meta = {
        "promptTokenCount": 10,  # includes the 5 cached tokens
        "candidatesTokenCount": 5,  # excludes the 2 thinking tokens
        "cachedContentTokenCount": 5,
        "thoughtsTokenCount": 2,
        "totalTokenCount": 17,
    }
    input_tokens, output_tokens = billable_token_counts(meta, has_search_grounding=False)
    assert input_tokens == 10
    assert output_tokens == 7


def test_search_grounding_excludes_all_tool_use_tokens():
    """Documented limitation: toolUsePromptTokenCount is an aggregate across
    ALL tools, so search grounding excludes function-call tool tokens too
    (input undercounted). The API exposes no per-tool breakdown."""
    meta = {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "toolUsePromptTokenCount": 3,  # 1 search-grounded + 2 function-call
        "totalTokenCount": 18,
    }
    input_tokens, output_tokens = billable_token_counts(meta, has_search_grounding=True)
    assert input_tokens == 10  # all 3 tool-use tokens excluded
    assert output_tokens == 5
