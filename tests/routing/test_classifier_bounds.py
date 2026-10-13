"""Bounded feature extraction: long inputs must not make classify() O(length).

Feature extraction scans text character by character in pure Python on the
event loop; the budget cap keeps the worst case constant instead of growing
with the size of a pasted document (:func:`llm_proxy.routing.classifier
._bounded_feature_text`).
"""

import time

from llm_proxy.routing import classifier


def _long_prompt(chars: int) -> str:
    return ("abcdefghij" * (chars // 10 + 1))[:chars]


def test_short_text_passes_through_unbounded():
    text = "What is 2+2?"
    assert classifier._bounded_feature_text(text) == text


def test_long_text_is_reduced_to_head_plus_tail():
    text = _long_prompt(100_000)
    bounded = classifier._bounded_feature_text(text)
    budget = classifier._FEATURE_BUDGET_HEAD_CHARS + classifier._FEATURE_BUDGET_TAIL_CHARS
    assert len(bounded) == budget
    # Head and tail preserved, middle dropped.
    assert bounded.startswith(text[:100])
    assert bounded.endswith(text[-100:])


def test_feature_extraction_receives_bounded_text():
    """The structural extractor sees the excerpt, not the full 100k prompt."""
    seen: list[str] = []
    original = classifier.extract_structural_features
    classifier.extract_structural_features = lambda text: seen.append(text) or original(text)
    try:
        classifier.classify(_long_prompt(100_000))
    finally:
        classifier.extract_structural_features = original
    assert seen, "structural extraction never ran"
    budget = classifier._FEATURE_BUDGET_HEAD_CHARS + classifier._FEATURE_BUDGET_TAIL_CHARS
    assert all(len(text) <= budget for text in seen)


def test_classify_stays_fast_on_huge_input():
    """The point of the budget: a 200k-char ask classifies in bounded time.

    The full pipeline used to cost ~90 ms at 50k chars and grew linearly;
    bounded extraction keeps it in single-digit milliseconds. The bound here
    is generous so the assertion holds on slow CI machines too.
    """
    classifier.classify("hi")  # force the model load out of the measurement
    start = time.perf_counter()
    classifier.classify(_long_prompt(200_000))
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 50


def test_length_sensitive_guards_still_see_the_full_text():
    """Trivial/soften logic reads the full prompt: a long ask is never 'trivial'."""
    # >100k tokens forces the structural COMPLEX short-circuit, which reads
    # the unbounded token estimate.
    result = classifier.classify(_long_prompt(500_000))
    assert result.tier is not None
    assert result.tier.value == "COMPLEX"
