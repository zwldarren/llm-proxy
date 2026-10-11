"""Dialect aliases folded at parse time, so the record holds one fact per field.

These two folds used to live beside the OpenAI response parser, but they are
dialect-alias normalization — exactly what usage intake owns (ADR-0025). They
are shared by every path that reads the chat dialect: the response parser, the
streaming converters, the protocol transformers that consume the converted
internal chat shape, and the adapters here.
"""

from typing import Any

__all__ = ["fold_deepseek_cache_hits", "fold_top_level_reasoning_tokens"]


def fold_top_level_reasoning_tokens(usage: dict[str, Any]) -> dict[str, Any]:
    """Fold SGLang-style top-level ``reasoning_tokens`` into the nested OpenAI details.

    SGLang reports reasoning tokens as a top-level ``usage.reasoning_tokens``
    field — on both the non-streaming response and the streaming terminal
    usage chunk — instead of OpenAI's nested
    ``usage.completion_tokens_details.reasoning_tokens``. The canonical record
    keeps thinking tokens flat, so the intake adapter lifts both expressions
    there (:func:`_lift_reasoning`); this fold serves the *dict* a streaming tier
    forwards to the client verbatim, where the OpenAI-dialect shape is what the
    client reads.

    A zero or negative report adds nothing (an absent details block already
    means "no reasoning tokens"), and an existing nested ``reasoning_tokens``
    wins. The input dict is never mutated — a new dict is returned when a fold
    applies, so callers may fold a response body without editing it.
    """
    reasoning_tokens = usage.get("reasoning_tokens")
    if not isinstance(reasoning_tokens, int) or reasoning_tokens <= 0:
        return usage

    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        details = {}
    if details.get("reasoning_tokens") is not None:
        return usage
    return {
        **usage,
        "completion_tokens_details": {**details, "reasoning_tokens": reasoning_tokens},
    }


def fold_deepseek_cache_hits(cached_tokens: int | None, cache_hits: Any) -> int | None:
    """Fold DeepSeek-style top-level ``prompt_cache_hit_tokens`` into ``cached_tokens``.

    DeepSeek reports ``prompt_tokens = hit + miss``, so hits are already
    included in ``prompt_tokens``; folding them into ``cached_tokens`` makes
    cache pricing apply instead of silently billing them at the full input
    rate. An explicit ``cached_tokens`` value wins — hits fold in only when
    the details carry no cached value.
    """
    if cached_tokens is not None:
        return cached_tokens
    if isinstance(cache_hits, int) and cache_hits > 0:
        return cache_hits
    return None
