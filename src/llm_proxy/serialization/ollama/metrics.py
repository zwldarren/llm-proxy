"""Ollama native duration metric extraction.

Ollama reports timings in nanoseconds alongside responses. Both the
non-streaming response parser and the streaming chunk converter surface these
in provider_info / usage so observability and billing can access them.
"""

from typing import Any

# Ollama native duration metrics (nanoseconds) preserved for observability.
_OLLAMA_DURATION_KEYS = (
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
)


def extract_ollama_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract Ollama native duration metrics from a response/chunk payload.

    Returns an empty dict when none of the known duration keys are present.
    """
    metrics = {
        key: payload.get(key) for key in _OLLAMA_DURATION_KEYS if payload.get(key) is not None
    }
    return metrics


def extract_ollama_token_counts(payload: dict[str, Any]) -> tuple[int, int] | None:
    """Extract Ollama's ``(input, output)`` token counts from a payload.

    Ollama reports ``prompt_eval_count`` and ``eval_count`` on the terminal
    response or chunk. Either counter may be absent while the other is
    present, and a reported ``0`` is a real value (the prompt was fully served
    from cache, or no output was generated), so the guard must not treat it as
    "missing". Returns ``None`` when neither counter is present — the caller
    builds no usage envelope in that case.
    """
    input_tokens = payload.get("prompt_eval_count")
    output_tokens = payload.get("eval_count")
    if input_tokens is None and output_tokens is None:
        return None
    return input_tokens or 0, output_tokens or 0


def extract_ollama_cached_tokens(payload: dict[str, Any], input_tokens: int) -> int | None:
    """Extract Ollama's cache-read prompt token count.

    ``prompt_eval_cached_count`` (Ollama v0.33.3+) reports how many prompt
    tokens were served from the KV cache. It is a *subset* of
    ``prompt_eval_count``, which Ollama keeps as the logical input total —
    the same contract as the canonical ``Usage`` record, where
    ``input_tokens`` includes cache tokens. The value is clamped into
    ``[0, input_tokens]`` so a provider violating that invariant yields a
    sane count instead of a negative or oversized one.

    Upstream the field is a nullable pointer and only exists on newer
    servers, so it is frequently absent. ``None`` is returned in that case
    rather than ``0``, keeping "provider reported no cache data" distinct
    from "provider reported zero cache hits".
    """
    cached = payload.get("prompt_eval_cached_count")
    if not isinstance(cached, int):
        return None
    return max(0, min(cached, input_tokens))
