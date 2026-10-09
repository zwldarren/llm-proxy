"""Ollama native duration metric extraction.

Ollama reports timings in nanoseconds alongside responses. Both the
non-streaming response parser and the streaming chunk converter surface these
in provider_info / usage so observability can access them. They are not billable
facts, so they stay out of the usage intake seam (ADR-0025): the token counters
are the seam's ``ollama`` dialect (``serialization/usage/adapters.py``).
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
