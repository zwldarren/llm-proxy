"""Finish-reason vocabulary and failure-detail extraction for the Gemini
Interactions API (shared).

``interaction.status`` → OpenAI finish reason, and failure-detail message
building for failed/cancelled interactions. Used by both the response parser
and the streaming converter so the two paths stay consistent.
"""

from typing import Any, Final

STATUS_TO_FINISH_REASON: Final[dict[str, str]] = {
    "completed": "stop",
    "requires_action": "tool_calls",
    "incomplete": "length",
    "budget_exceeded": "length",
}

# Statuses treated as terminal errors, propagated to the client.
FAILED_STATUSES: Final[frozenset[str]] = frozenset({"failed", "cancelled"})


def interaction_error_message(payload: dict[str, Any]) -> str:
    """Extract a human-readable failure message from an Interaction payload.

    The Interaction resource carries failure details in ``error`` (singular,
    per the SDK docs); ``errors`` is a defensive fallback. Returns "" when
    no details are present.
    """
    raw_error = payload.get("error") or payload.get("errors") or []
    error_items = raw_error if isinstance(raw_error, list) else [raw_error]
    return "; ".join(
        f"{e.get('code', '')}: {e.get('message', '')}" for e in error_items if isinstance(e, dict)
    )
