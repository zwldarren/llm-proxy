"""Finish-reason vocabulary for the Gemini Interactions API (shared).

``interaction.status`` → OpenAI finish reason. Used by both the response
parser and the streaming converter so the two paths stay consistent.
"""

from typing import Final

STATUS_TO_FINISH_REASON: Final[dict[str, str]] = {
    "completed": "stop",
    "requires_action": "tool_calls",
    "incomplete": "length",
    "budget_exceeded": "length",
}

# Statuses treated as terminal errors, propagated to the client.
FAILED_STATUSES: Final[frozenset[str]] = frozenset({"failed", "cancelled"})
