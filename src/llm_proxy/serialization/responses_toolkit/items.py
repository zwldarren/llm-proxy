"""Shared Responses-shaped item helpers.

Helpers for working with Responses-API-shaped items (item ids, reasoning
text extraction) that are used by the OpenResponses protocol module and by
provider-family serializers alike.
"""

import secrets
from typing import Any


def generate_item_id() -> str:
    return f"item_{secrets.token_hex(12)}"


def _extract_reasoning_text(content: Any) -> str:
    """Extract reasoning text from a reasoning item's ``content`` list."""
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in (
            "reasoning_text",
            "output_text",
            "summary_text",
        ):
            parts.append(part.get("text", ""))
    return "".join(parts)


def _extract_summary_text(summary: Any) -> str:
    """Extract reasoning summary text from a reasoning item's ``summary`` list.

    The OpenResponses/OpenAI reasoning item can carry its visible reasoning as
    ``summary`` parts (``summary_text``) instead of, or alongside, ``content``.
    Used as a fallback when ``content`` is empty so the reasoning context still
    round-trips to the provider.
    """
    if not isinstance(summary, list):
        return ""
    parts: list[str] = []
    for part in summary:
        if isinstance(part, dict) and part.get("type") in (
            "summary_text",
            "output_text",
            "reasoning_text",
        ):
            parts.append(part.get("text", ""))
    return "".join(parts)


__all__ = [
    "generate_item_id",
    "_extract_reasoning_text",
    "_extract_summary_text",
]
