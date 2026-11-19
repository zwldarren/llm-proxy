"""MiniMax wire vocabulary — service_tier translation.

The service tier vocabulary differs across wires: MiniMax speaks
``standard``/``priority``, Anthropic speaks ``auto``/``standard_only``, and
OpenAI speaks ``auto``/``default``/``flex``/``priority``. Thinking/reasoning
parameters pass through without proxy-side rewriting (ADR-0063).
"""

from typing import Any

#: Service tiers MiniMax accepts, and the client spellings that map onto
#: them. ``auto``/``flex``/``scale`` have no MiniMax spelling: ``auto`` means
#: "the provider picks" (MiniMax picks ``standard``), and the other two are
#: OpenAI-only tiers, so they are dropped rather than mistranslated.
_SERVICE_TIER_TO_MINIMAX: dict[str, str] = {
    "standard": "standard",
    "standard_only": "standard",
    "default": "standard",
    "priority": "priority",
}


def normalize_service_tier(body: dict[str, Any]) -> dict[str, Any]:
    """Map a request body's ``service_tier`` onto MiniMax's enum.

    A tier MiniMax does not know is dropped: ``auto`` means "the provider
    picks" and MiniMax picks ``standard`` (its default), so the field is
    simply omitted. The body is mutated in place and returned.
    """
    tier = body.get("service_tier")
    if tier is None:
        return body
    if not isinstance(tier, str):
        body.pop("service_tier", None)
        return body
    mapped = _SERVICE_TIER_TO_MINIMAX.get(tier.lower())
    if mapped is None:
        body.pop("service_tier", None)
    else:
        body["service_tier"] = mapped
    return body


__all__ = ["normalize_service_tier"]
