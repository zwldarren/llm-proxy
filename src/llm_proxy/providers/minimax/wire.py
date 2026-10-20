"""MiniMax wire vocabulary — translating the client's words into MiniMax's.

MiniMax's three wires do not speak the vocabulary the proxy's protocols use:

============  ===================================  =========================
wire          thinking                              depth
============  ===================================  =========================
Messages      ``{"type": "disabled" | "adaptive"}``  ``output_config.effort``
Chat          ``{"type": "disabled" | "adaptive"}``  ``reasoning_effort``
Responses     ``reasoning.effort`` (no thinking)     ``reasoning.effort``
============  ===================================  =========================

Anthropic's own vocabulary is richer: ``{"type": "enabled", "budget_tokens":
N}``, ``{"type": "between_tools"}`` and the effort values ``none``/``minimal``.
MiniMax rejects the parts it does not know with HTTP 400 (parameter error
2013): a ``thinking.type`` outside ``disabled|adaptive``, ``budget_tokens``,
``display``, an effort outside its enum, and — on the Chat wire —
``reasoning_effort: "none"``.

The service tier has the same problem: MiniMax speaks ``standard``/
``priority``, Anthropic speaks ``auto``/``standard_only`` and OpenAI speaks
``auto``/``default``/``flex``/``priority``.

Every rule below is a pure function over a request body, so each tier of the
proxy (the native Messages passthrough, the native Responses passthrough and
the rebuilt chat body) can share them.
"""

from typing import Any

#: Effort values MiniMax accepts on the Messages and Chat wires.
MINIMAX_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})

#: Anthropic's lowest non-zero effort tier; MiniMax spells it ``low``.
_MINIMAL_EFFORT = "low"

#: Model ids whose thinking cannot be turned off: MiniMax-M3.1-Flash always
#: thinks and answers HTTP 400 to ``thinking: {"type": "disabled"}`` (and to
#: ``reasoning.effort: "none"`` on the Responses wire). Matched
#: case-insensitively against the routed model id.
ALWAYS_THINKING_MODEL_MARKERS = ("m3.1",)

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


def always_thinks(model: str | None) -> bool:
    """Whether the model rejects every "thinking off" spelling."""
    if not model:
        return False
    lowered = model.lower()
    return any(marker in lowered for marker in ALWAYS_THINKING_MODEL_MARKERS)


def normalize_effort(effort: Any) -> str | None:
    """Map an effort value onto MiniMax's enum, or None when it cannot be kept.

    None means the value has no MiniMax spelling: the caller drops the effort
    (and, for ``none``, expresses "no reasoning" through ``thinking``).
    """
    if not isinstance(effort, str) or not effort:
        return None
    lowered = effort.lower()
    if lowered in MINIMAX_EFFORTS:
        return lowered
    if lowered == "minimal":
        return _MINIMAL_EFFORT
    return None


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


def _disable_thinking(body: dict[str, Any], model: str | None) -> None:
    """Express "no reasoning" on a MiniMax wire.

    ``thinking: {"type": "disabled"}`` is MiniMax's only spelling, and
    MiniMax-M3.1-Flash rejects it — on a model that always thinks the field is
    dropped instead, so the request stays valid.
    """
    if always_thinks(model):
        body.pop("thinking", None)
    else:
        body["thinking"] = {"type": "disabled"}


def normalize_messages_thinking(body: dict[str, Any], model: str | None) -> dict[str, Any]:
    """Normalize an Anthropic Messages body's thinking parameters for MiniMax.

    - ``thinking.type`` becomes ``adaptive`` (thinking on) or ``disabled``
      (thinking off); ``budget_tokens``/``display`` are dropped — MiniMax tunes
      depth with the effort, not with a token budget;
    - a ``disabled`` type is dropped on models that always think (they answer
      HTTP 400 to it);
    - ``output_config`` keeps only ``effort``, mapped onto MiniMax's enum; an
      effort of ``none`` becomes ``thinking: {"type": "disabled"}``, and a
      structured-output ``format`` (no MiniMax spelling) is dropped.

    The body is mutated in place and returned.
    """
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if thinking_type == "enabled":
            # Anthropic's "thinking on" — MiniMax spells it ``adaptive``.
            thinking_type = "adaptive"
        elif thinking_type == "between_tools":
            # Anthropic's lowest thinking setting; MiniMax has no equivalent,
            # and its closest is plain adaptive thinking.
            thinking_type = "adaptive"
        elif thinking_type not in ("adaptive", "disabled"):
            thinking_type = None

        if thinking_type is None or (thinking_type == "disabled" and always_thinks(model)):
            body.pop("thinking", None)
        else:
            body["thinking"] = {"type": thinking_type}
    elif thinking is not None:
        body.pop("thinking", None)

    output_config = body.get("output_config")
    if isinstance(output_config, dict):
        raw_effort = output_config.get("effort")
        if raw_effort is None:
            # MiniMax's output_config only carries the effort; a structured-
            # output format has no MiniMax spelling and would be rejected.
            body.pop("output_config", None)
        else:
            effort = normalize_effort(raw_effort)
            if effort is None:
                body.pop("output_config", None)
                if isinstance(raw_effort, str) and raw_effort.lower() == "none":
                    _disable_thinking(body, model)
            else:
                body["output_config"] = {"effort": effort}

    return body


def normalize_responses_reasoning(body: dict[str, Any], model: str | None) -> dict[str, Any]:
    """Normalize a Responses body's reasoning control for MiniMax.

    The Responses wire has no ``thinking`` field: the depth rides
    ``reasoning.effort``, whose enum is MiniMax's own (``minimal`` … ``max``)
    plus ``none``. ``none`` means "no reasoning" and is a valid value for
    MiniMax-M3 and the M2.x series, but MiniMax-M3.1-Flash always thinks and
    answers HTTP 400 to it — there the effort is dropped and the model's
    default depth applies.

    The body is mutated in place and returned.
    """
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        return body
    effort = reasoning.get("effort")
    if isinstance(effort, str) and effort.lower() == "none" and always_thinks(model):
        reasoning.pop("effort", None)
        if not reasoning:
            body.pop("reasoning", None)
    return body


def normalize_chat_thinking(body: dict[str, Any], model: str | None) -> dict[str, Any]:
    """Normalize the thinking parameters of a Chat Completions body for MiniMax.

    MiniMax's chat wire knows ``thinking: {"type": "disabled" | "adaptive"}``
    and ``reasoning_effort`` in ``low|medium|high|xhigh|max``. It answers HTTP
    400 to ``reasoning_effort: "none"`` and to any effort outside that enum,
    so an Anthropic client's "no reasoning" is expressed as
    ``thinking: {"type": "disabled"}`` instead.

    The body is mutated in place and returned.
    """
    thinking = body.get("thinking")
    thinking_type: str | None = None
    if isinstance(thinking, dict):
        raw_type = thinking.get("type")
        if raw_type in ("enabled", "between_tools"):
            # Anthropic's "thinking on" — MiniMax spells it ``adaptive``.
            thinking_type = "adaptive"
        elif raw_type in ("adaptive", "disabled"):
            thinking_type = raw_type

    effort = body.get("reasoning_effort")
    effort_is_none = isinstance(effort, str) and effort.lower() == "none"
    if effort is not None:
        normalized_effort = normalize_effort(effort)
        if normalized_effort is None:
            body.pop("reasoning_effort", None)
        else:
            body["reasoning_effort"] = normalized_effort

    if effort_is_none:
        if thinking_type == "adaptive":
            # The client asked for thinking on and for no depth: keep thinking
            # on and drop the effort MiniMax cannot spell.
            pass
        else:
            thinking_type = "disabled"

    if thinking_type is None or thinking_type == "disabled" and always_thinks(model):
        body.pop("thinking", None)
    else:
        body["thinking"] = {"type": thinking_type}
    return body


__all__ = [
    "ALWAYS_THINKING_MODEL_MARKERS",
    "MINIMAX_EFFORTS",
    "always_thinks",
    "normalize_chat_thinking",
    "normalize_effort",
    "normalize_messages_thinking",
    "normalize_responses_reasoning",
    "normalize_service_tier",
]
