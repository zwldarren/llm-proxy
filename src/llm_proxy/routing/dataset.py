"""Export of routing training data — feedback labels and judge shadow verdicts.

The two corpora the third content-reading signal will be trained on
(ADR-0029) already exist as operational records: explicit user feedback
(``ok``/``weak``/``strong`` on a routed request) and the judge's recorded
verdicts (the shadow ones especially). This module renders both as JSONL
rows — one JSON object per line, prompt text truncated to the same budget the
judge itself sees — so a training run reads them without touching the log
store's schema. Pure functions: the queries live in ``LogRepository``, the
endpoint in ``api.routers.routing``.
"""

import json
from collections.abc import Iterable
from typing import Any

from llm_proxy.routing.features import _message_text

__all__ = [
    "FEEDBACK_SIGNALS",
    "PROMPT_MAX_CHARS",
    "feedback_export_row",
    "jsonl_line",
    "judge_shadow_export_row",
    "to_jsonl",
]

#: The judge's ask budget: a prompt longer than this is truncated head-only
#: for the export (the judge's own excerpt keeps a head *and* a tail —
#: ADR-0029), so a training run reads at most what the router saw.
PROMPT_MAX_CHARS = 8000

#: Signals a feedback row may carry (mirrors ``ModelExperienceStore``).
FEEDBACK_SIGNALS = frozenset({"ok", "weak", "strong"})


def _truncate(text: str, limit: int = PROMPT_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def user_prompt_from_body(request_body: Any) -> str:
    """The routed prompt: the user-turn text of a stored request body.

    Tolerant of the wire formats the proxy stores (OpenAI ``messages``,
    Anthropic-style parts): non-user roles are skipped, structured content is
    flattened, and the result is truncated to :data:`PROMPT_MAX_CHARS`.
    """
    if not isinstance(request_body, dict):
        return ""
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_text(message.get("content")).strip()
        if text:
            parts.append(text)
    return _truncate("\n".join(parts))


def feedback_export_row(
    *,
    feedback: dict[str, Any],
    log: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """One training row per explicit feedback label.

    ``feedback`` is a ``feedback_records`` row shape (request_id, signal,
    model, created_at); ``log`` the routed request's log (``request_body`` and
    ``log_metadata``), when it still exists. A row without a routable prompt is
    skipped — the label needs its text to be trainable.
    """
    signal = str(feedback.get("signal") or "")
    if signal not in FEEDBACK_SIGNALS:
        return None
    prompt = user_prompt_from_body((log or {}).get("request_body"))
    if not prompt:
        return None
    metadata = (log or {}).get("log_metadata") or {}
    routing = metadata.get("routing") or {}
    return {
        "request_id": feedback.get("request_id"),
        "prompt": prompt,
        "model": feedback.get("model"),
        "signal": signal,
        "tier": routing.get("tier"),
        "complexity": routing.get("complexity"),
        "confidence": routing.get("confidence"),
        "created_at": feedback.get("created_at"),
    }


def judge_shadow_export_row(
    *,
    judge_log: dict[str, Any],
    parent_log: dict[str, Any] | None,
    signal: str | None = None,
) -> dict[str, Any] | None:
    """One training row per recorded judge verdict.

    ``judge_log`` is the judge call's own row (``log_type='judge'``); its
    metadata carries the flattened consultation
    (:meth:`JudgeConsultation.as_meta`): the verdict, its distribution, and
    whether it owned the tier. ``parent_log`` supplies the routed side — the
    ensemble's tier, complexity and confidence, and the prompt the excerpt was
    built from. The user's feedback signal on the parent request — the label a
    trainer needs — rides along when it exists.
    """
    metadata = judge_log.get("log_metadata") or {}
    judge = metadata.get("judge")
    if not isinstance(judge, dict):
        return None
    parent_metadata = (parent_log or {}).get("log_metadata") or {}
    routing = parent_metadata.get("routing") or {}
    prompt = user_prompt_from_body((parent_log or {}).get("request_body"))
    if not prompt:
        return None
    return {
        "request_id": metadata.get("parent_request_id"),
        "prompt": prompt,
        "resolved_model": routing.get("resolved_model") or metadata.get("resolved_model"),
        "ensemble_tier": routing.get("tier"),
        "ensemble_complexity": routing.get("complexity"),
        "ensemble_confidence": routing.get("confidence"),
        "judge": {
            "tier": judge.get("tier"),
            "probabilities": judge.get("probabilities"),
            "confidence": judge.get("confidence"),
            "escalation_probability": judge.get("escalation_probability"),
            "shadow": judge.get("shadow"),
            "owns_tier": judge.get("owns_tier"),
            "reason": judge.get("gate"),
            "rubric_version": judge.get("rubric_version"),
        },
        "signal": signal if signal in FEEDBACK_SIGNALS else None,
    }


def jsonl_line(row: dict[str, Any]) -> str:
    """One JSONL line for a row, newline-terminated and never ASCII-escaped."""
    return json.dumps(row, ensure_ascii=False) + "\n"


def to_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    """Render rows as JSONL (one JSON object per line, newline-terminated).

    Materializes the whole document; a caller that must not hold an export in
    memory serializes through :func:`jsonl_line` line by line instead.
    """
    return "".join(jsonl_line(row) for row in rows)
