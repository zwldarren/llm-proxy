"""Prompt sources for the label flywheel.

Two sources feed the same pipeline:

* an eval corpus (authored-tier JSONL) — the bootstrap source while served
  traffic is thin, and the validation set for label quality;
* ``request_logs`` — the production source. The sampler reads the SQLite log
  store read-only and never touches the serving path.
"""

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")

#: Prompts above this many characters are skipped: generation cost becomes
#: unbounded and the pairwise comparison below it is dominated by context
#: handling rather than by answer quality.
DEFAULT_MAX_PROMPT_CHARS = 200_000

#: A user turn shorter than this carries nothing to label.
MIN_ASK_CHARS = 15


@dataclass(frozen=True)
class PromptItem:
    """One candidate prompt to label."""

    source: str
    source_id: str
    messages: list[dict[str, Any]]
    authored_tier: str | None = None


def _message_text(message: dict[str, Any]) -> str:
    """Flatten one message's content to text, ignoring non-text parts."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "\n".join(parts)


def _has_image(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict)
        and part.get("type") in {"image_url", "image", "input_image", "inline_data"}
        for part in content
    )


def _prompt_is_labelable(messages: list[dict[str, Any]], *, max_chars: int) -> bool:
    """Shared eligibility filter for both sources."""
    if not messages:
        return False
    total = 0
    user_text = ""
    for message in messages:
        if not isinstance(message, dict) or _has_image(message):
            return False
        total += len(_message_text(message))
        if message.get("role") == "user":
            user_text = _message_text(message)
    return not (total > max_chars or len(user_text.strip()) < MIN_ASK_CHARS)


def _dedupe_key(messages: list[dict[str, Any]]) -> str:
    """Hash of the whitespace-normalized user turns — near-duplicate guard."""
    turns = [
        _WHITESPACE_RE.sub(" ", _message_text(m).strip().lower())
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    return hashlib.sha256("\n\u241f\n".join(turns).encode("utf-8")).hexdigest()


def iter_corpus_prompts(path: Path) -> Iterator[PromptItem]:
    """Read an authored-tier eval corpus (``id``/``messages``/``expected_tier``)."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                print(f"[corpus] skipping malformed line {line_no} of {path}")
                continue
            messages = case.get("messages")
            if not isinstance(messages, list) or not _prompt_is_labelable(
                messages, max_chars=DEFAULT_MAX_PROMPT_CHARS
            ):
                continue
            yield PromptItem(
                source=f"corpus:{Path(path).stem}",
                source_id=str(case.get("id", f"line-{line_no}")),
                messages=messages,
                authored_tier=case.get("expected_tier"),
            )


def iter_request_log_prompts(
    db_path: Path,
    *,
    limit: int = 1000,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    since_days: int | None = None,
) -> Iterator[PromptItem]:
    """Sample labelable chat requests from ``request_logs``, newest first.

    Read-only: the connection opens with ``mode=ro`` and never writes. Rows are
    filtered to completed text chat completions with bodies logged (the
    ``log_input_output`` switch replaces bodies with a marker when off), then
    deduplicated on user-turn content so one prompt answered many times labels
    once.
    """
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        since_clause = ""
        params: list[Any] = []
        if since_days is not None:
            since_clause = "AND timestamp >= strftime('%s','now', ?)"
            params.append(f"-{since_days} days")
        params.append(limit * 4)  # overfetch: filters and dedupe drop rows
        rows = connection.execute(
            "SELECT request_id, request_body FROM request_logs "
            "WHERE endpoint = '/v1/chat/completions' "
            "AND status_code BETWEEN 200 AND 299 "
            "AND request_body IS NOT NULL "
            "AND request_body NOT LIKE '%BODIES_DISABLED%' "
            "AND json_type(request_body, '$.messages') = 'array' "
            + since_clause
            + " ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
    finally:
        connection.close()

    seen: set[str] = set()
    emitted = 0
    for request_id, body_text in rows:
        if emitted >= limit:
            return
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError, TypeError:
            continue
        messages = body.get("messages")
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            continue
        if not _prompt_is_labelable(messages, max_chars=max_prompt_chars):
            continue
        key = _dedupe_key(messages)
        if key in seen:
            continue
        seen.add(key)
        emitted += 1
        yield PromptItem(
            source="request_logs",
            source_id=str(request_id),
            messages=messages,
        )
