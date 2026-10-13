"""Label-record schema and the streaming JSONL store.

One record per prompt. The store is append-only JSONL so a long labeling run
keeps partial progress and ``--resume`` can skip what already landed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson


@dataclass
class LabelRecord:
    """One labeled prompt: what was asked, what the roles answered, and the verdict.

    ``verdict_tier`` is ``None`` when the pairwise chain was inconsistent or a
    role failed to answer — such records stay on disk for inspection but are
    skipped by training.
    """

    source: str
    source_id: str
    messages: list[dict[str, Any]]
    models: dict[str, str] = field(default_factory=dict)
    efforts: dict[str, str] = field(default_factory=dict)
    answers: dict[str, str] = field(default_factory=dict)
    votes: list[dict[str, Any]] = field(default_factory=list)
    verdict_tier: str | None = None
    verdict_chain: str | None = None
    ensemble_tier: str | None = None
    ensemble_confidence_raw: float | None = None
    ensemble_signals: tuple[str, ...] = ()
    judge_model: str | None = None
    authored_tier: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "messages": self.messages,
            "models": self.models,
            "efforts": self.efforts,
            "answers": self.answers,
            "votes": self.votes,
            "verdict_tier": self.verdict_tier,
            "verdict_chain": self.verdict_chain,
            "ensemble_tier": self.ensemble_tier,
            "ensemble_confidence_raw": self.ensemble_confidence_raw,
            "ensemble_signals": list(self.ensemble_signals),
            "judge_model": self.judge_model,
            "authored_tier": self.authored_tier,
            "errors": self.errors,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> LabelRecord:
        return cls(
            source=str(data.get("source", "")),
            source_id=str(data.get("source_id", "")),
            messages=list(data.get("messages") or []),
            models=dict(data.get("models") or {}),
            efforts=dict(data.get("efforts") or {}),
            answers=dict(data.get("answers") or {}),
            votes=list(data.get("votes") or []),
            verdict_tier=data.get("verdict_tier"),
            verdict_chain=data.get("verdict_chain"),
            ensemble_tier=data.get("ensemble_tier"),
            ensemble_confidence_raw=data.get("ensemble_confidence_raw"),
            ensemble_signals=tuple(data.get("ensemble_signals") or ()),
            judge_model=data.get("judge_model"),
            authored_tier=data.get("authored_tier"),
            errors=list(data.get("errors") or []),
        )


class LabelStore:
    """Append-only JSONL store of :class:`LabelRecord` rows."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, record: LabelRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(orjson.dumps(record.to_json()))
            handle.write(b"\n")

    def read_all(self) -> list[LabelRecord]:
        if not self.path.exists():
            return []
        records: list[LabelRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(LabelRecord.from_json(orjson.loads(line)))
            except Exception:  # noqa: BLE001 - a torn line must not sink the file
                continue
        return records

    def seen_source_ids(self) -> set[str]:
        return {f"{r.source}:{r.source_id}" for r in self.read_all()}
