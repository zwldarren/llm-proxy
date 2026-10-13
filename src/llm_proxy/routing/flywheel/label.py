"""Labeling driver: generate per-role answers, judge the pairs, store records.

Offline only — this module never runs on the request path. Each prompt flows
through: three generations (weak / balanced / strong), four pairwise judge
calls (two pairs × two presentation orders), the verdict resolution, and one
in-process read of the current ensemble for later calibration re-fitting.
"""

import asyncio
import time
from pathlib import Path
from typing import Any

import orjson

from llm_proxy.routing.decision.calibration import PlattCalibrator
from llm_proxy.routing.flywheel.generate import GenerationClient, GenerationOutcome
from llm_proxy.routing.flywheel.pairs import (
    PAIR_BALANCED_VS_WEAK,
    PAIR_STRONG_VS_BALANCED,
    PairwiseJudge,
    VoteOutcome,
    build_pairwise_state,
    resolve_pair,
    resolve_verdict,
)
from llm_proxy.routing.flywheel.sources import PromptItem
from llm_proxy.routing.flywheel.store import LabelRecord, LabelStore

#: Presentation roles, ordered strong -> weak. The concrete model per role is
#: a CLI argument; these names are the record schema's vocabulary.
ROLES = ("strong", "balanced", "weak")

_ANSWER_STORE_CHARS = 8000

#: Wall-clock ceiling for one generation call. Generations on very long
#: prompts must fail fast rather than hold a concurrency slot indefinitely.
_GENERATION_DEADLINE_S = 240.0


def _truncate_answer(text: str) -> str:
    return text if len(text) <= _ANSWER_STORE_CHARS else text[:_ANSWER_STORE_CHARS] + "…"


def _offline_ensemble_read(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Read the current ensemble on this prompt, with identity calibration.

    The ensemble's confidence is Platt-calibrated on the request path; the
    flywheel needs the raw value to re-fit the temperature later, so the
    process-local calibrator is swapped for the identity before classifying.
    A failure degrades to an absent read, never to a lost label.
    """
    import llm_proxy.routing.api as routing_api

    saved = routing_api._calibrator
    routing_api._calibrator = PlattCalibrator(temperature=1.0)
    try:
        result = routing_api.classify(messages)
    except Exception as exc:  # noqa: BLE001 - a failed read must not stop labeling
        return {"tier": None, "confidence": None, "signals": [], "error": str(exc)}
    finally:
        routing_api._calibrator = saved
    # The v2 ensemble scores four tiers (low / mid / mid_high / high); the
    # flywheel label space is the three serving tiers, so both upper bands
    # collapse to COMPLEX — the same collapse the serving decision applies.
    tier = {0: "SIMPLE", 1: "MEDIUM", 2: "COMPLEX", 3: "COMPLEX"}.get(result.v2.tier_id)
    return {
        "tier": tier,
        "confidence": result.v2.confidence,
        "signals": list(result.v2.signals_text),
    }


async def label_one(
    item: PromptItem,
    *,
    generator: GenerationClient,
    judge: PairwiseJudge,
    models: dict[str, str],
    efforts: dict[str, str],
    upstream_slots: asyncio.Semaphore,
    judge_slots: asyncio.Semaphore,
) -> LabelRecord:
    """Label one prompt end to end; never raises.

    ``upstream_slots`` bounds concurrent GENERATION calls and ``judge_slots``
    concurrent judge calls, not items in flight: the deployment's model
    server, not this client, sets the parallelism ceiling, and exceeding it
    turns into 429 storms amplified by the proxy's retry policy. Separate
    pools keep the fast judge from queueing behind slow generations.
    """
    record = LabelRecord(
        source=item.source,
        source_id=item.source_id,
        messages=item.messages,
        models=dict(models),
        efforts=dict(efforts),
        authored_tier=item.authored_tier,
    )

    async def _generate(role: str) -> GenerationOutcome:
        # A hard wall-clock deadline: httpx's per-read timeouts reset on every
        # chunk, so a slow-drip upstream can hold a call open far past the
        # nominal timeout. wait_for bounds the whole call.
        async with upstream_slots:
            try:
                return await asyncio.wait_for(
                    generator.generate(
                        item.messages,
                        model=models[role],
                        role=role,
                        reasoning_effort=efforts.get(role),
                    ),
                    timeout=_GENERATION_DEADLINE_S,
                )
            except TimeoutError:
                return GenerationOutcome(
                    role=role,
                    model=models[role],
                    error=f"generation exceeded {_GENERATION_DEADLINE_S}s",
                )

    async with asyncio.TaskGroup() as group:
        tasks = {role: group.create_task(_generate(role)) for role in ROLES}
    outcomes = {role: task.result() for role, task in tasks.items()}
    for role in ROLES:
        outcome = outcomes[role]
        record.answers[role] = _truncate_answer(outcome.text)
        if outcome.error:
            record.errors.append(f"generate/{role}: {outcome.error}")

    # Every pair needs both answers; a missing role voids its pair only.

    async def _judge(state: dict[str, Any]) -> VoteOutcome:
        async with judge_slots:
            return await judge.evaluate(state)

    async with asyncio.TaskGroup() as group:
        pair_jobs: dict[str, list[asyncio.Task[VoteOutcome]]] = {}
        for pair, (high, low) in (
            (PAIR_STRONG_VS_BALANCED, ("strong", "balanced")),
            (PAIR_BALANCED_VS_WEAK, ("balanced", "weak")),
        ):
            if not (outcomes[high].ok and outcomes[low].ok):
                continue
            # Position swap: the high role leads in the first call.
            pair_jobs[pair] = [
                group.create_task(
                    _judge(
                        build_pairwise_state(item.messages, outcomes[high].text, outcomes[low].text)
                    )
                ),
                group.create_task(
                    _judge(
                        build_pairwise_state(item.messages, outcomes[low].text, outcomes[high].text)
                    )
                ),
            ]
    resolved: dict[str, str] = {}
    for pair, jobs in pair_jobs.items():
        first, second = jobs[0].result(), jobs[1].result()
        for vote in (first, second):
            record.votes.append(
                {
                    "pair": pair,
                    "choice": vote.choice,
                    "probabilities": vote.probabilities,
                    "latency_ms": round(vote.latency_ms, 1),
                    "error": vote.error,
                }
            )
        if first.error:
            record.errors.append(f"judge/{pair}: {first.error}")
        elif second.error:
            record.errors.append(f"judge/{pair}: {second.error}")
        resolved[pair] = resolve_pair(first, second)
    tier, chain = resolve_verdict(resolved)
    record.verdict_tier = tier
    record.verdict_chain = chain
    record.judge_model = judge.model

    read = _offline_ensemble_read(item.messages)
    record.ensemble_tier = read.get("tier")
    record.ensemble_confidence_raw = read.get("confidence")
    record.ensemble_signals = tuple(read.get("signals") or ())
    if read.get("error"):
        record.errors.append(f"ensemble: {read['error']}")
    return record


async def run_labeling(
    items: list[PromptItem],
    *,
    base_url: str,
    api_key: str,
    models: dict[str, str],
    efforts: dict[str, str] | None = None,
    judge_model: str,
    judge_timeout: float = 60.0,
    generation_timeout: float = 120.0,
    concurrency: int = 4,
    judge_concurrency: int = 2,
    out_path,
    resume: bool = True,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Label every item, streaming records to ``out_path``. Returns a summary."""
    store = LabelStore(out_path)
    seen = store.seen_source_ids() if resume else set()
    pending = [item for item in items if f"{item.source}:{item.source_id}" not in seen]

    generator = GenerationClient(base_url=base_url, api_key=api_key, timeout=generation_timeout)
    judge = PairwiseJudge(
        url=f"{base_url.rstrip('/')}/v1/systemone",
        model=judge_model,
        timeout=judge_timeout,
        api_key=api_key,
    )
    upstream_slots = asyncio.Semaphore(concurrency)
    efforts = efforts or {}
    judge_slots = asyncio.Semaphore(judge_concurrency)
    counts = {"labeled": 0, "skipped": len(seen), "verdict_tier": {}, "errors": 0}
    started = time.perf_counter()
    try:
        # All items run concurrently; the per-call upstream semaphore is what
        # paces them — upstream generation is seconds-latent, so throughput
        # comes from overlapping prompts at the model server's own ceiling.
        tasks = [
            asyncio.create_task(
                label_one(
                    item,
                    generator=generator,
                    judge=judge,
                    models=models,
                    efforts=efforts,
                    upstream_slots=upstream_slots,
                    judge_slots=judge_slots,
                )
            )
            for item in pending
        ]
        for index, future in enumerate(asyncio.as_completed(tasks), start=1):
            record = await future
            store.append(record)
            counts["labeled"] += 1
            if record.verdict_tier:
                counts["verdict_tier"][record.verdict_tier] = (
                    counts["verdict_tier"].get(record.verdict_tier, 0) + 1
                )
            if record.errors:
                counts["errors"] += 1
            if progress_every and index % progress_every == 0:
                rate = index / max(time.perf_counter() - started, 1e-6)
                remaining = (len(pending) - index) / max(rate, 1e-6)
                print(
                    f"[label] {index}/{len(pending)} ({rate:.2f}/s, ~{remaining:.0f}s left) "
                    f"verdicts={counts['verdict_tier']}",
                    flush=True,
                )
    finally:
        await generator.aclose()
        await judge.aclose()
    counts["elapsed_s"] = round(time.perf_counter() - started, 1)
    counts["output"] = str(out_path)
    return counts


def dump_items(items: list[PromptItem], path: Path) -> None:
    """Write prompt items to a JSONL file (the ``sample`` subcommand's output)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for item in items:
            handle.write(
                orjson.dumps(
                    {
                        "source": item.source,
                        "source_id": item.source_id,
                        "messages": item.messages,
                        "authored_tier": item.authored_tier,
                    }
                )
            )
            handle.write(b"\n")


def load_items(path: Path) -> list[PromptItem]:
    """Read prompt items back from a prompts JSONL file."""
    path = Path(path)
    items: list[PromptItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = orjson.loads(line)
        items.append(
            PromptItem(
                source=str(data.get("source", "")),
                source_id=str(data.get("source_id", "")),
                messages=list(data.get("messages") or []),
                authored_tier=data.get("authored_tier"),
            )
        )
    return items
