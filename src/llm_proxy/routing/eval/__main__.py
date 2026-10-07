"""Command line entry point for the offline routing evaluation rig.

    uv run python -m llm_proxy.routing.eval --embedding
    uv run python -m llm_proxy.routing.eval --judge ollama --judge-model tev1:0.8b --out ./

Development only: this is not installed as a console script and is not part of
the documented product surface (ADR-0018).
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

from llm_proxy.routing.eval.harness import (
    BOOTSTRAP_REPEATS,
    DEFAULT_CASES_PATH,
    DEFAULT_GATE_THRESHOLD,
    CaseResult,
    judge_state_for,
    load_cases,
    render_report,
    run_ensemble_leg,
    summarize,
)
from llm_proxy.routing.eval.judges import (
    OLLAMA_JUDGE_MODEL,
    OLLAMA_JUDGE_URL,
    OPENROUTER_JUDGE_MODEL,
    OPENROUTER_JUDGE_URL,
    JudgeTransport,
    OllamaSystemOneJudge,
    OpenRouterSystemOneJudge,
)
from llm_proxy.routing.types import RoutingMode


def _positive_int(value: str) -> int:
    """argparse type for a count that must be at least one."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {parsed}")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m llm_proxy.routing.eval",
        description="Measure routing tier accuracy against an authored corpus (ADR-0018).",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="JSONL corpus (default: the packaged seed cases)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Directory for metrics.json/report.md"
    )
    parser.add_argument("--mode", default="auto", choices=[m.value for m in RoutingMode])
    parser.add_argument(
        "--repeats", type=int, default=3, help="Seeded repetitions per case (the ADR uses three)"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=BOOTSTRAP_REPEATS,
        help="Bootstrap resamples per reported interval, over whole cases (0 disables)",
    )
    parser.add_argument("--seed", default="routing-eval", help="Seed for repeats and bandit draws")
    parser.add_argument(
        "--limit", type=_positive_int, default=None, help="Only run the first N cases"
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Enable the embedding signal (downloads/loads the BGE encoder on first use)",
    )
    parser.add_argument(
        "--judge",
        default="none",
        choices=["none", "ollama", "openrouter"],
        help="Judge transport to measure alongside the ensemble (openrouter measures Jev)",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=DEFAULT_GATE_THRESHOLD,
        help="Confidence below which the judge gate would fire (must be swept, not assumed)",
    )
    parser.add_argument(
        "--judge-url",
        default=None,
        help="Judge base URL (defaults per transport: Ollama localhost, OpenRouter api)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Upstream model id (defaults per transport: tev1:0.8b, ~typesafe/jev-latest)",
    )
    parser.add_argument(
        "--judge-api-key",
        default=None,
        help="Judge credential; falls back to OPENROUTER_API_KEY for the openrouter transport",
    )
    parser.add_argument("--judge-timeout", type=float, default=3.0)
    return parser.parse_args(argv)


async def _load_embedding_signal(enabled: bool) -> Any:
    if not enabled:
        return None
    from llm_proxy.routing.signals.embedding import get_embedding_signal

    signal = await get_embedding_signal(None)
    if signal is None:
        print("[routing-eval] embedding signal unavailable; measuring signals A+B only")
    return signal


def _build_judge(args: argparse.Namespace) -> JudgeTransport | None:
    """The transport ``--judge`` named, with per-transport defaults for url/model."""
    if args.judge == "none":
        return None
    if args.judge == "openrouter":
        return OpenRouterSystemOneJudge(
            base_url=args.judge_url or OPENROUTER_JUDGE_URL,
            model=args.judge_model or OPENROUTER_JUDGE_MODEL,
            timeout=args.judge_timeout,
            api_key=args.judge_api_key,
        )
    return OllamaSystemOneJudge(
        base_url=args.judge_url or OLLAMA_JUDGE_URL,
        model=args.judge_model or OLLAMA_JUDGE_MODEL,
        timeout=args.judge_timeout,
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the corpus and return the metrics document."""
    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    embedding_signal = await _load_embedding_signal(args.embedding)
    judge = _build_judge(args)
    mode = RoutingMode(args.mode)

    results: list[CaseResult] = []
    try:
        for repeat in range(max(1, args.repeats)):
            for case in cases:
                tier, complexity, confidence = run_ensemble_leg(
                    case,
                    mode=mode,
                    embedding_signal=embedding_signal,
                    rng=random.Random(f"{args.seed}:{case.id}:{repeat}"),
                )
                result = CaseResult(
                    case=case,
                    repeat=repeat,
                    ensemble_tier=tier,
                    ensemble_complexity=complexity,
                    ensemble_confidence=confidence,
                )
                if judge is not None:
                    result.judge_called = True
                    outcome = await judge.evaluate(judge_state_for(case))
                    result.judge_latency_ms = outcome.latency_ms
                    result.judge_error = outcome.error
                    result.judge_escalation = (
                        outcome.verdict.escalation_probability if outcome.verdict else None
                    )
                    if outcome.verdict is not None:
                        result.judge_tier = outcome.verdict.tier
                        result.judge_abstained = outcome.verdict.abstained
                        result.judge_probabilities = dict(outcome.verdict.probabilities)
                results.append(result)
    finally:
        if judge is not None:
            await judge.aclose()

    metrics = summarize(results, gate_threshold=args.gate_threshold, bootstrap=args.bootstrap)
    metrics["run"] = {
        "cases": str(args.cases),
        "mode": args.mode,
        "repeats": max(1, args.repeats),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "embedding_signal": embedding_signal is not None,
        "judge": judge.name if judge is not None else None,
        "judge_model": judge.model if judge is not None else None,
        "gate_threshold": args.gate_threshold,
    }
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Run the rig and print its report."""
    args = _parse_args(argv)
    try:
        metrics = asyncio.run(run(args))
    except (ValueError, OSError) as exc:
        # A transport that cannot be built (no judge credential) or a corpus path
        # that does not exist is a usage error, not a crash: the rig is a
        # development tool.
        print(f"[routing-eval] {exc}", file=sys.stderr)
        return 2
    report = render_report(metrics)
    print(report)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        (args.out / "report.md").write_text(report + "\n", encoding="utf-8")
        print(f"[routing-eval] wrote {args.out / 'metrics.json'} and {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
