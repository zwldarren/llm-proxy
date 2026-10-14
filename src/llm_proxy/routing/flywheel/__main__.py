"""CLI for the label flywheel: ``python -m llm_proxy.routing.flywheel``.

Subcommands:

* ``sample`` — collect prompts into a prompts JSONL (from eval corpora and/or
  the request-log store).
* ``label``  — generate per-role answers through a proxy deployment, pairwise-
  judge them, and stream label records to a labels JSONL.
* ``train``  — train the classifier on labeled records, evaluate against the
  shipped model on a seeded holdout, and write the hot-swap artifact only when
  the gate passes.
"""

import argparse
from pathlib import Path

import platformdirs

from llm_proxy.routing.flywheel.label import ROLES, dump_items, load_items, run_labeling
from llm_proxy.routing.flywheel.sources import iter_corpus_prompts, iter_request_log_prompts
from llm_proxy.routing.flywheel.store import LabelStore
from llm_proxy.routing.flywheel.train import render_report, train_and_gate


def _default_workdir() -> Path:
    return Path(platformdirs.user_data_dir("llm-proxy", "llm-proxy")) / "routing-flywheel"


def _default_db_path() -> Path:
    """The SQLite log store path, honoring the same env overrides the server uses."""
    import os

    from llm_proxy.database.db_config import DEFAULT_DB_PATH

    database_url = os.environ.get("DATABASE_URL")
    # Mirror get_db_url(): DATABASE_URL wins over LLM_PROXY_DB_PATH, and a
    # sqlite DATABASE_URL carries the store path. Checking LLM_PROXY_DB_PATH
    # first (as this once did) silently sampled the default database whenever
    # both were set, or whenever a custom sqlite URL path was configured.
    if database_url:
        if not database_url.startswith("sqlite"):
            raise SystemExit(
                "--from-logs supports the SQLite log store only; "
                f"DATABASE_URL is {database_url.split(':')[0]} — pass --db explicitly."
            )
        # ``sqlite+aiosqlite:///<path>`` / ``sqlite:///<path>``: the three
        # slashes precede an absolute path.
        path = database_url.split("///", 1)[-1]
        if path:
            return Path(path)
    override = os.environ.get("LLM_PROXY_DB_PATH")
    if override:
        return Path(override)
    return DEFAULT_DB_PATH


def cmd_sample(args: argparse.Namespace) -> int:
    items: list = []
    for corpus in args.corpus or []:
        corpus_items = list(iter_corpus_prompts(Path(corpus)))
        print(f"[sample] corpus {corpus}: {len(corpus_items)} prompts")
        items.extend(corpus_items)
    if args.from_logs:
        db_path = Path(args.db) if args.db else _default_db_path()
        log_items = list(
            iter_request_log_prompts(
                db_path,
                limit=args.limit,
                max_prompt_chars=args.max_prompt_chars,
                since_days=args.since_days,
            )
        )
        print(f"[sample] request_logs {db_path}: {len(log_items)} prompts")
        items.extend(log_items)
    dump_items(items, args.out)
    print(f"[sample] wrote {len(items)} prompts -> {args.out}")
    return 0


def _parse_efforts(raw: str | None) -> dict[str, str]:
    """Parse ``--efforts "weak=low,balanced=high,strong=max"`` into a dict."""
    if not raw:
        return {}
    efforts: dict[str, str] = {}
    for part in raw.split(","):
        role, _, effort = part.partition("=")
        role, effort = role.strip(), effort.strip()
        if not role or not effort:
            raise SystemExit(f"--efforts: expected role=effort pairs, got {part!r}")
        if role not in ROLES:
            raise SystemExit(f"--efforts: unknown role {role!r} (known: {', '.join(ROLES)})")
        efforts[role] = effort
    return efforts


def cmd_label(args: argparse.Namespace) -> int:
    import asyncio

    items = load_items(args.prompts)
    if args.limit:
        items = items[: args.limit]
    models = {"strong": args.strong, "balanced": args.balanced, "weak": args.weak}
    efforts = _parse_efforts(args.efforts)
    summary = asyncio.run(
        run_labeling(
            items,
            base_url=args.base_url,
            api_key=args.api_key,
            models=models,
            efforts=efforts,
            judge_model=args.judge_model,
            judge_timeout=args.judge_timeout,
            generation_timeout=args.generation_timeout,
            concurrency=args.concurrency,
            judge_concurrency=args.judge_concurrency,
            out_path=args.out,
            resume=not args.no_resume,
        )
    )
    print(
        f"[label] done: {summary['labeled']} labeled, {summary['skipped']} skipped, "
        f"{summary['errors']} with errors, verdicts={summary['verdict_tier']}, "
        f"{summary['elapsed_s']}s -> {summary['output']}"
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    records = LabelStore(args.labels).read_all()
    print(f"[train] loaded {len(records)} records from {args.labels}")
    report = train_and_gate(
        records,
        holdout_fraction=args.holdout,
        seed=args.seed,
        epochs=args.epochs,
        warm_start=not args.fresh,
        out_model=Path(args.out_model) if args.out_model else None,
        out_calibration=Path(args.out_calibration) if args.out_calibration else None,
    )
    print(render_report(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llm_proxy.routing.flywheel",
        description="Label flywheel: sample, pairwise-label, and train the routing classifier.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="collect candidate prompts into a prompts JSONL")
    sample.add_argument("--corpus", action="append", help="authored-tier corpus JSONL (repeatable)")
    sample.add_argument("--from-logs", action="store_true", help="also sample from request_logs")
    sample.add_argument("--db", help="request-log SQLite path (default: the configured store)")
    sample.add_argument("--limit", type=int, default=1000, help="max prompts from request_logs")
    sample.add_argument("--since-days", type=int, help="only logs from the last N days")
    sample.add_argument("--max-prompt-chars", type=int, default=200_000)
    sample.add_argument("--out", required=True, help="prompts JSONL output path")
    sample.set_defaults(func=cmd_sample)

    label = sub.add_parser("label", help="generate answers, pairwise-judge, write labels")
    label.add_argument("--prompts", required=True, help="prompts JSONL from `sample`")
    label.add_argument("--base-url", required=True, help="proxy deployment base URL")
    label.add_argument("--api-key", required=True, help="proxy API key")
    label.add_argument("--strong", required=True, help="concrete model name for the strong role")
    label.add_argument(
        "--balanced", required=True, help="concrete model name for the balanced role"
    )
    label.add_argument("--weak", required=True, help="concrete model name for the weak role")
    label.add_argument("--judge-model", required=True, help="System One model used for judging")
    label.add_argument(
        "--efforts",
        help="per-role reasoning effort, e.g. 'weak=low,balanced=high,strong=max' "
        "— a same-model effort ladder; omit for plain model roles",
    )
    label.add_argument("--judge-timeout", type=float, default=60.0)
    label.add_argument("--generation-timeout", type=float, default=120.0)
    label.add_argument("--concurrency", type=int, default=4, help="concurrent generation calls")
    label.add_argument("--judge-concurrency", type=int, default=2, help="concurrent judge calls")
    label.add_argument("--limit", type=int, help="cap the number of prompts labeled")
    label.add_argument("--no-resume", action="store_true", help="relabel even if already present")
    label.add_argument("--out", required=True, help="labels JSONL output path")
    label.set_defaults(func=cmd_label)

    train = sub.add_parser("train", help="train + evaluate + gate the classifier artifact")
    train.add_argument("--labels", required=True, help="labels JSONL from `label`")
    train.add_argument("--holdout", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=20261014)
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--fresh", action="store_true", help="train from zero, not warm start")
    train.add_argument("--out-model", help="model.json path to write when the gate passes")
    train.add_argument(
        "--out-calibration", help="calibration JSON path to write when the gate passes"
    )
    train.set_defaults(func=cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
