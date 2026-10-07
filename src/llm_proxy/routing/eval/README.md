# Routing evaluation rig

Development-only harness for measuring **routing tier accuracy** — how often the
router's tier matches an authored one — for the ensemble, for a routing judge, and
for the gated combination the judge ships behind (see [ADR-0018](../../../../../docs/adr/0018-routing-judge-owns-ambiguous-band.md)).

It is not a product surface: no API route, no console script, no documented
behavior. It lives inside the package so `ty` type-checks it and `pytest` covers it.

## What it measures, and what it does not

It measures **classification**: did the router put this request in the tier the
author says it belongs in. It does not measure whether the chosen model answered
well, and it cannot: a proxy never sees task success. Downstream quality needs
shadow mode plus outcome proxies on real traffic.

## Corpus format

One JSON object per line (`cases.jsonl`):

```json
{"id": "short-lookup", "subset": "short", "expected_tier": "SIMPLE", "rationale": "Why this tier.", "messages": [{"role": "user", "content": "..."}]}
```

- `subset` — one of `short`, `long`, `followup`, `tool-context`, `boundary`.
  `boundary` holds the cases where two tiers are defensible; they decide whether
  the judge earns its place, so keep them honest.
- `expected_tier` — `SIMPLE` / `MEDIUM` / `COMPLEX`, authored against the rubric in
  `llm_proxy.routing.judge.TIER_RUBRICS`.
- `rationale` — required in spirit if not by the loader: a label without a reason
  cannot be reviewed or overturned later.

The packaged cases are **seed examples**, not a corpus. Author 80–100 across the
subsets before trusting any number: one author, one pass, no independent review —
the same caveat the reference benchmark publishes about its own labels.

Two rules when extending it:

- Never add a case whose text (or the labels) came from Signal C's training data —
  the 621 seeds in `routing/assets/`. Leakage there would make the embedding signal
  look better than it is.
- Keep the boundary subset hard. A corpus that only contains easy labels flatters
  every decision path equally and tells you nothing.

## Running

```bash
# Ensemble baseline, no judge, signals A+B+C
uv run python -m llm_proxy.routing.eval --embedding --out ./

# With a local judge (Ollama serves /v1/systemone from local models only)
ollama pull tev1:0.8b
uv run python -m llm_proxy.routing.eval --embedding --judge ollama --judge-model tev1:0.8b --out ./

# With Jev, the reference judge ADR-0018 compares its numbers against
export OPENROUTER_API_KEY=sk-or-...
uv run python -m llm_proxy.routing.eval --embedding --judge openrouter --out ./

# Seeded repetitions (three is the default) and explicit bandit seed
uv run python -m llm_proxy.routing.eval --repeats 3 --seed 20261006 --out ./
```

`--out` writes `metrics.json` and `report.md`; the report always prints.

`--judge-model` and `--judge-url` default per transport (`tev1:0.8b` on
`http://localhost:11434`, `~typesafe/jev-latest` on OpenRouter). The judge model
name is **upstream**-specific: the rig calls HTTP directly, so nothing maps an
internal model name for it. `--judge-api-key` overrides `OPENROUTER_API_KEY`.

## Reading the report

- **Match rate** is over *decided* calls. A judge **abstention** (the `ambiguous`
  option, a timeout, a failure) is reported in its own column and excluded: the
  escape hatch is a designed answer, and the ensemble decides there.
- **Every match rate carries a 95% bootstrap interval over whole cases** (the
  reference protocol's convention). The resampling unit is the *case*, never the
  individual repeat row, so one case's repeats always travel together; intervals
  are seeded, so re-running reports identical bounds. Read the interval before
  declaring a gate or a judge better: on a small corpus a two-point spread can be
  noise.
- **Under vs over** are never averaged. Under-tiering serves a hard request from a
  cheaper band than it needs — the direction that costs quality, and the direction
  every mismatch went in the reference benchmark. Over-tiering only costs money.
- **Judge vs ensemble agreement** is not accuracy. A judge that agrees with a wrong
  ensemble scores perfectly, which is why every number here is against an authored
  label and never against another router.
- **Latency percentiles** are linear-interpolated over all calls, failures included.
  The gate's budget arithmetic depends on the judge's *own* p50, so read it before
  the tail.

## Why the gate matters to the numbers

The judge is only reached when the gate fires, and **which trigger is right is a measured
parameter, not a design constant** (ADR-0018). The rig reports every candidate trigger
side by side, each with the call rate it implies:

| gate | fires when |
| --- | --- |
| `confidence` | ensemble confidence < `--gate-threshold` |
| `confidence_q10/q25/q50` | ensemble confidence < the corpus' own 10th/25th/50th percentile |
| `band` | the ensemble's *complexity* lands in MEDIUM `[0.33, 0.67)` |
| `not_low_anchor` | complexity is anything but the obvious low anchor (`0.0`) |
| `always` | every gated-eligible request (the first-turn rule still applies) |

Do not assume a confidence threshold: read the reported confidence distribution first.
On the seed corpus this ensemble's confidence clustered at 0.82–0.89 — *including on the
cases it got wrong* — so a 0.55 gate fired on 12% of calls, all of them tool-loop steps
the first-turn rule excludes anyway. And because complexity snaps to the tier anchors
`{0.00, 0.40, 0.68, 0.90}`, a narrow band only ever catches the `0.40` anchor. Pick the
trigger from these rows, not from intuition.

`judge, every case` and `ensemble (no judge)` are the two extremes every gate sits
between; the best gate is whichever row lands closest to the judge's accuracy at an
acceptable call rate.

## Before you trust a latency number

Two effects will otherwise be mistaken for the judge's speed:

- **Cold start is inside the measurement.** A call cancelled at the deadline does not
  leave the model loaded, so a timeout shorter than the cold load makes *every* call time
  out and the judge never warms up: on this machine `tev1:0.8b` needs ~3.9 s to load and
  answer on CPU and ~0.24 s warm, which is why `--judge-timeout 3` reported 8/8 timeouts
  on the first run. Warm the model, or give the first call a longer timeout, before
  reading percentiles.
- **A deadline overshoots by its cancellation cost.** Measured locally, a 0.5 s deadline
  returned control at 0.55 s. Budget `deadline_s` plus that overhead, not the deadline
  alone.
