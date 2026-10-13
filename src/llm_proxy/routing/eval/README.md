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

- Never add a case whose text (or the labels) came from any signal's training data.
  (The removed embedding signal trained on 621 seed prompts once vendored under
  `routing/assets/`; the rule predates that asset and still applies to whatever
  classifier data the structural model was fitted on.) Leakage would make the
  signal under test look better than it is.
- Keep the boundary subset hard. A corpus that only contains easy labels flatters
  every decision path equally and tells you nothing.

## Dataset-derived corpora

For a bigger corpus without hand-writing 100 cases, `corpus/` ships two
JSONL corpora derived from public HuggingFace datasets — `open-source-103`
(the baseline sweep) and `complex-situations-98` (long context, multi-turn
memory, long tool loops) — mapped onto the rubric with per-case rationales:

```bash
uv run python -m llm_proxy.routing.eval \
  --cases src/llm_proxy/routing/eval/corpus/open-source-103.jsonl --out ./
uv run python -m llm_proxy.routing.eval \
  --cases src/llm_proxy/routing/eval/corpus/complex-situations-98.jsonl --out ./
```

Read `corpus/README.md` for provenance, the per-subset mapping and the honesty
caveats (limited raters, unverifiable training overlap, the corpus skew) before
quoting numbers from it. ADR-0028 records the gate decision the first corpus
produced; its README section "Measurements on record" carries the numbers that
decision and the complex corpus run rest on.

## Running

Every judge run goes through this repo's own dev server — the reference
configuration ships `jev-latest` (a System One model via OpenRouter) on it,
registered with `supports_systemone`: `uv run llm-proxy-dev`, then point the
rig at it. `--judge ollama` is the transport's historical name (it is a plain
System One HTTP client, not a daemon pick): it calls `<url>/v1/systemone`.

```bash
# Ensemble baseline, no judge (signals A+B)
uv run python -m llm_proxy.routing.eval --out ./

# With the configured judge through this repo's dev server — an API key of that
# deployment is a credential:
uv run python -m llm_proxy.routing.eval \
  --judge ollama --judge-url http://127.0.0.1:9911 \
  --judge-model jev-latest --judge-api-key $PROXY_KEY --judge-timeout 60 --out ./

# Through a real deployment the same way: replace --judge-url/--judge-model.
# Directly against OpenRouter (needs a key in the environment)
export OPENROUTER_API_KEY=sk-or-...
uv run python -m llm_proxy.routing.eval --judge openrouter --out ./

# Seeded repetitions (three is the default) and explicit bandit seed
uv run python -m llm_proxy.routing.eval --repeats 3 --seed 20261014 --out ./

# Consumption bars (ADR-0029): add an operator policy to the reported table
uv run python -m llm_proxy.routing.eval --apply-min-confidence 0.9 --apply-min-escalation 0.5 --out ./
```

`--out` writes `metrics.json` and `report.md`; the report always prints.

`--judge-model` and `--judge-url` default per transport (`~typesafe/jev-latest`
on OpenRouter). The judge model name is **upstream**-specific: the rig calls
HTTP directly, so nothing maps an internal model name for it — route through
the proxy to reuse its per-model mapping. `--judge-api-key` is sent as Bearer
credentials on either transport.

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
On the recorded seed run (24 × 3 repeats, seed 20261014) the repaired ensemble
matched 6 of 24 rows; its confidence still separates its own hits (0.77 on every
correct row) from its misses (0.39–0.67), but a 0.55 gate fired on 3 of 24 calls
and moved nothing beyond its own cost, and the MEDIUM-band trigger caught 9 of 24
rows for +1 case over the ensemble's own 25%. With the ensemble repaired, no gate
predicate met the bar for shipping on-by-default — see ADR-0029 for the decision
arithmetic. Pick the trigger from these rows, not from intuition.

`judge, every case` and `ensemble (no judge)` are the two extremes every gate sits
between; the best gate is whichever row lands closest to the judge's accuracy at an
acceptable call rate.

## Verdict consumption (ADR-0029)

A gate decides which turns are *consulted*; a consumption policy decides which
*verdicts are obeyed*. The rig reports the shipped rule
(`routing.judge.policy.verdict_consumable`) applied to every judged row, side by
side with the operator's bars:

| policy | verdict obeyed when |
| --- | --- |
| `obey-every-verdict` | always (the pre-amendment behavior, and the shipped default) |
| `conf>=0.9` | the verdict's own probability mass on the tier it chose reaches 0.9 |
| `esc>=0.5` | its escalation probability reaches 0.5 |
| `conf>=0.9 or esc>=0.5` | either bar — an operator's `--apply-*` values join here |

Every judged row was paid for — the call happens before the filter — so the
*apply rate* is how often the verdict replaced the ensemble's tier, not a call
rate. The `match` column is the **composite** the proxy would have scored: a
held verdict falls back to the ensemble's tier and stays decided; nothing is
excluded. The accuracy over the obeyed verdicts alone is reported separately
(`applied_overall` in `metrics.json`) — that is the bar's quality read.

Read the consumption table *against* the gate table: on the recorded corpora no
consultation gate beats the ensemble, while the measured bars beat it by ~13
points on their respective corpus (`conf>=0.9` on `open-source-103`, `esc>=0.5`
on `complex-situations-98`) and are neutral cross-wise. The bars ship off
(`apply_min_confidence`/`apply_min_escalation` unset); sweep them here over your
own shadow data before enabling — thresholds picked on a recorded corpus are a
starting point, not a constant.

## Before you trust a latency number

Two effects will otherwise be mistaken for the judge's speed:

- **Cold start is inside the measurement.** A call cancelled at the deadline does not
  leave the model loaded, so a timeout shorter than the cold load makes *every* call time
  out and the judge never warms up. The proxy warms the judge model at startup and on
  config changes, but a remote model can still cold-start on the upstream: give the first
  call a longer `--judge-timeout` before reading percentiles.
- **A deadline overshoots by its cancellation cost.** Budget the round trip, not the
  configured `deadline_s` alone.

## Cost–quality readout (CPT / APGR)

Beside the match-rate tables, the report renders RouteLLM's cost–quality
readout (see [RouteLLM](https://arxiv.org/abs/2406.18665)) over a binary
strong/weak collapse of the three-band ladder: the strong lane is the COMPLEX
band, the weak lane SIMPLE/MEDIUM. Serving at or above the authored tier keeps
quality intact (over-tiering is money, not quality); each rank below loses
half.

- **The curve** sweeps a threshold over the ensemble's continuous complexity,
  escalating the cases above it. It traces every (strong-call share, quality)
  point a threshold gate on this ensemble can reach.
- **CPT(x%)** is the smallest strong-call share on that curve recovering x% of
  the always-strong minus always-weak quality gap — the shipped readouts are
  CPT(50%) and CPT(80%). The random-router baseline recovers x% at a call
  share of x, so a CPT *above* the level means the score is a weak escalation
  predictor on this corpus.
- **APGR** is the area under the Pareto-gain curve against the random diagonal,
  normalized to the oracle: 0 ≈ random, 1 ≈ perfect. (A discrete or tiny corpus
  leaves interpolation residue — read the seed corpus' numbers as the
  consistency check they are.)
- **Paths** — the shipped ensemble, the judge-everywhere leg, every candidate
  gate and consumption policy — report where each sits on the (share,
  recovery) plane at its own call rate.

The readout needs label variety: a corpus with a single authored tier has no
gap and reports `None` everywhere.
