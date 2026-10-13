# Label flywheel: preference labels retrain the routing classifier

## Status

Accepted (2026-10). First iteration run; artifacts archived, candidate model
NOT installed (install criteria below).

## Context

ADR-0035 closed the long-text blind spot with response-format signals and
stated the remaining under-tier buckets (short competition math, multi-turn
memory, hard multiple-choice synthesis) are "the label flywheel's job, not
another heuristic". The classifier behind Signal B is a vendored averaged
perceptron (67,920 updates, external UncommonRoute provenance) whose training
code never lived in this repo; its calibration temperature is a frozen asset;
and the measured corpora show its confidence is saturated and close to
orthogonal to what the serving stack actually needs: an estimate of *where
the cheap tier stops being enough*.

The measured environment also showed the cold-start problem: the dev store's
8,123 `request_logs` rows are mostly judge-shadow calls, with only 27 real
chat completions. The flywheel needs a bootstrap source, and the eval corpora
(HF-derived, 201 unique prompts) are real data with authored tiers to compare
against.

## Decision

**A four-stage offline pipeline, `python -m llm_proxy.routing.flywheel`
(`routing/flywheel/`), turns prompts into preference labels and labels into a
classifier candidate.**

1. **sample** — prompts from an authored-tier corpus and/or from
   `request_logs` (read-only SQLite; completed text chat completions with
   bodies logged; deduplicated on user-turn content; text-only; size-capped).
2. **label** — per prompt, generate one answer per role through a proxy
   deployment (weak / balanced / strong), then pairwise-judge two role pairs
   (strong vs balanced, balanced vs weak) with **position swap**: a pair
   resolves only when the swapped votes agree. The chain of pair outcomes is
   the tier label — `strong>balanced>weak` → COMPLEX, `=,>` → MEDIUM,
   `=,=` → SIMPLE, any contradiction or abstention → no label. Each record
   also stores an in-process read of the current ensemble (raw, identity-
   calibrated) for temperature re-fitting.
3. **train** — features are built exactly as the request path builds them
   (closed-form ask when detected, else the last user message, compacted
   system prompt, the bounded extractor of ADR-0035); a fresh or warm-started
   averaged perceptron trains on the labels; a seeded holdout compares the
   candidate against the shipped model; the artifact is written **only when
   the gate passes**. `fit_platt_from_evals` re-fits the temperature from
   (raw ensemble confidence, correct) pairs on the same holdout.
4. **install** — the sanctioned hot-swap path is
   `user_data_dir/llm-proxy/routing-assets/model_online.json` (the packaged
   asset is never touched; deleting the file reverts). Tests pin the packaged
   asset explicitly (`tests/routing/conftest.py`) so operator hot-swaps
   cannot change repo test outcomes.

Concurrency is bounded **per upstream call**, not per item: the model server
sets the parallelism ceiling, and exceeding it produced 429 storms amplified
by the proxy's retry policy (measured). Generation carries a hard wall-clock
deadline — httpx's per-read timeouts reset on every chunk and let a slow-drip
upstream hold a slot indefinitely.

## First-iteration results (2026-10-13, three distinct models in the
premium / balanced / weak roles, through the dev server, judge deployment)

- 201 prompts labeled in ~3.5 h (6 generation + 4 judge calls each, zero
  transport errors after the concurrency fix; 1 long-prompt generation
  deadline-aborted).
- **92 usable labels** (SIMPLE 62, MEDIUM 24, COMPLEX 6); 109 abstain — 79
  from swap disagreement (judge position/noise), 29 from non-monotone chains
  such as `strong>balanced<weak`.
- The labels disagree with the authored tiers almost completely (1/20 exact
  agreement), and the disagreement is *systematic, not noise*: determinate
  GSM8K arithmetic (authored COMPLEX) is served correctly by the weak model
  → SIMPLE, while obscure SimpleQA lookups (authored SIMPLE) are
  hallucinated by the weak model and answered by the strong one → COMPLEX.
  The labels measure **this deployment's capability gradient**, which is the
  deployable objective; the authored tiers measure benchmark convention
  (and part company with ADR-0028's own rubric on determinate chains).
- On the flywheel holdout (18 cases, seed 20261014): **fresh-trained
  candidate 83.3% vs shipped model 5.6%** (worse than the constant-SIMPLE
  baseline of 67%). Platt grid search: candidate 1.55, shipped pinned at the
  3.0 ceiling — the shipped ensemble is systematically overconfident against
  these outcomes.
- **Warm-start training is structurally unable to adapt**: starting from the
  shipped average weights, the running average entrenches the old convention
  and the model stayed at exactly the shipped 5.6% for every epoch. When the
  label distribution shifts — which is the flywheel's premise — train fresh.

## Ladder correction (second run, same day)

The first run's role ladder — three distinct models — had a broken rung: the
balanced and weak models perform comparably, so the `balanced > weak`
comparison that MEDIUM labels ride on was mostly noise. Re-run with a
**same-model effort ladder** (`reasoning_effort` low / high / max per role —
the proxy's thinking layer maps the OpenAI field to the upstream `think`
field):

- Labeling cost fell 6x (35 min vs 3.5 h for 201 prompts) — one model, and
  the low rung is cheap.
- "Strong loses to a lower rung" anomaly chains halved (24 → 11).
- **MEDIUM labels collapsed 24 → 9**: the trio's MEDIUM band was largely a
  phantom created by the fake balanced-vs-weak gradient. What remains is the
  genuine middle (max ≈ high but high > low).
- COMPLEX labels rose 6 → 9 (max's extra thinking resolves more hard asks).
- Training on the effort labels: fresh candidate 77.8% vs shipped 16.7% on
  the holdout (gate passed again); COMPLEX support is still thin (9 labels),
  so the install criteria are unchanged.

The effort ladder is the recommended labeling setup going forward, and the
same mechanism (`parameter_overrides: {"reasoning_effort": ...}` on model
entries — the override stage re-parses the raw body, so the thinking layer
picks it up) is how the serving pool itself can be turned into a real ladder
without router changes.

## Consequences

- The pipeline, label schema and training/gating code are in-repo and
  tested; real traffic flows into the same machinery via `sample --from-logs`
  as it accumulates.
- **The candidate model was NOT installed.** The mechanical gate passed, but
  the label support is one-sided: 6 COMPLEX labels (5 in training) leave the
  candidate unable to produce COMPLEX (its one holdout COMPLEX case was
  predicted SIMPLE). Install criteria, to be re-checked per iteration:
  holdout accuracy beats shipped, **and** per-tier support ≥ ~30 labels per
  tier (COMPLEX especially), **and** no regression on the corpus rigs beyond
  the label-convention shift. Until then the candidate lives at
  `routing-flywheel/model-candidate-2026-10-13.json` (user data dir).
- The calibration finding (shipped temperature should be ~1.55–3.0, not
  0.75) is archived with the candidate; installing it touches the packaged
  asset and is an operator decision, pending more calibration samples.
- Judge-noise reduction is the highest-value pipeline improvement: the 54%
  abstention rate is dominated by swap disagreement. A third vote per pair
  (majority instead of strict both-agree) and an explicit "both answers
  wrong" rubric option (so all-fail tasks abstain instead of reading as
  SIMPLE) are the planned next iterations.
- Eval-rig numbers recorded before any future install are not comparable
  after it (ADR-0028's re-baseline caveat applies to the classifier swap as
  it did to the signal change).

## Considered Options

- **Train on authored corpus tiers**: rejected — it would re-teach the stale
  convention the flywheel exists to replace (authored COMPLEX on determinate
  arithmetic the weak model solves).
- **Score-style pairwise rubric (1–10)**: rejected, per the deep-research
  consensus — pairwise preference with position swap is more reliable than
  absolute scoring.
- **Write labels into `feedback_records`**: rejected — that table's PK and
  semantics are user feedback on served requests; flywheel labels are
  offline artifacts with a different lifecycle. The JSONL store keeps them
  inspectable and diffable.
- **Skip the gate and ship the candidate**: rejected — a router that cannot
  say COMPLEX would push real engineering work to the cheap lane; the floors
  would catch some of it, and "the model is right on average" is exactly the
  trap LLMRouterBench documents.
