# Dataset-derived eval corpora

Two corpora for the [eval rig](../README.md), built from public HuggingFace
datasets instead of authored from scratch. Pointed at the `--cases` flag of
`python -m llm_proxy.routing.eval`.

| corpus | n | built for |
| --- | --- | --- |
| `open-source-103.jsonl` | 103 | the baseline tier sweep (ADR-0028's gate decision) |
| `complex-situations-98.jsonl` | 98 | the sparse end: long context, multi-turn memory, long tool loops |

## Provenance and license — `open-source-103.jsonl`

| subset × tier | n | source dataset | mapping |
| --- | --- | --- | --- |
| short × SIMPLE | 22 | `google/simpleqa-verified` (eval) | factual lookup, `multi_step`/`requires_reasoning` false |
| short × COMPLEX | 14 | `openai/gsm8k` (test) | word problems stratified by solution steps (≥ 4) |
| short × MEDIUM | 20 | `openai/openai_humaneval` (test) | single function from a bounded spec |
| boundary × MEDIUM | 23 | GSM8K (2–3 steps) + SimpleQA (`multi_step`/`requires_reasoning`) + HumanEval variants with added invariants | genuinely two-tier-defensible asks |
| long × SIMPLE | 10 | `google/boolq` (train) | passage-supplied text, one-step yes/no |
| followup × SIMPLE | 8 | `stanfordnlp/coqa` (train) | contextual lookup, one step |
| tool-context × MEDIUM | 6 | `NousResearch/hermes-function-calling-v1` | tool-loop turn in the rig's seed shape |

Each case carries a `rationale` stating the mapping rule and the tier rule it
applies; `id` prefixes name the source row. The loader that produced the
mapping lives in `build_open_source_corpus.py` (same directory), which also
documents the per-dataset filters. Upstream datasets are pulled through the
datasets-server rows API at build time; the exact rows fetched are recorded in
the builder's offsets, and any rebuild must be diffed against the committed
JSONL before replacing it.

## Provenance and license — `complex-situations-98.jsonl`

| subset × tier | n | source | mapping |
| --- | --- | --- | --- |
| tool-context × SIMPLE | 8 | `NousResearch/hermes-function-calling-v1` (`func_calling`) | one recorded tool call; arguments read off the ask |
| tool-context × MEDIUM | 8 | Hermes `func_calling` | 2–5 dependent calls whose order reads straight off the task |
| boundary × MEDIUM | 4 | Hermes `func_calling` | ≥ 6 recorded calls with interleaved branching |
| boundary × MEDIUM | 6 | `openai/mrcr` (smallest `n_needles=2` rows, < 150k chars) | two-needle verbatim recall in order |
| boundary × MEDIUM | 6 | `hotpotqa/hotpot_qa` (`distractor` validation) | extraction chain (every hard pair has exactly two unique bridge documents) |
| boundary × MEDIUM | 6 | `nmayorga7/multichallenge` | 5–7-turn conversations, bounded axes |
| followup × COMPLEX | 12 | `nmayorga7/multichallenge` | ≥ 8-turn memory/coherence/version-editing axes |
| followup × MEDIUM | 6 | `nmayorga7/multichallenge` | < 5-turn conversations, bounded ask |
| long × COMPLEX | 8 | `zai-org/LongBench-v2` | author difficulty `hard`: multi-document reasoning, reading order is the plan |
| long × SIMPLE | 6 | `zai-org/LongBench-v2` | author difficulty `easy`: one-step lookup over a big supplied context |
| short × COMPLEX | 14 | `HuggingFaceH4/MATH-500` (level 5) + `AI-MO/aimo-validation-aime` | competition derivations with open plans |
| short × MEDIUM | 8 | `HuggingFaceH4/MATH-500` (level 3) | bounded derivations, routine techniques |
| boundary × MEDIUM | 6 | `HuggingFaceH4/MATH-500` (level 4) | mostly determinate chains; novel plans give COMPLEX a claim |

The loader lives in `build_complex_corpus.py` (same directory). Raw rows are
cached under `/tmp/routing-eval/raw2`; a rebuild downloads to a fresh cache and
must be diffed against the committed JSONL before replacing it.

### What the mapping does — and does not — claim

- **No recorded case is authored with a COMPLEX tool loop.** The recorded
  Hermes flows enumerate their calls up front, so the plan is readable in the
  ask, and the shipped v2 rubric reads a determinate chain as MEDIUM however
  long the chain is (ADR-0028). Open tool discovery and error-recovery
  branches are not sampled — a known gap, listed again in the caveats.
- **LongBench-v2's difficulty labels are borrowed, not re-rated**: their
  benchmark authors' hard/easy judged against their multi-choice protocol,
  mapped onto tier with a stated rule. This is one fewer independent rater, not
  an independent one.
- **MATH-500/AIME items are public competition problems** that are widely
  crawled training data. Both the ensemble and the judge meet them on the same
  footing, so no path is flattered uniquely — but do not read the short ×
  COMPLEX numbers as representative of novel COMPLEX traffic.

## Measurements on record (jev-latest over this repo's dev server)

The judge is the dev deployment's `jev-latest` (a System One model served
through the proxy at `127.0.0.1:9911`), called by the rig directly, one call
per case, HTTP only. The runs below were produced with the exact commands in
"Reproduce".

Recorded twice: **before** the ensemble's over-tier fixes (the ADR-0028
record) and **after** (ADR-0029: concordant votes no longer escalated, the
corroboration cap follows Signal A's vote, substance floors key on the
uncapped vote, gate defaults closed). The judge side is unchanged by the
fixes — any composite difference is the ensemble improving under it.

### `open-source-103` — single repeat, seed 20261010

| path | match before | match after (ADR-0029) | under/over before → after |
| --- | --- | --- | --- |
| ensemble (no judge) | 39.8% [30.1, 49.5] | **65.0% [55.3, 73.8]** | 12/50 → 15/21 |
| judge, every case | 57.3% [47.6, 68.0] | 56.3% [46.6, 67.0] | 44/0 → 45/0 (judge unchanged) |
| `not_low_anchor` composite | 57.3% @ 79.6% calls | 56.3% @ 79.6% calls | 44/0 → 45/0 |
| best gate after the fix | — | ensemble alone | — (no gate beats 65.0%) |

Per-subset ensemble, before → after: boundary 69.6% → **91.3%**, followup
100% → 100%, long 0% → 0%, short 30.4% → **57.1%**, tool-context 0% →
**100%**. The judge's latency over these runs: p50 223 → 302 ms, p95 406 →
418 ms; abstentions 0/103 both times.

### `complex-situations-98` — three repeats (294 rows), seed 20261014

| path | match before | match after (ADR-0029) | under/over before → after (per case) |
| --- | --- | --- | --- |
| ensemble (no judge) | 40.8% [31.6, 50.0] | 38.8% [29.6, 49.0] | 25/33 → 38/22 |
| judge, every case | 34.2% [25.1, 43.1] | 33.4% [24.0, 42.4] | 51/12 → 51/12 |
| confidence_q10 (best gate, 4.1% calls) | 42.9% [33.7, 53.1] | 40.8% [31.6, 51.0] | 25/31 → 38/20 |
| `not_low_anchor` (95.9% calls) | 34.0% [25.2, 43.5] | 32.7% [23.8, 41.5] | 52/12 → 53/13 |

Per-subset ensemble, before → after: boundary 57.1% → **82.1%**, followup
33.3% → 33.3%, long 50.0% → 0% (the LongBench hard strata moved to the
rubric-v2 boundary reading of a two-source chain; see ADR-0029), short 50% →
22.7% (the MATH/AIME strata moved to determinate-chain MEDIUM under the same
rubric decision ADR-0028 shipped — the corpus keeps its labels), tool-context
0% → 25%. The judge's latency: p50 299 → 274 ms, p95 417 → 412 ms;
abstentions 25 → 7 of 294.

### Reading these two tables together

Before the fix the story read as complementarity (judge wins shallow-
history traffic). After the fix the comparison is different in kind: the
ensemble alone takes the best row on both corpora, every high-call-rate
gate sits below it, and the miss families that remain line up with the
published rubric disagreement (the GSM8K/MATH strata the judge also graded
MEDIUM). What keeps the judge in the architecture is not accuracy on these
corpora: its bounded excerpt is measurably the constraint, and the honest
improvement levers on its side are budget (`context_turns`/`context_chars`)
and rubric+flag consumption — the ensemble cannot see past its own char
budget either, but the ensemble is cheaper to fix.

The seed corpus (24 × 3, seed 20261014) moved 12.5% → 25.0% for the
ensemble and stayed at 83.3% for the judge over its 24 abstention-decided
rows; with intervals [0, 37.5] and [50, 100] on that few cases it is a
consistency check, not a discriminator.

### Verdict consumption bars (rubric v3 excerpt, same seeds)

Re-measured after the excerpt gained head+tail truncation (rubric v3; criteria
text unchanged). The ensemble is untouched by any of this. New: the rig now
reports consumption policies — which *verdicts* are obeyed, not which turns are
consulted. The judge's sampling is nondeterministic run to run
(applied counts wobble ±25%); every direction below reproduced across four
independent runs.

| composite | `open-source-103` (1 rep) | `complex-situations-98` (3 reps) |
| --- | --- | --- |
| ensemble (no judge) | 65.0% [55.3, 73.8] | 38.8% [29.6, 49.0] |
| judge, every case (obey all) | 59.2% | 34.5% |
| apply when tier mass ≥ 0.9 | **77.7%** [68.9, 84.5] @ 81% applied | 37.1% @ 46% applied |
| apply when escalation ≥ 0.5 | 64.1% @ 56% applied | **50.7%** [41.2, 60.2] @ 47% applied |

Each corpus has a bar that beats the ensemble by ~13 points and is neutral on
the other corpus; the union is positive on both but below each corpus's best.
The bars ship off; these numbers are starting points for an operator's own
shadow-data sweep, not defaults. Per-family reads and the three measured no's
(rubric sharpening, a conversation-shape floor, confidence recalibration) are
recorded in the ADR.

## Honesty caveats

- **Limited raters, not two.** `open-source-103` was authored in one pass; the
  complex corpus re-rated each mapping against the rubric a second time and
  borrows LongBench-v2's difficulty for the long families — still short of the
  independent double-annotate the ideal would have. Read every number with the
  bootstrap CI, not the point estimate.
- **Not the signals' training data by intent, unverifiable by construction.**
  The structural signal's training corpus lives outside this repo
  (`routing/assets/README.md`), so overlap cannot be ruled out; treat elevated
  numbers as suspects, not proof. The MATH/AIME strata make this sharper than
  the caveat usually reads — see the mapping note above.
- **The tool-context subset measures the ask shape, not the loop.** The rig
  judge never sees assistant or tool content by design (ADR-0018), so what is
  actually measured is how a chain of dependent calls *reads* to a classifier
  that only sees user turns — on both corpora.
- **The corpus skew is a feature of the derivation, not of traffic.** Real
  proxy traffic is dominated by ordinary MEDIUM asks and multi-turn tool loops
  these corpora deliberately sample sparsely; the reverse also holds.

## Reproduce

```bash
# raw fetch (datasets-server pages, cached under /tmp/routing-eval/raw2)
# — see the builders' main methods for the page offsets actually used
python src/llm_proxy/routing/eval/corpus/build_open_source_corpus.py
python src/llm_proxy/routing/eval/corpus/build_complex_corpus.py

# the recorded numbers (dev server must be up: uv run llm-proxy-dev)
uv run python -m llm_proxy.routing.eval \
  --cases src/llm_proxy/routing/eval/corpus/open-source-103.jsonl --out /tmp/oss103 \
  --judge ollama --judge-url http://127.0.0.1:9911 --judge-model jev-latest \
  --judge-api-key "$PROXY_KEY" --judge-timeout 60 --repeats 1 --seed 20261010
uv run python -m llm_proxy.routing.eval \
  --cases src/llm_proxy/routing/eval/corpus/complex-situations-98.jsonl --out /tmp/cpx98 \
  --judge ollama --judge-url http://127.0.0.1:9911 --judge-model jev-latest \
  --judge-api-key "$PROXY_KEY" --judge-timeout 60 --repeats 3 --seed 20261014
```

`$PROXY_KEY` is a proxy API key on the dev deployment (any key with access to
the judge model works; the rig sends it as a Bearer credential).
