# Dataset-derived eval corpus (`open-source-103.jsonl`)

A 103-case tier-classification corpus for the [eval rig](../README.md), built
from public HuggingFace datasets instead of authored from scratch. Pointed at
the `--cases` flag of `python -m llm_proxy.routing.eval`.

## Provenance and license

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

The datasets carry their own licenses (MIT, Apache-2.0, CC BY/CC BY-SA). This
corpus is a development measurement tool: it is never shipped as product data
and never used for training.

## Honesty caveats

- **One rater.** Labels were authored in one pass against the rubric in
  `llm_proxy.routing.judge.TIER_RUBRICS` (the GSM8K stratification follows the
  pre-clarification "multi-step reasoning" reading of `complex`). No
  independent review — exactly the caveat the reference benchmark publishes
  about its own labels. Read every number with the bootstrap CI, not the point
  estimate.
- **Not the signals' training data by intent, unverifiable by construction.**
  The structural signal's training corpus lives outside this repo
  (`routing/assets/README.md`), so overlap cannot be ruled out; treat elevated
  numbers as suspects, not proof.
- **The corpus skew is a feature of the derivation, not of traffic.** Real
  proxy traffic is dominated by ordinary MEDIUM asks and multi-turn tool loops
  this corpus deliberately samples sparsely.

## Baseline to compare against (rid 20261010, 1 repeat per case)

Ensemble (signals A+B, no judge): 39.8% [30.1–49.5]; judge
(tev1:0.8b over Ollama, every case): 61.2% [51.5–70.9]; shipped-default gate
`not_low_anchor`: 74.8% [65.0–82.5] at an 80% call rate — recorded as
ADR-0028, alongside the observation that the ensemble's confidence carried no
signal weight across thresholds on this corpus.
