# The judge gate's shipped default (`not_low_anchor`) and the tier rubric's MEDIUM/COMPLEX boundary

## Status

Accepted (2026-10).

## Context

ADR-0018 made the judge gate a **measured parameter, not a design constant**: the
evaluation rig reports every candidate trigger side by side, and the shipped
default is whichever candidate wins. It shipped before any such measurement was
done over a corpus beyond the eight packaged seed cases, so the de-facto posture
was "no predicate set → judge every eligible first turn" (``gate_is_open``).

The gate has three expressible predicates on the ensemble's output:

- `confidence_below` — the ensemble's confidence under a threshold.
- `complexity_between` — the ensemble's complexity lands inside a half-open band.
- `not_low_anchor` — the ensemble did not place the turn at the trivial low
  anchor (complexity `0.0`, the SIMPLE anchor).

A 103-case corpus built from open-source datasets (SimpleQA-verified, GSM8K
stratified by solution steps, HumanEval, BoolQ passages, CoQA follow-ups,
Hermes tool tasks; one author, one pass, dataset-derived labels — the same
single-rater caveat the rig's README puts on any corpus) measured the ensemble
alone at 39.8% tier match [95% CI 30.1–49.5%], and the judge (tev1:0.8b over
Ollama) on every case at 60.8% [51.0–69.6%]. The candidate triggers over the
same run:

| gate | call rate | match 95% CI | under / over |
| --- | --- | --- | --- |
| ensemble (no judge) | 0% | 39.8% [30.1, 49.5] | 12 / 50 |
| confidence < 0.55 (and quantile-derived 0.61) | 5.8% | 45.6% — CI overlaps the ensemble's | 12 / 44 |
| band [0.33, 0.67) | 35.9% | 39.8% (unchanged) | 15 / 47 |
| not_low_anchor | 79.6% | **74.8% [65.0, 82.5]** | 18 / 8 |
| judge, every case | 100% | 61.2% [51.5, 70.9] | 17 / 23 |

## Decision

- Add `not_low_anchor` as a first-class config predicate on
  `RoutingJudgeConfig` and wire it through `plan_judge_consult`.
- **Make it the shipped default** (`not_low_anchor: True`), so a fresh install
  that turns the judge on runs the one posture the measurement justified. The
  confidence and band predicates stay configurable — they are the right shape
  to split on once a deployment's own shadow data says so.
- **Clarify the MEDIUM/COMPLEX boundary in the tier rubric** (rubric version
  bump 1 → 2). The measurement surfaced a real interpretive gap: the judge
  graded 13 of 14 stratified GSM8K multi-step word problems `medium` while the
  corpus had authored them `complex` on "multi-step reasoning". Both readings
  are defensible under the old text; the clarified text picks the structural
  one the tier system is for — a determinate chain of steps is medium however
  long the chain is, and complex is reserved for open plans. The corpus keeps
  its authored labels; the disagreement is part of the published result.
- Note what the corpus ruled out rather than "fixed": `confidence_below` is
  inert at the moment. The ensemble's confidence clustered at 0.61 median, on
  matches and misses alike, so every threshold swept to the same row (all
  quantiles equal). That is a confidence-calibration problem, not a gate-order
  one, and it stays on the list for a future ADR rather than being papered over
  by a threshold.

## Consequences

- The gate predicate is now the same concept on both sides of the
  measurement/production seam: the rig's `GateSpec` (``not_low_anchor``) and
  the config field carry the same name and the same rule
  (`not_low_anchor_fires` in the judge policy).
- The seed corpus' earlier conclusion ("no gate configured judges every
  eligible turn") still holds as the fallback for a deliberately open gate; it
  just no longer ships first.
- An open-eval cost line: the default consults ~80% of judged-eligible turns.
  Operators who turn the judge on accept one extra upstream call per such
  turn — that is the point of the gate, and shadow mode keeps recording without
  paying attention at all traffic scale.
- Judge latency reality: tev1:0.8b warm on CPU measures ~3.8 s p50, far above
  the configured default `deadline_s = 0.5`. A CPU-only deployment with the
  shipped deadline will abstain on every call and fall back to the ensemble;
  GPU serving or a deadline sized to hardware is a deployment prerequisite, not
  a code change.
