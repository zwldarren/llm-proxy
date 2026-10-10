# Drop the embedding signal (Signal C) from smart routing

## Status

Accepted (2026-10).

## Context

Smart routing's in-process ensemble (the UncommonRoute port) voted over three
signals: metadata (A, weight 0.50), structural (B, 0.10) and embedding (C,
0.40). Signal C embedded the last user message with `BAAI/bge-small-en-v1.5`
(ONNX Runtime) and classified the tier with a vendored XGBoost estimator over
the embedding plus 15 metadata features (KNN fallback otherwise). It brought
the heaviest dependency surface in the codebase:

- Python packages: `onnxruntime`, `transformers`, `xgboost`,
  `scikit-learn`, `huggingface-hub`, plus `numpy` for the seed index —
  the reason the optional `smart-routing` extra existed.
- Runtime behavior: a ~500 MB RSS warm-up per worker, a HuggingFace Hub
  download on cold start (~130 MB), and a startup warm-up step
  (`startup_embedding_signal`) whose only job was to mask that cost.
- Vendored binaries: four of the six routing assets
  (`seed_embeddings.npy`, `seed_labels.json`, `embedding_classifier.ubj`,
  `meta_scaler.pkl`, ~3 MB) existed only for Signal C.
- A packaging hazard: `onnxruntime` does not publish `cp315` wheels, which
  blocked the Python 3.15 upgrade for the whole `smart-routing` extra.

The routing judge (ADR-0018) changed the value calculus. ADR-0018's own
measurements showed Signal C's error mass concentrates on the MEDIUM boundary
— precisely the band where the judge was introduced as the authority, and
where the ensemble's calibrated confidence is highest exactly when it is
wrong. The signals the ensemble can no longer decide well are exactly the
turns the judge now settles; C's 0.40 weight was propping up the band the
judge was built to own.

## Decision

**Signal C is removed, and the ensemble decides on A+B alone.**

- `routing/signals/embedding.py`, its startup warm-up, its four vendored
  assets and the `smart-routing` extra are deleted. `numpy` leaves the main
  dependencies; no ML package remains anywhere in the install.
- The `--embedding` flag and `embedding_signal` parameters leave the routing
  API (`classify`, `route`, `run_ensemble_leg`) and the evaluation rig.
- Ensemble weights keep the source's static values for the surviving signals
  (A=0.50, B=0.10); the ensemble normalizes over active votes, so removing a
  signal is a vote-set change, not a re-weighting.
- Decision logic that read vote C is simplified to its A-only or structural
  remainder: `_cap_uncorroborated_structural_high` corroborates with Signal A
  and text substance; `_soften_tier_cap_for_agent_state` loses the
  `embedding_support` clause; the short-structural-medium floor drops its
  embedding-LOW gate and keeps the substance checks.
- `signal_votes` in the routing decision and its log metadata shrink to
  `metadata` and `structural` keys.
- ADR-0018's judge keeps the ambiguity band; nothing in the gate, the
  deadline, the fallback or the shadow-mode machinery changes.

## Consequences

- The Python 3.15 upgrade is unblocked: `requires-python` is `>=3.15` with no
  excluded dependency, and the Docker image no longer carries a partial ML
  stack.
- Startup drops the per-worker BGE/ONNX warm-up; `startup_judge_warmup`
  remains the only model warm-up smart routing performs.
- Routing quality on obvious ends rests on A+B (they were adequate there per
  ADR-0018's port measurements); the MEDIUM boundary is delegated to the
  judge, which is the band it owns by construction.
- The evaluation rig's ensemble baseline is now A+B; judge-agreement numbers
  from before this change are not comparable to runs after it. Re-baseline
  the authored corpus before trusting any new gate trigger measurement.
- `docs/api/routing.md`'s signal table is rewritten (two signals, no extra).
- The embedding-signal failure mode (a warning and abstention when the extra
  is absent) is gone entirely; the remaining fail-open paths are the judge's
  (deadline, HTTP error, malformed or ambiguous verdict, open circuit).

## Considered Options

- **Keep Signal C behind the extra on Python 3.14**: rejected. It kept a
  version-gated dependency matrix, a binary-asset surface and a warm-up cost
  for a signal whose distinctive contribution sits in the band the judge now
  owns. Operators who need offline embedding classification can add a
  purpose-built system when a use case names itself.
- **Replace C with a lighter semantic classifier**: rejected for now. Any
  replacement would need its own corpus, eval and ADR; nothing in hand beats
  "A+B + judge" on evidence.
- **Make the judge the only decision path (drop A+B too)**: rejected — see
  ADR-0018. The judge is gated (first turn, no tool loops, no background
  calls) and its p95 latency exceeds the added-p95 ceiling; the ensemble is
  both the fallback and the ground-truth baseline for judge evaluation.
