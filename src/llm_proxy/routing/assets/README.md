# Smart-routing trained assets

Vendored weight assets used by the smart-routing engine. These files ship
inside the package and are copied to a writable per-user data directory on
first use (see `llm_proxy.routing.assets`) so the routing engine can load them
by path.

## Provenance

These assets were **vendored from the external UncommonRoute project** in commit
`f358966f` ("feat(routing): vendor UncommonRoute trained assets and add ML
deps"). The training pipelines and source datasets that produced them live
**outside this repository**; they are not regenerated here. Treat the files in
this directory as build artifacts: committed for reproducibility and offline
operation, not edited by hand.

## Asset inventory

| File | Size | Format | Purpose |
|------|------|--------|---------|
| `model.json` | ~824 KB | JSON | Perceptron weights for the script-agnostic structural classifier |
| `calibration_params.json` | 35 B | JSON | Ensemble sampling-temperature calibration |

The embedding-signal assets (`seed_embeddings.npy`, `seed_labels.json`,
`embedding_classifier.ubj`, `meta_scaler.pkl`) were removed together with the
embedding signal itself (Signal C) and its heavy ML stack
(ONNX Runtime, transformers, XGBoost, scikit-learn).

## Per-asset detail

### `model.json` — structural Perceptron weights

- **Producer (in-repo):** `llm_proxy.routing.learned.ScriptAgnosticClassifier`
  via `ScriptAgnosticClassifier.train(...)` then `.save(path)`.
- **Loader:** `llm_proxy.routing.classifier._ensure_model_loaded()` loads it
  through `ScriptAgnosticClassifier.load(path)` (falling back to an online
  model path when present).
- **Schema:** `{ "avg_weights": {tier: {feature: weight}}, "weights": {tier:
  {feature: weight}}, "update_count": int, "use_ngrams": bool }` where `tier`
  is one of `SIMPLE` / `MEDIUM` / `COMPLEX` and weights below `1e-6` are pruned
  on save. Features are normalized structural/n-gram signals
  (`s_normalized_length`, `s_code_markers`, `s_nesting_depth`, …).

### `calibration_params.json` — ensemble calibration

- **Schema:** `{ "temperature": float, "version": int }` (currently
  `{"temperature": 0.75, "version": 1}`). Used to temperature-scale the
  ensemble's tier sampling. (Phase-2 Platt calibration was removed; the
  ensemble confidence is used directly — see `routing/api.py`.)

## Deployment

`llm_proxy.routing.assets.ensure_assets_deployed()` copies every name in
`_ASSET_NAMES` from this package directory to
`platformdirs.user_data_dir("llm-proxy", "llm-proxy") / "routing-assets"` on
first use, and `asset_path(name)` returns that deployed path. The copy is
skipped for files that already exist, so replacing an asset in the data dir is
the supported way to hot-swap a model locally without rebuilding the package.

## Regeneration

The full training pipeline is **not** in this repository (it lives in the
upstream UncommonRoute project). What the in-repo code does provide:

- **`model.json`** can be retrained end-to-end from labeled feature sets using
  `ScriptAgnosticClassifier.train(feature_sets, epochs=...)` followed by
  `.save(path)`. The labeled training examples themselves are produced
  upstream and are not checked in here.

To inspect/verify the committed artifacts without regenerating them, load each
asset directly (e.g. ``orjson.loads``) and check its schema — useful for
confirming an asset is not corrupted after a checkout or vendor bump.

## Git LFS

`model.json` is ~824 KB of data committed directly to Git — small enough not to
need [Git LFS](https://git-lfs.com/). If smart-routing assets ever begin
updating regularly or grow again, track large binaries under
`routing/assets/` with LFS (add a `.gitattributes` entry and
`git lfs migrate import`) and add a CI guard that fails when a committed
binary asset exceeds a size threshold and is not LFS-tracked.
