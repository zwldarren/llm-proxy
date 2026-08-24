# Smart-routing trained assets

Vendored ML/weight assets used by the smart-routing engine. These files ship
inside the package and are copied to a writable per-user data directory on
first use (see `llm_proxy.routing.assets`) so the ML libraries can load them by
path.

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
| `seed_embeddings.npy` | ~932 KB | NumPy `.npy` (float32, shape `(621, 384)`) | KNN seed index for the embedding signal |
| `seed_labels.json` | ~2 KB | JSON array of ints (length 621) | Tier labels for the KNN seed index |
| `embedding_classifier.ubj` | ~2.1 MB | native XGBoost model (UBJSON) | Trained embedding+metadata classifier (an `XGBClassifier`) |
| `meta_scaler.pkl` | ~800 B | pickled scikit-learn `StandardScaler` | Metadata-feature scaler used by the embedding classifier |
| `calibration_params.json` | 35 B | JSON | Ensemble sampling-temperature calibration |

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

### `seed_embeddings.npy` + `seed_labels.json` — KNN seed index

- **Consumer:** `llm_proxy.routing.signals.embedding.EmbeddingSignal` loads
  these as the KNN fallback when no trained classifier is available (or when
  `use_classifier=False`).
- **Embeddings:** 621 seed prompts embedded with `BAAI/bge-small-en-v1.5`
  (384-dim, float32). Loaded with `numpy.load`.
- **Labels:** 621 integer tier ids in `0..3`, consumed directly as
  `TierVote.tier_id` (0 = `SIMPLE`, 1 = `MEDIUM`, 2 = `COMPLEX`, 3 = the
  upstream agentic class retained from UncommonRoute training). Loaded with
  `orjson.loads`.

### `embedding_classifier.ubj` — trained embedding classifier

- **Consumer:** `EmbeddingSignal` (`use_classifier=True`); loaded with
  `xgboost.XGBClassifier.load_model`. When present it replaces the KNN vote
  with a trained estimator (an `xgboost.sklearn.XGBClassifier`) over the BGE
  embedding concatenated with 15 metadata features (see
  `EmbeddingSignal._extract_meta_features`).
- **Format:** native XGBoost serialization (UBJSON), migrated from the
  original pickled estimator. XGBoost warns when unpickling models saved by
  older versions, and the native format is the version-safe persistence
  recommended upstream. The loader keeps a one-time legacy fallback for
  deployments that still hold `embedding_classifier.pkl`, converting it to
  the native format on the fly. A `.ubj` file that fails native loading is
  also retried as a pickle (deployments that copied the asset before the
  migration carried a pickled estimator under the `.ubj` name) and
  converted to the native format in place.
- **Inputs:** 384-dim BGE embedding ++ 15-dim metadata vector
  (message count, tool presence/counts, user-text length/word count, code
  markers, question markers, …).

### `meta_scaler.pkl` — metadata-feature scaler

- **Consumer:** `EmbeddingSignal._try_load_scaler(directory)` loads it with
  `pickle.load` from the same directory as the classifier. A scikit-learn
  `StandardScaler` that normalizes the 15 metadata features before they are
  concatenated with the embedding for classification.

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
- **`seed_embeddings.npy` / `seed_labels.json`** require the seed prompt
  corpus plus the `BAAI/bge-small-en-v1.5` encoder
  (`EmbeddingSignal._default_embed_fn`).
- **`embedding_classifier.ubj` / `meta_scaler.pkl`** require the upstream
  scikit-learn training pipeline (estimator + `StandardScaler` fit on the
  embedding+metadata matrix).

To inspect/verify the committed artifacts without regenerating them, load each
asset directly (e.g. ``pickle.load`` / ``numpy.load``) and check its type,
shape, and schema — useful for confirming an asset is not corrupted after a
checkout or vendor bump.

## Git LFS

These files total ~3.8 MB of binary data committed directly to Git. The audit
recommends evaluating [Git LFS](https://git-lfs.com/) for `.pkl` and `.npy` if
they are expected to change frequently. They have been stable since the initial
vendor commit, so LFS is **not** adopted yet. If smart-routing assets begin
updating regularly, track `*.pkl` and `*.npy` under
`routing/assets/` with LFS (add a `.gitattributes` entry and
`git lfs migrate import`) and add a CI guard that fails when a committed
binary asset exceeds a size threshold and is not LFS-tracked.