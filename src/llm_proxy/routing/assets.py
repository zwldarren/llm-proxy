"""Trained-asset deployment for the smart routing engine.

Vendored assets (Perceptron weights, KNN index, XGBoost classifier, scaler,
Platt params) ship inside the package and are copied to a writable data dir
on first use so the ML libs can load them by path.
"""

import shutil
from pathlib import Path

import platformdirs

_ASSET_NAMES = (
    "model.json",
    "seed_embeddings.npy",
    "seed_labels.json",
    "embedding_classifier.ubj",
    "meta_scaler.pkl",
    "calibration_params.json",
)


def _package_assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def _data_dir() -> Path:
    return Path(platformdirs.user_data_dir("llm-proxy", "llm-proxy")) / "routing-assets"


def ensure_assets_deployed() -> Path:
    """Copy packaged assets to the data dir if missing; return that dir."""
    target = _data_dir()
    target.mkdir(parents=True, exist_ok=True)
    src = _package_assets_dir()
    for name in _ASSET_NAMES:
        t = target / name
        if not t.exists():
            shutil.copy2(src / name, t)
    return target


def asset_path(name: str) -> Path:
    """Return the deployed path for a named asset, deploying first."""
    return ensure_assets_deployed() / name
