import logging

import numpy as np
import pytest

from llm_proxy.routing.signals.embedding import EmbeddingSignal


def _fake_embed(text: str) -> np.ndarray:
    # deterministic 384-dim vector keyed by text length
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(384).astype(np.float32)


def test_embedding_signal_predict_with_injected_embed_fn():
    sig = EmbeddingSignal(embed_fn=_fake_embed)
    vote = sig.predict(
        {
            "messages": [{"role": "user", "content": "Write a haiku about routing."}],
        }
    )
    assert vote.tier_id in {0, 1, 2, 3, None}
    assert vote.confidence is None or 0.0 <= vote.confidence <= 1.0


def test_embedding_signal_predict_degrades_when_classifier_asset_broken(tmp_path):
    # A .ubj that fails both native XGBoost loading and the pickle fallback
    # (e.g. a legacy pickle saved by an incompatible xgboost version) must not
    # crash predict() — the signal degrades to KNN / abstain instead.
    broken = tmp_path / "embedding_classifier.ubj"
    broken.write_bytes(b"\x80\x04\x95" + b"garbage" * 100)

    sig = EmbeddingSignal(
        index_path=tmp_path / "seed_embeddings.npy",
        labels_path=tmp_path / "seed_labels.json",
        classifier_path=broken,
        embed_fn=_fake_embed,
    )
    assert sig._classifier is None
    vote = sig.predict(
        {
            "messages": [{"role": "user", "content": "Write a haiku about routing."}],
        }
    )
    assert vote.tier_id in {0, 1, 2, 3, None}
    assert vote.confidence is None or 0.0 <= vote.confidence <= 1.0


def test_scaler_load_failure_logs_warning(tmp_path, caplog):
    """A corrupt meta_scaler.pkl logs a warning instead of failing silently."""
    broken = tmp_path / "meta_scaler.pkl"
    broken.write_bytes(b"\x80\x04\x95" + b"garbage" * 100)

    sig = EmbeddingSignal(embed_fn=_fake_embed)
    sig._meta_scaler = None  # isolate from the real asset scaler loaded in __init__
    with caplog.at_level("WARNING"):
        sig._try_load_scaler(tmp_path)

    assert sig._meta_scaler is None
    assert any(
        record.levelno == logging.WARNING and "scaler" in record.message.lower()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_get_embedding_signal_returns_none_when_unavailable(monkeypatch):
    # Force the ONNX import to fail -> graceful degradation
    import llm_proxy.routing.signals.embedding as emb

    monkeypatch.setattr(
        emb,
        "_import_onnx_deps",
        lambda: (_ for _ in ()).throw(ImportError("no onnxruntime")),
    )
    monkeypatch.setattr(emb, "_cache", {})  # reset singleton
    assert await emb.get_embedding_signal(None) is None
