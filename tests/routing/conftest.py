"""Routing test isolation.

The classifier loader prefers the operator's hot-swap artifact
(``model_online.json`` in the user data dir) over the packaged asset. Tests
must measure the SHIPPED model: whatever a developer has installed locally is
operator state, not repo behavior. The autouse fixture points the loader at a
path that never exists so the packaged asset is always the one under test.
"""

from pathlib import Path

import platformdirs
import pytest


@pytest.fixture(autouse=True)
def _pin_classifier_to_packaged_asset(monkeypatch: pytest.MonkeyPatch):
    # The path is derived from the user data dir but named so it can never
    # exist; the loader then falls back to the packaged asset.
    never = (
        Path(platformdirs.user_data_dir("llm-proxy", "llm-proxy"))
        / "routing-assets"
        / "no-hot-swap-in-tests.json"
    )
    monkeypatch.setattr(
        "llm_proxy.routing.classifier._get_online_model_path",
        lambda: never,
    )
    # A model may already be cached in the module global from an earlier test;
    # reset both flags so the loader re-resolves with the pinned path.
    monkeypatch.setattr("llm_proxy.routing.classifier._model", None)
    monkeypatch.setattr("llm_proxy.routing.classifier._model_load_attempted", False)
