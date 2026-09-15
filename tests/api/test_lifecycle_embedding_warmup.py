"""The BGE embedding warm-up must be gated on smart routing being enabled.

Regression guard for a measured resource leak: ``startup_embedding_signal``
eagerly loaded the embedding model in *every* worker unconditionally, even
though ``smart_routing.enabled`` defaults to False. With the ``smart-routing``
extra installed that costs ~500 MB of RSS per worker, downloads ~130 MB from
HuggingFace Hub on a cold cache, and makes readiness depend on an external
network — for a code path that ``orchestrate_smart_routing`` can never reach
while routing is off.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.api.lifecycle import startup_embedding_signal


def _app(*, smart_routing_enabled: bool | None) -> SimpleNamespace:
    """Build a stand-in app whose config manager reports the routing flag.

    ``smart_routing_enabled=None`` models "no config manager yet".
    """
    if smart_routing_enabled is None:
        return SimpleNamespace(state=SimpleNamespace())
    config_manager = MagicMock()
    config_manager.get_smart_routing_config = AsyncMock(
        return_value=SimpleNamespace(enabled=smart_routing_enabled)
    )
    return SimpleNamespace(state=SimpleNamespace(config_manager=config_manager))


@pytest.mark.asyncio
async def test_warmup_skipped_when_smart_routing_disabled(monkeypatch) -> None:
    get_embedding_signal = AsyncMock()
    monkeypatch.setattr(
        "llm_proxy.routing.signals.embedding.get_embedding_signal",
        get_embedding_signal,
    )

    await startup_embedding_signal(_app(smart_routing_enabled=False))

    get_embedding_signal.assert_not_awaited()


@pytest.mark.asyncio
async def test_warmup_runs_when_smart_routing_enabled(monkeypatch) -> None:
    get_embedding_signal = AsyncMock()
    monkeypatch.setattr(
        "llm_proxy.routing.signals.embedding.get_embedding_signal",
        get_embedding_signal,
    )

    app = _app(smart_routing_enabled=True)
    await startup_embedding_signal(app)

    get_embedding_signal.assert_awaited_once_with(app.state)


@pytest.mark.asyncio
async def test_warmup_skipped_without_config_manager(monkeypatch) -> None:
    get_embedding_signal = AsyncMock()
    monkeypatch.setattr(
        "llm_proxy.routing.signals.embedding.get_embedding_signal",
        get_embedding_signal,
    )

    await startup_embedding_signal(_app(smart_routing_enabled=None))

    get_embedding_signal.assert_not_awaited()


@pytest.mark.asyncio
async def test_warmup_failure_never_breaks_startup(monkeypatch) -> None:
    """A broken ML stack must degrade, not abort the lifespan."""
    config_manager = MagicMock()
    config_manager.get_smart_routing_config = AsyncMock(side_effect=RuntimeError("boom"))
    app = SimpleNamespace(state=SimpleNamespace(config_manager=config_manager))

    await startup_embedding_signal(app)  # must not raise
