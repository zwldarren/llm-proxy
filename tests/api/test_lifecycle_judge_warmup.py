"""The routing judge must be warm before a real gated turn needs it.

Regression guard for a measured failure mode: a judge call cancelled at its
deadline does not leave the model loaded, so a proxy whose gate fires every few
minutes pays a cold model load on *every* judged turn, loses it to the deadline,
and falls back to the ensemble — the judge never answers at all. The warm-up
exists so the first real turn can meet its own deadline (ADR-0018).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.api.lifecycle import startup_judge_warmup
from llm_proxy.config.types.server import ProxyAuthConfig, ServerParams
from llm_proxy.config.types.smart_routing import RoutingJudgeConfig, SmartRoutingConfig


def _config(*, smart_routing_enabled=True, judge: RoutingJudgeConfig | None = None):
    """A stand-in ProxyConfig: only the fields the judge path reads are real."""
    judge = judge or RoutingJudgeConfig(enabled=True, model="tev1:0.8b")
    return SimpleNamespace(
        models={"tev1:0.8b": object()},
        provider_configs={"local": object()},
        server_params=ServerParams(auth=ProxyAuthConfig(jwt_secret="a" * 32)),
        smart_routing=SmartRoutingConfig(enabled=smart_routing_enabled, judge=judge),
    )


def _app(config) -> SimpleNamespace:
    config_manager = MagicMock()
    config_manager.get_config = AsyncMock(return_value=config)
    config_manager.add_reload_listener = MagicMock()
    return SimpleNamespace(
        state=SimpleNamespace(
            config_manager=config_manager,
            http_client=object(),
            circuit_breaker=None,
            provider_stats=None,
        )
    )


async def _drain(app) -> None:
    """Let the scheduled warm-up run and be collected."""
    task = getattr(app.state, "judge_warmup_task", None)
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_warmup_is_scheduled_in_the_background_and_not_awaited(monkeypatch):
    """Readiness must not wait on the judge: startup returns before the call does."""
    started = asyncio.Event()

    async def warm_judge(*, config, services):  # noqa: ARG001 - test double
        started.set()
        await asyncio.Event().wait()  # never finishes

    monkeypatch.setattr("llm_proxy.routing.judge.consult.warm_judge", warm_judge)

    app = _app(_config())
    await startup_judge_warmup(app)

    # Returned without waiting: the scheduled task had not even started yet.
    assert not app.state.judge_warmup_task.done()
    await asyncio.wait_for(started.wait(), timeout=1)
    # ... and the warm-up is still in flight while startup is long done.
    assert not app.state.judge_warmup_task.done()
    app.state.judge_warmup_task.cancel()
    await asyncio.gather(app.state.judge_warmup_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_warmup_runs_with_a_generous_deadline(monkeypatch):
    warm_judge = AsyncMock(return_value=None)
    monkeypatch.setattr("llm_proxy.routing.judge.consult.warm_judge", warm_judge)

    app = _app(_config())
    await startup_judge_warmup(app)
    await _drain(app)

    warm_judge.assert_awaited_once()
    assert warm_judge.await_args.kwargs["config"].smart_routing.judge.model == "tev1:0.8b"


@pytest.mark.asyncio
async def test_a_reload_warms_the_judge_again(monkeypatch):
    warm_judge = AsyncMock(return_value=None)
    monkeypatch.setattr("llm_proxy.routing.judge.consult.warm_judge", warm_judge)

    app = _app(_config())
    await startup_judge_warmup(app)
    await _drain(app)
    assert warm_judge.await_count == 1

    # The listener the app registered is what turns the UI toggle into a warm judge.
    listener = app.state.config_manager.add_reload_listener.call_args.args[0]
    await listener(_config(judge=RoutingJudgeConfig(enabled=True, model="typesafe/jev")))
    await _drain(app)

    assert warm_judge.await_count == 2
    assert warm_judge.await_args.kwargs["config"].smart_routing.judge.model == "typesafe/jev"


@pytest.mark.asyncio
async def test_an_in_flight_warmup_is_not_stacked(monkeypatch):
    started = asyncio.Event()

    async def warm_judge(*, config, services):  # noqa: ARG001 - test double
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("llm_proxy.routing.judge.consult.warm_judge", warm_judge)

    app = _app(_config())
    await startup_judge_warmup(app)
    first = app.state.judge_warmup_task
    await started.wait()

    listener = app.state.config_manager.add_reload_listener.call_args.args[0]
    await listener(_config())
    assert app.state.judge_warmup_task is first  # still the same, single warm-up

    first.cancel()
    await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_failing_warmup_never_escapes(monkeypatch):
    async def boom(*, config, services):  # noqa: ARG001 - test double
        raise RuntimeError("judge unreachable")

    monkeypatch.setattr("llm_proxy.routing.judge.consult.warm_judge", boom)

    app = _app(_config())
    await startup_judge_warmup(app)
    await _drain(app)  # would raise if the task leaked the exception

    assert app.state.judge_warmup_task.done()


@pytest.mark.asyncio
async def test_no_config_manager_means_no_warmup(monkeypatch):
    warm_judge = AsyncMock()
    monkeypatch.setattr("llm_proxy.routing.judge.consult.warm_judge", warm_judge)

    app = SimpleNamespace(state=SimpleNamespace())
    await startup_judge_warmup(app)

    warm_judge.assert_not_awaited()
    assert not hasattr(app.state, "judge_warmup_task")
