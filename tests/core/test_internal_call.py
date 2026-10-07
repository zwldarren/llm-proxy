"""Tests for the request-free internal-call seam (ADR-0018).

The seam is what lets the routing layer reach a provider without importing the
API package, so its contract is tested directly: the failures are abstentions,
the deadline is enforced rather than trusted to the transport defaults, and
internal calls never retry inside a client's request.
"""

import asyncio

import pytest

from llm_proxy.config.types.model import ModelConfig, ModelProviderConfig
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.adapter import BaseAdapter
from llm_proxy.core.exceptions import NotFoundError, ValidationError
from llm_proxy.core.internal_call import (
    INTERNAL_CALL_ATTEMPTS,
    build_provider_adapter,
    call_systemone,
    select_internal_provider,
)
from llm_proxy.models.systemone import InternalSystemOneRequest


def _model(*providers: tuple[str, int]) -> ModelConfig:
    return ModelConfig(
        providers=[
            ModelProviderConfig(provider=name, priority=priority) for name, priority in providers
        ]
    )


def _provider_config(name: str, *, type_: str = "openai-compatible") -> ProviderConfig:
    return ProviderConfig(name=name, type=type_, api_key="k", base_url="http://upstream.test/v1")


def _request() -> InternalSystemOneRequest:
    return InternalSystemOneRequest(
        model="tev1:0.8b",
        state={"current_request": "hello"},
        questions={"tier": {"type": "choice", "instructions": "classify"}},
    )


class _StubAdapter:
    """An adapter whose ``systemone`` behaves as the test asks."""

    def __init__(self, behaviour: str) -> None:
        self._behaviour = behaviour
        self.closed = False

    async def systemone(self, request):  # noqa: ANN001, ANN201 - test double
        if self._behaviour == "boom":
            raise ValueError("upstream exploded")
        if self._behaviour == "slow":
            await asyncio.sleep(5)
        return object()

    async def close(self) -> None:
        if self._behaviour == "close-boom":
            raise RuntimeError("close exploded")
        if self._behaviour == "close-hang":
            await asyncio.sleep(5)
        self.closed = True


class _StubPool:
    """Stands in for the shared HTTP client manager."""

    def __init__(self) -> None:
        self.requested: list[str] = []
        self.session = object()

    async def get_client(self, provider_name: str) -> object:
        self.requested.append(provider_name)
        return self.session


def _patch_builder(monkeypatch, behaviour: str) -> dict:
    """Replace adapter construction with a stub, capturing its keyword args."""
    captured: dict = {}
    adapters: list[_StubAdapter] = []

    def fake_build(provider_name, provider_config, **kwargs):  # noqa: ANN001, ANN202 - test double
        captured.update(kwargs)
        captured["provider_name"] = provider_name
        adapter = _StubAdapter(behaviour)
        adapters.append(adapter)
        captured["adapter"] = adapter
        return adapter

    monkeypatch.setattr("llm_proxy.core.internal_call.build_provider_adapter", fake_build)
    captured["adapters"] = adapters
    return captured


class TestBuildProviderAdapter:
    def test_provider_without_a_type_is_a_validation_error(self):
        # ``ProviderConfig`` already rejects an empty type, so this guard only
        # fires for a config that never went through that validation (a raw row,
        # a fixture built by attribute). Keep it exercised anyway: the failure it
        # prevents is an opaque 500 instead of a named provider.
        config = _provider_config("broken")
        config.type = ""
        with pytest.raises(ValidationError, match="has no type configured"):
            build_provider_adapter("broken", config)

    def test_unknown_provider_type_is_not_found(self):
        with pytest.raises(NotFoundError, match="not found"):
            build_provider_adapter("p", _provider_config("p", type_="does-not-exist"))

    def test_builds_a_real_adapter_without_a_request(self):
        adapter = build_provider_adapter("p", _provider_config("p"))
        assert isinstance(adapter, BaseAdapter)


class TestSelectInternalProvider:
    def test_picks_the_highest_priority_provider(self):
        selection = select_internal_provider(
            _model(("low", 0), ("high", 5)),
            {"low": _provider_config("low"), "high": _provider_config("high")},
        )
        assert selection is not None
        assert selection.provider_name == "high"
        # One attempt, never a retry: internal calls must not stall a client's
        # request hoping an upstream recovers. (RetryPolicy counts attempts, so
        # this is 1 — 0 would mean the call is never attempted at all.)
        assert selection.max_retries == INTERNAL_CALL_ATTEMPTS == 1

    def test_returns_none_when_no_provider_is_configured(self):
        assert select_internal_provider(_model(("missing", 1)), {}) is None


class TestCallSystemOne:
    async def test_success_returns_the_response_and_latency(self, monkeypatch):
        captured = _patch_builder(monkeypatch, "ok")
        outcome = await call_systemone(
            _request(),
            model_config=_model(("p", 1)),
            provider_configs={"p": _provider_config("p")},
        )
        assert outcome.ok is True
        assert outcome.error is None
        assert outcome.latency_ms >= 0.0
        assert outcome.provider_name == "p"
        assert captured["max_retries"] == INTERNAL_CALL_ATTEMPTS

    async def test_deadline_is_enforced_rather_than_trusted_to_the_transport(self, monkeypatch):
        _patch_builder(monkeypatch, "slow")
        outcome = await call_systemone(
            _request(),
            model_config=_model(("p", 1)),
            provider_configs={"p": _provider_config("p")},
            deadline_s=0.01,
        )
        assert outcome.ok is False
        assert outcome.error is not None and "deadline exceeded" in outcome.error
        assert outcome.latency_ms < 1000.0

    async def test_the_pooled_client_is_used_and_left_open(self, monkeypatch):
        """A manager session belongs to real traffic: use it, do not close it."""
        captured = _patch_builder(monkeypatch, "ok")
        pool = _StubPool()
        outcome = await call_systemone(
            _request(),
            model_config=_model(("p", 1)),
            provider_configs={"p": _provider_config("p")},
            http_client_manager=pool,
        )
        assert outcome.ok is True
        assert pool.requested == ["p"]
        assert captured["http_client"] is pool.session
        assert captured["adapter"].closed is False

    async def test_an_unpooled_call_closes_the_client_it_caused(self, monkeypatch):
        """No client and no pool means the adapter opened one; close it or leak."""
        captured = _patch_builder(monkeypatch, "ok")
        outcome = await call_systemone(
            _request(),
            model_config=_model(("p", 1)),
            provider_configs={"p": _provider_config("p")},
        )
        assert outcome.ok is True
        assert captured["http_client"] is None
        assert captured["adapter"].closed is True

    async def test_a_failing_close_does_not_escape_the_never_raising_contract(self, monkeypatch):
        """Cleanup is transport I/O too; a close failure is not the caller's problem."""
        _patch_builder(monkeypatch, "close-boom")
        outcome = await call_systemone(
            _request(),
            model_config=_model(("p", 1)),
            provider_configs={"p": _provider_config("p")},
        )
        assert outcome.ok is True

    async def test_a_hanging_close_gets_its_own_small_budget(self, monkeypatch):
        _patch_builder(monkeypatch, "close-hang")
        monkeypatch.setattr("llm_proxy.core.internal_call.INTERNAL_CLOSE_TIMEOUT_S", 0.01)
        started = asyncio.get_running_loop().time()
        outcome = await call_systemone(
            _request(),
            model_config=_model(("p", 1)),
            provider_configs={"p": _provider_config("p")},
        )
        elapsed = asyncio.get_running_loop().time() - started
        assert outcome.ok is True
        assert elapsed < 1.0

    async def test_upstream_failure_is_an_abstention_not_an_exception(self, monkeypatch):
        _patch_builder(monkeypatch, "boom")
        outcome = await call_systemone(
            _request(),
            model_config=_model(("p", 1)),
            provider_configs={"p": _provider_config("p")},
        )
        assert outcome.ok is False
        assert outcome.error is not None and "ValueError" in outcome.error

    async def test_unknown_model_abstains_without_touching_a_provider(self):
        outcome = await call_systemone(
            _request(), model_config=None, provider_configs={}, http_client=None
        )
        assert outcome.ok is False
        assert outcome.error == "unknown model"
        assert outcome.latency_ms == 0.0

    async def test_model_without_an_available_provider_abstains(self):
        outcome = await call_systemone(
            _request(), model_config=_model(("p", 1)), provider_configs={}
        )
        assert outcome.ok is False
        assert outcome.error == "no available provider"
