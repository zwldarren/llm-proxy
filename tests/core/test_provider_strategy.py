"""Tests for pluggable provider-selection strategies (provider_strategy.py)."""

import random

import pytest

from llm_proxy.config.types.model import (
    ModelConfig,
    ModelProviderConfig,
    ProviderSelectionStrategy,
)
from llm_proxy.core.provider_stats import ProviderStatsStore
from llm_proxy.core.provider_strategy import (
    STICKY_TTL_SECONDS,
    StrategyContext,
    order_providers,
    pick_provider,
    resolve_sticky_key,
)


class FakeRedis:
    """Minimal dict-backed async Redis double."""

    def __init__(self, data: dict | None = None, fail: bool = False):
        self.data = dict(data or {})
        self.fail = fail
        self.setex_calls: list[tuple[str, int, str]] = []

    async def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.data.get(key)

    async def setex(self, key, ttl, value):
        if self.fail:
            raise ConnectionError("redis down")
        self.setex_calls.append((key, ttl, value))
        self.data[key] = value


def _provider(
    name: str,
    priority: int = 1,
    input_cost: float | None = None,
    output_cost: float | None = None,
) -> ModelProviderConfig:
    return ModelProviderConfig(
        provider=name,
        priority=priority,
        input_cost_per_1m=input_cost,
        output_cost_per_1m=output_cost,
    )


def _model(
    providers: list[ModelProviderConfig],
    input_cost: float | None = None,
    output_cost: float | None = None,
) -> ModelConfig:
    return ModelConfig(
        providers=providers,
        model_name="test-model",
        input_cost_per_1m=input_cost,
        output_cost_per_1m=output_cost,
    )


def _candidates(*names: str, costs: dict[str, tuple[float, float]] | None = None) -> tuple:
    """Build (available, model_config) for same-priority providers.

    ``costs`` maps provider name → (input_cost, output_cost); missing entries
    get no per-mapping pricing.
    """
    providers = []
    for name in names:
        ic = oc = None
        if costs and name in costs:
            ic, oc = costs[name]
        providers.append(_provider(name, input_cost=ic, output_cost=oc))
    available = [(p, f"{p.provider}::0") for p in providers]
    return available, _model(providers)


def _ctx(
    model_config: ModelConfig,
    conversation_key: str | None = None,
    redis=None,
    stats_store: ProviderStatsStore | None = None,
    rng: random.Random | None = None,
) -> StrategyContext:
    return StrategyContext(
        model_config=model_config,
        model_name="test-model",
        conversation_key=conversation_key,
        redis=redis,
        stats_store=stats_store,
        rng=rng,
    )


class TestRandomStrategy:
    def test_pick_returns_member(self):
        available, model = _candidates("a", "b", "c")
        picked = pick_provider(
            available, ProviderSelectionStrategy.RANDOM, _ctx(model, rng=random.Random(1))
        )
        assert picked in available

    def test_empty_available_returns_none(self):
        _, model = _candidates("a")
        assert pick_provider([], ProviderSelectionStrategy.RANDOM, _ctx(model)) is None

    def test_eventually_covers_all(self):
        available, model = _candidates("a", "b")
        rng = random.Random(42)
        picks = {
            pick_provider(available, ProviderSelectionStrategy.RANDOM, _ctx(model, rng=rng))[
                0
            ].provider
            for _ in range(50)
        }
        assert picks == {"a", "b"}


class TestCostOptimizedStrategy:
    def test_cheapest_first(self):
        available, model = _candidates(
            "expensive", "cheap", costs={"expensive": (10.0, 30.0), "cheap": (1.0, 2.0)}
        )
        picked = pick_provider(available, ProviderSelectionStrategy.COST_OPTIMIZED, _ctx(model))
        assert picked[0].provider == "cheap"

    def test_full_ordering_is_ascending_cost(self):
        available, model = _candidates(
            "c", "a", "b", costs={"a": (1.0, 1.0), "b": (5.0, 5.0), "c": (50.0, 50.0)}
        )
        ordered = order_providers(available, ProviderSelectionStrategy.COST_OPTIMIZED, _ctx(model))
        assert [p.provider for p, _ in ordered] == ["a", "b", "c"]

    def test_model_level_pricing_fallback(self):
        # "b" has no per-mapping pricing → falls back to model-level (1+2=3),
        # which beats "a"'s explicit 10+20=30.
        providers = [
            _provider("a", input_cost=10.0, output_cost=20.0),
            _provider("b"),
        ]
        model = _model(providers, input_cost=1.0, output_cost=2.0)
        available = [(p, f"{p.provider}::0") for p in providers]
        picked = pick_provider(available, ProviderSelectionStrategy.COST_OPTIMIZED, _ctx(model))
        assert picked[0].provider == "b"

    def test_unpriced_sorted_last(self):
        providers = [
            _provider("priced", input_cost=100.0, output_cost=100.0),
            _provider("unpriced"),  # no mapping-level, no model-level pricing
        ]
        model = _model(providers)  # no model-level pricing
        available = [(p, f"{p.provider}::0") for p in providers]
        ordered = order_providers(available, ProviderSelectionStrategy.COST_OPTIMIZED, _ctx(model))
        assert ordered[0][0].provider == "priced"
        assert ordered[-1][0].provider == "unpriced"

    def test_deterministic_across_calls(self):
        available, model = _candidates("a", "b", costs={"a": (2.0, 2.0), "b": (1.0, 1.0)})
        for seed in range(10):
            picked = pick_provider(
                available,
                ProviderSelectionStrategy.COST_OPTIMIZED,
                _ctx(model, rng=random.Random(seed)),
            )
            assert picked[0].provider == "b"


class TestSessionStickyStrategy:
    def test_same_conversation_same_provider(self):
        available, model = _candidates("a", "b", "c")
        ctx = _ctx(model, conversation_key="conv-1")
        first = pick_provider(available, ProviderSelectionStrategy.SESSION_STICKY, ctx)
        for _ in range(10):
            assert pick_provider(available, ProviderSelectionStrategy.SESSION_STICKY, ctx) == first

    def test_distribution_across_conversations(self):
        available, model = _candidates("a", "b")
        picks = {
            pick_provider(
                available,
                ProviderSelectionStrategy.SESSION_STICKY,
                _ctx(model, conversation_key=f"conv-{i}"),
            )[0].provider
            for i in range(20)
        }
        # With 20 distinct conversations, rendezvous hashing should use both.
        assert picks == {"a", "b"}

    def test_no_conversation_key_degrades_to_random(self):
        available, model = _candidates("a", "b")
        rng = random.Random(7)
        picks = {
            pick_provider(
                available, ProviderSelectionStrategy.SESSION_STICKY, _ctx(model, rng=rng)
            )[0].provider
            for _ in range(50)
        }
        assert picks == {"a", "b"}

    def test_sticky_key_pinned_first(self):
        available, model = _candidates("a", "b", "c")
        sticky = available[1][1]  # pin "b" explicitly
        ordered = order_providers(
            available,
            ProviderSelectionStrategy.SESSION_STICKY,
            _ctx(model, conversation_key="conv-1"),
            sticky_key=sticky,
        )
        assert ordered[0][1] == sticky

    def test_fallback_ordering_deterministic(self):
        # After the pinned provider is used, the remaining ordering is the
        # deterministic rendezvous rank — the whole conversation drifts to the
        # same backup provider.
        available, model = _candidates("a", "b", "c")
        ctx = _ctx(model, conversation_key="conv-1")
        first = pick_provider(available, ProviderSelectionStrategy.SESSION_STICKY, ctx)
        remaining = [pk for pk in available if pk[1] != first[1]]
        second = pick_provider(remaining, ProviderSelectionStrategy.SESSION_STICKY, ctx)
        for _ in range(10):
            assert pick_provider(remaining, ProviderSelectionStrategy.SESSION_STICKY, ctx) == second

    def test_sticky_key_not_in_available_ignored(self):
        available, model = _candidates("a", "b")
        ordered = order_providers(
            available,
            ProviderSelectionStrategy.SESSION_STICKY,
            _ctx(model, conversation_key="conv-1"),
            sticky_key="ghost::9",
        )
        assert {k for _, k in ordered} == {k for _, k in available}


class TestResolveStickyKey:
    @pytest.mark.asyncio
    async def test_miss_persists_rendezvous_head(self):
        available, model = _candidates("a", "b", "c")
        redis = FakeRedis()
        ctx = _ctx(model, conversation_key="conv-1", redis=redis)
        key = await resolve_sticky_key(available, ctx)
        assert key in {k for _, k in available}
        assert redis.setex_calls == [
            ("routing:conv:conv-1:provider:test-model", STICKY_TTL_SECONDS, key)
        ]

    @pytest.mark.asyncio
    async def test_hit_returns_stored_when_available(self):
        available, model = _candidates("a", "b")
        stored = available[1][1]
        redis = FakeRedis({"routing:conv:conv-1:provider:test-model": stored})
        ctx = _ctx(model, conversation_key="conv-1", redis=redis)
        key = await resolve_sticky_key(available, ctx)
        assert key == stored
        assert redis.setex_calls == []  # no re-persist on hit

    @pytest.mark.asyncio
    async def test_stored_not_available_resolves_new(self):
        available, model = _candidates("a", "b")
        redis = FakeRedis({"routing:conv:conv-1:provider:test-model": "ghost::9"})
        ctx = _ctx(model, conversation_key="conv-1", redis=redis)
        key = await resolve_sticky_key(available, ctx)
        assert key in {k for _, k in available}
        # Persisted the fresh choice over the stale mapping.
        assert redis.data["routing:conv:conv-1:provider:test-model"] == key

    @pytest.mark.asyncio
    async def test_redis_error_degrades_without_raising(self):
        available, model = _candidates("a", "b")
        redis = FakeRedis(fail=True)
        ctx = _ctx(model, conversation_key="conv-1", redis=redis)
        key = await resolve_sticky_key(available, ctx)
        assert key in {k for _, k in available}

    @pytest.mark.asyncio
    async def test_no_key_or_no_redis_returns_none(self):
        available, model = _candidates("a", "b")
        no_key = _ctx(model, conversation_key=None, redis=FakeRedis())
        assert await resolve_sticky_key(available, no_key) is None
        no_redis = _ctx(model, conversation_key="c", redis=None)
        assert await resolve_sticky_key(available, no_redis) is None
        empty = _ctx(model, conversation_key="c", redis=FakeRedis())
        assert await resolve_sticky_key([], empty) is None


class TestBalancedStrategy:
    def test_cold_start_degrades_to_cost_order(self):
        available, model = _candidates("a", "b", costs={"a": (5.0, 5.0), "b": (1.0, 1.0)})
        store = ProviderStatsStore()  # no samples
        picked = pick_provider(
            available,
            ProviderSelectionStrategy.BALANCED,
            _ctx(model, stats_store=store),
        )
        assert picked[0].provider == "b"

    def test_lower_latency_wins_when_costs_equal(self):
        available, model = _candidates(
            "slow", "fast", costs={"slow": (2.0, 2.0), "fast": (2.0, 2.0)}
        )
        store = ProviderStatsStore()
        store.observe("slow::0", 2000.0)
        store.observe("fast::0", 200.0)
        picked = pick_provider(
            available,
            ProviderSelectionStrategy.BALANCED,
            _ctx(model, stats_store=store),
        )
        assert picked[0].provider == "fast"

    def test_latency_can_outweigh_cost(self):
        # "a" is cheap (score cost 0) but very slow; "b" is pricey but instant.
        available, model = _candidates("a", "b", costs={"a": (1.0, 1.0), "b": (10.0, 10.0)})
        store = ProviderStatsStore()
        store.observe("a::0", 5000.0)
        store.observe("b::0", 50.0)
        picked = pick_provider(
            available,
            ProviderSelectionStrategy.BALANCED,
            _ctx(model, stats_store=store),
        )
        assert picked[0].provider == "b"

    def test_cost_can_outweigh_latency(self):
        # Latencies nearly equal; the much cheaper provider should win.
        available, model = _candidates("a", "b", costs={"a": (1.0, 1.0), "b": (50.0, 50.0)})
        store = ProviderStatsStore()
        store.observe("a::0", 100.0)
        store.observe("b::0", 90.0)
        picked = pick_provider(
            available,
            ProviderSelectionStrategy.BALANCED,
            _ctx(model, stats_store=store),
        )
        assert picked[0].provider == "a"

    def test_partial_samples_use_median(self):
        # Only one sampled provider; the unsampled one takes the median and
        # the call must not crash or exclude anyone.
        available, model = _candidates(
            "a", "b", "c", costs={"a": (1.0, 1.0), "b": (2.0, 2.0), "c": (3.0, 3.0)}
        )
        store = ProviderStatsStore()
        store.observe("c::0", 10.0)  # cheap-latency for the priciest provider
        ordered = order_providers(
            available,
            ProviderSelectionStrategy.BALANCED,
            _ctx(model, stats_store=store),
        )
        assert {p.provider for p, _ in ordered} == {"a", "b", "c"}

    def test_no_stats_store_degrades_to_cost_order(self):
        available, model = _candidates("a", "b", costs={"a": (5.0, 5.0), "b": (1.0, 1.0)})
        picked = pick_provider(
            available,
            ProviderSelectionStrategy.BALANCED,
            _ctx(model, stats_store=None),
        )
        assert picked[0].provider == "b"


class TestProviderStatsStore:
    def test_ewma_updates(self):
        store = ProviderStatsStore()
        assert store.get("k") is None
        store.observe("k", 100.0)
        assert store.get("k") == 100.0
        store.observe("k", 200.0)
        assert store.get("k") == pytest.approx(130.0)  # 0.3*200 + 0.7*100
        assert store.sample_count("k") == 2

    def test_negative_ignored(self):
        store = ProviderStatsStore()
        store.observe("k", -5.0)
        assert store.get("k") is None

    def test_reset(self):
        store = ProviderStatsStore()
        store.observe("a", 1.0)
        store.observe("b", 2.0)
        store.reset("a")
        assert store.get("a") is None
        assert store.get("b") == 2.0
        store.reset()
        assert store.key_count == 0
