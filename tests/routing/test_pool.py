"""Tests for routing.pool — build candidate pool from ProxyConfig."""

from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.config.types.main import ProxyConfig, ServerParams
from llm_proxy.config.types.model import ModelConfig, ModelProviderConfig
from llm_proxy.routing.pool import build_candidate_pool
from llm_proxy.routing.types import ServedQuality


def _proxy_config() -> ProxyConfig:
    eligible = ModelConfig(
        providers=[
            ModelProviderConfig(
                provider="p1",
                priority=0,
                provider_model_name="up-1",
                input_cost_per_1m=1.0,
                output_cost_per_1m=4.0,
            )
        ],
        auto_eligible=True,
        quality_tier="BALANCED",
    )
    not_eligible = ModelConfig(
        providers=[ModelProviderConfig(provider="p1", provider_model_name="up-2")]
    )
    return ProxyConfig(
        server_params=ServerParams(auth=ProxyAuthConfig(jwt_secret="test-secret")),
        models={"good": eligible, "other": not_eligible},
    )


def test_build_pool_only_includes_eligible():
    pool = build_candidate_pool(_proxy_config())
    assert pool.available_models == ["good"]
    assert "good" in pool.pricing
    assert pool.pricing["good"].input_price == 1.0
    assert pool.served_qualities["good"] == ServedQuality.BALANCED


def test_build_pool_defaults_to_economy_when_unset():
    cfg = _proxy_config()
    cfg.models["good"].quality_tier = None
    pool = build_candidate_pool(cfg)
    assert pool.served_qualities["good"] == ServedQuality.ECONOMY


def test_build_pool_omits_none_and_empty_routing_assignments():
    """None and [] both mean "all modes" and are omitted from the pool."""
    cfg = _proxy_config()
    cfg.models["good"].routing_assignments = None
    cfg.models["other"].auto_eligible = True
    cfg.models["other"].routing_assignments = []

    pool = build_candidate_pool(cfg)

    assert "good" not in pool.routing_assignments
    assert "other" not in pool.routing_assignments


def test_build_pool_populates_context_lengths():
    """context_lengths dict should be populated from model.context_length."""
    cfg = _proxy_config()
    cfg.models["good"].context_length = 200_000
    cfg.models["other"].auto_eligible = True
    cfg.models["other"].context_length = None

    pool = build_candidate_pool(cfg)

    assert pool.context_lengths["good"] == 200_000
    assert pool.context_lengths["other"] is None
