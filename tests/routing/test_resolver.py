from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.routing.resolver import resolve_virtual_model
from llm_proxy.routing.types import RoutingMode


@pytest.fixture(autouse=True)
def _mock_embedding_signal(monkeypatch):
    """Prevent ML model loading in get_embedding_signal.

    The real implementation downloads a HuggingFace ONNX model + tokenizer
    which is slow (~2s) and unnecessary for unit tests.
    """
    monkeypatch.setattr(
        "llm_proxy.routing.resolver.get_embedding_signal",
        AsyncMock(return_value=None),
    )


@pytest.mark.asyncio
async def test_resolve_returns_decision(monkeypatch):
    # Fake config with one eligible model.
    fake_config = MagicMock()
    fake_config.models = {
        "good": MagicMock(
            auto_eligible=True,
            quality_tier="BALANCED",
            providers=[
                MagicMock(
                    priority=0,
                    provider_model_name="up",
                    input_cost_per_1m=1.0,
                    output_cost_per_1m=4.0,
                    cached_read_cost_per_1m=None,
                    cached_write_cost_per_1m=None,
                )
            ],
            input_cost_per_1m=None,
            output_cost_per_1m=None,
            cached_read_cost_per_1m=None,
            cached_write_cost_per_1m=None,
        ),
    }
    fake_config.provider_configs = {}

    cm = MagicMock()
    cm.get_smart_routing_config = AsyncMock(return_value=MagicMock(enabled=True))

    # Stub route() to a fixed decision.
    from llm_proxy.routing import api as routing_api

    monkeypatch.setattr(
        routing_api,
        "route",
        lambda **kw: MagicMock(
            model="good",
            tier=MagicMock(),
            complexity=0.4,
            confidence=0.7,
            reasoning={"method": "test"},
            cost_estimate=0.001,
            savings=0.0,
            fallback_chain=["good"],
            candidate_scores={"good": 1.0},
        ),
    )

    decision = await resolve_virtual_model(
        mode=RoutingMode.AUTO,
        messages=[{"role": "user", "content": "hi"}],
        request=MagicMock(),
        config=fake_config,
        config_manager=cm,
        app_state=MagicMock(),
        session=MagicMock(),
    )
    assert decision.model == "good"


@pytest.mark.asyncio
async def test_resolve_raises_when_pool_empty():
    fake_config = MagicMock()
    fake_config.models = {}
    fake_config.provider_configs = {}
    cm = MagicMock()
    cm.get_smart_routing_config = AsyncMock(return_value=MagicMock(enabled=True))
    from llm_proxy.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        await resolve_virtual_model(
            mode=RoutingMode.AUTO,
            messages=[{"role": "user", "content": "hi"}],
            request=MagicMock(),
            config=fake_config,
            config_manager=cm,
            app_state=MagicMock(),
            session=MagicMock(),
        )
