import pytest

from llm_proxy.routing.config import (
    DEFAULT_CONFIG,
    VIRTUAL_MODEL_IDS,
    routing_mode_from_model,
)
from llm_proxy.routing.types import RoutingMode


@pytest.mark.parametrize(
    "name,expected",
    [
        ("auto", RoutingMode.AUTO),
        ("AUTO", RoutingMode.AUTO),
        ("fast", RoutingMode.FAST),
        ("best", RoutingMode.BEST),
        ("uncommon-route/auto", None),  # we use bare ids, not the upstream prefix
        ("gpt-4o", None),
        ("", None),
    ],
)
def test_routing_mode_from_model(name, expected):
    assert routing_mode_from_model(name) is expected


def test_virtual_model_ids_cover_three_modes():
    assert set(VIRTUAL_MODEL_IDS) == {RoutingMode.AUTO, RoutingMode.FAST, RoutingMode.BEST}


def test_default_config_has_per_mode_weights():
    auto_q = DEFAULT_CONFIG.modes[RoutingMode.AUTO].selection.quality_alignment
    fast_q = DEFAULT_CONFIG.modes[RoutingMode.FAST].selection.quality_alignment
    best_q = DEFAULT_CONFIG.modes[RoutingMode.BEST].selection.quality_alignment
    assert auto_q > 0
    assert fast_q < auto_q
    assert best_q >= auto_q
