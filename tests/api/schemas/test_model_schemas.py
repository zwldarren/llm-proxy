"""Tests for the model schemas: status vocabulary, capabilities, pricing tiers."""

import pytest
from pydantic import ValidationError

from llm_proxy.api.schemas.models import (
    ModelCreate,
    ModelProviderMapping,
    ModelRead,
    ModelUpdate,
)


def test_model_update_rejects_unknown_status() -> None:
    """PATCH must enforce the same models.dev vocabulary as POST."""
    with pytest.raises(ValidationError, match="status must be 'beta' or 'deprecated'"):
        ModelUpdate(status="active")


def test_model_update_normalizes_status() -> None:
    update = ModelUpdate(status=" Deprecated ")
    assert update.status == "deprecated"


def test_model_create_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError, match="status must be 'beta' or 'deprecated'"):
        ModelCreate(name="m", providers=[], status="bogus")


def test_model_read_derives_capabilities() -> None:
    """ModelRead exposes backend-derived capabilities in plaza badge order."""
    read = ModelRead(
        id=1,
        name="m",
        providers=[],
        supports_images=True,
        supports_tts=True,
        supports_systemone=True,
        reasoning=True,
        experimental=True,
    )
    assert read.capabilities == ["vision", "tts", "systemone", "reasoning", "experimental"]


def test_model_update_sorts_pricing_tiers_by_threshold() -> None:
    update = ModelUpdate(
        pricing_tiers=[
            {"threshold": 272000, "input_cost_per_1m": 5.0},
            {"threshold": 128000, "input_cost_per_1m": 2.0},
        ]
    )
    assert update.pricing_tiers is not None
    assert [tier.threshold for tier in update.pricing_tiers] == [128000, 272000]


def test_model_update_rejects_duplicate_tier_thresholds() -> None:
    with pytest.raises(ValidationError, match="unique"):
        ModelUpdate(
            pricing_tiers=[
                {"threshold": 1000, "input_cost_per_1m": 1.0},
                {"threshold": 1000, "input_cost_per_1m": 2.0},
            ]
        )


def test_model_provider_mapping_carries_pricing_tiers() -> None:
    mapping = ModelProviderMapping(
        provider_name="openai",
        provider_model_name="gpt-5.4",
        pricing_tiers=[{"threshold": 272000, "input_cost_per_1m": 5.0, "output_cost_per_1m": 22.5}],
    )
    assert mapping.pricing_tiers is not None
    assert mapping.pricing_tiers[0].threshold == 272000
    assert mapping.pricing_tiers[0].output_cost_per_1m == 22.5


def test_model_create_accepts_pricing_tiers() -> None:
    model = ModelCreate(
        name="m",
        providers=[{"provider_name": "openai", "provider_model_name": "m"}],
        pricing_tiers=[{"threshold": 200000, "input_cost_per_1m": 6.0}],
    )
    assert model.pricing_tiers is not None
    assert model.pricing_tiers[0].threshold == 200000


def test_model_update_rejects_an_empty_provider_list() -> None:
    """An update replaces the whole list, so empty would strip every provider."""
    with pytest.raises(ValidationError, match="at least 1 item"):
        ModelUpdate(providers=[])


def test_model_update_accepts_leaving_providers_alone() -> None:
    assert ModelUpdate(name="m").providers is None


@pytest.mark.parametrize(
    "field",
    [
        "input_cost_per_1m",
        "output_cost_per_1m",
        "cached_read_cost_per_1m",
        "cached_write_cost_per_1m",
        "audio_input_cost_per_1m",
        "audio_output_cost_per_1m",
        "image_input_cost_per_1m",
        "reasoning_cost_per_1m",
        "cost_per_image",
        "audio_cost_per_minute",
        "tts_cost_per_1m_chars",
        "web_search_cost_per_1k",
    ],
)
def test_model_update_rejects_negative_costs(field: str) -> None:
    """Every cost dimension is non-negative on update, as it is on create."""
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ModelUpdate(**{field: -0.5})


def test_model_create_accepts_reasoning_rate() -> None:
    model = ModelCreate(
        name="gpt-5.4",
        providers=[ModelProviderMapping(provider_name="openai", provider_model_name="gpt-5.4")],
        reasoning_cost_per_1m=30.0,
    )

    assert model.reasoning_cost_per_1m == 30.0


def test_provider_mapping_carries_time_windows() -> None:
    mapping = ModelProviderMapping(
        provider_name="ark",
        provider_model_name="doubao-pro",
        pricing_time_windows=[
            {
                "start": "22:00",
                "end": "08:00",
                "days": [0, 1],
                "timezone": "Asia/Shanghai",
                "input_cost_per_1m": 0.5,
                "reasoning_cost_per_1m": 1.0,
            }
        ],
    )

    assert mapping.pricing_time_windows is not None
    window = mapping.pricing_time_windows[0]
    assert window.start == "22:00"
    assert window.days == [0, 1]
    assert window.reasoning_cost_per_1m == 1.0


def test_provider_mapping_rejects_overlapping_time_windows() -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        ModelProviderMapping(
            provider_name="ark",
            provider_model_name="doubao-pro",
            pricing_time_windows=[
                {"start": "08:00", "end": "20:00"},
                {"start": "19:00", "end": "23:00"},
            ],
        )


def test_model_update_normalizes_time_windows_order() -> None:
    update = ModelUpdate(
        pricing_time_windows=[
            {"start": "20:00", "end": "21:00"},
            {"start": "06:00", "end": "07:00"},
        ]
    )

    assert update.pricing_time_windows is not None
    assert [window.start for window in update.pricing_time_windows] == ["06:00", "20:00"]


def test_model_create_accepts_service_tier_pricing() -> None:
    model = ModelCreate(
        name="gpt-5.4",
        providers=[ModelProviderMapping(provider_name="openai", provider_model_name="gpt-5.4")],
        service_tier_pricing={"FLEX": {"input_cost_per_1m": 1.25, "output_cost_per_1m": 7.5}},
    )

    assert model.service_tier_pricing is not None
    assert model.service_tier_pricing["flex"].output_cost_per_1m == 7.5
