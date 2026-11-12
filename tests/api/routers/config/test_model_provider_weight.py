"""Admin model config: the provider mapping ``weight`` on the wire."""

import pytest
from pydantic import ValidationError

from llm_proxy.api.routers.config.models import _build_providers_list
from llm_proxy.api.schemas.models import ModelProviderMapping


def test_build_providers_list_carries_the_weight():
    built = _build_providers_list(
        [
            {"provider_name": "a", "provider_model_name": "m", "weight": 2},
            {"provider_name": "b", "provider_model_name": "m"},
        ]
    )

    # A row submitted without a weight falls back to the neutral 1.
    assert [entry["weight"] for entry in built] == [2, 1.0]


def test_mapping_schema_defaults_to_a_neutral_weight():
    mapping = ModelProviderMapping(provider_name="a", provider_model_name="m")

    assert mapping.weight == 1.0


def test_mapping_schema_accepts_a_fractional_weight():
    mapping = ModelProviderMapping(provider_name="a", provider_model_name="m", weight=0.5)

    assert mapping.weight == 0.5


def test_mapping_schema_rejects_a_negative_weight():
    with pytest.raises(ValidationError):
        ModelProviderMapping(provider_name="a", provider_model_name="m", weight=-1)
