"""Tests for admin Pydantic schemas (API key MCP fields, model status)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from llm_proxy.api.schemas.admin import (
    ApiKeyCreate,
    ApiKeyRead,
    ApiKeyResponse,
    ApiKeyUpdate,
    ModelCreate,
    ModelRead,
    ModelUpdate,
)

# --- API key MCP field tests ---


def test_api_key_create_accepts_mcp_fields() -> None:
    key = ApiKeyCreate(
        name="agent",
        allowed_models=None,
        allowed_mcp_servers=["github_mcp"],
    )
    assert key.allowed_mcp_servers == ["github_mcp"]


def test_api_key_read_has_mcp_fields() -> None:
    key = ApiKeyRead(
        name="agent",
        allowed_models=None,
        allowed_mcp_servers=["github_mcp"],
        user_id=1,
        created_at=datetime.now(UTC),
        last_used_at=None,
        is_active=True,
    )
    assert key.allowed_mcp_servers == ["github_mcp"]


def test_api_key_response_has_mcp_fields() -> None:
    resp = ApiKeyResponse(
        name="agent",
        key="sk_abc",
        allowed_models=None,
        allowed_mcp_servers=["github_mcp"],
        created_at=datetime.now(UTC),
    )
    assert resp.allowed_mcp_servers == ["github_mcp"]


def test_api_key_update_accepts_mcp_fields() -> None:
    update = ApiKeyUpdate(
        allowed_mcp_servers=["github_mcp"],
    )
    assert update.allowed_mcp_servers == ["github_mcp"]


# --- Model status vocabulary tests ---


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
        reasoning=True,
        experimental=True,
    )
    assert read.capabilities == ["vision", "tts", "reasoning", "experimental"]


# --- Context-tier pricing tests ---


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
    from llm_proxy.api.schemas.admin import ModelProviderMapping

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
