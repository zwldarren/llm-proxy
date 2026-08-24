"""Tests for the global provider-selection configuration."""

import pytest
from pydantic import ValidationError

from llm_proxy.config.types.model import ProviderSelectionStrategy
from llm_proxy.config.types.provider_selection import ProviderSelectionConfig


class TestProviderSelectionConfig:
    def test_default_strategy_is_random(self):
        """The default global strategy preserves historical behavior."""
        cfg = ProviderSelectionConfig()
        assert cfg.strategy is ProviderSelectionStrategy.RANDOM

    @pytest.mark.parametrize(
        "raw",
        [
            "random",
            "session_sticky",
            "cost_optimized",
            "balanced",
        ],
    )
    def test_from_row_accepts_each_strategy(self, raw: str):
        """from_row parses every supported strategy value."""
        cfg = ProviderSelectionConfig.from_row({"strategy": raw})
        assert cfg.strategy == ProviderSelectionStrategy(raw)

    def test_from_row_empty_and_none(self):
        """from_row falls back to defaults for empty/missing rows."""
        assert ProviderSelectionConfig.from_row(None).strategy is ProviderSelectionStrategy.RANDOM
        assert ProviderSelectionConfig.from_row({}).strategy is ProviderSelectionStrategy.RANDOM

    def test_from_row_unknown_strategy_raises(self):
        """Unknown strategies surface as validation errors (manager degrades gracefully)."""
        with pytest.raises(ValidationError):
            ProviderSelectionConfig.from_row({"strategy": "unknown"})

    def test_to_row_roundtrip(self):
        """to_row produces a plain dict that from_row can re-parse."""
        cfg = ProviderSelectionConfig(strategy=ProviderSelectionStrategy.COST_OPTIMIZED)
        row = cfg.to_row()
        assert row == {"strategy": "cost_optimized"}
        assert (
            ProviderSelectionConfig.from_row(row).strategy
            is ProviderSelectionStrategy.COST_OPTIMIZED
        )
