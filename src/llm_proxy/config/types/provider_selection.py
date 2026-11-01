"""Global provider-selection configuration (stored in server_config)."""

from typing import Any

from pydantic import BaseModel, Field

from llm_proxy.config.types.model import (
    DEFAULT_PROVIDER_SELECTION_STRATEGY,
    ProviderSelectionStrategy,
)


class ProviderSelectionConfig(BaseModel):
    """Global provider-selection strategy applied to every model.

    The strategy orders same-priority provider candidates of a model when
    the selector picks the next provider; higher-priority groups are always
    tried first regardless of strategy.
    """

    strategy: ProviderSelectionStrategy = Field(
        default=DEFAULT_PROVIDER_SELECTION_STRATEGY,
        description=(
            "How to pick among same-priority providers: 'session_sticky' (default) "
            "pins a conversation to one provider for cache affinity, "
            "'random' rotates every request, 'cost_optimized' picks the cheapest first, "
            "'balanced' weighs cost against observed latency"
        ),
    )

    @staticmethod
    def from_row(value: dict[str, Any] | None) -> ProviderSelectionConfig:
        return ProviderSelectionConfig(**(value or {}))

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()
