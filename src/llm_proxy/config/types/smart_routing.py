"""Global smart routing configuration (stored in server_config)."""

from typing import Any

from pydantic import BaseModel, Field


class SmartRoutingConfig(BaseModel):
    enabled: bool = Field(default=False)
    mode_weights: dict[str, float] = Field(
        default_factory=lambda: {"fast": 0.35, "auto": 0.65, "best": 1.0}
    )

    @staticmethod
    def from_row(value: dict[str, Any] | None) -> SmartRoutingConfig:
        return SmartRoutingConfig(**(value or {}))

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()
