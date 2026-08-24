"""Provider model information types.

These types are used by both the providers layer and the API layer,
so they live in the models layer to avoid reverse dependencies
(providers importing from api).
"""

from pydantic import BaseModel, Field


class ProviderModelInfo(BaseModel):
    """Schema for a model available from a provider."""

    id: str = Field(..., description="Model ID to use in API requests")
    name: str = Field(..., description="Human-readable model name")
    description: str | None = Field(None, description="Model description")
    owned_by: str | None = Field(None, description="Organization that owns the model")
