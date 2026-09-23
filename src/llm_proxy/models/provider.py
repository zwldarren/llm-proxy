"""Provider model information types.

These types are used by both the providers layer and the API layer,
so they live in the models layer to avoid reverse dependencies
(providers importing from api).
"""

from pydantic import BaseModel, Field


class ProviderModelArchitecture(BaseModel):
    """Modalities a model accepts and produces.

    Upstream ``/models`` endpoints describe this as
    ``architecture.input_modalities`` / ``architecture.output_modalities``
    (OpenRouter, vLLM); providers without the block leave it ``None``.
    """

    input_modalities: list[str] = Field(
        default_factory=list, description="Modalities the model accepts (text, image, audio, ...)"
    )
    output_modalities: list[str] = Field(
        default_factory=list, description="Modalities the model produces (text, image, audio, ...)"
    )


class ProviderModelPricing(BaseModel):
    """Upstream-reported price per unit, verbatim.

    Values are USD strings upstream and are surfaced as floats in the unit the
    upstream declares — per token for the text dimensions, per image/audio unit
    for the rest — so no conversion happens here. Display code that needs a
    per-1M figure converts at render time.
    """

    prompt: float | None = Field(None, description="Price per input token (USD)")
    completion: float | None = Field(None, description="Price per output token (USD)")
    request: float | None = Field(None, description="Price per request (USD)")


class ProviderModelInfo(BaseModel):
    """Schema for a model available from a provider."""

    id: str = Field(..., description="Model ID to use in API requests")
    name: str = Field(..., description="Human-readable model name")
    description: str | None = Field(None, description="Model description")
    owned_by: str | None = Field(None, description="Organization that owns the model")
    context_length: int | None = Field(None, description="Maximum context window in tokens")
    architecture: ProviderModelArchitecture | None = Field(
        None, description="Accepted and produced modalities"
    )
    supported_parameters: list[str] = Field(
        default_factory=list, description="Request parameters the model accepts"
    )
    pricing: ProviderModelPricing | None = Field(None, description="Upstream-reported pricing")
