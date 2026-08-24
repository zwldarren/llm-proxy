"""Pydantic schemas for model pricing sync from models.dev API."""

from pydantic import BaseModel, Field


class ModelPricingInfo(BaseModel):
    """Schema for model pricing information from models.dev."""

    model_id: str = Field(..., description="Model ID")
    provider: str = Field(..., description="Provider name")
    input_cost_per_1m: float | None = Field(None, description="Input cost per 1M tokens")
    output_cost_per_1m: float | None = Field(None, description="Output cost per 1M tokens")
    cached_read_cost_per_1m: float | None = Field(
        None, description="Cached read cost per 1M tokens"
    )
    cached_write_cost_per_1m: float | None = Field(
        None, description="Cached write cost per 1M tokens"
    )
    audio_input_cost_per_1m: float | None = Field(
        None, description="Audio input cost per 1M tokens"
    )
    audio_output_cost_per_1m: float | None = Field(
        None, description="Audio output cost per 1M tokens"
    )


class PricingSourceSelection(BaseModel):
    """Schema for selecting a pricing source for a model."""

    model_name: str = Field(..., description="Model name in proxy")
    pricing_source: str = Field(..., description="Pricing source key (models.dev provider key)")


class SyncPricingRequest(BaseModel):
    """Request schema for syncing model pricing."""

    dry_run: bool = Field(
        default=False,
        description="If true, only return what would be updated without making changes",
    )
    preserve_custom_pricing: bool = Field(
        default=True,
        description="If true, skip providers that already have custom pricing set",
    )
    pricing_selections: list[PricingSourceSelection] = Field(
        default_factory=list,
        description="Optional: specify which pricing source to use for each model",
    )


class PricingOption(BaseModel):
    """Schema for an available pricing option."""

    source: str = Field(..., description="Pricing source (models.dev provider key)")
    input_cost_per_1m: float | None = Field(None, description="Input cost per 1M tokens")
    output_cost_per_1m: float | None = Field(None, description="Output cost per 1M tokens")
    cached_read_cost_per_1m: float | None = Field(
        None, description="Cached read cost per 1M tokens"
    )
    cached_write_cost_per_1m: float | None = Field(
        None, description="Cached write cost per 1M tokens"
    )
    audio_input_cost_per_1m: float | None = Field(
        None, description="Audio input cost per 1M tokens"
    )
    audio_output_cost_per_1m: float | None = Field(
        None, description="Audio output cost per 1M tokens"
    )


class SyncPricingResult(BaseModel):
    """Result schema for a single model pricing update."""

    mapping_id: int = Field(..., description="ModelProviderRecord ID (used to apply updates)")
    model_name: str = Field(..., description="Model name in proxy")
    provider_model_name: str = Field(..., description="Provider's model name")
    provider: str = Field(..., description="Provider name")
    old_input_cost: float | None = Field(None, description="Previous input cost per 1M tokens")
    old_output_cost: float | None = Field(None, description="Previous output cost per 1M tokens")
    new_input_cost: float | None = Field(None, description="New input cost per 1M tokens")
    new_output_cost: float | None = Field(None, description="New output cost per 1M tokens")
    old_cached_read_cost: float | None = Field(
        None, description="Previous cached read cost per 1M tokens"
    )
    new_cached_read_cost: float | None = Field(
        None, description="New cached read cost per 1M tokens"
    )
    old_cached_write_cost: float | None = Field(
        None, description="Previous cached write cost per 1M tokens"
    )
    new_cached_write_cost: float | None = Field(
        None, description="New cached write cost per 1M tokens"
    )
    old_audio_input_cost: float | None = Field(
        None, description="Previous audio input cost per 1M tokens"
    )
    new_audio_input_cost: float | None = Field(
        None, description="New audio input cost per 1M tokens"
    )
    old_audio_output_cost: float | None = Field(
        None, description="Previous audio output cost per 1M tokens"
    )
    new_audio_output_cost: float | None = Field(
        None, description="New audio output cost per 1M tokens"
    )
    updated: bool = Field(..., description="Whether the model was updated")
    message: str = Field(..., description="Status message")
    available_sources: list[PricingOption] = Field(
        default_factory=list,
        description="All available pricing sources for this model",
    )
    selected_source: str | None = Field(None, description="Which pricing source was used")


class PricingUpdateItem(BaseModel):
    """Explicit pricing update for a single model-provider mapping.

    Only fields explicitly provided are written; a provided ``null`` clears
    the field. Omitted fields are left untouched.

    Note: the consumer must use ``model_dump(exclude_unset=True)`` to honor
    the partial-update contract; plain attribute access will return ``None``
    for both omitted and explicitly-null fields.
    """

    mapping_id: int = Field(..., description="ModelProviderRecord ID to update")
    input_cost_per_1m: float | None = Field(None, description="Input cost per 1M tokens")
    output_cost_per_1m: float | None = Field(None, description="Output cost per 1M tokens")
    cached_read_cost_per_1m: float | None = Field(
        None, description="Cached read cost per 1M tokens"
    )
    cached_write_cost_per_1m: float | None = Field(
        None, description="Cached write cost per 1M tokens"
    )
    audio_input_cost_per_1m: float | None = Field(
        None, description="Audio input cost per 1M tokens"
    )
    audio_output_cost_per_1m: float | None = Field(
        None, description="Audio output cost per 1M tokens"
    )
    image_input_cost_per_1m: float | None = Field(
        None, ge=0, description="Cost per 1M image input tokens in USD"
    )
    cost_per_image: float | None = Field(None, ge=0, description="Cost per generated image in USD")
    audio_cost_per_minute: float | None = Field(
        None, ge=0, description="Cost per minute of audio (STT) in USD"
    )
    tts_cost_per_1m_chars: float | None = Field(
        None, ge=0, description="Cost per 1M characters (TTS) in USD"
    )
    web_search_cost_per_1k: float | None = Field(
        None, ge=0, description="Cost per 1k web search requests in USD"
    )


class ApplyPricingRequest(BaseModel):
    """Request schema for applying reviewed pricing updates."""

    updates: list[PricingUpdateItem] = Field(
        ..., description="Explicit per-mapping pricing updates to apply"
    )


class ApplyPricingResult(BaseModel):
    """Result for a single applied pricing update."""

    mapping_id: int = Field(..., description="ModelProviderRecord ID")
    applied: bool = Field(..., description="Whether the update was applied")
    message: str = Field(..., description="Status message")


class ApplyPricingResponse(BaseModel):
    """Response schema for applying pricing updates."""

    success: bool = Field(..., description="Whether all updates were applied")
    applied_count: int = Field(..., description="Number of mappings updated")
    failed_count: int = Field(..., description="Number of mappings that failed")
    results: list[ApplyPricingResult] = Field(
        default_factory=list, description="Per-mapping results"
    )


class SyncPricingResponse(BaseModel):
    """Response schema for syncing model pricing."""

    success: bool = Field(..., description="Whether the sync was successful")
    dry_run: bool = Field(..., description="Whether this was a dry run")
    total_models: int = Field(..., description="Total number of unique models checked")
    total_provider_mappings: int = Field(
        ..., description="Total number of provider mappings checked"
    )
    updated_count: int = Field(..., description="Number of provider mappings updated")
    skipped_count: int = Field(
        ..., description="Number of provider mappings skipped (no pricing found)"
    )
    unchanged_count: int = Field(
        ..., description="Number of provider mappings with unchanged pricing"
    )
    results: list[SyncPricingResult] = Field(
        default_factory=list, description="Details for each mapping"
    )
    error: str | None = Field(default=None, description="Error message if sync failed")
