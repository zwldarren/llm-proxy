"""Pydantic schemas for model metadata/capability sync from models.dev API."""

from pydantic import BaseModel, Field


class ModelMetadataOption(BaseModel):
    """models.dev metadata snapshot for one catalog (provider, model) entry.

    Field names mirror the ``ModelRecord`` columns they would be written to.
    ``None`` means the catalog entry carries no value for the field — diffs
    never propose clearing a stored value down to ``None`` from missing data.

    The stored-side snapshot uses ``source=""``.
    """

    source: str = Field(..., description="models.dev provider key ('' for stored values)")
    attachment: bool | None = Field(None, description="Supports file/image attachments")
    reasoning: bool | None = Field(None, description="Supports extended reasoning")
    tool_call: bool | None = Field(None, description="Supports tool/function calling")
    structured_output: bool | None = Field(None, description="Supports structured output")
    temperature: bool | None = Field(None, description="Supports temperature control")
    open_weights: bool | None = Field(None, description="Open weights model")
    status: str | None = Field(None, description="models.dev status: 'beta' | 'deprecated'")
    family: str | None = Field(None, description="Model family (e.g. 'claude-sonnet')")
    knowledge: str | None = Field(None, description="Knowledge cutoff date")
    release_date: str | None = Field(None, description="Release date")
    context_length: int | None = Field(None, description="Context window in tokens")
    max_output_tokens: int | None = Field(
        None, description="Maximum output tokens in a single response"
    )
    # Derived from modalities.input: True/False when modalities are published,
    # None when the entry carries no modalities information at all.
    supports_images: bool | None = Field(None, description="Accepts image input (vision)")


class SyncMetadataResult(BaseModel):
    """Result schema for a single model's metadata preview."""

    model_name: str = Field(..., description="Model name in proxy")
    old: ModelMetadataOption = Field(..., description="Currently stored metadata values")
    available_sources: list[ModelMetadataOption] = Field(
        default_factory=list, description="All matching models.dev entries, one per provider key"
    )
    selected_source: str | None = Field(
        None, description="models.dev provider key selected by default for the preview"
    )
    message: str = Field(..., description="Status message")


class SyncMetadataResponse(BaseModel):
    """Response schema for the metadata sync preview."""

    success: bool = Field(..., description="Whether the preview was built successfully")
    total_models: int = Field(..., description="Total number of models checked")
    changed_count: int = Field(..., description="Models with metadata changes")
    unchanged_count: int = Field(..., description="Models whose metadata matches models.dev")
    nodata_count: int = Field(..., description="Models with no matching models.dev entry")
    results: list[SyncMetadataResult] = Field(
        default_factory=list, description="Details for each model"
    )
    error: str | None = Field(default=None, description="Error message if the preview failed")


class MetadataUpdateItem(BaseModel):
    """Explicit metadata update for a single model.

    Only fields explicitly provided are written (a provided ``null`` clears
    the field); omitted fields are left untouched.

    Note: the consumer must use ``model_dump(exclude_unset=True)`` to honor
    the partial-update contract; plain attribute access will return ``None``
    for both omitted and explicitly-null fields.
    """

    model_name: str = Field(..., description="Model name in proxy to update")
    supports_images: bool | None = Field(None, description="Accepts image input (vision)")
    attachment: bool | None = Field(None, description="Supports file/image attachments")
    reasoning: bool | None = Field(None, description="Supports extended reasoning")
    tool_call: bool | None = Field(None, description="Supports tool/function calling")
    structured_output: bool | None = Field(None, description="Supports structured output")
    temperature: bool | None = Field(None, description="Supports temperature control")
    open_weights: bool | None = Field(None, description="Open weights model")
    status: str | None = Field(None, description="models.dev status: 'beta' | 'deprecated'")
    family: str | None = Field(None, description="Model family")
    knowledge: str | None = Field(None, description="Knowledge cutoff date")
    release_date: str | None = Field(None, description="Release date")
    context_length: int | None = Field(None, ge=0, description="Context window in tokens")
    max_output_tokens: int | None = Field(
        None, ge=0, description="Maximum output tokens in a single response"
    )


class ApplyMetadataRequest(BaseModel):
    """Request schema for applying reviewed metadata updates."""

    updates: list[MetadataUpdateItem] = Field(
        ..., description="Explicit per-model metadata updates to apply"
    )


class ApplyMetadataResult(BaseModel):
    """Result for a single applied metadata update."""

    model_name: str = Field(..., description="Model name in proxy")
    applied: bool = Field(..., description="Whether the update was applied")
    message: str = Field(..., description="Status message")


class ApplyMetadataResponse(BaseModel):
    """Response schema for applying metadata updates."""

    success: bool = Field(..., description="Whether all updates were applied")
    applied_count: int = Field(..., description="Number of models updated")
    failed_count: int = Field(..., description="Number of models that failed")
    results: list[ApplyMetadataResult] = Field(
        default_factory=list, description="Per-model results"
    )
