"""Model metadata/capability sync from models.dev API.

Mirrors the pricing sync flow (``pricing.py``) for the models.dev-mirrored
attribute columns on ``ModelRecord``: a preview endpoint builds a per-model
diff against the catalog, and an apply endpoint writes exactly the fields the
caller reviewed. Proxy-bound ``supports_*`` flags stay admin-managed except
``supports_images`` (vision), which the catalog derives from modalities.
"""

from typing import Any

import httpx2
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_async_session_dep, get_http_client, require_admin_role
from llm_proxy.api.routers.config.helpers import commit_and_reload, get_config_repository
from llm_proxy.api.routers.config.metadata_schemas import (
    ApplyMetadataRequest,
    ApplyMetadataResponse,
    ApplyMetadataResult,
    ModelMetadataOption,
    SyncMetadataResponse,
    SyncMetadataResult,
)
from llm_proxy.core.models_dev import build_model_index, fetch_models_dev_data
from llm_proxy.database.tables import ModelRecord

router = APIRouter(
    prefix="/models", tags=["configuration"], dependencies=[Depends(require_admin_role)]
)

# ModelRecord columns synced from models.dev, in stable diff/display order.
_SYNCED_FIELDS: tuple[str, ...] = (
    "supports_images",
    "attachment",
    "reasoning",
    "tool_call",
    "structured_output",
    "temperature",
    "open_weights",
    "status",
    "family",
    "knowledge",
    "release_date",
    "context_length",
    "max_output_tokens",
)


def _coerce_int(value: Any) -> int | None:
    """Best-effort int coercion for optional models.dev payload fields."""
    try:
        return int(value) if value is not None else None
    except ValueError, TypeError:
        return None


def _coerce_str(value: Any) -> str | None:
    """Pass through only genuine string fields; anything else is 'no data'."""
    return value if isinstance(value, str) else None


def _coerce_bool(value: Any) -> bool | None:
    """Pass through only genuine boolean fields; anything else is 'no data'."""
    return value if isinstance(value, bool) else None


def _entry_to_option(provider_key: str, entry: dict[str, Any]) -> ModelMetadataOption:
    """Map one models.dev model entry to a metadata option."""
    limit = entry.get("limit")
    limit = limit if isinstance(limit, dict) else {}
    modalities = entry.get("modalities")
    modalities = modalities if isinstance(modalities, dict) else {}
    input_modalities = modalities.get("input")

    # Vision is derivable only when the entry publishes input modalities:
    # "image" present -> True, modalities without "image" -> False, no
    # modalities at all -> None (no signal, never a diff).
    if isinstance(input_modalities, list):
        supports_images: bool | None = "image" in input_modalities
    else:
        supports_images = None

    return ModelMetadataOption(
        source=provider_key,
        attachment=_coerce_bool(entry.get("attachment")),
        reasoning=_coerce_bool(entry.get("reasoning")),
        tool_call=_coerce_bool(entry.get("tool_call")),
        structured_output=_coerce_bool(entry.get("structured_output")),
        temperature=_coerce_bool(entry.get("temperature")),
        open_weights=_coerce_bool(entry.get("open_weights")),
        status=_coerce_str(entry.get("status")),
        family=_coerce_str(entry.get("family")),
        knowledge=_coerce_str(entry.get("knowledge")),
        release_date=_coerce_str(entry.get("release_date")),
        context_length=_coerce_int(limit.get("context")),
        max_output_tokens=_coerce_int(limit.get("output")),
        supports_images=supports_images,
    )


async def _fetch_models_dev_metadata(request: Request) -> dict[str, list[ModelMetadataOption]]:
    client = await get_http_client(request)
    data = await fetch_models_dev_data(client)

    catalog: dict[str, list[ModelMetadataOption]] = {}
    for model_id, entries in build_model_index(data).items():
        catalog[model_id] = [
            _entry_to_option(provider_key, entry) for provider_key, entry in entries
        ]
    return catalog


def _stored_option(model: ModelRecord) -> ModelMetadataOption:
    """Snapshot the currently stored metadata values (source='' marker)."""
    return ModelMetadataOption(
        source="",
        attachment=model.attachment,
        reasoning=model.reasoning,
        tool_call=model.tool_call,
        structured_output=model.structured_output,
        temperature=model.temperature,
        open_weights=model.open_weights,
        status=model.status,
        family=model.family,
        knowledge=model.knowledge,
        release_date=model.release_date,
        context_length=model.context_length,
        max_output_tokens=model.max_output_tokens,
        supports_images=model.supports_images,
    )


def _changed_fields(old: ModelMetadataOption, new: ModelMetadataOption) -> list[str]:
    """Fields where the catalog candidate differs from the stored values.

    Catalog fields with no data (``None``) never count as a change: missing
    upstream data must not clear stored values.
    """
    changed: list[str] = []
    for field in _SYNCED_FIELDS:
        new_value = getattr(new, field)
        if new_value is None:
            continue
        if new_value != getattr(old, field):
            changed.append(field)
    return changed


def _candidate_sources(
    model: ModelRecord, catalog: dict[str, list[ModelMetadataOption]]
) -> dict[str, ModelMetadataOption]:
    """Collect candidate catalog entries for a model, deduped by source key.

    Every provider mapping's provider_model_name is looked up (falling back to
    the proxy model name); the same models.dev provider key reached through
    multiple mappings contributes a single entry.
    """
    candidates: dict[str, ModelMetadataOption] = {}
    for mapping in model.provider_mappings:
        provider_model_name = mapping.provider_model_name or model.name
        for option in catalog.get(provider_model_name, []):
            candidates.setdefault(option.source, option)
    return candidates


@router.post("/sync-metadata", response_model=SyncMetadataResponse)
async def sync_model_metadata(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> SyncMetadataResponse:
    """Build a reviewable preview of models.dev metadata for every model.

    Preview-only: nothing is written here. Apply reviewed changes via
    ``POST /models/metadata/apply``. Candidates are collected from every
    provider mapping's provider_model_name and deduplicated by models.dev
    provider key; the alphabetically-first source is selected by default,
    mirroring the pricing sync.
    """
    try:
        catalog = await _fetch_models_dev_metadata(request)
    except httpx2.HTTPStatusError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        return SyncMetadataResponse(
            success=False,
            total_models=0,
            changed_count=0,
            unchanged_count=0,
            nodata_count=0,
            error=f"Failed to fetch metadata from models.dev: {status_code}",
        )
    except httpx2.RequestError as e:
        return SyncMetadataResponse(
            success=False,
            total_models=0,
            changed_count=0,
            unchanged_count=0,
            nodata_count=0,
            error=f"Network error when fetching metadata: {e!s}",
        )

    repo = get_config_repository(session)
    models = await repo.get_all_models()

    results: list[SyncMetadataResult] = []
    changed_count = 0
    unchanged_count = 0
    nodata_count = 0

    for model in models:
        old = _stored_option(model)
        candidates = _candidate_sources(model, catalog)

        if not candidates:
            nodata_count += 1
            results.append(
                SyncMetadataResult(
                    model_name=model.name,
                    old=old,
                    available_sources=[],
                    selected_source=None,
                    message="No models.dev entry found",
                )
            )
            continue

        ordered = [candidates[key] for key in sorted(candidates)]
        selected = ordered[0]
        fields = _changed_fields(old, selected)

        if not fields:
            unchanged_count += 1
            results.append(
                SyncMetadataResult(
                    model_name=model.name,
                    old=old,
                    available_sources=ordered,
                    selected_source=selected.source,
                    message="Metadata unchanged",
                )
            )
            continue

        changed_count += 1
        results.append(
            SyncMetadataResult(
                model_name=model.name,
                old=old,
                available_sources=ordered,
                selected_source=selected.source,
                message="Would update: " + ", ".join(fields),
            )
        )

    return SyncMetadataResponse(
        success=True,
        total_models=len(models),
        changed_count=changed_count,
        unchanged_count=unchanged_count,
        nodata_count=nodata_count,
        results=results,
    )


@router.post("/metadata/apply", response_model=ApplyMetadataResponse)
async def apply_model_metadata(
    request_data: ApplyMetadataRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> ApplyMetadataResponse:
    """Apply explicitly reviewed metadata updates to models.

    Unlike the bulk sync endpoint, this applies exactly the updates provided
    by the caller — typically after the user has reviewed a preview and
    selected individual models. For each update item, only the fields
    explicitly provided are written (a provided ``null`` clears the field);
    omitted fields are left untouched.
    """
    repo = get_config_repository(session)

    results: list[ApplyMetadataResult] = []
    applied_count = 0
    failed_count = 0

    for item in request_data.updates:
        updates: dict[str, object] = item.model_dump(exclude_unset=True)
        updates.pop("model_name", None)

        if not updates:
            failed_count += 1
            results.append(
                ApplyMetadataResult(
                    model_name=item.model_name,
                    applied=False,
                    message="No metadata fields provided",
                )
            )
            continue

        updated = await repo._models.apply_model_metadata(item.model_name, updates)
        if updated is None:
            failed_count += 1
            results.append(
                ApplyMetadataResult(
                    model_name=item.model_name,
                    applied=False,
                    message="Model not found",
                )
            )
            continue

        applied_count += 1
        results.append(
            ApplyMetadataResult(
                model_name=item.model_name,
                applied=True,
                message="Applied",
            )
        )

    if applied_count > 0:
        await commit_and_reload(session, request)

    return ApplyMetadataResponse(
        success=failed_count == 0,
        applied_count=applied_count,
        failed_count=failed_count,
        results=results,
    )
