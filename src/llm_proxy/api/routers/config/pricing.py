"""Model pricing sync from models.dev API."""

import httpx2
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_async_session_dep, get_http_client, require_admin_role
from llm_proxy.api.routers.config.helpers import commit_and_reload, get_config_repository
from llm_proxy.api.routers.config.pricing_schemas import (
    ApplyPricingRequest,
    ApplyPricingResponse,
    ApplyPricingResult,
    ModelPricingInfo,
    PricingOption,
    SyncPricingRequest,
    SyncPricingResponse,
    SyncPricingResult,
)
from llm_proxy.http.client import fetch_json

router = APIRouter(
    prefix="/models", tags=["configuration"], dependencies=[Depends(require_admin_role)]
)

# models.dev API URL for fetching model pricing
_MODELS_DEV_API_URL = "https://models.dev/api.json"


def _is_valid_pricing(input_cost: float | None, output_cost: float | None) -> bool:
    """Check if pricing is valid (not both zero or missing)."""
    return bool(input_cost) or bool(output_cost)


async def _fetch_models_dev_pricing(request: Request) -> dict[str, list[ModelPricingInfo]]:
    client = await get_http_client(request)
    data = await fetch_json(client, _MODELS_DEV_API_URL)

    pricing_data: dict[str, list[ModelPricingInfo]] = {}

    for provider_key, provider_data in data.items():
        if not isinstance(provider_data, dict):
            continue

        models = provider_data.get("models", {})
        if not isinstance(models, dict):
            continue

        for model_id, model_info in models.items():
            if not isinstance(model_info, dict):
                continue

            cost = model_info.get("cost", {})
            if not isinstance(cost, dict):
                continue

            input_cost = cost.get("input")
            output_cost = cost.get("output")
            cache_read = cost.get("cache_read")
            cache_write = cost.get("cache_write")
            input_audio = cost.get("input_audio")
            output_audio = cost.get("output_audio")

            def _to_float(v):
                try:
                    return float(v) if v is not None else None
                except ValueError, TypeError:
                    return None

            input_cost_f = _to_float(input_cost)
            output_cost_f = _to_float(output_cost)
            cache_read_f = _to_float(cache_read)
            cache_write_f = _to_float(cache_write)
            input_audio_f = _to_float(input_audio)
            output_audio_f = _to_float(output_audio)

            if not _is_valid_pricing(input_cost_f, output_cost_f):
                continue

            pricing_info = ModelPricingInfo(
                model_id=model_id,
                provider=provider_key,
                input_cost_per_1m=input_cost_f,
                output_cost_per_1m=output_cost_f,
                cached_read_cost_per_1m=cache_read_f,
                cached_write_cost_per_1m=cache_write_f,
                audio_input_cost_per_1m=input_audio_f,
                audio_output_cost_per_1m=output_audio_f,
            )

            if model_id not in pricing_data:
                pricing_data[model_id] = []
            pricing_data[model_id].append(pricing_info)

            if "/" in model_id:
                _, suffix = model_id.rsplit("/", 1)
                if suffix not in pricing_data:
                    pricing_data[suffix] = []
                pricing_data[suffix].append(pricing_info)

    return pricing_data


def _find_model_pricing(
    provider_model_name: str,
    pricing_data: dict[str, list[ModelPricingInfo]],
    preferred_source: str | None = None,
) -> tuple[ModelPricingInfo | None, list[ModelPricingInfo]]:
    """Find pricing for a model from the models.dev data.

    Matches only by model ID, ignoring provider.

    Args:
        provider_model_name: The model name used with the provider
        pricing_data: The pricing data from models.dev (dict: model_id -> list of pricing)
        preferred_source: Optional preferred pricing source

    Returns:
        Tuple of (selected pricing, all available pricing options)
    """
    all_options = pricing_data.get(provider_model_name, [])

    if not all_options:
        return None, []

    if preferred_source:
        for option in all_options:
            if option.provider == preferred_source:
                return option, all_options

    return min(all_options, key=lambda x: x.provider), all_options


@router.post("/sync-pricing", response_model=SyncPricingResponse)
async def sync_model_pricing(
    request_data: SyncPricingRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> SyncPricingResponse:
    """Sync model pricing from models.dev API.

    This endpoint fetches the latest model pricing data from models.dev
    and updates the pricing for all configured models in the database.

    The pricing is matched based on the provider_model_name configured
    for each model-provider mapping. Pricing is stored at the provider
    mapping level (per-provider pricing).

    Users can specify which pricing source to use for each model via
    the pricing_selections parameter.
    """
    try:
        # Fetch pricing data from models.dev
        pricing_data = await _fetch_models_dev_pricing(request)
    except httpx2.HTTPStatusError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        return SyncPricingResponse(
            success=False,
            dry_run=request_data.dry_run,
            total_models=0,
            total_provider_mappings=0,
            updated_count=0,
            skipped_count=0,
            unchanged_count=0,
            error=f"Failed to fetch pricing from models.dev: {status_code}",
        )
    except httpx2.RequestError as e:
        return SyncPricingResponse(
            success=False,
            dry_run=request_data.dry_run,
            total_models=0,
            total_provider_mappings=0,
            updated_count=0,
            skipped_count=0,
            unchanged_count=0,
            error=f"Network error when fetching pricing: {e!s}",
        )

    # Build a lookup for user's pricing source selections
    selections_map: dict[str, str] = {}
    for selection in request_data.pricing_selections:
        selections_map[selection.model_name] = selection.pricing_source

    repo = get_config_repository(session)
    models = await repo.get_all_models()

    total_provider_mappings = sum(len(m.provider_mappings) for m in models)

    results: list[SyncPricingResult] = []
    updated_count = 0
    skipped_count = 0
    unchanged_count = 0

    for model in models:
        # Check if user has specified a preferred source for this model
        preferred_source = selections_map.get(model.name)

        # For each provider mapping, find and update pricing
        for mapping in model.provider_mappings:
            provider_model_name = mapping.provider_model_name or model.name
            provider_name = mapping.provider.name if mapping.provider else ""

            # Get current pricing from provider mapping (per-provider pricing)
            old_input_cost = mapping.input_cost_per_1m
            old_output_cost = mapping.output_cost_per_1m
            old_cached_read = mapping.cached_read_cost_per_1m
            old_cached_write = mapping.cached_write_cost_per_1m
            old_audio_input = mapping.audio_input_cost_per_1m
            old_audio_output = mapping.audio_output_cost_per_1m

            if request_data.preserve_custom_pricing and (
                old_input_cost is not None
                or old_output_cost is not None
                or old_cached_read is not None
                or old_cached_write is not None
                or old_audio_input is not None
                or old_audio_output is not None
            ):
                skipped_count += 1
                results.append(
                    SyncPricingResult(
                        mapping_id=mapping.id,
                        model_name=model.name,
                        provider_model_name=provider_model_name,
                        provider=provider_name,
                        old_input_cost=old_input_cost,
                        old_output_cost=old_output_cost,
                        new_input_cost=None,
                        new_output_cost=None,
                        updated=False,
                        message="Skipped: custom pricing preserved",
                        available_sources=[],
                        selected_source=None,
                    )
                )
                continue

            # Find pricing based on provider_model_name
            found_pricing, all_options = _find_model_pricing(
                provider_model_name, pricing_data, preferred_source
            )

            # Convert options to PricingOption schema
            available_sources = [
                PricingOption(
                    source=opt.provider,
                    input_cost_per_1m=opt.input_cost_per_1m,
                    output_cost_per_1m=opt.output_cost_per_1m,
                    cached_read_cost_per_1m=opt.cached_read_cost_per_1m,
                    cached_write_cost_per_1m=opt.cached_write_cost_per_1m,
                    audio_input_cost_per_1m=opt.audio_input_cost_per_1m,
                    audio_output_cost_per_1m=opt.audio_output_cost_per_1m,
                )
                for opt in all_options
            ]

            if not found_pricing:
                skipped_count += 1
                results.append(
                    SyncPricingResult(
                        mapping_id=mapping.id,
                        model_name=model.name,
                        provider_model_name=provider_model_name,
                        provider=provider_name,
                        old_input_cost=old_input_cost,
                        old_output_cost=old_output_cost,
                        new_input_cost=None,
                        new_output_cost=None,
                        updated=False,
                        message="No pricing found in models.dev",
                        available_sources=[],
                        selected_source=None,
                    )
                )
                continue

            # Check if any pricing field has changed
            pricing_changed = (
                old_input_cost != found_pricing.input_cost_per_1m
                or old_output_cost != found_pricing.output_cost_per_1m
                or old_cached_read != found_pricing.cached_read_cost_per_1m
                or old_cached_write != found_pricing.cached_write_cost_per_1m
                or old_audio_input != found_pricing.audio_input_cost_per_1m
                or old_audio_output != found_pricing.audio_output_cost_per_1m
            )

            if not pricing_changed:
                unchanged_count += 1
                results.append(
                    SyncPricingResult(
                        mapping_id=mapping.id,
                        model_name=model.name,
                        provider_model_name=provider_model_name,
                        provider=provider_name,
                        old_input_cost=old_input_cost,
                        old_output_cost=old_output_cost,
                        new_input_cost=found_pricing.input_cost_per_1m,
                        new_output_cost=found_pricing.output_cost_per_1m,
                        old_cached_read_cost=old_cached_read,
                        new_cached_read_cost=found_pricing.cached_read_cost_per_1m,
                        old_cached_write_cost=old_cached_write,
                        new_cached_write_cost=found_pricing.cached_write_cost_per_1m,
                        old_audio_input_cost=old_audio_input,
                        new_audio_input_cost=found_pricing.audio_input_cost_per_1m,
                        old_audio_output_cost=old_audio_output,
                        new_audio_output_cost=found_pricing.audio_output_cost_per_1m,
                        updated=False,
                        message="Pricing unchanged",
                        available_sources=available_sources,
                        selected_source=found_pricing.provider,
                    )
                )
                continue

            # Update pricing at the provider mapping level if not dry run
            if not request_data.dry_run:
                await repo._models.update_model_provider_pricing(
                    mapping.id,
                    input_cost_per_1m=found_pricing.input_cost_per_1m,
                    output_cost_per_1m=found_pricing.output_cost_per_1m,
                    cached_read_cost_per_1m=found_pricing.cached_read_cost_per_1m,
                    cached_write_cost_per_1m=found_pricing.cached_write_cost_per_1m,
                    audio_input_cost_per_1m=found_pricing.audio_input_cost_per_1m,
                    audio_output_cost_per_1m=found_pricing.audio_output_cost_per_1m,
                )

            updated_count += 1
            results.append(
                SyncPricingResult(
                    mapping_id=mapping.id,
                    model_name=model.name,
                    provider_model_name=provider_model_name,
                    provider=provider_name,
                    old_input_cost=old_input_cost,
                    old_output_cost=old_output_cost,
                    new_input_cost=found_pricing.input_cost_per_1m,
                    new_output_cost=found_pricing.output_cost_per_1m,
                    old_cached_read_cost=old_cached_read,
                    new_cached_read_cost=found_pricing.cached_read_cost_per_1m,
                    old_cached_write_cost=old_cached_write,
                    new_cached_write_cost=found_pricing.cached_write_cost_per_1m,
                    old_audio_input_cost=old_audio_input,
                    new_audio_input_cost=found_pricing.audio_input_cost_per_1m,
                    old_audio_output_cost=old_audio_output,
                    new_audio_output_cost=found_pricing.audio_output_cost_per_1m,
                    updated=not request_data.dry_run,
                    message="Updated" if not request_data.dry_run else "Would update",
                    available_sources=available_sources,
                    selected_source=found_pricing.provider,
                )
            )

    # Commit changes and reload config if not dry run
    if not request_data.dry_run and updated_count > 0:
        await commit_and_reload(session, request)

    return SyncPricingResponse(
        success=True,
        dry_run=request_data.dry_run,
        total_models=len(models),
        total_provider_mappings=total_provider_mappings,
        updated_count=updated_count,
        skipped_count=skipped_count,
        unchanged_count=unchanged_count,
        results=results,
    )


@router.post("/pricing/apply", response_model=ApplyPricingResponse)
async def apply_model_pricing(
    request_data: ApplyPricingRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> ApplyPricingResponse:
    """Apply explicitly reviewed pricing updates to model-provider mappings.

    Unlike the bulk sync endpoint, this applies exactly the updates provided
    by the caller — typically after the user has reviewed a dry-run preview
    and selected/edited individual mappings.

    For each update item, only the pricing fields explicitly provided are
    written (a provided ``null`` clears the field); omitted fields are left
    untouched.
    """
    repo = get_config_repository(session)

    results: list[ApplyPricingResult] = []
    applied_count = 0
    failed_count = 0

    for item in request_data.updates:
        updates = item.model_dump(exclude_unset=True)
        updates.pop("mapping_id", None)

        if not updates:
            failed_count += 1
            results.append(
                ApplyPricingResult(
                    mapping_id=item.mapping_id,
                    applied=False,
                    message="No pricing fields provided",
                )
            )
            continue

        updated = await repo._models.apply_mapping_pricing(item.mapping_id, updates)
        if updated is None:
            failed_count += 1
            results.append(
                ApplyPricingResult(
                    mapping_id=item.mapping_id,
                    applied=False,
                    message="Mapping not found",
                )
            )
            continue

        applied_count += 1
        results.append(
            ApplyPricingResult(
                mapping_id=item.mapping_id,
                applied=True,
                message="Applied",
            )
        )

    if applied_count > 0:
        await commit_and_reload(session, request)

    return ApplyPricingResponse(
        success=failed_count == 0,
        applied_count=applied_count,
        failed_count=failed_count,
        results=results,
    )
