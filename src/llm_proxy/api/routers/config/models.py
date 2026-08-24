"""Model configuration API endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, Path, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import (
    get_async_session_dep,
    require_admin_role,
    require_authenticated,
)
from llm_proxy.api.routers.config.helpers import (
    _extract_metadata_fields,
    commit_and_reload,
    get_config_repository,
)
from llm_proxy.api.schemas.admin import (
    ModelCreate,
    ModelProviderMapping,
    ModelRead,
    ModelUpdate,
)
from llm_proxy.core.exceptions import ConflictError, NotFoundError, ValidationError
from llm_proxy.core.identity import get_request_identity
from llm_proxy.database import UserRepository
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/models", tags=["configuration"], dependencies=[Depends(require_admin_role)]
)


# Public, read-only endpoint accessible to any authenticated user (admin or
# viewer). It exposes only model names so non-admin users can populate the
# API-key model allowlist dropdown without seeing full config (pricing,
# provider mappings, etc.). A non-admin with a per-user model allowlist sees
# only the names within that allowlist (filtered to configured models).
public_router = APIRouter(
    prefix="/model-names", tags=["configuration"], dependencies=[Depends(require_authenticated)]
)


@public_router.get("", response_model=list[str])
async def list_model_names(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> list[str]:
    """List model names available to the current user for API-key configuration.

    Accessible to any authenticated user. Admins and unrestricted users see
    every configured model name; a non-admin with a per-user model allowlist
    sees only the names within that allowlist (filtered to models that still
    exist, preserving the admin-defined order).
    """
    repo = get_config_repository(session)
    models = await repo.get_all_models()
    all_names = [m.name for m in models]
    existing = set(all_names)

    identity = get_request_identity(request)
    if identity.user:
        user = await UserRepository(session).get_by_username(identity.user)
        if user and user.role != "admin" and user.allowed_models is not None:
            return [name for name in user.allowed_models if name in existing]
    return all_names


def model_record_to_read(model) -> ModelRead:
    """Convert a ModelRecord to ModelRead schema with provider mappings."""
    providers_list = [
        ModelProviderMapping(
            provider_name=mapping.provider.name,
            priority=mapping.priority,
            provider_model_name=mapping.provider_model_name or model.name,
            input_cost_per_1m=mapping.input_cost_per_1m,
            output_cost_per_1m=mapping.output_cost_per_1m,
            cached_read_cost_per_1m=mapping.cached_read_cost_per_1m,
            cached_write_cost_per_1m=mapping.cached_write_cost_per_1m,
            audio_input_cost_per_1m=mapping.audio_input_cost_per_1m,
            audio_output_cost_per_1m=mapping.audio_output_cost_per_1m,
            image_input_cost_per_1m=mapping.image_input_cost_per_1m,
            cost_per_image=mapping.cost_per_image,
            audio_cost_per_minute=mapping.audio_cost_per_minute,
            tts_cost_per_1m_chars=mapping.tts_cost_per_1m_chars,
            web_search_cost_per_1k=mapping.web_search_cost_per_1k,
            parameter_overrides=mapping.parameter_overrides or {},
        )
        for mapping in model.provider_mappings
    ]
    providers_list.sort(key=lambda p: p.priority, reverse=True)

    remaining, extracted = _extract_metadata_fields(model.model_metadata, ["parameter_overrides"])

    return ModelRead(
        id=model.id,
        name=model.name,
        providers=providers_list,
        timeout=model.timeout,
        max_retries=model.max_retries,
        model_metadata=remaining,
        parameter_overrides=extracted.get("parameter_overrides", {}),
        input_cost_per_1m=model.input_cost_per_1m,
        output_cost_per_1m=model.output_cost_per_1m,
        cached_read_cost_per_1m=model.cached_read_cost_per_1m,
        cached_write_cost_per_1m=model.cached_write_cost_per_1m,
        audio_input_cost_per_1m=model.audio_input_cost_per_1m,
        audio_output_cost_per_1m=model.audio_output_cost_per_1m,
        image_input_cost_per_1m=model.image_input_cost_per_1m,
        cost_per_image=model.cost_per_image,
        audio_cost_per_minute=model.audio_cost_per_minute,
        tts_cost_per_1m_chars=model.tts_cost_per_1m_chars,
        web_search_cost_per_1k=model.web_search_cost_per_1k,
        icon_url=model.icon_url,
        auto_eligible=model.auto_eligible,
        quality_tier=model.quality_tier,
        routing_assignments=model.routing_assignments,
        supports_images=model.supports_images,
        supports_image_generation=model.supports_image_generation,
        supports_tts=model.supports_tts,
        supports_stt=model.supports_stt,
        supports_embedding=model.supports_embedding,
        supports_realtime=model.supports_realtime,
        description=model.description,
        homepage_url=model.homepage_url,
        context_length=model.context_length,
    )


def _build_providers_list(
    providers_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert provider data from API schema to repository format."""
    return [
        {
            "provider_name": p["provider_name"],
            "priority": p.get("priority", 0),
            "provider_model_name": p.get("provider_model_name"),
            "input_cost_per_1m": p.get("input_cost_per_1m"),
            "output_cost_per_1m": p.get("output_cost_per_1m"),
            "cached_read_cost_per_1m": p.get("cached_read_cost_per_1m"),
            "cached_write_cost_per_1m": p.get("cached_write_cost_per_1m"),
            "audio_input_cost_per_1m": p.get("audio_input_cost_per_1m"),
            "audio_output_cost_per_1m": p.get("audio_output_cost_per_1m"),
            "image_input_cost_per_1m": p.get("image_input_cost_per_1m"),
            "cost_per_image": p.get("cost_per_image"),
            "audio_cost_per_minute": p.get("audio_cost_per_minute"),
            "tts_cost_per_1m_chars": p.get("tts_cost_per_1m_chars"),
            "web_search_cost_per_1k": p.get("web_search_cost_per_1k"),
            "parameter_overrides": p.get("parameter_overrides"),
        }
        for p in providers_data
    ]


@router.get("", response_model=list[ModelRead])
async def list_models(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> list[ModelRead]:
    """List all models with their provider mappings."""
    repo = get_config_repository(session)
    models = await repo.get_all_models()
    return [model_record_to_read(m) for m in models]


@router.get("/{name:path}", response_model=ModelRead)
async def get_model(
    request: Request,
    session: AsyncSession = get_async_session_dep,
    name: str = Path(...),
) -> ModelRead:
    """Get a specific model with its provider mappings."""
    repo = get_config_repository(session)
    model = await repo.get_model_with_provider(name)
    if not model:
        raise NotFoundError(message=f"Model '{name}' not found")
    return model_record_to_read(model)


@router.post("", response_model=ModelRead)
async def create_model(
    model_data: ModelCreate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> ModelRead:
    """Create a new model with multi-provider support."""
    repo = get_config_repository(session)
    try:
        data = model_data.model_dump()

        name = data.pop("name", None)
        if not name:
            raise ValidationError(message="name is required")

        providers_data = data.pop("providers", None)
        if not providers_data:
            raise ValidationError(message="At least one provider is required")

        providers_list = _build_providers_list(providers_data)

        model = await repo.create_model(
            name=name,
            providers=providers_list,
            **data,
        )
        if not model:
            raise ValidationError(message="No valid providers found")

        loaded_model = await repo.get_model_with_provider(model.name)
        if not loaded_model:
            raise NotFoundError(message="Failed to load created model")

        await commit_and_reload(session, request)

        return model_record_to_read(loaded_model)
    except IntegrityError:
        raise ConflictError(
            message=(
                f"Model '{model_data.name}' already exists. "
                "Please use a unique name or update the existing model."
            ),
        ) from None
    except Exception as e:
        logger.error(f"Failed to create model '{model_data.name}': {e}", exc_info=e)
        raise ValidationError(
            message=f"Failed to create model: {e}",
        ) from None


@router.put("/{name:path}", response_model=ModelRead)
async def update_model(
    model_data: ModelUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    name: str = Path(...),
) -> ModelRead:
    """Update a model configuration.

    Can update providers list to add/remove/reorder providers.
    """
    repo = get_config_repository(session)
    data = model_data.model_dump(exclude_unset=True)

    new_name = data.pop("name", None)
    providers_data = data.pop("providers", None)
    providers_list = _build_providers_list(providers_data) if providers_data is not None else None

    model = await repo.update_model(name, providers=providers_list, new_name=new_name, **data)
    if not model:
        raise NotFoundError(message=f"Model '{name}' not found")

    loaded_model = await repo.get_model_with_provider(model.name)

    await commit_and_reload(session, request)

    return model_record_to_read(loaded_model) if loaded_model else model_record_to_read(model)


@router.delete("/{name:path}")
async def delete_model(
    request: Request,
    session: AsyncSession = get_async_session_dep,
    name: str = Path(...),
) -> dict[str, str]:
    """Delete a model."""
    repo = get_config_repository(session)
    success = await repo.delete_model(name)
    if not success:
        raise NotFoundError(message=f"Model '{name}' not found")

    await commit_and_reload(session, request)

    return {"message": f"Model '{name}' deleted successfully"}
