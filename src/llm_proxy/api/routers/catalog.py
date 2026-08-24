"""Public model catalog endpoint.

Provides display-oriented model information for the model plaza, accessible to
any authenticated user (including viewers). Excludes sensitive pricing and
admin configuration details.
"""

from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_async_session_dep, require_authenticated
from llm_proxy.api.routers.config.helpers import get_config_repository
from llm_proxy.api.schemas.admin import ModelCatalogEntry
from llm_proxy.core.identity import get_request_identity
from llm_proxy.database import UserRepository
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/catalog", tags=["catalog"], dependencies=[Depends(require_authenticated)]
)


def _derive_capabilities(model: Any) -> list[str]:
    """Collect the display capabilities an admin configured for a model."""
    capabilities: list[str] = []
    if model.supports_images:
        capabilities.append("vision")
    if model.supports_image_generation:
        capabilities.append("image_generation")
    if model.supports_tts:
        capabilities.append("tts")
    if model.supports_stt:
        capabilities.append("stt")
    if model.supports_embedding:
        capabilities.append("embedding")
    if model.supports_realtime:
        capabilities.append("realtime")
    return capabilities


@router.get("/models", response_model=list[ModelCatalogEntry])
async def list_model_catalog(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> list[ModelCatalogEntry]:
    """List all models with display-oriented metadata for the model plaza.

    Accessible to any authenticated user (admin or viewer). Returns only the
    fields needed to present a model in the catalog; pricing and admin-only
    configuration are not exposed.

    A non-admin user with a per-user model allowlist only sees the models
    within that allowlist, mirroring the restriction enforced at request time.
    """
    repo = get_config_repository(session)
    models = await repo.get_all_models()

    # Resolve the per-user allowlist for non-admin users (None = unrestricted).
    allowed_names: set[str] | None = None
    identity = get_request_identity(request)
    if identity.user:
        user = await UserRepository(session).get_by_username(identity.user)
        if user and user.role != "admin" and user.allowed_models is not None:
            allowed_names = set(user.allowed_models)

    entries: list[ModelCatalogEntry] = []
    for model in models:
        if allowed_names is not None and model.name not in allowed_names:
            continue
        provider_names = [
            mapping.provider.name
            for mapping in model.provider_mappings
            if mapping.provider is not None
        ]
        entries.append(
            ModelCatalogEntry(
                name=model.name,
                icon_url=model.icon_url,
                description=model.description,
                homepage_url=model.homepage_url,
                context_length=model.context_length,
                capabilities=_derive_capabilities(model),
                quality_tier=model.quality_tier,
                provider_names=provider_names,
            )
        )
    return entries
