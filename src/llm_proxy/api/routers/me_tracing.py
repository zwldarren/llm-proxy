"""Self-service tracing configuration for the current user.

Tracing is strictly per-user: every authenticated user (admin included)
configures their own tracing backends, stored on ``users.tracing_config``.
A user's config applies only to their own requests — no user's tracing can
record another user's activity. There is no system-level tracing config.

These endpoints mirror the shape of the former global tracing API but read
and write the requesting user's ``users.tracing_config`` column. They require
only ``require_authenticated`` (any logged-in user), not the admin role.
"""

from fastapi import APIRouter, Depends

from llm_proxy.api.dependencies import get_current_user, require_authenticated
from llm_proxy.api.routers.tracing import (
    _build_config_read,
    _validate_tracing_provider_urls_async,
    build_persisted_tracing_dict,
)
from llm_proxy.api.schemas.tracing import (
    TracingConfigRead,
    TracingConfigWrite,
    TracingGetResponse,
    TracingProviderDetails,
    TracingProviderField,
    TracingProvidersResponse,
    TracingStatus,
    TracingUpdateResponse,
)
from llm_proxy.core.exceptions import ValidationError
from llm_proxy.database import UserRepository, get_async_session_context
from llm_proxy.database.tables import UserRecord
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.tracing.handlers import list_handler_types
from llm_proxy.observability.tracing_config import TracingConfig
from llm_proxy.observability.user_tracing import get_user_tracing_manager

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/me/tracing",
    tags=["Tracing"],
    dependencies=[Depends(require_authenticated)],
)


@router.get("/", response_model=TracingGetResponse)
async def get_my_tracing(user: UserRecord = Depends(get_current_user)) -> TracingGetResponse:
    """Get the current user's personal tracing configuration."""
    async with get_async_session_context() as session:
        repo = UserRepository(session)
        db_config = await repo.get_tracing_config(user.id)

    if db_config:
        config_read = _build_config_read(db_config)
        config = TracingConfig.from_dict(db_config)
        is_configured = config.is_configured
    else:
        # No personal config: report defaults. Requests fall back to the
        # admin-managed global tracing registry.
        config_read = TracingConfigRead(enabled=False, providers=[])
        is_configured = False

    return TracingGetResponse.model_validate(
        {
            "config": config_read,
            "status": {
                "enabled": config_read.enabled,
                "providers": [
                    {"name": p.name, "provider": p.provider} for p in config_read.providers
                ],
                "is_configured": is_configured,
            },
        }
    )


@router.put("/", response_model=TracingUpdateResponse)
async def update_my_tracing(
    config_data: TracingConfigWrite,
    user: UserRecord = Depends(get_current_user),
) -> TracingUpdateResponse:
    """Update the current user's personal tracing configuration."""
    try:
        async with get_async_session_context() as session:
            repo = UserRepository(session)
            existing_config = await repo.get_tracing_config(user.id)

        config_dict = build_persisted_tracing_dict(config_data, existing_config)
        config = TracingConfig.from_dict(config_dict)

        # Validate provider URLs before persisting to prevent SSRF.
        await _validate_tracing_provider_urls_async(config.providers)

        async with get_async_session_context() as session:
            repo = UserRepository(session)
            await repo.set_tracing_config(user.id, config.to_dict())
            await session.commit()

        # Drop any cached per-user registry so the next request rebuilds it
        # from the updated config.
        await get_user_tracing_manager().invalidate(user.id)

        config_read = _build_config_read(config.to_dict())

        return TracingUpdateResponse.model_validate(
            {
                "config": config_read,
                "status": {
                    "enabled": config.enabled,
                    "providers": [
                        {"name": p.name, "provider": p.provider} for p in config.providers
                    ],
                    "is_configured": config.is_configured,
                },
                "message": "Tracing configuration updated successfully",
            }
        )

    except ValueError as e:
        raise ValidationError(message=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to update tracing configuration for user {user.id}: {e}", exc_info=e)
        raise ValidationError(
            message="Failed to update tracing configuration", code="tracing_config_error"
        ) from e


@router.get("/status", response_model=TracingStatus)
async def get_my_tracing_status(
    user: UserRecord = Depends(get_current_user),
) -> TracingStatus:
    """Get the current user's personal tracing status."""
    async with get_async_session_context() as session:
        repo = UserRepository(session)
        db_config = await repo.get_tracing_config(user.id)

    if not db_config:
        return TracingStatus(enabled=False, providers=[], is_configured=False)

    config = TracingConfig.from_dict(db_config)
    return TracingStatus.model_validate(
        {
            "enabled": config.enabled,
            "providers": [{"name": p.name, "provider": p.provider} for p in config.providers],
            "is_configured": config.is_configured,
        }
    )


@router.get("/providers", response_model=TracingProvidersResponse)
async def list_my_tracing_providers() -> TracingProvidersResponse:
    """List all supported tracing providers (same set as the admin endpoint)."""
    handler_types = list_handler_types()
    provider_details = [
        TracingProviderDetails(
            name=cls.provider_name,
            required_fields=cls.required_settings,
            optional_fields=cls.optional_settings,
            description=cls.description,
            fields=[TracingProviderField(**field) for field in cls.field_metadata],
        )
        for cls in handler_types
    ]
    return TracingProvidersResponse(providers=provider_details)


__all__ = ["router"]
