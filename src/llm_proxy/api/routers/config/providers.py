"""Provider configuration API endpoints."""

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Path, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_async_session_dep, require_admin_role
from llm_proxy.api.routers.config.helpers import (
    _extract_metadata_fields,
    commit_and_reload,
    get_config_repository,
)
from llm_proxy.api.routers.config.models import model_record_to_read
from llm_proxy.api.schemas.admin import (
    ProviderCreate,
    ProviderDetails,
    ProviderKeyReveal,
    ProviderRead,
    ProviderTypeRead,
    ProviderUpdate,
)
from llm_proxy.core.adapter import list_provider_types
from llm_proxy.core.exceptions import (
    ConflictError,
    EncryptionError,
    NotFoundError,
    ValidationError,
)
from llm_proxy.core.identity import get_request_identity
from llm_proxy.http.client import validate_server_url
from llm_proxy.observability.audit_helpers import write_provider_key_reveal_audit_log
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/providers", tags=["configuration"], dependencies=[Depends(require_admin_role)]
)


def provider_record_to_read(provider) -> ProviderRead:
    """Convert a ProviderRecord to ProviderRead schema."""
    remaining, extracted = _extract_metadata_fields(
        provider.provider_metadata,
        ["parameter_overrides", "endpoint_base_urls", "native_web_search"],
    )

    endpoint_base_urls = extracted.get("endpoint_base_urls")
    endpoint_base_urls_typed = {}
    if isinstance(endpoint_base_urls, dict):
        endpoint_base_urls_typed = {
            k: str(v) for k, v in endpoint_base_urls.items() if isinstance(v, str)
        }

    return ProviderRead(
        id=provider.id,
        name=provider.name,
        type=provider.type,
        base_url=provider.base_url,
        api_key=provider.api_key or "",
        api_version=provider.api_version,
        timeout=provider.timeout,
        rate_limit=provider.rate_limit,
        custom_headers=provider.custom_headers or {},
        provider_models=provider.provider_models or [],
        enabled=provider.enabled,
        priority=provider.priority,
        provider_metadata=remaining or {},
        parameter_overrides=extracted.get("parameter_overrides", {}),
        endpoint_base_urls=endpoint_base_urls_typed,
        native_web_search=extracted.get("native_web_search", False),
        icon_url=provider.icon_url,
    )


def _validate_provider_urls(data: dict[str, Any]) -> None:
    """Validate provider base_url, endpoint_base_urls, and icon_url for SSRF."""
    base_url = data.get("base_url")
    if base_url:
        validate_server_url(base_url, label="provider base_url")

    endpoint_base_urls = data.get("endpoint_base_urls")
    if isinstance(endpoint_base_urls, dict):
        for endpoint, url in endpoint_base_urls.items():
            if url:
                validate_server_url(
                    str(url),
                    label=f"endpoint base_url '{endpoint}'",
                )

    icon_url = data.get("icon_url")
    if icon_url:
        validate_server_url(icon_url, label="provider icon_url")


async def _validate_provider_urls_async(data: dict[str, Any]) -> None:
    """Async wrapper that runs SSRF validation in a thread.

    DNS resolution is blocking, so offload it to avoid stalling the event loop
    during infrequent admin configuration changes.
    """
    await asyncio.to_thread(_validate_provider_urls, data)


@router.get("", response_model=list[ProviderRead])
async def list_providers(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> list[ProviderRead]:
    """List all providers."""
    repo = get_config_repository(session)
    providers = await repo.get_all_providers()
    return [provider_record_to_read(p) for p in providers]


@router.get("/provider-types", response_model=list[ProviderTypeRead])
async def list_provider_types_endpoint(
    request: Request,
) -> list[ProviderTypeRead]:
    """List available provider types with branding metadata.

    Derived from the adapter registry (not from configured providers), so the
    admin UI can render the create/edit type selector and the list filter
    without a per-provider frontend list: adding a provider adapter is a
    backend-only change.

    Registered before the ``/{name:path}`` catch-all so the literal
    ``provider-types`` path is not swallowed as a provider name.
    """
    return [ProviderTypeRead.model_validate(info) for info in list_provider_types()]


@router.get("/{name:path}", response_model=ProviderDetails)
async def get_provider(
    request: Request,
    session: AsyncSession = get_async_session_dep,
    name: str = Path(...),
) -> ProviderDetails:
    """Get a specific provider."""
    repo = get_config_repository(session)
    provider = await repo.get_provider_with_models(name)
    if not provider:
        raise NotFoundError(message=f"Provider '{name}' not found")

    models_data = [model_record_to_read(m.model) for m in provider.model_provider_mappings]

    read = provider_record_to_read(provider)
    return ProviderDetails(
        **read.model_dump(),
        # model_dump() omits the excluded api_key field; pass it through so
        # the masked_api_key computed field stays populated (it is never
        # serialized).
        api_key=read.api_key,
        models=models_data,
    )


@router.post("", response_model=ProviderRead)
async def create_provider(
    provider_data: ProviderCreate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> ProviderRead:
    """Create a new provider."""
    data = provider_data.model_dump()
    await _validate_provider_urls_async(data)

    repo = get_config_repository(session)
    try:
        provider = await repo.create_provider(**data)

        await commit_and_reload(session, request)

        return provider_record_to_read(provider)
    except EncryptionError:
        # Fail closed: encryption failures must surface as a sanitized 5xx,
        # not be masked as a 400 validation error.
        raise
    except IntegrityError:
        raise ConflictError(
            message=(
                f"Provider '{data.get('name')}' already exists. "
                "Please use a unique name or update the existing provider."
            ),
        ) from None
    except Exception as e:
        logger.error(f"Failed to create provider '{data.get('name')}': {e}", exc_info=e)
        raise ValidationError(
            message=f"Failed to create provider: {e}",
        ) from e


@router.post("/{name:path}/api-key/reveal", response_model=ProviderKeyReveal)
async def reveal_provider_api_key(
    request: Request,
    session: AsyncSession = get_async_session_dep,
    name: str = Path(...),
) -> ProviderKeyReveal:
    """Reveal a provider's plaintext API key.

    The plaintext key is only ever returned by this explicit endpoint —
    list/detail/update responses carry ``masked_api_key`` instead. Every
    reveal is recorded in the audit log (``event_type=DATA_ACCESS``).
    """
    repo = get_config_repository(session)
    provider = await repo.get_provider(name)
    if not provider:
        raise NotFoundError(message=f"Provider '{name}' not found")

    identity = get_request_identity(request)
    await write_provider_key_reveal_audit_log(
        request,
        actor=identity.display_name or "unknown",
        provider_name=name,
    )

    return ProviderKeyReveal(name=provider.name, api_key=provider.api_key or "")


@router.put("/{name:path}", response_model=ProviderRead)
async def update_provider(
    provider_data: ProviderUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    name: str = Path(...),
) -> ProviderRead:
    """Update a provider."""
    data = provider_data.model_dump(exclude_unset=True)
    await _validate_provider_urls_async(data)

    repo = get_config_repository(session)
    provider = await repo.update_provider(name, **data)
    if not provider:
        raise NotFoundError(message=f"Provider '{name}' not found")

    await commit_and_reload(session, request)

    # update_provider returns the record with the key still encrypted; re-read
    # it so the response carries a meaningful masked key (the plaintext never
    # leaves the server).
    provider = await repo.get_provider(name)
    return provider_record_to_read(provider)


@router.delete("/{name:path}")
async def delete_provider(
    request: Request,
    session: AsyncSession = get_async_session_dep,
    name: str = Path(...),
) -> dict[str, str]:
    """Delete a provider."""
    repo = get_config_repository(session)
    success = await repo.delete_provider(name)
    if not success:
        raise NotFoundError(message=f"Provider '{name}' not found")

    await commit_and_reload(session, request)

    return {"message": f"Provider '{name}' deleted successfully"}
