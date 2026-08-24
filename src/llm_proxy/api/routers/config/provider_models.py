"""Provider model listing from upstream APIs."""

import httpx2
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_async_session_dep, get_http_client, require_admin_role
from llm_proxy.api.routers.config.helpers import get_config_repository
from llm_proxy.api.schemas.admin import ProviderModelsResponse
from llm_proxy.core.adapter import get_adapter
from llm_proxy.core.exceptions import (
    AdapterNotFoundError,
    NotFoundError,
    ProviderError,
    ValidationError,
)
from llm_proxy.providers.base import BaseHttpProvider

router = APIRouter(
    prefix="/providers/{name}/models",
    tags=["configuration"],
    dependencies=[Depends(require_admin_role)],
)

# Providers that don't require authentication for listing models
_NO_AUTH_PROVIDERS: set[str] = {"ollama"}


@router.get("", response_model=ProviderModelsResponse)
async def list_provider_models(
    name: str,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> ProviderModelsResponse:
    """Fetch available models from a provider's upstream API.

    This endpoint calls the provider's models API to retrieve the list of
    available models. The response includes model IDs that can be used
    when configuring model mappings.
    """
    repo = get_config_repository(session)
    provider = await repo.get_provider(name, decrypt=True)
    if not provider:
        raise NotFoundError(message=f"Provider '{name}' not found")

    provider_type = provider.type.strip().lower()
    api_key = provider.api_key
    base_url = provider.base_url

    # Check if authentication is required for this provider
    if provider_type not in _NO_AUTH_PROVIDERS and not api_key:
        raise ValidationError(message=f"Provider '{name}' requires an API key to fetch models.")

    try:
        # Get the HTTP client from app state
        http_client = await get_http_client(request)

        # Create adapter instance for the provider
        try:
            adapter = get_adapter(
                provider_type,
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
            )
        except AdapterNotFoundError as e:
            raise ValidationError(message=f"Unsupported provider type: {provider_type}") from e

        # Fetch models using the adapter's list_models method
        if not isinstance(adapter, BaseHttpProvider):
            raise ValidationError(
                message=f"Provider type '{provider_type}' does not support model listing."
            )
        models = await adapter.list_models(client=http_client)

        return ProviderModelsResponse(
            provider_name=name,
            provider_type=provider_type,
            models=models,
        )
    except httpx2.HTTPStatusError:
        raise ProviderError(
            message="Failed to fetch models from provider API. Check provider configuration.",
            error_type="api_error",
            status_code=502,
        ) from None
    except httpx2.RequestError:
        raise ProviderError(
            message="Network error when connecting to provider API.",
            error_type="network_error",
            status_code=502,
        ) from None
    except ValueError:
        raise ValidationError(message="Invalid provider response format.") from None
