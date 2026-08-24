"""FastAPI dependency-injection providers.

Pure DI wiring (HTTP clients, config manager, auth, adapter factory) and
request-introspection helpers. Request-context assembly lives in
:mod:`llm_proxy.api.context` and smart-routing orchestration lives in
:mod:`llm_proxy.routing.orchestrator`; this module is imported by both, so it
must not depend on either (one-directional layering).
"""

from fastapi import Depends, Request

from llm_proxy.config.manager import DatabaseConfigManager
from llm_proxy.config.types.main import ProxyConfig
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.adapter import BaseAdapter, list_providers
from llm_proxy.core.exceptions import (
    AdapterNotFoundError as CoreAdapterNotFoundError,
)
from llm_proxy.core.exceptions import (
    AuthenticationFailedError,
    ConfigurationError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from llm_proxy.core.identity import get_request_identity
from llm_proxy.core.provider_selector import ProviderSelectionResult
from llm_proxy.database import UserRecord, UserRepository, get_async_session
from llm_proxy.http.client import AsyncSession, ProviderHTTPClientManager

get_async_session_dep = Depends(get_async_session)


def extract_trace_id(request: Request) -> str | None:
    return request.headers.get("x-langfuse-trace-id") or request.headers.get("x-trace-id")


def extract_session_id(request: Request) -> str | None:
    return request.headers.get("x-session-id")


def extract_user_id(request: Request) -> str | None:
    """Extract user ID from the authenticated identity only.

    Only an identity established by a trusted authentication mechanism (e.g.
    JWT) may provide a user ID. API-key-authenticated requests do not carry
    a user claim, so this function returns None for them. The raw
    ``x-user-id`` request header is intentionally ignored to prevent any
    API-key holder from spoofing user attribution in audit logs, usage
    records, and telemetry.
    """
    return get_request_identity(request).user


def _get_http_manager(request: Request) -> ProviderHTTPClientManager:
    manager = getattr(request.app.state, "http_client", None)
    if manager is None:
        raise ConfigurationError(
            "HTTP client not initialized. Ensure lifespan is properly configured."
        )
    return manager


async def get_http_client(request: Request) -> AsyncSession:
    manager = _get_http_manager(request)
    if isinstance(manager, ProviderHTTPClientManager):
        return await manager.get_client("__default__")
    return manager.client


async def get_provider_http_client(request: Request, provider_name: str) -> AsyncSession:
    manager = _get_http_manager(request)
    if isinstance(manager, ProviderHTTPClientManager):
        return await manager.get_client(provider_name)
    return manager.client


def get_config_manager(request: Request) -> DatabaseConfigManager:
    config_manager = getattr(request.app.state, "config_manager", None)
    if config_manager is None:
        raise ConfigurationError(
            "Config manager not initialized. Ensure lifespan is properly configured."
        )
    return config_manager


async def get_config_from_state(request: Request) -> ProxyConfig:
    return await get_config_manager(request).get_config()


async def get_auth_config(request: Request):
    return (await get_config_from_state(request)).server_params.auth


async def require_authenticated(request: Request):
    """Require the request to be authenticated (any auth method).

    Unlike require_admin_role, this does not check for admin privileges.
    It is suitable for self-service endpoints where the user only needs
    to be logged in.
    """
    if not get_request_identity(request).is_authenticated:
        raise AuthenticationFailedError(message="Authentication required")


async def get_current_user(
    request: Request,
    session=Depends(get_async_session),
) -> UserRecord:
    """Resolve the authenticated user from the request identity.

    Must run after require_authenticated (or any auth dependency that sets the identity).
    Returns the UserRecord or raises AuthenticationFailedError.
    """
    identity = get_request_identity(request)
    if not identity.user:
        raise AuthenticationFailedError(message="Authentication required")
    repo = UserRepository(session)
    user = await repo.get_by_username(identity.user)
    if not user:
        raise ForbiddenError(message="User not found")
    if not user.is_active:
        raise ForbiddenError(message="Account is disabled")
    return user


async def require_api_key_auth(request: Request):
    if not get_request_identity(request).api_key_name:
        raise AuthenticationFailedError(
            message="Authentication required.",
            code="api_key_required",
        )


async def require_admin_role(request: Request, session=Depends(get_async_session)):
    """Require JWT auth AND admin role. Must run after require_authenticated."""
    identity = get_request_identity(request)
    if not identity.is_authenticated:
        raise AuthenticationFailedError(message="Authentication required")
    if not identity.user:
        raise AuthenticationFailedError(message="Authentication required")
    repo = UserRepository(session)
    user = await repo.get_by_username(identity.user)
    if not user or user.role != "admin":
        raise AuthenticationFailedError(
            message="Admin role required",
            code="forbidden",
            status_code=403,
        )
    if not user.is_active:
        raise AuthenticationFailedError(message="Account is disabled")
    return user


def _create_adapter(
    provider_name: str,
    provider_config: ProviderConfig,
    http_client: AsyncSession,
    unknown_fields_policy: str,
    unsupported_block_policy: str,
    http_client_manager: ProviderHTTPClientManager | None = None,
    max_retries: int = 3,
) -> BaseAdapter:
    from llm_proxy.core.adapter import get_adapter

    if not provider_config.type or not provider_config.type.strip():
        raise ValidationError(
            message=(
                f"Provider '{provider_name}' has no type configured. "
                f"Please ensure provider_config.type is set (e.g., 'ollama', 'openai')."
            ),
        )

    try:
        return get_adapter(
            provider_config.type,
            provider_name=provider_name,
            api_key=provider_config.get_api_key(),
            base_url=provider_config.base_url,
            timeout=provider_config.timeout,
            max_retries=max_retries,
            custom_headers=provider_config.custom_headers,
            parameter_overrides=provider_config.parameter_overrides,
            endpoint_base_urls=provider_config.endpoint_base_urls,
            # Kill switch for providers with native-protocol endpoints
            # (e.g. DeepSeek's Anthropic/Responses passthrough); forwarded
            # opaquely through AdapterConfig.extra — only adapters that read
            # it are affected.
            native_passthrough=provider_config.metadata.get("native_passthrough", True),
            # Upstream API dialect for Gemini ("generate_content" default |
            # "interactions") — forwarded opaquely through AdapterConfig.extra;
            # only the Gemini adapter reads it.
            api_variant=provider_config.metadata.get("api_variant", "generate_content"),
            unknown_fields_policy=unknown_fields_policy,
            unsupported_block_policy=unsupported_block_policy,
            http_client=http_client,
            http_client_manager=http_client_manager,
        )
    except CoreAdapterNotFoundError as e:
        raise NotFoundError(
            message=f"Provider '{provider_name}' not found. Available: {list_providers()}",
        ) from e
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(
            message=f"Failed to create provider adapter: {e}",
        ) from e


async def create_adapter_for_provider(
    request: Request,
    selection: ProviderSelectionResult,
    http_client: AsyncSession | None = None,
) -> BaseAdapter:
    manager = _get_http_manager(request)
    if http_client is None:
        http_client = await get_provider_http_client(request, selection.provider_name)

    manager_instance = manager if isinstance(manager, ProviderHTTPClientManager) else None

    # Read global policy values from server config
    config = await get_config_manager(request).get_config()
    ufp = config.server_params.unknown_fields_policy
    ubp = config.server_params.unsupported_block_policy

    return _create_adapter(
        selection.provider_name,
        selection.provider_config,
        http_client,
        unknown_fields_policy=ufp,
        unsupported_block_policy=ubp,
        http_client_manager=manager_instance,
        max_retries=selection.max_retries,
    )
