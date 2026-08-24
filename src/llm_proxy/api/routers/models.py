"""Model listing endpoint."""

from fastapi import APIRouter, Depends, Request

from llm_proxy.api.dependencies import (
    get_config_from_state,
    get_config_manager,
    get_request_identity,
    require_api_key_auth,
)
from llm_proxy.api.schemas.admin import OpenAIModel, OpenAIModelList

router = APIRouter(prefix="/v1", tags=["models"])


async def require_any_auth(request: Request):
    """Require either API key or admin JWT authentication."""
    identity = get_request_identity(request)
    if not identity.is_authenticated:
        await require_api_key_auth(request)


@router.get("/models", response_model=OpenAIModelList, dependencies=[Depends(require_any_auth)])
async def list_available_models(request: Request) -> OpenAIModelList:
    """List all available models from configured providers.

    Returns a list of all models that are configured in the system,
    including their provider information and type.

    Requires API key authentication when auth is enabled.
    """
    config = await get_config_from_state(request)

    # Per-API-key model allowlist set by the auth middleware.
    # None means unrestricted; an empty list is a valid deny-all restriction.
    allowed_models: list[str] | None = getattr(request.state, "allowed_models", None)

    def is_allowed(model_id: str) -> bool:
        return allowed_models is None or model_id in allowed_models

    models = []
    for model_name in config.models:
        if not is_allowed(model_name):
            continue
        model_cfg = config.models.get(model_name)
        provider = None
        if model_cfg and model_cfg.providers:
            sorted_providers = model_cfg.get_providers_by_priority()
            if sorted_providers:
                provider = sorted_providers[0].provider
        models.append(OpenAIModel(id=model_name, provider=provider))

    # Add virtual models when smart routing is enabled
    config_manager = get_config_manager(request)
    smart_cfg = await config_manager.get_smart_routing_config()
    if smart_cfg.enabled:
        for vid in ("auto", "fast", "best"):
            if is_allowed(vid):
                models.append(OpenAIModel(id=vid, provider="routing"))

    return OpenAIModelList(data=models)
