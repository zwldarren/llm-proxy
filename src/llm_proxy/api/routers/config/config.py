"""Configuration management API endpoints.

This module aggregates sub-routers from the config package.
"""

from fastapi import APIRouter

# Import all sub-routers
from llm_proxy.api.routers.config.circuit_breaker import (
    router as circuit_breaker_router,
)
from llm_proxy.api.routers.config.models import (
    public_router as models_public_router,
)
from llm_proxy.api.routers.config.models import router as models_router
from llm_proxy.api.routers.config.pricing import router as pricing_router
from llm_proxy.api.routers.config.provider_models import (
    router as provider_models_router,
)
from llm_proxy.api.routers.config.providers import (
    router as providers_router,
)
from llm_proxy.api.routers.config.server import router as server_router

# Create the main router — individual sub-routers carry their own auth dependencies
router = APIRouter(
    prefix="/api/config",
    tags=["configuration"],
)

# Include sub-routers — more specific routes MUST be registered before greedy ones
# (provider_models_router uses /providers/{name}/models which must match
# before providers_router's /providers/{name:path} catch-all)
router.include_router(provider_models_router)
router.include_router(providers_router)
router.include_router(models_router)
router.include_router(models_public_router)
router.include_router(server_router)
router.include_router(circuit_breaker_router)
router.include_router(pricing_router)

__all__ = ["router"]
