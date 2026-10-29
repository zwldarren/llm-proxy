"""Server routers package."""

# Import the protocol router factory for dynamic protocol registration
from llm_proxy.api.routers.protocol import (
    create_all_protocol_routers,
    create_protocol_list_router,
    import_registered_protocol_modules,
)

# Import routers
from .api_keys import router as api_keys_router
from .auth import router as auth_router
from .catalog import router as catalog_router
from .config import router as config_router
from .feedback import router as feedback_router
from .health import router as health_router
from .invitations import public_router as invitations_public_router
from .invitations import router as invitations_router
from .logs import router as logs_router
from .mcp import mcp_proxy_app
from .mcp import public_router as mcp_public_router
from .mcp import router as mcp_router
from .me import router as me_router
from .me_tracing import router as me_tracing_router
from .models import router as models_router
from .oauth import public_router as oauth_public_router
from .oauth import router as oauth_router
from .openresponses import router as openresponses_router
from .openresponses import ws_router as openresponses_ws_router
from .passkeys import public_router as passkeys_public_router
from .passkeys import router as passkeys_router
from .realtime import ws_router as realtime_ws_router
from .routing import router as routing_export_router
from .system import router as system_router
from .team import router as team_router
from .two_factor import router as two_factor_router

__all__ = [
    # Dynamic protocol registration utilities
    "create_all_protocol_routers",
    "create_protocol_list_router",
    "import_registered_protocol_modules",
    # Router exports
    "api_keys_router",
    "auth_router",
    "catalog_router",
    "config_router",
    "feedback_router",
    "routing_export_router",
    "health_router",
    "invitations_public_router",
    "invitations_router",
    "logs_router",
    "mcp_proxy_app",
    "mcp_public_router",
    "mcp_router",
    "me_router",
    "me_tracing_router",
    "models_router",
    "oauth_public_router",
    "oauth_router",
    "passkeys_public_router",
    "passkeys_router",
    "two_factor_router",
    "openresponses_router",
    "openresponses_ws_router",
    "realtime_ws_router",
    "system_router",
    "team_router",
]
