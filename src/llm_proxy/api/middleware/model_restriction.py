"""Model restriction middleware.

Handles per-API-key model restrictions for client API endpoints (/v1/*, /servers/*).
"""

import orjson
from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.api.middleware.api_key_auth import add_auth_failure_delay
from llm_proxy.api.middleware.security import get_api_key_lockout_manager
from llm_proxy.core.identity import get_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


def get_model_from_request_body(body: bytes | None) -> str | None:
    """Extract model name from request body."""
    if not body:
        return None
    try:
        data = orjson.loads(body)
        return data.get("model") or data.get("model_id")
    except orjson.JSONDecodeError, AttributeError:
        return None


def check_model_restriction(
    api_key_name: str,
    allowed_models: list[str] | None,
    requested_model: str | None,
) -> tuple[bool, str | None]:
    """Check if requested model is allowed for the API key.

    Semantics: ``None`` means unrestricted; an empty list means deny-all
    (e.g. the key/user constraint intersection left nothing).

    Returns:
        tuple of (is_allowed, error_message)
    """
    if allowed_models is None:
        return True, None  # No restrictions, allow all

    if not requested_model:
        return True, None  # No model specified, allow

    if requested_model in allowed_models:
        return True, None

    if not allowed_models:
        return False, (
            f"API key '{api_key_name}' is not authorized to access any model "
            "(empty allowlist after applying user-level restrictions)."
        )

    return False, (
        f"API key '{api_key_name}' is not authorized to access model '{requested_model}'. "
        f"Allowed models: {', '.join(allowed_models)}"
    )


async def model_restriction_middleware(request: Request, call_next):
    """Check API key model restrictions.

    Validates that the requested model is allowed for the provided API key.
    Applies to both regular API keys and session API keys (sk-ui-).
    """
    # Skip if JWT authenticated (JWT is only for /api/* admin panel, not /v1/*)
    identity = get_request_identity(request)
    if identity.auth_method == "jwt":
        return await call_next(request)

    path = request.url.path
    if not path.startswith("/v1/") and not path.startswith("/servers/"):
        return await call_next(request)

    # /servers/* uses JSON-RPC (no "model" field) and reading the body
    # would starve the mounted MCP proxy ASGI app.
    if path.startswith("/servers/"):
        return await call_next(request)

    allowed_models: list[str] | None = getattr(request.state, "allowed_models", None)

    # None means unrestricted; an empty list means deny-all and must still
    # be enforced below.
    if allowed_models is None:
        return await call_next(request)

    api_key_name: str = getattr(request.state, "api_key_name", None) or "unknown"

    # Extract model name from request body, handling both JSON and multipart/form-data.
    if "multipart/form-data" in request.headers.get("content-type", "").lower():
        try:
            form = await request.form()
            model_field = form.get("model")
            # form.get() returns UploadFile | str | None for multipart;
            # we only want the string value.
            requested_model = model_field if isinstance(model_field, str) else None
        except Exception:
            requested_model = None
    else:
        requested_model = get_model_from_request_body(await request.body())
    is_allowed, error_msg = check_model_restriction(
        api_key_name,
        allowed_models,
        requested_model,
    )
    if not is_allowed:
        client_ip = get_client_ip(request)
        lockout_manager = get_api_key_lockout_manager()
        lockout_manager.record_failed_attempt(client_ip)
        await add_auth_failure_delay()

        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "message": error_msg,
                    "type": "forbidden",
                    "code": "model_not_allowed",
                }
            },
        )

    return await call_next(request)
