"""Server configuration API endpoints.

Every route is generated from the server settings registry
(``llm_proxy.config.server_settings``): one descriptor per UI-managed
``server_config`` key declares its path, verbs, request model, write merge
mode, and response shape, so the handler logic exists once instead of being
copied per key. Paths, verbs, and payloads are byte-identical to the
hand-written endpoints this replaces (ADR-0026).
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import (
    get_async_session_dep,
    get_config_from_state,
    require_admin_role,
    require_recent_auth_for_mutations,
)
from llm_proxy.api.middleware.rate_limiting import get_rate_limiter
from llm_proxy.api.routers.config.helpers import (
    commit_and_reload,
    get_config_repository,
)
from llm_proxy.api.schemas.server_config import TestEmailRequest
from llm_proxy.config.server_settings import (
    SERVER_SETTINGS,
    ServerSetting,
    apply_write,
    build_response,
    request_model_for,
)
from llm_proxy.notifications.email import send_email

router = APIRouter(
    prefix="/server", tags=["configuration"], dependencies=[Depends(require_admin_role)]
)
limiter = get_rate_limiter()


def _make_get_handler(setting: ServerSetting):
    async def get_server_setting(
        request: Request,
        session: AsyncSession = get_async_session_dep,
    ) -> dict:
        """Return the stored value rendered per the setting's response mode."""
        repo = get_config_repository(session)
        row = await repo.get_server_config(setting.key)
        return build_response(setting, row.value if row else None)

    get_server_setting.__name__ = f"get_server_setting_{setting.key}"
    get_server_setting.__doc__ = f"Get {setting.description}."
    return get_server_setting


def _make_put_handler(setting: ServerSetting):
    request_model = request_model_for(setting)

    async def put_server_setting(
        config_data: request_model,  # ty: ignore[invalid-type-form]
        request: Request,
        session: AsyncSession = get_async_session_dep,
    ) -> dict:
        """Validate, store per the setting's write mode, reload, then respond."""
        if setting.validate is not None:
            try:
                setting.validate(config_data)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        repo = get_config_repository(session)
        row = await repo.get_server_config(setting.key)
        stored = apply_write(setting, config_data, row.value if row else None)
        await repo.set_server_config(setting.key, stored, description=setting.description)
        await commit_and_reload(session, request)
        return build_response(setting, stored)

    put_server_setting.__name__ = f"put_server_setting_{setting.key}"
    put_server_setting.__doc__ = f"Update {setting.description}."
    return put_server_setting


def _make_delete_handler(setting: ServerSetting):
    async def delete_server_setting(
        request: Request,
        session: AsyncSession = get_async_session_dep,
    ) -> dict:
        """Delete the stored row and reload."""
        repo = get_config_repository(session)
        await repo.delete_server_config(setting.key)
        await commit_and_reload(session, request)
        return {"message": setting.delete_message}

    delete_server_setting.__name__ = f"delete_server_setting_{setting.key}"
    delete_server_setting.__doc__ = f"Delete {setting.description}."
    return delete_server_setting


for _setting in SERVER_SETTINGS:
    if "GET" in _setting.verbs:
        router.add_api_route(
            _setting.path,
            _make_get_handler(_setting),
            methods=["GET"],
            name=f"Get {_setting.description}",
        )
    if "PUT" in _setting.verbs:
        router.add_api_route(
            _setting.path,
            _make_put_handler(_setting),
            methods=["PUT"],
            name=f"Update {_setting.description}",
            dependencies=(
                [Depends(require_recent_auth_for_mutations)] if _setting.sensitive else None
            ),
        )
    if "DELETE" in _setting.verbs:
        router.add_api_route(
            _setting.path,
            _make_delete_handler(_setting),
            methods=["DELETE"],
            name=f"Delete {_setting.description}",
        )


@router.post(
    "/console-auth/test-email",
    dependencies=[Depends(require_recent_auth_for_mutations)],
)
@limiter.limit("config.test_email")
async def send_console_auth_test_email(payload: TestEmailRequest, request: Request) -> dict:
    """Send one message through the configured email transport.

    Reads the live runtime config, so the test exercises exactly what
    invitations, email verification and password resets use. The response
    names the transport that delivered it, which is how an operator finds out
    whether the Resend block or the SMTP block was selected.

    Sending exercises the stored Resend key or SMTP password, so it takes the
    same step-up window as the ``PUT`` that stores them.
    """
    console_auth = (await get_config_from_state(request)).server_params.console_auth
    transport = console_auth.email_transport()
    if transport is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No email transport is configured; enable SMTP or Resend with a "
                "complete set of fields and save first."
            ),
        )
    sent = await send_email(
        transport,
        payload.to,
        "LLM Proxy test email",
        "This is a test message from the LLM Proxy console. Email delivery works.\n",
    )
    if not sent:
        raise HTTPException(
            status_code=502,
            detail=(
                f"{transport.kind.upper()} did not accept the test message; "
                "check the saved credentials and sender address."
            ),
        )
    return {"sent": True, "transport": transport.kind}
