"""Server configuration API endpoints.

Every route is generated from the server settings registry
(``llm_proxy.config.server_settings``): one descriptor per UI-managed
``server_config`` key declares its path, verbs, request model, write merge
mode, and response shape, so the handler logic exists once instead of being
copied per key. Paths, verbs, and payloads are byte-identical to the
hand-written endpoints this replaces (ADR-0025).
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_async_session_dep, require_admin_role
from llm_proxy.api.routers.config.helpers import (
    commit_and_reload,
    get_config_repository,
)
from llm_proxy.config.server_settings import (
    SERVER_SETTINGS,
    ServerSetting,
    apply_write,
    build_response,
    request_model_for,
)

router = APIRouter(
    prefix="/server", tags=["configuration"], dependencies=[Depends(require_admin_role)]
)


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
        )
    if "DELETE" in _setting.verbs:
        router.add_api_route(
            _setting.path,
            _make_delete_handler(_setting),
            methods=["DELETE"],
            name=f"Delete {_setting.description}",
        )
