"""FastAPI WebSocket router for the OpenAI Realtime API relay.

Clients connect to ``WS /v1/realtime?model=<proxy model>`` and authenticate
with a proxy API key. The proxy resolves the model to a provider, opens a
WebSocket to the provider's native Realtime endpoint, and relays messages
verbatim in both directions while observing ``response.done`` events for
per-turn usage logging (see :mod:`llm_proxy.realtime`).
"""

import time
import uuid
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from llm_proxy.api.dependencies import get_config_manager
from llm_proxy.api.middleware.model_restriction import check_model_restriction
from llm_proxy.core.provider_selector import create_provider_selector
from llm_proxy.core.ws_common import (
    WS_CLOSE_AUTH_FAILED,
    WS_CLOSE_FORBIDDEN,
    WS_CLOSE_UPSTREAM_FAILURE,
    authenticate_ws,
    build_ws_request,
)
from llm_proxy.observability.logger import get_logger
from llm_proxy.realtime.relay import (
    RealtimeRelay,
    StarletteWebSocketAdapter,
    WebsocketsClientAdapter,
)
from llm_proxy.realtime.upstream import (
    REALTIME_SUBPROTOCOL,
    build_realtime_url,
    build_upstream_headers,
    connect_upstream,
)
from llm_proxy.realtime.usage import RealtimeSessionContext, RealtimeUsageObserver

logger = get_logger(__name__)

# Provider types that natively speak the OpenAI Realtime WebSocket protocol.
_REALTIME_PROVIDER_TYPES = frozenset({"openai", "openai-compatible"})

# Subprotocol prefix used by browser clients that cannot set headers
# (openai-insecure-api-key.<key>).
_INSECURE_KEY_SUBPROTOCOL_PREFIX = "openai-insecure-api-key."

# WebSocket close codes used by this endpoint. Two codes follow the official
# OpenAI Realtime close-code scheme (4000-4009 client errors, 4100-4108
# server errors) where a semantic match exists (4004 invalid model, 4007
# rate limited); the rest are the proxy-wide close-code language shared with
# the OpenResponses WebSocket transport (see :mod:`llm_proxy.core.ws_common`)
# — the official 4005 invalid-authentication and 4100-4108 server-error
# codes are intentionally not used so both proxy WS transports speak one
# close-code language. See the "Realtime relay" entry in CONTEXT.md for the
# full table.
_CLOSE_INVALID_MODEL = 4004  # official: invalid model
_CLOSE_RATE_LIMITED = 4007  # official: rate limited (budget cap reached)

ws_router = APIRouter(tags=["realtime"])


def _realtime_error_event(
    message: str,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> dict[str, Any]:
    """Build a Realtime-shaped error event (``{"type": "error", ...}``).

    The top-level ``event_id`` is a required non-nullable string in the
    official error-event schema, so one is generated per event (mirroring
    OpenAI's own ``event_...`` ids).
    """
    error: dict[str, Any] = {"type": error_type, "message": message}
    if code is not None:
        error["code"] = code
    return {"type": "error", "event_id": f"event_{uuid.uuid4().hex[:24]}", "error": error}


async def _ws_send_error(
    websocket: WebSocket,
    message: str,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> None:
    """Send a Realtime error event, swallowing send failures."""
    with suppress(Exception):
        await websocket.send_json(_realtime_error_event(message, error_type, code))


async def _ws_fail(
    websocket: WebSocket,
    message: str,
    *,
    close_code: int,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> None:
    """Send a Realtime error event, then close the socket.

    Both steps are best-effort: the client may already be gone (e.g. it
    disconnected while the endpoint was authenticating), and neither a
    failed send nor a failed close may raise out of the endpoint.
    """
    await _ws_send_error(websocket, message, error_type, code)
    with suppress(Exception):
        await websocket.close(code=close_code)


def _negotiated_subprotocol(websocket: WebSocket) -> str | None:
    """Select the ``realtime`` subprotocol when the client offered it."""
    offered = websocket.scope.get("subprotocols") or []
    return REALTIME_SUBPROTOCOL if REALTIME_SUBPROTOCOL in offered else None


async def _ws_check_budget(websocket: WebSocket, auth_info: dict[str, Any]) -> bool:
    """Enforce the key/account spending caps; returns True to proceed.

    Mirrors the budget enforcement shared by the /v1/* and /servers/*
    middlewares (the HTTP middleware never sees WebSocket scopes, so the
    check must happen here): a capped key or account is rejected with a
    Realtime error event, and spend that cannot be confirmed fails closed.
    """
    from llm_proxy.api.middleware.mcp_proxy import BudgetCheckStatus, check_key_budget

    status = await check_key_budget(auth_info)
    if status is BudgetCheckStatus.OK:
        return True
    if status is BudgetCheckStatus.UNAVAILABLE:
        await _ws_fail(
            websocket,
            "Budget check unavailable; request rejected (fail closed).",
            close_code=WS_CLOSE_UPSTREAM_FAILURE,
            error_type="server_error",
            code="budget_unavailable",
        )
        return False
    await _ws_fail(
        websocket,
        "Budget limit exceeded for this API key or account.",
        close_code=_CLOSE_RATE_LIMITED,
        code="budget_exceeded",
    )
    return False


@ws_router.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket) -> None:
    """OpenAI Realtime API transparent relay.

    Clients connect with ``?model=<proxy model>`` and authenticate with a
    proxy API key (``Authorization`` header, ``x-api-key`` header, the
    ``openai-insecure-api-key.<key>`` subprotocol for browser clients, or the
    ``api_key`` query parameter). The model must be explicitly marked as
    Realtime-capable and routed to an ``openai`` / ``openai-compatible``
    provider. Messages are relayed verbatim to the selected provider's
    native Realtime endpoint; ``response.done`` events are observed for
    per-turn usage logging.
    """
    await websocket.accept(subprotocol=_negotiated_subprotocol(websocket))

    auth = await authenticate_ws(
        websocket, insecure_key_subprotocol_prefix=_INSECURE_KEY_SUBPROTOCOL_PREFIX
    )
    if auth is None:
        await _ws_fail(
            websocket,
            "Authentication required.",
            close_code=WS_CLOSE_AUTH_FAILED,
            code="authentication_failed",
        )
        return
    identity, auth_info = auth

    if not await _ws_check_budget(websocket, auth_info):
        return

    model_name = websocket.query_params.get("model") or ""
    if not model_name:
        await _ws_fail(
            websocket,
            "Missing required query parameter 'model'.",
            close_code=_CLOSE_INVALID_MODEL,
            code="invalid_request",
        )
        return

    # Enforce the per-API-key model allowlist (None means unrestricted).
    allowed_models = auth_info.get("allowed_models")
    if isinstance(allowed_models, list):
        is_allowed, error_msg = check_model_restriction(
            identity.api_key_name or "unknown",
            allowed_models,
            model_name,
        )
        if not is_allowed:
            await _ws_fail(
                websocket,
                error_msg or "Model not allowed",
                close_code=WS_CLOSE_FORBIDDEN,
                code="forbidden",
            )
            return

    # Build a Request-like object from the websocket scope so the standard
    # config/dependency accessors work unchanged.
    request = build_ws_request(websocket, identity)

    config_manager = get_config_manager(request)
    config = await config_manager.get_config()

    model_config = await config_manager.get_model_config(model_name)
    if model_config is None:
        await _ws_fail(
            websocket,
            f"Model '{model_name}' not found.",
            close_code=_CLOSE_INVALID_MODEL,
            code="model_not_found",
        )
        return

    if not model_config.supports_realtime:
        await _ws_fail(
            websocket,
            f"Model '{model_name}' is not marked as Realtime-capable.",
            close_code=_CLOSE_INVALID_MODEL,
            code="model_not_supported",
        )
        return

    redis_wrapper = getattr(request.app.state, "redis_client", None)
    redis = redis_wrapper.client if redis_wrapper is not None else None
    orchestrator = create_provider_selector(
        model_config=model_config,
        provider_configs=config.provider_configs,
        max_fallback_attempts=config.server_params.max_fallback_attempts,
        default_max_retries=config.server_params.max_retries,
        circuit_breaker=getattr(request.app.state, "circuit_breaker", None),
        strategy=config.provider_selection.strategy,
        model_name=model_name,
        redis=redis,
        stats_store=getattr(request.app.state, "provider_stats", None),
    )
    await orchestrator.prepare()
    selection = orchestrator.select_next_provider()
    if selection is None:
        await _ws_fail(
            websocket,
            f"No available provider for model '{model_name}'.",
            close_code=WS_CLOSE_UPSTREAM_FAILURE,
            code="provider_unavailable",
        )
        return

    provider_config = selection.provider_config
    if provider_config.type not in _REALTIME_PROVIDER_TYPES:
        await _ws_fail(
            websocket,
            f"Provider '{selection.provider_name}' (type '{provider_config.type}') "
            "does not support the Realtime API.",
            close_code=WS_CLOSE_UPSTREAM_FAILURE,
            code="provider_not_supported",
        )
        return

    upstream_model = selection.provider_model_name or model_name
    url = build_realtime_url(provider_config, upstream_model)
    headers = build_upstream_headers(
        provider_config,
        model_name=upstream_model,
        safety_identifier=websocket.headers.get("openai-safety-identifier"),
    )
    try:
        upstream_conn = await connect_upstream(url, headers, timeout=provider_config.timeout)
    except Exception as exc:  # noqa: BLE001 - map any connect failure to an error event
        logger.warning(f"Upstream Realtime connection failed: {exc}")
        await _ws_fail(
            websocket,
            f"Upstream connection failed: {exc}",
            close_code=WS_CLOSE_UPSTREAM_FAILURE,
            error_type="server_error",
            code="upstream_connection_failed",
        )
        return

    observer = RealtimeUsageObserver(
        context=RealtimeSessionContext(
            model=model_name,
            provider=selection.provider_name,
            api_key_name=identity.api_key_name or "",
            request_id=request.state.request_id,
            client_ip=websocket.client.host if websocket.client is not None else None,
            user_agent=websocket.headers.get("user-agent"),
            session_id=request.state.request_id,
            user_id=identity.user_id,
        ),
        config_manager=config_manager,
    )

    relay = RealtimeRelay(
        client=StarletteWebSocketAdapter(websocket),
        upstream=WebsocketsClientAdapter(upstream_conn),
        on_upstream_message=observer.on_upstream_message,
        on_client_error=lambda code, message: _ws_send_error(websocket, message, code=code),
    )
    session_started_at = time.monotonic()
    try:
        await relay.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning(f"Realtime relay error: {exc}")
    finally:
        # Teardown metadata: record the session duration, turn count, and
        # upstream session id so operators can correlate closed sessions with
        # their request logs.
        logger.debug(
            "Realtime session closed",
            extra={
                "request_id": request.state.request_id,
                "session_id": observer.session_id,
                "turns": observer.turns,
                "duration_s": round(time.monotonic() - session_started_at, 3),
            },
        )
        with suppress(Exception):
            await websocket.close()


__all__ = ["ws_router"]
