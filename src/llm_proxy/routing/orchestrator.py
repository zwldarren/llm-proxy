"""Smart-routing orchestration: virtual-model resolution with continuity.

Extracted from ``llm_proxy.api.dependencies._build_request_context``. Owns the
application-layer routing workflow that runs when a client requests a virtual
model (``auto``/``fast``/``best``): conversation-continuity lookup, resolution
against the candidate pool with route-record persistence, and storing the
chosen model for the next turn.

The per-API-key model-restriction re-check intentionally stays in the API
context layer (``llm_proxy.api.context``) so this module never depends on
``llm_proxy.api`` and remains a pure routing concern.
"""

import logging
from dataclasses import dataclass
from typing import Any

from llm_proxy.config.manager import DatabaseConfigManager
from llm_proxy.config.types.main import ProxyConfig
from llm_proxy.core.conversation_key import conversation_key
from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.core.request_type import RequestType
from llm_proxy.database.connection import get_async_session_context
from llm_proxy.routing.config import routing_mode_from_model
from llm_proxy.routing.message_extract import extract_messages_for_routing
from llm_proxy.routing.resolver import resolve_virtual_model
from llm_proxy.routing.types import RoutingDecision

logger = logging.getLogger("llm-proxy.routing.orchestrator")


@dataclass
class SmartRoutingResult:
    """Outcome of resolving a virtual model request.

    ``resolved_model`` is the concrete model chosen by the router;
    ``requested_model`` is the virtual model name the client asked for;
    ``routing_decision`` carries the full routing decision for telemetry.
    """

    resolved_model: str
    routing_decision: RoutingDecision
    requested_model: str


async def orchestrate_smart_routing(
    *,
    model_name: str,
    request: Any,
    request_type: RequestType | None,
    config: ProxyConfig,
    config_manager: DatabaseConfigManager,
    app_state: Any,
    request_id: str | None,
    session_id: str | None,
    redis: Any | None,
) -> SmartRoutingResult | None:
    """Resolve a virtual model request to a concrete model.

    Returns ``None`` when ``model_name`` is not a virtual model (no routing
    needed). Otherwise performs conversation-continuity lookup, resolves the
    virtual model (persisting the route record), stores the chosen model for
    the next turn, and returns the resolution result. Raises
    :class:`ConfigurationError` when smart routing is disabled or a non-chat
    request targets a virtual model.
    """
    routing_mode = routing_mode_from_model(model_name)
    if routing_mode is None:
        return None

    smart_cfg = await config_manager.get_smart_routing_config()
    if not smart_cfg.enabled:
        raise ConfigurationError(
            f"Smart routing is disabled; '{model_name}' is a virtual model. "
            "Enable smart routing in the admin UI or request a concrete model."
        )
    if request_type != RequestType.CHAT:
        raise ConfigurationError(f"Virtual model '{model_name}' only supports chat completions.")

    messages = extract_messages_for_routing(request)

    # ─── Conversation continuity: retrieve previous model ───
    previous_model: str | None = None
    conv_key: str | None = None
    if redis is not None:
        conv_key = conversation_key(session_id, messages)
        if conv_key:
            try:
                prev = await redis.get(f"routing:conv:{conv_key}:last_model")
                if prev:
                    previous_model = prev.decode("utf-8") if isinstance(prev, bytes) else prev
            except Exception as exc:
                # Degrade gracefully: continue without continuity, but leave a
                # diagnostic trace so Redis outages/wrong-types are visible.
                logger.warning(
                    "Redis conversation-continuity lookup failed for key %s: %s",
                    conv_key,
                    exc,
                )

    # Resolve against a short-lived session for route_records persistence.
    async with get_async_session_context() as routing_session:
        routing_decision = await resolve_virtual_model(
            mode=routing_mode,
            messages=messages,
            request=request,
            config=config,
            config_manager=config_manager,
            app_state=app_state,
            session=routing_session,
            request_id=request_id,
            mode_weights=smart_cfg.mode_weights,
            previous_model=previous_model,
        )
        await routing_session.commit()  # persist the route_record row

    # ─── Store the chosen model for next turn ───
    if redis is not None and conv_key:
        try:
            await redis.setex(
                f"routing:conv:{conv_key}:last_model",
                1800,  # 30 min TTL
                routing_decision.model,
            )
        except Exception as exc:
            # Next turn loses continuity; log so the failure is diagnosable.
            logger.warning(
                "Redis conversation-continuity persist failed for key %s: %s",
                conv_key,
                exc,
            )

    return SmartRoutingResult(
        resolved_model=routing_decision.model,
        routing_decision=routing_decision,
        requested_model=model_name,
    )
