"""Request-context assembly for the unified processing pipeline.

Extracted from ``llm_proxy.api.dependencies``. Builds the per-request
:class:`RequestContext` from the incoming FastAPI request: resolves virtual
models via :mod:`llm_proxy.routing.orchestrator`, re-checks per-API-key model
restrictions against the resolved concrete model, wires the provider selector,
response store, web-search interceptor, and per-user tracing registry.

Pure FastAPI dependency providers (HTTP clients, config manager, auth, adapter
factory) remain in :mod:`llm_proxy.api.dependencies`; this module imports them
one-directionally to avoid import cycles.
"""

import time
import uuid
from typing import Any, Protocol

from fastapi import Request

from llm_proxy.api.dependencies import (
    create_adapter_for_provider,
    extract_session_id,
    extract_trace_id,
    extract_user_id,
    get_config_manager,
)
from llm_proxy.api.middleware.model_restriction import check_model_restriction
from llm_proxy.config.types.model import ProviderSelectionStrategy
from llm_proxy.core.conversation_key import conversation_key
from llm_proxy.core.exceptions import (
    AuthenticationFailedError,
    ModelNotFoundError,
    ValidationError,
)
from llm_proxy.core.identity import get_request_identity
from llm_proxy.core.processing import RequestContext
from llm_proxy.core.request_type import RequestType
from llm_proxy.core.request_utils import peer_is_trusted_proxy
from llm_proxy.observability.logger import get_logger
from llm_proxy.routing.orchestrator import orchestrate_smart_routing
from llm_proxy.services import runtime_services

logger = get_logger(__name__)


def _peer_is_trusted_proxy(request: Request) -> bool:
    """Check whether the immediate TCP peer is in TRUSTED_PROXIES.

    Thin wrapper over :func:`llm_proxy.core.request_utils.peer_is_trusted_proxy`
    kept for call-site readability in this module.
    """
    return peer_is_trusted_proxy(request)


class HasModel(Protocol):
    """Protocol for objects that have a model attribute."""

    model: str


def _get_model(request: Any) -> str:
    if isinstance(request, dict):
        return request.get("model", "")
    return getattr(request, "model", "")


def _optional_number(value: Any) -> float | None:
    """Return ``value`` as a float, or None when it is not a real number.

    Booleans are excluded deliberately: they are ints in Python, and routing
    telemetry is JSON that may round-trip through the log store before it is read
    back. None means "not reported", which is not the same fact as 0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _log_judge_call(
    req: Request,
    decision: Any,
    *,
    request_id: str,
    requested_model: str | None,
    session_id: str | None,
) -> None:
    """Record a routing judge consultation as its own log row and usage record.

    Called as soon as routing returns, before the request is served: the call has
    already happened and been billed, so its record must not depend on the parent
    request succeeding (a 403 on the resolved model, for instance). The judge's
    spend is attributed to the request that caused it, but stays out of that
    request's own cost; the row is the single accounting source (ADR-0018).

    A consultation without a provider never reached an upstream: nothing was
    billed, and its diagnostic stays on the parent's routing metadata.
    """
    judge = getattr(decision, "judge", None)
    if not isinstance(judge, dict) or not judge.get("provider"):
        return

    from llm_proxy.core.request_utils import get_client_ip
    from llm_proxy.observability.log_intake import InternalCallLogEntry, record_internal_call

    input_tokens = _optional_number(judge.get("input_tokens"))
    output_tokens = _optional_number(judge.get("output_tokens"))
    total_tokens = (
        int((input_tokens or 0.0) + (output_tokens or 0.0))
        if input_tokens is not None or output_tokens is not None
        else None
    )
    latency_ms = _optional_number(judge.get("latency_ms"))
    started_at = _optional_number(judge.get("started_at"))

    identity = get_request_identity(req)
    entry = InternalCallLogEntry(
        # Deterministic child id: the same parent can never write two judge rows,
        # and the pair stays greppable next to each other.
        request_id=f"{request_id}:judge",
        timestamp=started_at if started_at is not None else time.time(),
        model=str(judge.get("model") or ""),
        provider=judge["provider"],
        provider_model_name=judge.get("provider_model_name"),
        # A billed HTTP call is a success even when the verdict abstains; only a
        # call that never came back (deadline, transport error) has no status.
        status_code=200 if judge.get("error") is None else None,
        response_time_ms=int(latency_ms) if latency_ms is not None else None,
        prompt_tokens=int(input_tokens) if input_tokens is not None else None,
        completion_tokens=int(output_tokens) if output_tokens is not None else None,
        total_tokens=total_tokens,
        cost_usd=_optional_number(judge.get("cost")),
        error_message=judge.get("error"),
        log_metadata={
            "internal_call": "routing_judge",
            "parent_request_id": request_id,
            "requested_model": requested_model,
            "resolved_model": getattr(decision, "model", None),
            "conversation_key": getattr(decision, "conversation_key", None),
            "judge": dict(judge),
        },
    )
    try:
        record_internal_call(
            entry,
            user_id=identity.user_id,
            user_identity=identity.user,
            api_key_name=identity.api_key_name,
            auth_method=identity.auth_method,
            session_id=session_id,
            client_ip=get_client_ip(req),
            user_agent=req.headers.get("user-agent"),
        )
    except Exception:  # noqa: BLE001 - accounting must never fail the request
        logger.warning("Failed to record routing judge call", exc_info=True)


async def _build_request_context(
    request: Any,
    req: Request,
    request_type: RequestType | None = None,
    protocol_name: str | None = None,
) -> RequestContext:
    from llm_proxy.core.processing.base import ServiceDependencies
    from llm_proxy.core.provider_selector import create_provider_selector
    from llm_proxy.protocols.openresponses.store import ResponseStore

    model_name = _get_model(request)
    config_manager = get_config_manager(req)
    config = await config_manager.get_config()

    # Fetch once at the top for reuse throughout
    peer_trusted = _peer_is_trusted_proxy(req)
    session_id = extract_session_id(req) if peer_trusted else None
    services = runtime_services(req)
    redis = services.redis_client()

    # Mint the request id before routing so a judge call made during routing can be
    # logged as its own row and still name the request it belongs to. The pipeline
    # reuses it from ``request.state`` when it builds the request's EventContext.
    # ``isinstance`` rather than truthiness: a test double's state may answer with
    # an arbitrary object where a real Starlette State raises AttributeError.
    existing_id = getattr(req.state, "request_id", None)
    request_id = existing_id if isinstance(existing_id, str) and existing_id else str(uuid.uuid4())
    req.state.request_id = request_id

    # Intercept virtual models (auto/fast/best) and resolve to real models.
    routing_decision = None
    requested_model = None
    routing_result = await orchestrate_smart_routing(
        model_name=model_name,
        request=request,
        request_type=request_type,
        config=config,
        config_manager=config_manager,
        services=services,
        request_id=request_id,
        session_id=session_id,
        redis=redis,
    )
    if routing_result is not None:
        routing_decision = routing_result.routing_decision
        requested_model = routing_result.requested_model
        model_name = routing_result.resolved_model

        # Record the judge call before any later failure can discard it: the call
        # was made and billed while routing ran.
        _log_judge_call(
            req,
            routing_decision,
            request_id=request_id,
            requested_model=requested_model,
            session_id=session_id,
        )

        # Enforce per-API-key model restrictions against the resolved concrete
        # model. The middleware only validates the virtual model name requested
        # by the client, so we must re-check after smart routing resolves it.
        allowed_models: list[str] | None = getattr(req.state, "allowed_models", None)
        if isinstance(allowed_models, list):
            api_key_name: str = getattr(req.state, "api_key_name", None) or "unknown"
            is_allowed, error_msg = check_model_restriction(
                api_key_name,
                allowed_models,
                model_name,
            )
            if not is_allowed:
                raise AuthenticationFailedError(
                    message=error_msg or "Model not allowed",
                    code="forbidden",
                    status_code=403,
                )
    model_config = await config_manager.get_model_config(model_name)

    if model_config is None:
        raise ModelNotFoundError(model_name)

    # System One and Decisions are evaluation endpoints, not chat: only models
    # explicitly marked as decision models may serve them. One flag gates both —
    # they are two envelopes over the same primitives — so a decision model on a
    # System One upstream answers /v1/decisions and one on OpenAI answers
    # /v1/systemone, each bridged by the provider (see
    # ``llm_proxy.models.decisions_bridge``). Without this check a chat model
    # would resolve to an adapter that does not implement the endpoint and fail
    # with an opaque 500 instead of a clear 400.
    if (
        request_type in (RequestType.SYSTEMONE, RequestType.DECISIONS)
        and not model_config.supports_systemone
    ):
        raise ValidationError(
            message=(
                f"Model '{model_name}' is not marked as a decision model "
                "(supports_systemone). Mark it as a decision model in Models "
                f"to serve /v1/{request_type.value}."
            ),
            code="invalid_request_error",
            status_code=400,
        )

    # ``get_model_config`` may have refreshed the snapshot when a peer worker
    # added the model (multi-worker deployments). Re-read it so the provider
    # configs and selection/retry settings below match the resolved model.
    config = await config_manager.get_config()

    circuit_breaker = services.circuit_breaker()

    # Resolve per-request inputs for the provider-selection strategy. The
    # strategy itself is a single global setting (server_config:
    # provider_selection); the conversation key is only needed by
    # session_sticky: session id when the peer is a trusted proxy, otherwise
    # a hash of the first user message.
    strategy = config.provider_selection.strategy
    conv_key: str | None = None
    if strategy is ProviderSelectionStrategy.SESSION_STICKY:
        from llm_proxy.routing.message_extract import extract_messages_for_routing

        conv_key = conversation_key(session_id, extract_messages_for_routing(request))

    orchestrator = create_provider_selector(
        model_config=model_config,
        provider_configs=config.provider_configs,
        max_fallback_attempts=config.server_params.max_fallback_attempts,
        default_max_retries=config.server_params.max_retries,
        circuit_breaker=circuit_breaker,
        strategy=strategy,
        model_name=model_name,
        conversation_key=conv_key,
        redis=redis,
        stats_store=services.provider_stats(),
    )
    # Resolve async per-request selection inputs (the session_sticky provider
    # mapping) before the pipeline starts. No-op for other strategies.
    await orchestrator.prepare()

    trace_id = extract_trace_id(req) if peer_trusted else None
    user_id = extract_user_id(req)

    response_store: ResponseStore | None = None
    if redis is not None:
        response_store = ResponseStore(redis_client=redis)

    ctx = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(
            adapter_factory=lambda r, s: create_adapter_for_provider(r, s),
            config_manager=config_manager,
        ),
        process_request=None,
        process_response=None,
        trace_id=trace_id,
        session_id=session_id,
        user_id=user_id,
        response_store=response_store,
        verbose_routing_logs=getattr(
            getattr(config.server_params, "logging", None), "verbose_routing_logs", False
        ),
    )

    # Set routing decision fields if virtual model was resolved
    if routing_decision is not None:
        ctx.routing_decision = routing_decision
        # Post-request EWMA observation lives in the routing package; wire it
        # here as a callback so core never imports routing (keeps the
        # dependency direction routing -> core only).
        from llm_proxy.routing.model_experience import observe_model_experience

        ctx.on_request_completed = lambda event_context, success: observe_model_experience(
            event_context, ctx, success=success
        )
    if requested_model is not None:
        ctx.requested_model = requested_model
        # Preserve original virtual model on request state so early-failure
        # error logs record the user-facing model even before ProviderSelectionStage.
        req.state.model = requested_model

    # The interceptor is process-local, so resolve it from the runtime against
    # the snapshot fetched above instead of reading app state directly: a reload
    # listener may still be swapping it, and the runtime rebuilds under its own
    # lock. In steady state this is a single equality check that returns the
    # cached interceptor.
    from llm_proxy.web_search.runtime import ensure_web_search_interceptor

    web_search_interceptor = await ensure_web_search_interceptor(
        req.app, config.server_params.web_search
    )
    if web_search_interceptor is not None:
        ctx.web_search_interceptor = web_search_interceptor
        if hasattr(request, "tools") and request.tools:
            from llm_proxy.web_search.interceptor import WebSearchInterceptor

            if isinstance(web_search_interceptor, WebSearchInterceptor):
                ctx.web_search_tool_config = web_search_interceptor.extract_web_search_tool_config(
                    request.tools
                )

    # Resolve a per-user tracing registry for the authenticated owner. When the
    # owner has a personal tracing config, their requests are traced through a
    # dedicated registry (their backends + the shared internal handlers); otherwise
    # ctx.tracing_registry stays None and the request falls back to the global
    # registry, which holds only internal handlers (no tracing backends) — so a
    # user without personal config is never traced by anyone else's Langfuse.
    owner_user_id = get_request_identity(req).user_id
    if owner_user_id is not None:
        from llm_proxy.observability.user_tracing import get_user_tracing_manager

        user_registry = await get_user_tracing_manager().get_registry(owner_user_id)
        if user_registry is not None:
            ctx.tracing_registry = user_registry

    if request_type:
        ctx.request_type = request_type
    if protocol_name:
        ctx.protocol_name = protocol_name
    return ctx


async def build_embeddings_request_context(
    request: HasModel,
    req: Request,
) -> RequestContext:
    """Build RequestContext for embeddings request processing."""
    return await _build_request_context(request, req, request_type=RequestType.EMBEDDING)


async def build_images_request_context(
    request: HasModel,
    req: Request,
    *,
    request_type: RequestType = RequestType.IMAGE_GENERATION,
) -> RequestContext:
    """Build RequestContext for image request processing."""
    return await _build_request_context(request, req, request_type=request_type)


async def build_speech_request_context(
    request: HasModel,
    req: Request,
) -> RequestContext:
    """Build RequestContext for speech request processing."""
    return await _build_request_context(
        request, req, request_type=RequestType.SPEECH, protocol_name="speech"
    )


async def build_transcription_request_context(
    request: HasModel,
    req: Request,
) -> RequestContext:
    """Build RequestContext for transcription request processing."""
    return await _build_request_context(
        request, req, request_type=RequestType.TRANSCRIPTION, protocol_name="transcription"
    )


async def build_translation_request_context(
    request: HasModel,
    req: Request,
) -> RequestContext:
    """Build RequestContext for translation request processing."""
    return await _build_request_context(
        request, req, request_type=RequestType.TRANSLATION, protocol_name="translation"
    )


async def build_systemone_request_context(
    request: HasModel,
    req: Request,
) -> RequestContext:
    """Build RequestContext for System One evaluation request processing."""
    return await _build_request_context(
        request, req, request_type=RequestType.SYSTEMONE, protocol_name="systemone"
    )


async def build_decisions_request_context(
    request: HasModel,
    req: Request,
) -> RequestContext:
    """Build RequestContext for Decisions evaluation request processing."""
    return await _build_request_context(
        request, req, request_type=RequestType.DECISIONS, protocol_name="openai_decisions"
    )


async def build_request_context(
    request: HasModel,
    req: Request,
    protocol_name: str | None = None,
) -> RequestContext:
    """Build RequestContext for regular request processing."""
    return await _build_request_context(
        request, req, request_type=RequestType.CHAT, protocol_name=protocol_name
    )
