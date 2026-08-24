"""Dynamic protocol router factory.

This module provides a factory function to create routers for protocol endpoints,
preserving OpenAPI documentation by creating specific endpoints for each protocol
rather than using wildcard routes.

The factory dynamically creates APIRouter instances with properly typed endpoints
that integrate with the UnifiedProcessor infrastructure.
"""

import asyncio
import contextlib
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from llm_proxy.api.context import (
    build_embeddings_request_context,
    build_images_request_context,
    build_request_context,
    build_speech_request_context,
    build_transcription_request_context,
    build_translation_request_context,
)
from llm_proxy.api.dependencies import get_request_identity, require_api_key_auth
from llm_proxy.api.keepalive import await_with_keepalive, supports_keepalive
from llm_proxy.api.utils import (
    add_info_endpoint,
    create_standard_router,
    create_traced_handler,
)
from llm_proxy.config.manager import resolve_keepalive_params
from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.core.request_type import RequestType
from llm_proxy.observability.logger import get_logger
from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openresponses.store import ResponseStore
from llm_proxy.protocols.registry import get_protocols_info

logger = get_logger(__name__)

# Lookup table for non-chat protocol context builders.
# Add new protocol types here instead of adding boolean flags + elif branches.
_NON_CHAT_CONTEXT_BUILDERS: dict[str, Any] = {
    "embeddings": build_embeddings_request_context,
    "image_generations": lambda r, fr: build_images_request_context(
        r, fr, request_type=RequestType.IMAGE_GENERATION
    ),
    "image_edits": lambda r, fr: build_images_request_context(
        r, fr, request_type=RequestType.IMAGE_EDIT
    ),
    "speech": build_speech_request_context,
    "transcription": build_transcription_request_context,
    "translation": build_translation_request_context,
}


async def require_any_auth(request: Request):
    """Require either API key or admin JWT authentication."""
    identity = get_request_identity(request)
    if not identity.is_authenticated:
        await require_api_key_auth(request)


def _create_endpoint_fn(
    endpoint: ProtocolEndpoint,
    middleware: list[Callable[[Any, Request], Awaitable[None]]],
) -> Callable[[Any, Request], Awaitable[Any]]:
    """Create a request handler function for a protocol endpoint.

    Applies protocol-specific middleware and processes the request
    through the UnifiedProcessor.

    Args:
        endpoint: The protocol endpoint configuration
        middleware: List of middleware functions to apply

    Returns:
        An async handler function
    """
    protocol_name = endpoint.name

    async def handle_protocol_request(request: Any, fastapi_request: Request) -> Any:
        """Handle a protocol request using the UnifiedProcessor."""
        # For protocols with custom request parsing (e.g., multipart uploads)
        if request is None:
            if endpoint.parse_http_request is not None:
                request = await endpoint.parse_http_request(fastapi_request)
            else:
                raise ConfigurationError(
                    f"Protocol '{protocol_name}' has no request_model and no "
                    "parse_http_request function"
                )

        # Stash parsed body for early-failure logging (multipart protocols
        # bypass the stash in create_traced_handler).
        if request is not None and hasattr(request, "model_dump"):
            with contextlib.suppress(Exception):
                fastapi_request.state.parsed_request_body = request.model_dump()

        for mw in middleware:
            await mw(request, fastapi_request)

        processor = getattr(fastapi_request.app.state, f"{protocol_name}_processor", None)
        if processor is None:
            raise ConfigurationError(
                f"{protocol_name}_processor not initialized. "
                "Ensure lifespan is properly configured."
            )

        builder = _NON_CHAT_CONTEXT_BUILDERS.get(protocol_name)
        if builder is not None:
            context = await builder(request, fastapi_request)
        else:
            context = await build_request_context(
                request, fastapi_request, protocol_name=protocol_name
            )

        process_coro = processor.process(
            protocol_request=request, req=fastapi_request, context=context
        )

        # OpenResponses ``background`` mode: return immediately with an
        # in_progress response and run the request as a background task, then
        # persist the result so GET /v1/responses/{id} can retrieve it.
        if protocol_name == "openresponses" and getattr(request, "background", False):
            return await _run_background_openresponses(
                request, fastapi_request, context, process_coro
            )

        # CDN proxies (e.g. Cloudflare) abort requests that stay silent for
        # ~100s. Slow non-streaming requests (long-reasoning models) can opt
        # into whitespace heartbeats to survive that budget. See keepalive.py.
        keepalive = resolve_keepalive_params(
            getattr(fastapi_request.app.state, "config_manager", None)
        )
        is_stream = bool(getattr(request, "stream", False))
        if keepalive.enabled and supports_keepalive(protocol_name, is_stream):
            return await await_with_keepalive(
                process_coro,
                grace_seconds=keepalive.grace_seconds,
                interval_seconds=keepalive.interval_seconds,
            )

        return await process_coro

    return handle_protocol_request


async def _run_background_openresponses(
    request: Any,
    fastapi_request: Request,
    context: Any,
    process_coro: Any,
) -> JSONResponse:
    """Run an OpenResponses request in the background and return immediately.

    The spec's ``background`` field asks the server to run the request in the
    background and return immediately. The proxy generates a response id
    upfront, persists an ``in_progress`` placeholder so the id is pollable
    right away, spawns the processing as a background task, and overwrites the
    stored response with the final result so the client can poll
    ``GET /v1/responses/{id}``.
    """
    response_id = f"resp_{secrets.token_hex(12)}"
    model = getattr(request, "model", "") or ""
    created_at = int(time.time())

    # Background responses must be persisted for polling; force store on the
    # request so the response body carries it. Streaming has no consumer in
    # background mode, so the request is executed non-streaming.
    if hasattr(request, "store"):
        request.store = True
    if hasattr(request, "stream"):
        request.stream = False

    redis_wrapper = getattr(fastapi_request.app.state, "redis_client", None)
    api_key_name = getattr(fastapi_request.state, "api_key_name", None)
    store: ResponseStore | None = None
    if redis_wrapper is not None and redis_wrapper.client is not None and api_key_name:
        store = ResponseStore(redis_client=redis_wrapper.client)

    if store is None:
        # Background responses must be pollable via GET /v1/responses/{id};
        # without response storage the in_progress placeholder would never
        # resolve and the client would poll a dangling id forever. Fail fast
        # instead of returning a 200 that can never complete.
        process_coro.close()
        raise ConfigurationError(
            message="Background mode requires response storage. Redis must be enabled "
            "and the request must be authenticated with an API key.",
            code="redis_not_available",
            status_code=503,
        )

    in_progress_body = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": None,
        "status": "in_progress",
        "model": model,
        "output": [],
        "store": True,
        "background": True,
    }

    # Persist the in_progress placeholder immediately so polling
    # GET /v1/responses/{id} succeeds while the request is still running.
    if store is not None and api_key_name:
        try:
            await store.store(api_key_name, response_id, in_progress_body)
        except Exception:  # noqa: BLE001 - storage failure must not fail the request
            logger.error(f"Failed to store in_progress response {response_id}", exc_info=True)

    async def _background_execute() -> None:
        try:
            response = await process_coro
            body = orjson.loads(response.body)
        except Exception as exc:  # noqa: BLE001 - persist failures too
            logger.error(f"Background response {response_id} failed: {exc}")
            body = None
        # The pipeline may succeed at the HTTP level while the upstream
        # failed (provider 4xx/5xx -> error-handler body {"error": ...}).
        # Such a body is not a response object: storing it would make
        # GET /v1/responses/{id} 500 on missing required fields, so it is
        # normalized into a spec-shaped failed response carrying the error.
        if not isinstance(body, dict) or body.get("object") != "response":
            error_payload: dict[str, Any] | None = None
            if isinstance(body, dict):
                raw_error = body.get("error")
                if isinstance(raw_error, dict):
                    error_payload = {
                        "code": raw_error.get("code") or "server_error",
                        "message": raw_error.get("message") or "Background request failed",
                        "type": raw_error.get("type") or "server_error",
                        "param": raw_error.get("param"),
                    }
            body = {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "completed_at": int(time.time()),
                "status": "failed",
                "model": model,
                "output": [],
                "error": error_payload
                or {
                    "code": "server_error",
                    "message": "Background request failed",
                    "type": "server_error",
                    "param": None,
                },
            }
        body["id"] = response_id
        body["store"] = True
        body["background"] = True
        body["created_at"] = created_at
        if body.get("status") == "in_progress":
            body["status"] = "completed"
            body["completed_at"] = int(time.time())
        # Stored responses must carry the input so follow-up continuations can
        # replay the conversation (the formatted body does not include it).
        input_items = await _background_input_items(request, store, api_key_name)
        if input_items is not None:
            body["input"] = input_items
        if store is not None and api_key_name:
            try:
                await store.store(api_key_name, response_id, body)
            except Exception:  # noqa: BLE001 - storage failure must not crash the task
                logger.error(f"Failed to store background response {response_id}", exc_info=True)

    task = asyncio.create_task(_background_execute())
    # Keep a strong reference to the task (the event loop only holds weak
    # references, so an unreferenced task may be garbage collected mid-run) and
    # retrieve its result so unexpected failures are logged instead of surfacing
    # as "Task exception was never retrieved".
    background_tasks: set[asyncio.Task] | None = getattr(
        fastapi_request.app.state, "background_tasks", None
    )
    if background_tasks is None:
        background_tasks = set()
        fastapi_request.app.state.background_tasks = background_tasks
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    task.add_done_callback(_log_background_task_failure)

    return JSONResponse(status_code=200, content=in_progress_body)


async def _background_input_items(
    request: Any, store: Any, api_key_name: str | None
) -> list[dict[str, Any]] | None:
    """Materialize the input items to store with a background response.

    Mirrors the WebSocket transport's continuation logic: the previous
    response's input + output are prepended to the new input, so a follow-up
    ``previous_response_id`` referencing this response replays the full
    conversation. Falls back to the raw new input when the previous response
    cannot be resolved.
    """
    try:
        raw_input = request.model_dump(exclude_none=True).get("input")
    except Exception:
        raw_input = getattr(request, "input", None)
    if raw_input is None:
        return None
    if isinstance(raw_input, str):
        raw_items: list[dict[str, Any]] = [
            {"type": "message", "role": "user", "content": raw_input}
        ]
    elif isinstance(raw_input, list):
        raw_items = list(raw_input)
    else:
        raw_items = [raw_input]

    prev_id = getattr(request, "previous_response_id", None)
    if prev_id and store is not None and api_key_name:
        try:
            prev = await store.retrieve(api_key_name, prev_id)
        except Exception:
            prev = None
        if prev is not None:
            prev_input = prev.get("input") or []
            if isinstance(prev_input, str):
                prev_input = [{"type": "message", "role": "user", "content": prev_input}]
            elif not isinstance(prev_input, list):
                prev_input = []
            return [*(prev_input or []), *(prev.get("output") or []), *raw_items]
    return raw_items


def _log_background_task_failure(task: asyncio.Task) -> None:
    """Log unexpected background-task failures and retrieve their result."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(f"Background OpenResponses task failed: {exc}", exc_info=exc)


def create_protocol_router(
    endpoint: ProtocolEndpoint,
    *,
    include_docs_endpoint: bool = False,
) -> APIRouter:
    """Create a FastAPI router for a protocol endpoint.

    This function creates a router with endpoints that preserve OpenAPI
    documentation by using the protocol's specific request/response models.

    Args:
        endpoint: The protocol endpoint configuration
        include_docs_endpoint: Whether to include a protocol info endpoint

    Returns:
        Configured APIRouter instance
    """
    paths = endpoint.paths
    path = paths[0] if paths else ""
    tags = endpoint.tags
    response_model = endpoint.response_model
    middleware = endpoint.middleware

    request_model = endpoint.request_model

    endpoint_fn = _create_endpoint_fn(endpoint, middleware)

    # create_standard_router wraps the handler with create_traced_handler
    # itself; pre-wrapping here would double-wrap request_model=None endpoints
    # (the outer wrapper calls handler_func(None, request), which the inner
    # single-argument wrapper cannot accept).
    router = create_standard_router(
        request_name=f"{endpoint.name}_endpoint",
        request_model=request_model,
        path=path,
        handler=endpoint_fn,
        response_model=response_model,
        tags=tags,
        dependencies=[Depends(require_any_auth)],
    )

    # Register path aliases (paths[1:]) on the same handler: clients with a
    # misconfigured base_url that omits or double-writes /v1 still reach the
    # endpoint. Wrap endpoint_fn once (create_standard_router wraps its own
    # handler; aliases need the same single wrap).
    for alias_path in paths[1:]:
        alias_handler = create_traced_handler(
            f"http_request:{endpoint.name}:alias", endpoint_fn, request_model
        )
        alias_kwargs: dict[str, Any] = {
            "path": alias_path,
            "name": f"{endpoint.name}_endpoint_alias_{alias_path.replace('/', '_')}",
        }
        if response_model is not None:
            alias_kwargs["response_model"] = response_model
        router.post(**alias_kwargs)(alias_handler)
        logger.debug(f"Added path alias '{alias_path}' to protocol '{endpoint.name}'")

    for (
        route_path,
        route_request_model,
        route_response_model,
        route_handler,
    ) in endpoint.additional_routes:
        route_trace_name = f"http_request:{endpoint.name}:{route_path}"
        traced_route_handler = create_traced_handler(
            route_trace_name, route_handler, route_request_model
        )

        endpoint_kwargs: dict[str, Any] = {
            "path": route_path,
            "name": f"{endpoint.name}_{route_path.replace('/', '_')}",
        }
        if route_response_model is not None:
            endpoint_kwargs["response_model"] = route_response_model

        router.post(**endpoint_kwargs)(traced_route_handler)
        logger.debug(f"Added additional route '{route_path}' to protocol '{endpoint.name}'")

    if include_docs_endpoint:
        _add_protocol_info_endpoint(router, endpoint)

    return router


def _add_protocol_info_endpoint(router: APIRouter, endpoint: ProtocolEndpoint) -> None:
    """Add an endpoint that returns protocol information."""
    path = endpoint.paths[0] if endpoint.paths else ""

    info_data = {
        "name": endpoint.name,
        "path": path,
        "description": endpoint.description,
        "request_model": (endpoint.request_model.__name__ if endpoint.request_model else None),
    }

    add_info_endpoint(router, "/info/", endpoint.name, info_data, ["protocols"])


def create_protocol_list_router() -> APIRouter:
    """Create a router with an endpoint listing all registered protocols.

    Returns:
        APIRouter with a /protocols endpoint
    """

    async def list_all_protocols() -> dict[str, Any]:
        """Return information about all registered protocols."""
        return {"protocols": get_protocols_info()}

    router = APIRouter(prefix="/v1", tags=["protocols"])
    router.get("/protocols")(list_all_protocols)

    return router


def create_all_protocol_routers(
    include_docs_endpoint: bool = False,
) -> list[APIRouter]:
    """Create routers for all registered protocols.

    This function iterates through all registered protocols and creates
    routers for each one.

    Args:
        include_docs_endpoint: Whether to include protocol info endpoints

    Returns:
        List of APIRouter instances, one per registered protocol
    """
    from llm_proxy.protocols.registry import get_protocol

    routers = []

    for protocol_info in get_protocols_info():
        protocol_name = protocol_info["name"]
        endpoint = get_protocol(protocol_name)

        if endpoint is None:
            logger.warning(f"Could not get protocol endpoint for '{protocol_name}'")
            continue

        router = create_protocol_router(endpoint, include_docs_endpoint=include_docs_endpoint)
        routers.append(router)

    logger.debug(f"Created {len(routers)} protocol routers")
    return routers


def import_registered_protocol_modules(
    base_module: str = "llm_proxy.protocols",
) -> None:
    """Import all protocol modules from the base package.

    This triggers the registration of all protocols that use the
    @register_protocol decorator in their __init__.py files.

    Args:
        base_module: The base module name for protocol packages
    """
    import importlib
    import pkgutil

    try:
        package = importlib.import_module(base_module)
        package_path = package.__path__

        skip_paths = {"registry", "base_handler", "__pycache__", "base", "streaming"}

        for _, module_name, _ in pkgutil.iter_modules(package_path):
            if module_name in skip_paths:
                continue

            try:
                importlib.import_module(f"{base_module}.{module_name}")
            except ImportError as e:
                logger.warning(f"Failed to import protocol module '{module_name}': {e}")
            except Exception as e:
                logger.error(
                    f"Error importing protocol module '{module_name}': {e}",
                    exc_info=e,
                )

    except ImportError as e:
        logger.error(f"Failed to import base protocol module '{base_module}': {e}")


__all__ = [
    "create_protocol_router",
    "create_protocol_list_router",
    "create_all_protocol_routers",
    "import_registered_protocol_modules",
]
