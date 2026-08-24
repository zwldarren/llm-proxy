"""FastAPI router for OpenResponses protocol endpoints."""

import logging
import time
from contextlib import suppress
from typing import Annotated, Any, Literal, cast

import orjson
from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.requests import Request as StarletteRequest

from llm_proxy.api.context import HasModel, build_request_context
from llm_proxy.api.dependencies import get_request_identity, require_api_key_auth
from llm_proxy.core.exceptions import AuthenticationFailedError, ConfigurationError, NotFoundError
from llm_proxy.core.ws_common import (
    WS_CLOSE_AUTH_FAILED,
    WS_MAX_CONNECTION_SECONDS,
    WS_MAX_MESSAGE_BYTES,
    WebSocketConnectionLimitError,
    authenticate_ws,
    build_ws_request,
    receive_with_connection_cap,
)
from llm_proxy.protocols.openresponses.compaction import build_compaction_response
from llm_proxy.protocols.openresponses.schemas import (
    ReasoningParam,
    ResponsesResponse,
    TextFormatParam,
    TextParam,
    ToolParam,
)
from llm_proxy.protocols.openresponses.store import ResponseStore
from llm_proxy.providers.openai.client_headers import capture_client_headers

logger = logging.getLogger(__name__)


async def require_any_auth(request: Request):
    """Require either API key or admin JWT authentication."""
    identity = get_request_identity(request)
    if not identity.is_authenticated:
        await require_api_key_auth(request)


router = APIRouter(tags=["responses"], dependencies=[Depends(require_any_auth)])

# WebSocket transport router: no router-level HTTP auth dependency (the
# WebSocket endpoint authenticates the connection itself after accept).
ws_router = APIRouter(tags=["responses"])


async def require_api_key_name(request: Request) -> str:
    """Require API key authentication and return the authenticated key name."""
    identity = get_request_identity(request)
    api_key_name = identity.api_key_name
    if not api_key_name:
        raise AuthenticationFailedError(message="Response storage requires API key authentication.")
    return api_key_name


ApiKeyNameDep = Annotated[str, Depends(require_api_key_name)]


def get_response_store_required(request: Request) -> ResponseStore:
    """Dependency to get ResponseStore instance from app state (required).

    Args:
        request: The FastAPI request object

    Returns:
        ResponseStore instance with Redis client from app state

    Raises:
        HTTPException: 503 if Redis is not available
    """
    redis_client_wrapper = getattr(request.app.state, "redis_client", None)
    if redis_client_wrapper is None or redis_client_wrapper.client is None:
        raise ConfigurationError(
            message="Response storage is not available. Redis must be enabled.",
            code="redis_not_available",
            status_code=503,
        )
    return ResponseStore(redis_client=redis_client_wrapper.client)


ResponseStoreRequiredDep = Annotated[ResponseStore, Depends(get_response_store_required)]


class CompactionRequest(BaseModel):
    """Request body for POST /v1/responses/compact.

    Accepts the OpenResponses spec fields plus the Codex client extras
    (tools / reasoning / text / service_tier / parallel_tool_calls / cache
    knobs) so real-world clients such as Codex are not rejected. Compaction is
    performed locally by the proxy, so the extras are accepted but not used.
    """

    model: str = Field(..., description="Model ID used to generate the response")
    input: str | list[Any] = Field(
        ..., description="Text, image, or file inputs to the model, used to generate a response"
    )
    previous_response_id: str | None = Field(
        None, description="The unique ID of the previous response to the model"
    )
    instructions: str | None = Field(
        None, description="A system (or developer) message inserted into the model's context"
    )
    prompt_cache_key: str | None = Field(
        None, description="A key to use when reading from or writing to the prompt cache"
    )
    # ── Codex client parity ──
    tools: list[ToolParam] | None = Field(None, description="Tools available to the model")
    parallel_tool_calls: bool | None = Field(
        None, description="Whether the model may call multiple tools in parallel"
    )
    reasoning: ReasoningParam | None = Field(None, description="Reasoning configuration")
    service_tier: Literal["auto", "default", "flex", "priority"] | None = Field(
        None, description="Service tier"
    )
    prompt_cache_options: dict[str, Any] | None = Field(
        None, description="Prompt cache options (opaque, accepted for compatibility)"
    )
    prompt_cache_retention: Literal["in_memory", "24h"] | None = Field(
        None, description="Prompt cache retention policy"
    )
    text: TextFormatParam | TextParam | None = Field(None, description="Text output configuration")


async def _try_native_compact_passthrough(request: Request) -> JSONResponse | None:
    """Forward a compact request to a native Responses upstream when possible.

    When the selected provider speaks the Responses API natively, the
    upstream performs real model-driven compaction, so the raw
    request body is forwarded verbatim and the upstream response (status +
    body) is returned as-is. Returns None for non-native providers (or when
    the passthrough attempt itself fails), letting the caller fall back to
    the proxy's local lossless packing.
    """
    from llm_proxy.core.processing.stages.previous_response import (
        _is_native_responses_upstream,
    )

    try:
        raw_body = orjson.loads(await request.body())
    except orjson.JSONDecodeError:
        return None
    if not isinstance(raw_body, dict) or not raw_body.get("model"):
        return None
    # A previous_response_id that the proxy holds in its own store may be
    # unknown to the upstream (store=false request, upstream TTL expiry, or a
    # response created before the proxy). Forwarding it would make the
    # upstream answer 400 previous_response_not_found; the local compaction
    # below can materialize it from the store, so skip the passthrough.
    prev_id = raw_body.get("previous_response_id")
    api_key_name = getattr(request.state, "api_key_name", None)
    if prev_id and api_key_name:
        redis_wrapper = getattr(request.app.state, "redis_client", None)
        if redis_wrapper is not None and redis_wrapper.client is not None:
            from llm_proxy.protocols.openresponses.store import ResponseStore

            store = ResponseStore(redis_client=redis_wrapper.client)
            try:
                if await store.retrieve(api_key_name, prev_id) is not None:
                    return None
            except Exception:
                logger.debug("Failed to check local store for compact passthrough")
    # Capture Codex client headers so the adapter can forward them upstream
    # (the compact endpoint bypasses the protocol middleware that normally
    # does this).
    capture_client_headers(request.headers)
    try:
        context = await build_request_context(
            cast(HasModel, raw_body), request, protocol_name="openresponses"
        )
        selection = context.orchestrator.select_next_provider()
        if selection is None:
            return None
        adapter = await context.adapter_factory(request, selection)
        async with adapter:
            if not _is_native_responses_upstream(adapter):
                return None
            compact = getattr(adapter, "compact_response", None)
            if compact is None:
                return None
            status, payload = await compact(raw_body)
        # Upstream 4xx (e.g. previous_response_not_found for an id the proxy
        # does not hold either): the local lossless packing is a strictly
        # better answer than surfacing the upstream error, so fall back to it.
        if status >= 400:
            logger.warning(
                "Native compact passthrough returned %s; falling back to local compaction",
                status,
            )
            return None
        return JSONResponse(status_code=status, content=payload)
    except Exception:
        logger.warning(
            "Native compact passthrough failed; falling back to local compaction",
            exc_info=True,
        )
        return None


@router.post("/v1/responses/compact")
async def compact_response(
    body: CompactionRequest,
    request: Request,
    api_key_name: ApiKeyNameDep,
) -> Any:
    """Compact a conversation into a reusable compaction item.

    Returns a ``response.compaction`` object whose output carries a single
    ``compaction`` item with an opaque ``encrypted_content`` blob. Sending that
    item back as input on a follow-up request rehydrates the conversation.
    """
    # Native upstream passthrough first (real model-driven compaction);
    # the local packing below remains the fallback for non-native providers.
    passthrough = await _try_native_compact_passthrough(request)
    if passthrough is not None:
        return passthrough

    items: list[dict[str, Any]] = []

    if body.previous_response_id:
        redis_client_wrapper = getattr(request.app.state, "redis_client", None)
        if redis_client_wrapper is None or redis_client_wrapper.client is None:
            raise ConfigurationError(
                message="Response storage is not available. Redis must be enabled.",
                code="redis_not_available",
                status_code=503,
            )
        store = ResponseStore(redis_client=redis_client_wrapper.client)
        prev = await store.retrieve(api_key_name, body.previous_response_id)
        if prev is None:
            # OpenAI parity: previous_response_not_found is an HTTP 400 error
            # (matching the WebSocket transport's error envelope status).
            raise NotFoundError(
                message=f"Previous response with id '{body.previous_response_id}' not found.",
                code="previous_response_not_found",
                status_code=400,
            )
        items.extend(_as_input_items(prev.get("input")))
        items.extend(prev.get("output") or [])

    items.extend(_as_input_items(body.input))

    return build_compaction_response(model=body.model, items=items)


# Path aliases: clients whose base_url omits /v1 or double-writes it
# still reach the compact endpoint.
router.post("/responses/compact")(compact_response)
router.post("/v1/v1/responses/compact")(compact_response)


@router.get("/v1/responses/{response_id}")
async def get_response(
    response_id: str,
    response_store: ResponseStoreRequiredDep,
    api_key_name: ApiKeyNameDep,
) -> ResponsesResponse:
    """Retrieve a stored response by ID.

    Args:
        response_id: The response ID
        response_store: Response storage backend
        api_key_name: Name of the API key that owns this response

    Returns:
        The stored response

    Raises:
        HTTPException: 404 if not found
    """
    stored = await response_store.retrieve(api_key_name, response_id)
    if stored is None:
        raise NotFoundError(
            message=f"Response '{response_id}' not found or expired. "
            "Stored responses expire after 24 hours."
        )
    # The stored body carries the materialized ``input`` so follow-up
    # previous_response_id continuations (and /v1/responses/compact) can
    # replay the conversation. That is internal proxy state — the spec's
    # ResponseResource has no ``input`` field — so strip it before returning.
    stored = {k: v for k, v in stored.items() if k != "input"}
    return ResponsesResponse(**stored)


@router.delete("/v1/responses/{response_id}")
async def delete_response(
    response_id: str,
    response_store: ResponseStoreRequiredDep,
    api_key_name: ApiKeyNameDep,
) -> dict:
    """Delete a stored response by ID.

    Args:
        response_id: The response ID
        response_store: Response storage backend
        api_key_name: Name of the API key that owns this response

    Returns:
        Deletion confirmation
    """
    deleted = await response_store.delete(api_key_name, response_id)
    if not deleted:
        # OpenAI parity: deleting an unknown/expired response id is a 404,
        # not a 200 with deleted=false.
        raise NotFoundError(
            message=f"Response '{response_id}' not found or expired. "
            "Stored responses expire after 24 hours."
        )
    return {
        "id": response_id,
        "deleted": True,
        "object": "response.deleted",
    }


def _as_input_items(value: Any) -> list[Any]:
    """Normalize a raw ``input`` value (string or item list) into an item list.

    A plain-string input must be wrapped, not extended — extending a string
    would shred it into individual characters.
    """
    if isinstance(value, str):
        return [{"type": "message", "role": "user", "content": value}]
    if isinstance(value, list):
        return value
    return []


# =============================================================================
# WebSocket transport (spec 2026-04-24)
# =============================================================================


def _ws_error_envelope(
    status: int, code: str, message: str, param: str | None = None
) -> dict[str, Any]:
    """Build the WebSocket error envelope per the spec."""
    error: dict[str, Any] = {"code": code, "message": message}
    if param is not None:
        error["param"] = param
    return {"type": "error", "status": status, "error": error}


async def _ws_send_error(
    websocket: WebSocket, status: int, code: str, message: str, param: str | None = None
) -> None:
    """Send a WebSocket error envelope, swallowing send failures."""
    with suppress(Exception):
        await websocket.send_json(_ws_error_envelope(status, code, message, param))


def _parse_sse_blocks(buffer: str) -> tuple[list[dict[str, Any]], str]:
    """Parse complete SSE blocks from a buffer, returning (events, remainder).

    Handles both ``event: x\ndata: {...}`` and ``data: {...}`` blocks. The
    ``[DONE]`` terminal marker is an SSE transport artifact and is skipped on
    the WebSocket transport. Blocks may arrive split across multiple chunks, so
    the trailing partial block is kept as the remainder for the next chunk.
    """
    if "\n\n" not in buffer:
        return [], buffer
    blocks = buffer.split("\n\n")
    complete, remainder = blocks[:-1], blocks[-1]
    events: list[dict[str, Any]] = []
    for block in complete:
        data_line = None
        for line in block.splitlines():
            if line.startswith("data: "):
                data_line = line[6:]
                break
        if data_line is None or data_line == "[DONE]":
            continue
        try:
            parsed = orjson.loads(data_line)
        except orjson.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events, remainder


def _exception_status_code(exc: Exception) -> int:
    return getattr(exc, "status_code", None) or 500


def _exception_error_code(exc: Exception) -> str:
    return getattr(exc, "code", None) or getattr(exc, "error_type", None) or "server_error"


def _validate_continuation_call_ids(items: list[dict[str, Any]]) -> str | None:
    """Validate function_call_output items against the materialized context.

    On a continuation turn, every ``function_call_output`` must reference a
    ``function_call`` present in the effective context (previous input +
    previous output + new input). Returns the offending ``call_id`` when a
    dangling tool output is found, else None.
    """
    known_call_ids: set[str] = set()
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id")
            if call_id:
                known_call_ids.add(call_id)
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        call_id = item.get("call_id")
        if call_id and call_id not in known_call_ids:
            return call_id
    return None


async def _ws_resolve_previous_response(
    websocket: WebSocket,
    body: dict[str, Any],
    local_state: dict[str, dict[str, Any]],
    api_key_name: str,
) -> bool:
    """Resolve previous_response_id against connection-local state, then Redis.

    On success, prepends the prior input+output to ``body["input"]`` and drops
    ``previous_response_id`` (the context is materialized). On failure, sends
    the ``previous_response_not_found`` error and evicts the id from the
    connection-local cache (spec: failed continuations evict).

    Returns True when the turn may proceed.
    """
    prev_id = body.get("previous_response_id")
    if not prev_id:
        return True

    prev = local_state.get(prev_id)
    if prev is None:
        # store=true: MAY hydrate older response IDs from persisted state.
        redis_wrapper = getattr(websocket.app.state, "redis_client", None)
        if redis_wrapper is not None and redis_wrapper.client is not None:
            store = ResponseStore(redis_client=redis_wrapper.client)
            try:
                prev = await store.retrieve(api_key_name, prev_id)
            except Exception:
                prev = None

    if prev is None:
        local_state.pop(prev_id, None)
        await _ws_send_error(
            websocket,
            400,
            "previous_response_not_found",
            f"Previous response with id '{prev_id}' not found.",
            param="previous_response_id",
        )
        return False

    input_items = _as_input_items(body.get("input"))
    prev_input = _as_input_items(prev.get("input"))
    body["input"] = [*prev_input, *(prev.get("output") or []), *input_items]
    body.pop("previous_response_id", None)

    # Spec: continuation turns that fail must evict the referenced id from the
    # connection-local cache. A dangling function_call_output (no matching
    # function_call in the materialized context) is such a failure: reject it
    # with an invalid_request error before the model is called.
    dangling = _validate_continuation_call_ids(body["input"])
    if dangling is not None:
        local_state.pop(prev_id, None)
        await _ws_send_error(
            websocket,
            400,
            "invalid_request",
            f"function_call_output references unknown call_id '{dangling}'. "
            "No matching function_call exists in the previous response.",
            param="call_id",
        )
        return False
    return True


async def _ws_handle_response_create(
    websocket: WebSocket,
    request: StarletteRequest,
    processor: Any,
    msg: dict[str, Any],
    local_state: dict[str, dict[str, Any]],
    api_key_name: str,
) -> None:
    """Process a single response.create turn over the WebSocket."""
    # HTTP/SSE transport-specific fields MUST NOT be sent on WebSocket requests.
    body = {
        k: v for k, v in msg.items() if k not in ("type", "stream", "stream_options", "background")
    }
    body["stream"] = True

    # Spec: a continuation turn that fails with a 4xx/5xx error MUST evict the
    # referenced previous_response_id from the connection-local cache.
    referenced_previous_id = body.get("previous_response_id")

    if not await _ws_resolve_previous_response(websocket, body, local_state, api_key_name):
        return

    try:
        context = await build_request_context(
            cast(HasModel, body), request, protocol_name="openresponses"
        )
        response = await processor.process(protocol_request=body, req=request, context=context)
    except Exception as exc:
        if referenced_previous_id:
            local_state.pop(referenced_previous_id, None)
        await _ws_send_error(
            websocket,
            _exception_status_code(exc),
            _exception_error_code(exc),
            str(exc),
        )
        return

    response_id: str | None = None
    final_output: list[dict[str, Any]] | None = None
    sse_buffer = ""
    try:
        async for chunk in response.body_iterator:
            if not isinstance(chunk, str):
                continue
            sse_buffer += chunk
            events, sse_buffer = _parse_sse_blocks(sse_buffer)
            for event in events:
                event_type = event.get("type")
                if event_type == "response.created":
                    response_id = (event.get("response") or {}).get("id")
                elif event_type == "response.completed":
                    final_output = (event.get("response") or {}).get("output")
                elif event_type == "response.failed" and referenced_previous_id:
                    # A failed continuation turn also evicts the referenced
                    # previous_response_id from the connection-local cache.
                    local_state.pop(referenced_previous_id, None)
                await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.warning("WebSocket stream error: %s", exc)
        await _ws_send_error(websocket, 500, "server_error", str(exc))
        return

    # Keep the most recent previous-response state in connection-local memory
    # so store=false continuations work on the same socket.
    if response_id and final_output is not None:
        local_state[response_id] = {
            "input": body.get("input"),
            "output": final_output,
        }


@ws_router.websocket("/v1/responses")
async def responses_websocket(websocket: WebSocket) -> None:
    """OpenResponses WebSocket transport.

    Clients start each turn with ``{"type": "response.create", ...}``. Progress
    is sent using the same streaming event objects as text/event-stream
    responses; failures use the error envelope. At most one in-flight response
    is processed at a time; messages are handled sequentially.
    """
    await websocket.accept()

    auth = await authenticate_ws(websocket)
    if auth is None:
        await _ws_send_error(websocket, 401, "authentication_failed", "Authentication required.")
        await websocket.close(code=WS_CLOSE_AUTH_FAILED)
        return
    identity, _ = auth

    # Build a Request-like object from the websocket scope so the standard
    # context builder and pipeline stages work unchanged.
    request = build_ws_request(websocket, identity)

    processor = getattr(websocket.app.state, "openresponses_processor", None)
    if processor is None:
        await _ws_send_error(
            websocket,
            503,
            "server_error",
            "openresponses_processor not initialized. Ensure lifespan is configured.",
        )
        await websocket.close()
        return

    # Connection-local previous-response state (spec: keeps store=false
    # continuations working on the same socket without persisted storage).
    local_state: dict[str, dict[str, Any]] = {}
    connected_at = time.monotonic()

    try:
        while True:
            try:
                raw = await receive_with_connection_cap(
                    websocket.receive_text,
                    connected_at=connected_at,
                    max_seconds=WS_MAX_CONNECTION_SECONDS,
                )
            except WebSocketConnectionLimitError:
                await _ws_send_error(
                    websocket,
                    400,
                    "websocket_connection_limit_reached",
                    "WebSocket connection limit of 60 minutes reached.",
                )
                break
            # NOTE: ASGI delivers each WebSocket message fully assembled, so
            # the size check runs after the message is already in memory. The
            # limit guards against accidental oversized turns, not against a
            # deliberate memory-exhaustion attack — operators needing the
            # latter should also enforce a body/message cap at the edge.
            if len(raw.encode("utf-8")) > WS_MAX_MESSAGE_BYTES:
                await _ws_send_error(
                    websocket,
                    400,
                    "invalid_request",
                    "WebSocket message exceeds the 64 MiB size limit.",
                )
                continue
            try:
                msg = orjson.loads(raw)
            except orjson.JSONDecodeError:
                await _ws_send_error(websocket, 400, "invalid_request", "Invalid JSON message.")
                continue
            if not isinstance(msg, dict) or msg.get("type") != "response.create":
                await _ws_send_error(
                    websocket,
                    400,
                    "invalid_request",
                    "Client message must have type 'response.create'.",
                )
                continue

            await _ws_handle_response_create(
                websocket,
                request,
                processor,
                msg,
                local_state,
                identity.api_key_name or "",
            )
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
        await _ws_send_error(websocket, 500, "server_error", str(exc))
    finally:
        with suppress(Exception):
            await websocket.close()
