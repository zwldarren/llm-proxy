"""Pipeline stage: resolve openresponses previous_response_id after overrides."""

from typing import Any

from llm_proxy.core import reasoning_cache
from llm_proxy.core.identity import get_request_identity
from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.stages.base import PipelineStage, PipelineState
from llm_proxy.models.content_blocks import ThinkingBlock, ToolUseBlock
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


def _repair_encrypted_blocks(state: PipelineState) -> None:
    """Patch ThinkingBlocks using call_id → reasoning cache.

    Handles two cases per assistant message that carries tool calls:
    1. An existing ThinkingBlock with empty thinking → restore reasoning from
       cache keyed by an adjacent ToolUseBlock id.
    2. No ThinkingBlock at all → insert one (with cached reasoning) before the
       first tool call, so the chat-completions request carries reasoning
       instead of the "tool call" placeholder. The Responses→Chat serializer
       can split a single assistant turn's reasoning and function_call into
       separate messages, so the ThinkingBlock may live in a different message
       than the ToolUseBlock it belongs to.
    """
    # Guard against requests without conversation (e.g., image generation, embeddings)
    conversation = getattr(state.unified_request, "conversation", None)
    if conversation is None or not hasattr(conversation, "messages"):
        return

    restored = 0
    inserted = 0
    for msg in conversation.messages:
        if msg.role != "assistant":
            continue
        tool_blocks = [b for b in msg.content if isinstance(b, ToolUseBlock)]
        if not tool_blocks:
            continue

        # Case 1: existing ThinkingBlock with empty thinking → restore.
        for i, block in enumerate(msg.content):
            if isinstance(block, ThinkingBlock) and not block.thinking:
                next_tool = next(
                    (b for b in msg.content[i + 1 :] if isinstance(b, ToolUseBlock)),
                    None,
                )
                if next_tool:
                    reasoning = reasoning_cache.get(next_tool.id)
                    if reasoning:
                        block.thinking = reasoning
                        restored += 1
                        logger.debug(
                            "Repair: restored reasoning for call_id="
                            f"{next_tool.id} ({len(reasoning)} chars)"
                        )
                    else:
                        logger.debug(
                            "Repair: no cached reasoning for call_id="
                            f"{next_tool.id}, will use placeholder"
                        )

        # Case 2: message has tool calls but no ThinkingBlock at all → insert.
        has_thinking = any(isinstance(b, ThinkingBlock) and b.thinking for b in msg.content)
        if not has_thinking:
            for tool in tool_blocks:
                reasoning = reasoning_cache.get(tool.id)
                if reasoning:
                    insert_idx = min(
                        idx for idx, b in enumerate(msg.content) if isinstance(b, ToolUseBlock)
                    )
                    msg.content.insert(
                        insert_idx,
                        ThinkingBlock(thinking=reasoning),
                    )
                    inserted += 1
                    logger.debug(
                        "Repair: inserted reasoning ThinkingBlock for call_id="
                        f"{tool.id} ({len(reasoning)} chars)"
                    )
                    break
    if restored or inserted:
        logger.debug(f"Repair: {restored} restored, {inserted} inserted from cache")
    else:
        logger.debug("Repair: no reasoning restored/inserted from cache")


class PreviousResponseResolutionStage(PipelineStage):
    """Prepend stored previous response items and repair encrypted reasoning.

    Primary: previous_response_id → Redis ResponseStore → prepend + repair.
    Fallback: in-memory reasoning cache (populated by streaming/serialization)
    with ambiguity detection to prevent stale reasoning after rewind/undo.
    """

    async def process(self, state: PipelineState, context: RequestContext) -> None:
        req = state.unified_request

        if not hasattr(req, "extra"):
            return

        # ── primary: previous_response_id → ResponseStore ──
        prev_id = req.extra.get("previous_response_id")
        if prev_id and context.response_store is None:
            # No local storage is configured, so the proxy cannot materialize
            # the referenced response. Forwarding the id only makes sense for a
            # native Responses upstream (which may hold it server-side); for
            # every other provider the chat request builder would silently
            # strip the id and the client would lose the prior context without
            # any error. Per the OpenResponses spec, fail the turn loudly with
            # previous_response_not_found instead.
            if _is_native_responses_upstream(state.adapter):
                logger.info(
                    f"previous_response_id '{prev_id}' not resolvable locally "
                    "(response storage disabled); forwarding to native Responses upstream"
                )
                return
            from llm_proxy.core.exceptions import NotFoundError

            raise NotFoundError(
                message=f"Previous response with id '{prev_id}' not found.",
                code="previous_response_not_found",
                status_code=400,
            )
        if prev_id and context.response_store is not None:
            identity = get_request_identity(state.req) if state.req is not None else None
            api_key_name = getattr(identity, "api_key_name", None) if identity else None
            if not api_key_name:
                logger.warning("previous_response_id requires API key; skipping")
                return
            try:
                prev_response = await context.response_store.retrieve(api_key_name, prev_id)
            except Exception:
                logger.warning(f"Failed to retrieve {prev_id}", exc_info=True)
                return
            if prev_response is None:
                # Native upstream fallback: when the selected provider speaks
                # the Responses API natively, it may hold the referenced
                # response server-side (created outside this proxy, streamed
                # before persistence existed, or stored upstream-only).
                # Forward the id instead of failing.
                if _is_native_responses_upstream(state.adapter):
                    logger.info(
                        f"previous_response_id '{prev_id}' not in local store; "
                        "forwarding to native Responses upstream"
                    )
                    return
                # Spec: when the referenced response is not available, the
                # server MUST fail the turn with previous_response_not_found
                # (instead of silently continuing without the prior context).
                # OpenAI returns HTTP 400 for this code (matching the WebSocket
                # transport's error envelope status).
                from llm_proxy.core.exceptions import NotFoundError

                raise NotFoundError(
                    message=f"Previous response with id '{prev_id}' not found.",
                    code="previous_response_not_found",
                    status_code=400,
                )
            from llm_proxy.protocols.openresponses import replay_stored_response

            # Replay the stored response's items into the conversation and
            # splice any item_reference targets that point into the stored turn.
            unresolved = req._unresolved_item_references
            replay_stored_response(prev_response, req.conversation, unresolved)
            # The raw protocol body still carries the proxy-local
            # previous_response_id (popped from ``extra`` below) and none of
            # the materialized items; disable native request passthrough so
            # the rebuilt body is what reaches the upstream (the flag is read
            # by BaseAdapter.allows_native_request).
            state.unified_request.previous_response_materialized = True
            # A native Responses upstream (OpenAI) rebuilds a Responses-shaped
            # body via its serializer, so its stream may stay native even with
            # a rebuilt request body. Any other upstream (e.g. DeepSeek, whose
            # rebuilt body is Chat Completions-shaped) cannot consume a native
            # Responses stream with a rebuilt body — disable native handling on
            # both sides so the whole request falls back to translation.
            if not _is_native_responses_upstream(state.adapter):
                state.unified_request.native_request_disabled = True
            # The prior context is now materialized in the conversation; drop
            # the id so a native Responses provider does not load it again and
            # double-apply the previous turn.
            req.extra.pop("previous_response_id", None)
            # Populate cache from stored response output
            _populate_cache_from_response(prev_response)
            _repair_encrypted_blocks(state)
            return

        # ── fallback: in-memory cache (populated by streaming/serialization) ──
        _repair_encrypted_blocks(state)


def _populate_cache_from_response(prev_response: dict) -> None:
    """Extract reasoning→call_id pairs from a stored response and fill cache."""
    response_id = prev_response.get("id", "")
    if not response_id:
        return
    pending: str | None = None
    for item in prev_response.get("output", []):
        item_type = item.get("type")
        if item_type == "reasoning":
            for part in item.get("content", []):
                if isinstance(part, dict) and part.get("type") in (
                    "reasoning_text",
                    "output_text",
                    "summary_text",
                ):
                    text = part.get("text", "")
                    if text:
                        pending = text
        elif item_type == "function_call" and pending:
            call_id = item.get("call_id", "")
            if call_id:
                reasoning_cache.store(response_id, call_id, pending)
            pending = None
        else:
            pending = None


def _is_native_responses_upstream(adapter: Any) -> bool:
    """Whether the selected adapter talks to a native OpenAI Responses API endpoint."""
    target = getattr(adapter, "_target_endpoint", None)
    if not callable(target):
        return False
    try:
        return target() == "responses"
    except Exception:
        return False
