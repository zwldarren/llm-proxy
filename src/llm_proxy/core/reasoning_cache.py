"""In-memory reasoning cache keyed by function call_id, with response_id scoping.

Codex encrypts reasoning content with ``encrypted_content`` but preserves
``call_id`` on the adjacent ``function_call`` item.  By caching the plaintext
reasoning against the call_id it precedes in the response, we can restore it on
subsequent turns even after rewind / undo / message reordering.

Uses response_id-scoped storage with ambiguity detection: a call_id is only
returned if it appears in exactly one cached response, preventing stale
reasoning from being used after rewind/undo (cc-switch style).
"""

from collections import OrderedDict, defaultdict
from contextlib import suppress
from typing import Any

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

_MAX_ENTRIES = 4096
# response_id -> {call_id -> reasoning_text}
_cache: OrderedDict[str, dict[str, str]] = OrderedDict()
# call_id -> set of response_ids (for ambiguity detection)
_call_index: dict[str, set[str]] = defaultdict(set)


def store(response_id: str, call_id: str, reasoning_text: str) -> None:
    """Cache reasoning text against a function call_id, scoped by response_id.

    Call once per (reasoning, function_call) pair in the response output.
    """
    if not call_id or not reasoning_text or not response_id:
        return
    # Ensure response_id entry exists (LRU eviction at response level)
    if response_id not in _cache:
        if len(_cache) >= _MAX_ENTRIES:
            _cache.popitem(last=False)
        _cache[response_id] = {}
    _cache[response_id][call_id] = reasoning_text
    # Track call_id -> response_ids for ambiguity detection
    _call_index[call_id].add(response_id)


def get(call_id: str) -> str | None:
    """Return cached reasoning text for a call_id, only if unique.

    Returns None if the call_id appears in multiple cached responses
    (ambiguous) or is not found at all.  This prevents stale reasoning
    from being used after rewind/undo.
    """
    response_ids = _call_index.get(call_id)
    if not response_ids or len(response_ids) != 1:
        return None
    response_id = next(iter(response_ids))
    return _cache.get(response_id, {}).get(call_id)


def cache_reasoning_from_blocks(blocks: list[Any] | None, response_id: str) -> None:
    """Cache real reasoning text from internal response output blocks.

    Pairs each tool-use block with the nearest ``ThinkingBlock`` in the output
    (nearest by index — parsers may emit the tool call before or after the
    reasoning block) and stores it keyed by the tool call id. Subsequent turns
    can then restore the real reasoning via :func:`get` even when the client
    did not echo it — e.g. another provider served an intermediate turn, or
    the client stripped reasoning to save tokens.

    Placeholder reasoning injected for DeepSeek-style echo never appears in
    ``InternalResponse.output``, so only genuine model reasoning is cached.

    Args:
        blocks: The response's ``output`` content blocks.
        response_id: Response id used for ambiguity-scoped storage.
    """
    if not blocks or not response_id:
        return
    # Imported lazily: core.reasoning_cache is imported from serialization code
    # (converter, streaming paths), and content blocks live in the models layer
    # which must not be imported at module scope here.
    from llm_proxy.models.content_blocks import (
        CustomToolUseBlock,
        ThinkingBlock,
        ToolUseBlock,
    )

    thinking_idx = [
        i
        for i, b in enumerate(blocks)
        if isinstance(b, ThinkingBlock) and getattr(b, "thinking", None)
    ]
    if not thinking_idx:
        return
    for i, block in enumerate(blocks):
        if not isinstance(block, (ToolUseBlock, CustomToolUseBlock)):
            continue
        call_id = getattr(block, "id", None)
        if not call_id:
            continue
        nearest = min(thinking_idx, key=lambda ti: abs(ti - i))
        thinking = getattr(blocks[nearest], "thinking", None)
        if thinking:
            store(response_id, call_id, thinking)


def cache_reasoning_from_response(response: Any) -> None:
    """Cache real reasoning from a parsed ``InternalResponse``.

    Convenience wrapper over :func:`cache_reasoning_from_blocks` for adapter
    call sites that already hold a parsed response.
    """
    if response is None:
        return
    cache_reasoning_from_blocks(
        getattr(response, "output", None), getattr(response, "id", "") or ""
    )


def try_cache_reasoning_from_response(response: Any) -> None:
    """Best-effort variant of :func:`cache_reasoning_from_response`; never raises.

    See :func:`try_cache_reasoning_from_blocks` for the contract.
    """
    with suppress(Exception):
        cache_reasoning_from_response(response)


def _extract_responses_reasoning_text(item: dict[str, Any]) -> str:
    """Extract reasoning text from a Responses API ``reasoning`` output item."""
    for part in item.get("content", []):
        if isinstance(part, dict) and part.get("type") == "reasoning_text":
            text = part.get("text", "")
            if text:
                return text
    for part in item.get("summary", []):
        if isinstance(part, dict) and part.get("type") == "summary_text":
            text = part.get("text", "")
            if text:
                return text
    return ""


def cache_reasoning_from_responses_output(
    output: list[dict[str, Any]] | None,
    response_id: str,
    logger_prefix: str = "ReasoningCache",
) -> None:
    """Cache reasoning from raw Responses API output items (native paths).

    A model may emit the function_call before its reasoning (or carry
    reasoning in ``summary``), so each function_call is associated with the
    nearest reasoning text in either direction. Used by the native
    passthrough paths — verbatim bodies never become parsed blocks, so the
    cache is written straight from the wire shape.
    """
    if not response_id or not output:
        return

    last_reasoning: str | None = None
    pending_calls: list[str] = []
    cached_count = 0

    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            text = _extract_responses_reasoning_text(item)
            if text:
                last_reasoning = text
                for cid in pending_calls:
                    store(response_id, cid, last_reasoning)
                    cached_count += 1
                pending_calls = []
        elif item_type == "function_call":
            call_id = item.get("call_id", "")
            if call_id:
                if last_reasoning:
                    store(response_id, call_id, last_reasoning)
                    cached_count += 1
                else:
                    pending_calls.append(call_id)

    for cid in pending_calls:
        if last_reasoning:
            store(response_id, cid, last_reasoning)
            cached_count += 1

    if cached_count:
        logger.info(
            f"{logger_prefix}: cached {cached_count} reasoning item(s) "
            f"for response_id={response_id}",
            cached_count=cached_count,
            response_id=response_id,
        )
    else:
        # The common case: most responses carry no reasoning items, so this
        # is only useful when debugging reasoning restoration.
        logger.debug(
            f"{logger_prefix}: no reasoning items to cache for response_id={response_id} "
            f"(output has {len(output)} items)",
            output_item_count=len(output),
            response_id=response_id,
        )


def cache_reasoning_from_chat_completion_body(body: dict[str, Any]) -> None:
    """Cache reasoning from a raw Chat Completions response body.

    Wire-reuse responses never become parsed blocks, so the cache is written
    straight from the wire shape: the message's ``reasoning_content`` (after
    the response-side reasoning-field rename) pairs with its ``tool_calls``
    ids. Built as blocks so the pairing rules stay in one place
    (:func:`cache_reasoning_from_blocks`).
    """
    if not isinstance(body, dict):
        return
    response_id = body.get("id", "") or ""
    choices = body.get("choices")
    if not response_id or not isinstance(choices, list):
        return

    from llm_proxy.models.content_blocks import ThinkingBlock, ToolUseBlock

    blocks: list[Any] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            blocks.append(ThinkingBlock(thinking=reasoning))
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                call_id = tc.get("id")
                function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                if call_id:
                    blocks.append(ToolUseBlock(id=call_id, name=function.get("name", ""), input={}))

    cache_reasoning_from_blocks(blocks, response_id)


def try_cache_reasoning_from_blocks(blocks: list[Any] | None, response_id: str) -> None:
    """Best-effort variant of :func:`cache_reasoning_from_blocks`; never raises.

    The reasoning cache is an optimization — a write failure must never fail
    a request — so every raw/stream write site shares this contract.
    """
    with suppress(Exception):
        cache_reasoning_from_blocks(blocks, response_id)


def try_cache_reasoning_from_responses_output(
    output: list[dict[str, Any]] | None,
    response_id: str,
    logger_prefix: str = "ReasoningCache",
) -> None:
    """Best-effort variant of :func:`cache_reasoning_from_responses_output`.

    Never raises — see :func:`try_cache_reasoning_from_blocks`.
    """
    with suppress(Exception):
        cache_reasoning_from_responses_output(output, response_id, logger_prefix=logger_prefix)


def try_cache_reasoning_from_chat_completion_body(body: dict[str, Any]) -> None:
    """Best-effort variant of :func:`cache_reasoning_from_chat_completion_body`.

    Never raises — see :func:`try_cache_reasoning_from_blocks`.
    """
    with suppress(Exception):
        cache_reasoning_from_chat_completion_body(body)


def clear() -> None:
    """Clear all entries (tests)."""
    _cache.clear()
    _call_index.clear()
