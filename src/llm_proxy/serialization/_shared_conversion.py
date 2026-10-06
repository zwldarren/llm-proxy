"""Shared conversion logic: convert unsupported blocks to natively-supported ones."""

import base64
from typing import Any, cast

import orjson

from llm_proxy.models import ContentBlock


def try_convert_block(block: ContentBlock) -> ContentBlock | None:
    """Try to convert a block to a natively-supported type.

    Conversion is lossless and happens BEFORE applying unknown_fields_policy.
    If a block is converted, it will be processed by the provider's native
    handler instead of falling through to degradation/error logic.

    Returns:
        The converted block, or None if no conversion is possible.
    """
    from llm_proxy.models import (
        RedactedThinkingBlock,
        RefusalBlock,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
    )
    from llm_proxy.models.content_blocks.anthropic_builtin import (
        BashCodeExecutionToolResultBlock,
        CodeExecutionToolResultBlock,
        ContainerUploadBlock,
        MidConversationSystemBlock,
        SearchResultBlock,
        TextEditorCodeExecutionToolResultBlock,
        ToolSearchToolResultBlock,
        WebFetchToolResultBlock,
        WebSearchResultContentBlock,
        WebSearchToolResultBlock,
    )

    if isinstance(block, RefusalBlock):
        return TextBlock(text=block.refusal)

    if isinstance(block, RedactedThinkingBlock):
        return ThinkingBlock(thinking="[Redacted thinking]")

    # NOTE: We intentionally do NOT convert ServerToolUseBlock or CustomToolUseBlock
    # to ToolUseBlock here. Converting them would cause downstream models to emit
    # tool_calls for tools that may not be configured on that provider. Instead they
    # fall through to degradation, which keeps the name/input as human-readable text.

    if isinstance(block, MidConversationSystemBlock):
        text = _content_blocks_to_text(block.content)
        return TextBlock(text=text) if text else None

    if isinstance(block, SearchResultBlock):
        text = _content_blocks_to_text(block.content) if block.content else block.title or ""
        return TextBlock(text=text) if text else None

    if isinstance(block, WebSearchResultContentBlock):
        if block.title and block.url:
            return TextBlock(text=f"[{block.title}]({block.url})")
        return TextBlock(text=block.title or block.url or "")

    if isinstance(block, ContainerUploadBlock):
        text = block.content or block.filename or block.file_id
        if text:
            return TextBlock(text=text)
        return None

    # A client replaying a prior turn echoes its ``web_search_tool_result``
    # block verbatim. Providers without native web search can only take plain
    # tool-result text, so render the results instead of degrading each item to
    # a title placeholder (which would make the model re-search for facts it
    # already fetched).
    if isinstance(block, WebSearchToolResultBlock):
        decoded = _decode_web_search_results(block.content)
        if decoded is not None:
            return ToolResultBlock(
                tool_use_id=block.tool_use_id,
                content=decoded,
                is_error=block.is_error,
            )

    # Tool result variants: all structurally match ToolResultBlock.
    # Some variants carry raw server-tool payloads (a dict such as
    # ``web_fetch_result``/``web_search_tool_result_error``, or a list of
    # ``web_search_result`` dicts); those are JSON-encoded so the payload is
    # preserved instead of being dropped, keeping ToolResultBlock type-safe.
    if isinstance(
        block,
        (
            WebSearchToolResultBlock,
            WebFetchToolResultBlock,
            CodeExecutionToolResultBlock,
            BashCodeExecutionToolResultBlock,
            TextEditorCodeExecutionToolResultBlock,
            ToolSearchToolResultBlock,
        ),
    ):
        raw_content = block.content
        is_raw_payload = isinstance(raw_content, dict) or (
            isinstance(raw_content, list) and bool(raw_content) and isinstance(raw_content[0], dict)
        )
        content = orjson.dumps(raw_content).decode() if is_raw_payload else raw_content
        return ToolResultBlock(
            tool_use_id=block.tool_use_id,
            content=cast("str | list[Any]", content),
            is_error=block.is_error,
        )

    return None


def _decode_web_search_results(items: Any) -> str | None:
    """Decode replayed ``web_search_tool_result`` items into the proxy's result JSON.

    Mirrors ``WebSearchInterceptor.decode_search_results`` — the exact shape the
    proxy's own continuation sends upstream — so a replayed result reaches
    providers without native web search exactly like a fresh search does:
    ``{"results": [{"url", "title", "snippet"}]}``, with the snippet decoded
    from the base64 ``encrypted_content`` the interceptor stored.

    Returns None when the items are not per-result objects, so error payloads
    keep the generic verbatim handling.
    """
    from llm_proxy.models.content_blocks.anthropic_builtin import (
        WebSearchResultContentBlock,
    )

    if not isinstance(items, list) or not items:
        return None
    if not all(isinstance(item, WebSearchResultContentBlock) for item in items):
        return None

    results = [
        {
            "url": item.url,
            "title": item.title,
            "snippet": _decode_encrypted_content(item.encrypted_content),
        }
        for item in items
    ]
    return orjson.dumps({"results": results}).decode()


def _decode_encrypted_content(value: str | None) -> str:
    """Decode the proxy's base64 ``encrypted_content`` to its snippet text.

    Returns an empty string when the payload is absent, not valid base64, or not
    UTF-8 text (a real Anthropic encrypted payload), so a native result still
    renders without its snippet instead of failing.
    """
    if not value:
        return ""
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError:
        return ""
    try:
        return decoded.decode("utf-8").strip()
    except UnicodeDecodeError:
        return ""


def _content_blocks_to_text(blocks: list[Any] | None) -> str:
    """Extract plain text from a list of content blocks."""
    if not blocks:
        return ""

    from llm_proxy.models import (
        TextBlock,
        ToolResultBlock,
    )
    from llm_proxy.models.content_blocks.anthropic_builtin import (
        WebSearchResultContentBlock,
    )

    parts: list[str] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            parts.append(b.text)
        elif isinstance(b, ToolResultBlock):
            if isinstance(b.content, str):
                parts.append(b.content)
            elif isinstance(b.content, list):
                parts.append(_content_blocks_to_text(b.content))
        elif isinstance(b, WebSearchResultContentBlock):
            if b.title and b.url:
                parts.append(f"[{b.title}]({b.url})")
            else:
                parts.append(b.title or b.url or "")
    return "".join(parts)
