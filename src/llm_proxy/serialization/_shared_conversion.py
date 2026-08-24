"""Shared conversion logic: convert unsupported blocks to natively-supported ones."""

from typing import Any

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

    # Tool result variants: all structurally match ToolResultBlock.
    # Some variants carry list[dict[str, Any]] content (e.g. WebSearchToolResultBlock);
    # we stringify those so the resulting ToolResultBlock stays type-safe.
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
        content: str | list[Any] = block.content
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
            import orjson

            content = orjson.dumps(content).decode()
        return ToolResultBlock(
            tool_use_id=block.tool_use_id,
            content=content,  # type: ignore[arg-type]
            is_error=block.is_error,
        )

    return None


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
