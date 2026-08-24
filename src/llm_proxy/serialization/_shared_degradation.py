"""Shared degradation logic for content block handling across serializers."""

from collections.abc import Callable
from typing import Any

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    AudioBlock,
    ContentBlock,
    CustomToolUseBlock,
    DocumentBlock,
    FileBlock,
    ImageBlock,
    RedactedThinkingBlock,
    RefusalBlock,
    ServerToolUseBlock,
    VideoBlock,
)
from llm_proxy.models.content_blocks.anthropic_builtin import (
    BashCodeExecutionToolResultBlock,
    CodeExecutionToolResultBlock,
    ContainerUploadBlock,
    SearchResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    ToolReferenceBlock,
    ToolSearchToolResultBlock,
    WebFetchToolResultBlock,
    WebSearchResultContentBlock,
    WebSearchToolResultBlock,
)

_BLOCK_DEGRADERS: dict[type[ContentBlock], Callable[[Any], str | None]] = {}


def register_block_degrader[T: ContentBlock](
    block_type: type[T],
) -> Callable[[Callable[[T], str | None]], Callable[[T], str | None]]:
    """Decorator to register a text degradation function for a content block type.

    The decorated function receives a block instance and returns a human-readable
    text representation, or None if it cannot be degraded.
    """

    def decorator(
        func: Callable[[T], str | None],
    ) -> Callable[[T], str | None]:
        _BLOCK_DEGRADERS[block_type] = func  # type: ignore[arg-type]
        return func

    return decorator


def degrade_block_to_text(block: ContentBlock) -> str | None:
    """Convert an unsupported block type to a text placeholder via the registry.

    Args:
        block: The content block to degrade.

    Returns:
        A human-readable text representation, or None if not degradable.
    """
    func = _BLOCK_DEGRADERS.get(type(block))
    if func is not None:
        return func(block)
    return None


def should_degrade_block(
    policy: str,
    block: ContentBlock,
    provider_name: str,
    *,
    supported_blocks: frozenset[type[ContentBlock]] = frozenset(),
) -> bool:
    """Decide whether an unsupported content block should be degraded to text.

    Capability-based checking is used: blocks whose type is in
    ``supported_blocks`` are considered supported; all others trigger
    degradation according to the policy.

    Args:
        policy: The unsupported_block_policy - 'drop', 'degrade', or 'error'.
        block: The content block being evaluated.
        provider_name: Provider name, used in error messages.
        supported_blocks: Set of block types this provider supports.

    Returns:
        True if the block should be degraded to text.

    Raises:
        ProviderError: If policy is 'error' and the block type is unsupported.
    """
    if type(block) in supported_blocks:
        return False
    match policy:
        case "drop":
            return False
        case "degrade":
            return True
        case "error":
            block_type = type(block).__name__
            raise ProviderError(
                message=(
                    f"Provider '{provider_name}' does not support content block type "
                    f"'{block_type}'. Remove or replace these blocks, or set "
                    "unsupported_block_policy to 'drop' or 'degrade'."
                ),
                error_type="invalid_request_error",
                provider_name=provider_name,
            )
        case _:
            return False


# ---------------------------------------------------------------------------
# Register degraders for all known block types
# ---------------------------------------------------------------------------


@register_block_degrader(ServerToolUseBlock)
def _degrade_server_tool_use(block: ServerToolUseBlock) -> str | None:
    return f"[Server tool use: {block.name}({block.input})]"


@register_block_degrader(RedactedThinkingBlock)
def _degrade_redacted_thinking(block: RedactedThinkingBlock) -> str | None:
    return "[Redacted thinking]"


@register_block_degrader(RefusalBlock)
def _degrade_refusal(block: RefusalBlock) -> str | None:
    return f"[Refusal: {block.refusal}]"


@register_block_degrader(CustomToolUseBlock)
def _degrade_custom_tool_use(block: CustomToolUseBlock) -> str | None:
    return f"[Custom tool use: {block.name}]"


@register_block_degrader(WebSearchToolResultBlock)
def _degrade_web_search_tool_result(
    block: WebSearchToolResultBlock,
) -> str | None:
    if isinstance(block.content, str):
        return f"[Web search: {block.content[:200]}]"
    return "[Web search result]"


@register_block_degrader(WebFetchToolResultBlock)
def _degrade_web_fetch_tool_result(
    block: WebFetchToolResultBlock,
) -> str | None:
    if isinstance(block.content, str):
        return f"[Web fetch: {block.content[:200]}]"
    return "[Web fetch result]"


@register_block_degrader(WebSearchResultContentBlock)
def _degrade_web_search_result_content(
    block: WebSearchResultContentBlock,
) -> str | None:
    title = block.title or ""
    return f"[Web search: {title}]" if title else "[Web search result]"


@register_block_degrader(CodeExecutionToolResultBlock)
def _degrade_code_execution_tool_result(
    block: CodeExecutionToolResultBlock,
) -> str | None:
    if isinstance(block.content, str):
        return f"[Code execution: {block.content[:200]}]"
    return "[Code execution result]"


@register_block_degrader(BashCodeExecutionToolResultBlock)
def _degrade_bash_code_execution_tool_result(
    block: BashCodeExecutionToolResultBlock,
) -> str | None:
    return "[Bash code execution result]"


@register_block_degrader(TextEditorCodeExecutionToolResultBlock)
def _degrade_text_editor_code_execution_tool_result(
    block: TextEditorCodeExecutionToolResultBlock,
) -> str | None:
    return "[Text editor code execution result]"


@register_block_degrader(ToolSearchToolResultBlock)
def _degrade_tool_search_tool_result(
    block: ToolSearchToolResultBlock,
) -> str | None:
    return "[Tool search result]"


@register_block_degrader(ToolReferenceBlock)
def _degrade_tool_reference(block: ToolReferenceBlock) -> str | None:
    name = block.tool_name or block.tool_id
    return f"[Tool reference: {name}]"


@register_block_degrader(ContainerUploadBlock)
def _degrade_container_upload(block: ContainerUploadBlock) -> str | None:
    name = block.filename or ""
    return f"[Container upload: {name}]" if name else "[Container upload]"


@register_block_degrader(SearchResultBlock)
def _degrade_search_result(block: SearchResultBlock) -> str | None:
    title = block.title or ""
    return f"[Search result: {title}]" if title else "[Search result]"


# Multimedia blocks: provide basic text fallback so they are not silently dropped.
@register_block_degrader(ImageBlock)
def _degrade_image(block: ImageBlock) -> str | None:
    return f"[Image: {block.source.media_type or 'image'}]"


@register_block_degrader(AudioBlock)
def _degrade_audio(block: AudioBlock) -> str | None:
    return f"[Audio: {block.source.media_type or 'audio'}]"


@register_block_degrader(DocumentBlock)
def _degrade_document(block: DocumentBlock) -> str | None:
    title = block.title or ""
    return f"[Document: {title}]" if title else "[Document]"


@register_block_degrader(FileBlock)
def _degrade_file(block: FileBlock) -> str | None:
    filename = block.filename or block.file_id or ""
    return f"[File: {filename}]" if filename else "[File]"


@register_block_degrader(VideoBlock)
def _degrade_video(block: VideoBlock) -> str | None:
    return f"[Video: {block.source.media_type or 'video'}]"
