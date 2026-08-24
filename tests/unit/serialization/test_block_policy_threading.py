"""Test that unsupported_block_policy is threaded through call sites.

Verifies that content_to_openai_parts reads context.unsupported_block_policy
instead of the old context.unknown_fields_policy for block degradation decisions.
"""

from llm_proxy.models import CustomToolUseBlock
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.converter import content_to_openai_parts


def _ctx(block_policy: str) -> BuildContext:
    """Build a context with just the block policy and no supported blocks."""
    return BuildContext(
        unsupported_block_policy=block_policy,
        supported_content_blocks=frozenset(),
    )


def test_openai_converter_drop_skips_block() -> None:
    """With 'drop' policy, CustomToolUseBlock is silently skipped."""
    parts = content_to_openai_parts(
        [CustomToolUseBlock(id="ctu_1", name="my_tool", input='{"k":"v"}')],
        context=_ctx("drop"),
    )
    assert parts in ([], "")


def test_openai_converter_degrade_emits_text() -> None:
    """With 'degrade' policy, CustomToolUseBlock is text-degraded."""
    parts = content_to_openai_parts(
        [CustomToolUseBlock(id="ctu_1", name="my_tool", input='{"k":"v"}')],
        context=_ctx("degrade"),
    )
    assert any("my_tool" in str(p) for p in (parts if isinstance(parts, list) else [parts]))
