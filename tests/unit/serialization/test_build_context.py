"""Tests for BuildContext policy enums."""

from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.serialization.context import BuildContext


def _req() -> InternalRequest:
    return InternalRequest(
        model="m",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
    )


def test_context_defaults():
    ctx = BuildContext.from_request(_req())
    assert ctx.unknown_fields_policy == "ignore"
    assert ctx.unsupported_block_policy == "drop"


def test_context_accepts_both_policies():
    ctx = BuildContext.from_request(
        _req(), unknown_fields_policy="passthrough", unsupported_block_policy="degrade"
    )
    assert ctx.unknown_fields_policy == "passthrough"
    assert ctx.unsupported_block_policy == "degrade"
