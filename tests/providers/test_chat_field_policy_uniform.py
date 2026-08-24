"""Test that chat field policy applies uniformly through the dispatch chokepoint."""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase


def _adapter(policy):
    return OpenAICompatibleBase(
        api_key="k", base_url="https://api.openai.com/v1", unknown_fields_policy=policy
    )


def _req(extra):
    return InternalRequest(
        model="gpt-4o",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        extra=extra,
    )


def test_openai_compatible_passthrough_keeps_extra():
    a = _adapter("passthrough")
    body = a._build_request_body(_req({"x_custom_flag": True}))
    assert body["x_custom_flag"] is True


def test_openai_compatible_ignore_strips_extra():
    a = _adapter("ignore")
    body = a._build_request_body(_req({"x_custom_flag": True}))
    assert "x_custom_flag" not in body


def test_openai_compatible_error_raises():
    a = _adapter("error")
    with pytest.raises(ProviderError, match="unknown request fields"):
        a._build_request_body(_req({"x_custom_flag": True}))


def test_build_chat_raw_dispatch_invoked_with_policies():
    """Verify _build_request_body routes through _build_chat_raw dispatch.

    Subclass OpenAICompatibleBase and spy on _build_chat_raw to confirm the
    dispatch chokepoint (_build_outbound_body) is actually exercised, not
    bypassed by inline body construction.  Also confirm the BuildContext
    carries the configured policies.
    """
    call_log: dict = {}

    class SpyAdapter(OpenAICompatibleBase):
        def _build_chat_raw(self, request, context):
            call_log["called"] = True
            call_log["context"] = context
            return super()._build_chat_raw(request, context)

    a = SpyAdapter(
        api_key="k",
        base_url="https://api.openai.com/v1",
        unknown_fields_policy="passthrough",
        unsupported_block_policy="error",
    )
    body = a._build_request_body(_req({"x_custom_flag": True}))

    assert call_log.get("called") is True, (
        "_build_chat_raw was not invoked -- dispatch chokepoint was bypassed"
    )
    ctx = call_log["context"]
    assert ctx.unknown_fields_policy == "passthrough"
    assert ctx.unsupported_block_policy == "error"
    # passthrough keeps the extra field in the body
    assert body["x_custom_flag"] is True
