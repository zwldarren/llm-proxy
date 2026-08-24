"""Tests for BaseProvider field policy chokepoint (Task 3)."""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    ConversationContext,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase


def _adapter(policy):
    return OpenAICompatibleBase(
        api_key="k", base_url="https://api.openai.com/v1", unknown_fields_policy=policy
    )


def _req(extra):
    return InternalRequest(
        model="m",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        extra=extra,
    )


def test_apply_field_policy_passthrough_keeps():
    a = _adapter("passthrough")
    assert a._apply_field_policy({"model": "m", "x": 1}, {"x": 1}) == {
        "model": "m",
        "x": 1,
    }


def test_apply_field_policy_ignore_strips():
    a = _adapter("ignore")
    body = a._apply_field_policy({"model": "m", "x": 1, "y": 2}, {"x": 1, "y": 2})
    assert "x" not in body and "y" not in body and body["model"] == "m"


def test_apply_field_policy_error_raises():
    a = _adapter("error")
    with pytest.raises(ProviderError, match="unknown request fields"):
        a._apply_field_policy({"model": "m", "x": 1}, {"x": 1})


def test_apply_field_policy_exempt_keys_kept_under_ignore():
    a = _adapter("ignore")
    body = a._apply_field_policy(
        {"model": "m", "keep_me": 1, "x": 2},
        {"keep_me": 1, "x": 2},
        exempt_keys={"keep_me"},
    )
    assert body["keep_me"] == 1 and "x" not in body


def test_finalize_body_merges_then_applies():
    a = _adapter("ignore")
    req = _req({"temperature": 0.5})
    body = a._finalize_body({"model": "m"}, req, merge_extra=True)
    # ignore strips the extra that was merged in
    assert "temperature" not in body and body["model"] == "m"


def test_finalize_body_passthrough_keeps_merged():
    a = _adapter("passthrough")
    req = _req({"temperature": 0.5})
    body = a._finalize_body({"model": "m"}, req, merge_extra=True)
    assert body["temperature"] == 0.5


def test_build_outbound_body_chat_applies_policy():
    a = OpenAICompatibleBase(
        api_key="k",
        base_url="https://api.openai.com/v1",
        unknown_fields_policy="error",
    )
    req = _req({"__unknown__": True})
    with pytest.raises(ProviderError, match="unknown request fields"):
        a._build_outbound_body(req, request_type="chat")
