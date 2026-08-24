"""Tests for Anthropic format_content_blocks block degradation via unsupported_block_policy."""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import VideoBlock
from llm_proxy.models.types import VideoSource
from llm_proxy.serialization.anthropic.mixin import AnthropicContentMixin
from llm_proxy.serialization.context import BuildContext


def _ctx(policy: str) -> BuildContext:
    return BuildContext(
        unsupported_block_policy=policy,
        provider_name="anthropic",
        supported_content_blocks=frozenset(),
    )


def test_anthropic_degrade_emits_text():
    mixin = AnthropicContentMixin()
    parts = mixin.format_content_blocks(
        [VideoBlock(source=VideoSource(type="base64", data="abc"))],
        context=_ctx("degrade"),
    )
    assert any("Video" in str(p) for p in parts)


def test_anthropic_drop_skips():
    mixin = AnthropicContentMixin()
    parts = mixin.format_content_blocks(
        [VideoBlock(source=VideoSource(type="base64", data="abc"))],
        context=_ctx("drop"),
    )
    # The VideoBlock should be dropped, leaving only the empty-text invariant.
    assert len(parts) == 1
    assert parts[0] == {"type": "text", "text": ""}


def test_anthropic_error_raises():
    mixin = AnthropicContentMixin()
    with pytest.raises(ProviderError, match="does not support content block type"):
        mixin.format_content_blocks(
            [VideoBlock(source=VideoSource(type="base64", data="abc"))],
            context=_ctx("error"),
        )
