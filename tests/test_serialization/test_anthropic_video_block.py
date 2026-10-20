"""Anthropic ``video`` content block mapping tests.

Anthropic itself has no video block; MiniMax's Anthropic-compatible Messages
endpoint does (``{"type": "video", "source": {...}}``). The proxy parses the
block into the shared ``VideoBlock`` so a chat wire can carry it as
``video_url`` — and it must not leak the block to an Anthropic upstream that
would reject it with HTTP 400.
"""

import pytest

from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
    VideoBlock,
)
from llm_proxy.models.types import VideoSource
from llm_proxy.serialization.anthropic.serializer import AnthropicProviderSerializer
from llm_proxy.serialization.context import BuildContext, UnsupportedBlockPolicy

_SERIALIZER = AnthropicProviderSerializer()


def _content(blocks, policy: UnsupportedBlockPolicy = "drop", supported=None) -> list[dict]:
    """Build an Anthropic message content list for the given internal blocks."""
    request = InternalRequest(
        model="claude-sonnet-4-6",
        conversation=ConversationContext(messages=[Message(role="user", content=list(blocks))]),
        params=GenerationParams(max_tokens=64),
    )
    context = BuildContext(
        provider_name="anthropic",
        unsupported_block_policy=policy,
        supported_content_blocks=supported or _SERIALIZER.supported_content_blocks,
    )
    return _SERIALIZER.build_provider_request(request, context)["messages"][0]["content"]


def _url_video(**extra) -> VideoBlock:
    return VideoBlock(
        source=VideoSource(type="url", data="https://example.com/clip.mp4"),
        extra=dict(extra),
    )


class TestVideoBlockOnTheAnthropicWire:
    """Only a dialect that declares video support receives the block."""

    def test_video_block_is_not_sent_to_an_anthropic_upstream(self):
        content = _content([_url_video()])
        assert all(part.get("type") != "video" for part in content)

    def test_video_block_degrades_to_text_when_the_policy_asks_for_it(self):
        content = _content([_url_video()], policy="degrade")
        assert all(part.get("type") != "video" for part in content)
        assert any("Video" in part.get("text", "") for part in content)

    def test_video_block_is_rendered_when_the_dialect_supports_it(self):
        supported = frozenset(_SERIALIZER.supported_content_blocks | {VideoBlock})
        content = _content([_url_video(fps=2, max_long_side_pixel=1024)], supported=supported)
        assert content == [
            {
                "type": "video",
                "source": {
                    "type": "url",
                    "url": "https://example.com/clip.mp4",
                    "fps": 2,
                    "max_long_side_pixel": 1024,
                },
            }
        ]

    def test_base64_video_keeps_its_media_type(self):
        supported = frozenset(_SERIALIZER.supported_content_blocks | {VideoBlock})
        block = VideoBlock(
            source=VideoSource(type="base64", data="AAAA", media_type="video/mp4"),
        )
        content = _content([block], supported=supported)
        assert content == [
            {
                "type": "video",
                "source": {"type": "base64", "media_type": "video/mp4", "data": "AAAA"},
            }
        ]


class TestVideoBlockParsing:
    """The Anthropic and OpenAI protocols both parse the block into a VideoBlock."""

    @pytest.mark.parametrize("protocol", ["anthropic", "openai"])
    def test_video_block_is_parsed(self, protocol):
        from llm_proxy.protocols.registry import get_protocol_serializer

        if protocol == "anthropic":
            from llm_proxy.protocols.anthropic import serializer  # noqa: F401
        else:
            from llm_proxy.protocols.openai import serializer  # noqa: F401

        serializer_instance = get_protocol_serializer(protocol)
        blocks = serializer_instance.parse_content_blocks(
            [
                {"type": "text", "text": "what?"},
                {
                    "type": "video",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/clip.mp4",
                        "fps": 2,
                    },
                },
            ]
        )
        video_blocks = [b for b in blocks if isinstance(b, VideoBlock)]
        assert len(video_blocks) == 1
        assert video_blocks[0].source.type == "url"
        assert video_blocks[0].source.data == "https://example.com/clip.mp4"
        assert video_blocks[0].extra == {"fps": 2}
