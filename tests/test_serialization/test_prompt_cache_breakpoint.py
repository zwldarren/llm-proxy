"""Explicit prompt-cache breakpoints survive parse -> emit on both protocols.

OpenAI's per-content-part ``prompt_cache_breakpoint`` (``{"mode": "explicit"}``)
was parsed into an Anthropic-shaped ``CacheControl`` only, so it never reached a
native OpenAI upstream. These tests pin the field on the unified blocks and the
Chat/Responses emit paths that write it back.
"""

from llm_proxy.models import (
    AudioBlock,
    FileBlock,
    ImageBlock,
    TextBlock,
)
from llm_proxy.models.types import AudioSource, ImageSource
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.serialization.openai.converter import content_to_openai_parts
from llm_proxy.serialization.openai.serializer import OpenAIResponsesProviderSerializer

BP = {"mode": "explicit"}
IMAGE = ImageSource(type="url", data="https://x/a.png", media_type=None)


# ---------------------------------------------------------------------------
# Parse: wire part -> unified block
# ---------------------------------------------------------------------------


def test_chat_parse_keeps_breakpoint_on_text_block():
    blocks = OpenAIProtocolSerializer().parse_content_blocks(
        [{"type": "text", "text": "hi", "prompt_cache_breakpoint": BP}]
    )
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].prompt_cache_breakpoint == BP


def test_chat_parse_keeps_breakpoint_on_image_block():
    blocks = OpenAIProtocolSerializer().parse_content_blocks(
        [
            {"type": "text", "text": "look"},
            {
                "type": "image_url",
                "image_url": {"url": "https://x/a.png"},
                "prompt_cache_breakpoint": BP,
            },
        ]
    )
    assert isinstance(blocks[1], ImageBlock)
    assert blocks[1].prompt_cache_breakpoint == BP


def test_chat_parse_keeps_breakpoint_on_audio_block():
    blocks = OpenAIProtocolSerializer().parse_content_blocks(
        [
            {
                "type": "input_audio",
                "input_audio": {"data": "AAAA", "format": "wav"},
                "prompt_cache_breakpoint": BP,
            }
        ]
    )
    assert isinstance(blocks[0], AudioBlock)
    assert blocks[0].prompt_cache_breakpoint == BP


def test_chat_parse_keeps_breakpoint_on_file_block():
    blocks = OpenAIProtocolSerializer().parse_content_blocks(
        [{"type": "file", "file": {"file_id": "file_1"}, "prompt_cache_breakpoint": BP}]
    )
    assert isinstance(blocks[0], FileBlock)
    assert blocks[0].prompt_cache_breakpoint == BP


def test_responses_parse_keeps_breakpoint_on_input_text():
    request = get_protocol_serializer("openresponses").parse_request(
        {
            "model": "gpt-6-luna",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "hi",
                            "prompt_cache_breakpoint": BP,
                        }
                    ],
                }
            ],
        }
    )
    block = request.conversation.messages[0].content[0]
    assert isinstance(block, TextBlock)
    assert block.prompt_cache_breakpoint == BP


def test_responses_parse_keeps_breakpoint_on_input_image():
    request = get_protocol_serializer("openresponses").parse_request(
        {
            "model": "gpt-6-luna",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "https://x/a.png",
                            "prompt_cache_breakpoint": BP,
                        }
                    ],
                }
            ],
        }
    )
    block = request.conversation.messages[0].content[0]
    assert isinstance(block, ImageBlock)
    assert block.prompt_cache_breakpoint == BP


def test_breakpoint_absent_is_none():
    blocks = OpenAIProtocolSerializer().parse_content_blocks([{"type": "text", "text": "hi"}])
    assert blocks[0].prompt_cache_breakpoint is None


# ---------------------------------------------------------------------------
# Emit: unified block -> wire part
# ---------------------------------------------------------------------------


def test_chat_emit_writes_breakpoint_on_text_and_image():
    parts = content_to_openai_parts(
        [
            TextBlock(text="hi", prompt_cache_breakpoint=BP),
            ImageBlock(source=IMAGE, prompt_cache_breakpoint=BP),
        ]
    )
    assert isinstance(parts, list)
    assert parts[0] == {"type": "text", "text": "hi", "prompt_cache_breakpoint": BP}
    assert parts[1]["prompt_cache_breakpoint"] == BP


def test_chat_emit_drops_breakpoint_for_non_openai_provider():
    # ``prompt_cache_breakpoint`` is an OpenAI-dialect content field; forwarding
    # it to DeepSeek/OpenRouter would be an unknown key. Non-OpenAI targets keep
    # the previous behavior of dropping it.
    from llm_proxy.serialization.context import BuildContext

    parts = content_to_openai_parts(
        [TextBlock(text="hi", prompt_cache_breakpoint=BP), ImageBlock(source=IMAGE)],
        BuildContext(provider_name="deepseek"),
    )
    assert isinstance(parts, list)
    assert all("prompt_cache_breakpoint" not in part for part in parts)


def test_responses_normalizer_carries_breakpoint_on_image():
    normalized = OpenAIResponsesProviderSerializer()._normalize_responses_content(
        [
            {
                "type": "image_url",
                "image_url": {"url": "https://x/a.png"},
                "prompt_cache_breakpoint": BP,
            }
        ],
        "user",
    )
    assert normalized == [
        {
            "type": "input_image",
            "image_url": "https://x/a.png",
            "prompt_cache_breakpoint": BP,
        }
    ]


def test_responses_normalizer_carries_breakpoint_on_file():
    normalized = OpenAIResponsesProviderSerializer()._normalize_responses_content(
        [
            {
                "type": "file",
                "file": {"file_id": "file_1"},
                "prompt_cache_breakpoint": BP,
            }
        ],
        "user",
    )
    assert normalized == [
        {"type": "input_file", "file_id": "file_1", "prompt_cache_breakpoint": BP}
    ]


def test_responses_normalizer_carries_breakpoint_on_text():
    # ``text`` is renamed to ``input_text``; the sibling field must ride along.
    normalized = OpenAIResponsesProviderSerializer()._normalize_responses_content(
        [{"type": "text", "text": "hi", "prompt_cache_breakpoint": BP}],
        "user",
    )
    assert normalized == [{"type": "input_text", "text": "hi", "prompt_cache_breakpoint": BP}]


def test_audio_source_breakpoint_is_threaded():
    # Sanity: the field lives on the block independently of the source payload.
    block = AudioBlock(
        source=AudioSource(type="base64", data="AAAA", media_type="audio/wav"),
        prompt_cache_breakpoint=BP,
    )
    assert block.prompt_cache_breakpoint == BP
