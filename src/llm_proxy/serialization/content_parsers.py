"""Shared content block parsing utilities.

Extracted as standalone functions so that both ProtocolSerializer and
ProviderSerializer classes can use them without inheriting from a common base.
"""

from llm_proxy.core.utils import create_image_source_from_url
from llm_proxy.models import (
    AudioBlock,
    FileBlock,
    ImageBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    VideoBlock,
)
from llm_proxy.models.content_blocks.anthropic_builtin import CacheControl
from llm_proxy.models.types import AudioSource, ImageSource, VideoSource


def parse_reasoning_content(msg: dict) -> ThinkingBlock | RedactedThinkingBlock | None:
    """Parse OpenAI-style reasoning_content into a ThinkingBlock.

    Reasoning precedes the answer text, so callers insert the result at the
    front of the block list. Returns None when the message carries no
    reasoning_content.
    """
    reasoning_text = msg.get("reasoning_content")
    if not reasoning_text or not isinstance(reasoning_text, str):
        return None
    if msg.get("reasoning_is_redacted", False):
        return RedactedThinkingBlock(data=reasoning_text)
    return ThinkingBlock(thinking=reasoning_text, signature=msg.get("reasoning_signature"))


def parse_text_block(part: dict) -> TextBlock | None:
    part_type = part.get("type", "text")
    if part_type == "text":
        text = part.get("text", "")
        cache_control = _parse_text_block_cache_control(part)
        return TextBlock(text=text, cache_control=cache_control)
    return None


def _parse_text_block_cache_control(part: dict) -> CacheControl | None:
    """Extract cache control for a text content part.

    Supports Anthropic-style ``cache_control`` and OpenAI Responses-style
    ``prompt_cache_breakpoint`` (treated as an ephemeral Anthropic breakpoint).
    """
    raw = part.get("cache_control")
    if raw:
        if isinstance(raw, dict):
            return CacheControl(
                type=raw.get("type", "ephemeral"),
                ttl=raw.get("ttl"),
            )
        return CacheControl(type="ephemeral")
    if part.get("prompt_cache_breakpoint"):
        return CacheControl(type="ephemeral")
    return None


def parse_image_block_openai(part: dict) -> ImageBlock | None:
    if part.get("type") != "image_url":
        return None
    image_url = part.get("image_url") or {}
    url = image_url.get("url", "")
    detail = image_url.get("detail")
    source = create_image_source_from_url(url)
    if source:
        return ImageBlock(source=source, detail=detail)
    return None


def parse_image_block_anthropic(part: dict) -> ImageBlock | None:
    if part.get("type") != "image":
        return None
    source = part.get("source", {})
    source_type = source.get("type", "base64")
    if source_type == "file":
        source_type = "file_id"
    data = source.get("data", source.get("url", source.get("file_id", "")))
    return ImageBlock(
        source=ImageSource(
            type=source_type,
            data=data,
            media_type=source.get("media_type"),
        )
    )


def parse_audio_block_anthropic(part: dict) -> AudioBlock | None:
    if part.get("type") != "audio":
        return None
    source = part.get("source", {})
    source_type = source.get("type", "base64")
    if source_type == "file":
        source_type = "file_id"
    data = source.get("data", source.get("url", source.get("file_id", "")))
    return AudioBlock(
        source=AudioSource(
            type=source_type,
            data=data,
            media_type=source.get("media_type"),
        )
    )


def parse_audio_block_openai(part: dict) -> AudioBlock | None:
    if part.get("type") != "input_audio":
        return None
    audio = part.get("input_audio") or {}
    data = audio.get("data", "")
    audio_format = audio.get("format", "wav")
    _MEDIA_TYPE_MAP = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
    }
    media_type = _MEDIA_TYPE_MAP.get(audio_format, f"audio/{audio_format}")
    return AudioBlock(source=AudioSource(type="base64", data=data, media_type=media_type))


def parse_file_block_openai(part: dict) -> FileBlock | None:
    if part.get("type") != "file":
        return None
    file_info = part.get("file") or {}
    # Accept both the OpenAI nested shape ({"file": {...}}) and the
    # DeepSeek top-level shape ({"file_id": ..., "file_data": ...});
    # the nested shape wins when both are present.
    return FileBlock(
        file_data=file_info.get("file_data") or part.get("file_data"),
        file_id=file_info.get("file_id") or part.get("file_id"),
        filename=file_info.get("filename") or part.get("filename"),
    )


def parse_file_block_anthropic(part: dict) -> FileBlock | None:
    if part.get("type") != "file":
        return None
    return FileBlock(
        file_data=part.get("file_data"),
        file_id=part.get("file_id"),
        filename=part.get("filename"),
    )


def parse_video_block_openai(part: dict) -> VideoBlock | None:
    """Parse an OpenAI-format video_url content part into a VideoBlock."""
    if part.get("type") != "video_url":
        return None
    video_url = part.get("video_url") or {}
    url = video_url.get("url", "")

    source = create_image_source_from_url(url)
    if source:
        return VideoBlock(
            source=VideoSource(
                type=source.type,
                data=source.data,
                media_type=source.media_type,
            )
        )
    return None
