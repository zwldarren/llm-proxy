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

#: Placeholder text emitted for ``RedactedThinkingBlock`` summaries on the
#: client wire. The opaque payload rides ``encrypted_content`` (written by the
#: OpenResponses, OpenAI and Anthropic formatters/converter), while this marker
#: is only a display placeholder the parse side recognizes to restore the block
#: type on round-trip. Single source for every site that emits or recognizes it.
REDACTED_THINKING_TEXT = "[redacted]"


def parse_reasoning_content(msg: dict) -> ThinkingBlock | RedactedThinkingBlock | None:
    """Parse OpenAI-style reasoning_content into a ThinkingBlock.

    Reasoning precedes the answer text, so callers insert the result at the
    front of the block list. Returns None when the message carries no reasoning
    at all. Accepts both OpenAI's ``reasoning_content`` and the
    OpenRouter/NanoGPT ``reasoning`` spelling.

    A segment with no text but a ``reasoning_signature`` or ``encrypted_content``
    is kept: those are integrity payloads the upstream verifies on replay, so
    dropping a "signature-only" thinking block breaks interleaved-thinking
    continuations (the Anthropic formatter preserves the same shape).
    """
    reasoning_text = msg.get("reasoning_content")
    if not isinstance(reasoning_text, str) or not reasoning_text:
        alternate = msg.get("reasoning")
        if isinstance(alternate, str) and alternate:
            reasoning_text = alternate
    if not isinstance(reasoning_text, str):
        reasoning_text = ""
    encrypted = msg.get("encrypted_content")
    if not isinstance(encrypted, str):
        encrypted = None
    signature = msg.get("reasoning_signature")
    if not isinstance(signature, str):
        signature = None
    if msg.get("reasoning_is_redacted", False):
        # The opaque payload rides ``encrypted_content`` (written by the
        # streaming formatter); ``reasoning_content`` is only the
        # "[redacted]" display placeholder, so prefer the real payload.
        data = encrypted or reasoning_text
        return RedactedThinkingBlock(data=data) if data else None
    if not reasoning_text and not signature and not encrypted:
        return None
    return ThinkingBlock(
        thinking=reasoning_text,
        signature=signature,
        encrypted_content=encrypted,
    )


def parse_reasoning_segments(
    msg: dict,
) -> list[tuple[ThinkingBlock | RedactedThinkingBlock, int]]:
    """Parse reasoning segments plus their position relative to tool calls.

    Returns ``(block, after_tool_calls)`` pairs in original order. The position
    is the number of tool calls that preceded the segment in the assistant turn
    (written by the OpenAI protocol formatter as ``after_tool_calls``); callers
    use it to re-interleave reasoning and tool calls the way the model emitted
    them instead of flattening every reasoning block to the front. The
    single-segment fallback has no recorded position and reports ``0``, which
    keeps the historical "reasoning precedes the answer" behavior.
    """
    segments = msg.get("reasoning_segments")
    if isinstance(segments, list) and segments:
        parsed: list[tuple[ThinkingBlock | RedactedThinkingBlock, int]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            position = segment.get("after_tool_calls")
            position = position if isinstance(position, int) and position > 0 else 0
            if segment.get("type") == "redacted":
                data = segment.get("data")
                if isinstance(data, str) and data:
                    parsed.append((RedactedThinkingBlock(data=data), position))
                continue
            text = segment.get("text")
            signature = segment.get("signature")
            encrypted = segment.get("encrypted_content")
            if text or signature or encrypted:
                parsed.append(
                    (
                        ThinkingBlock(
                            thinking=text if isinstance(text, str) else "",
                            signature=signature if isinstance(signature, str) else None,
                            encrypted_content=encrypted if isinstance(encrypted, str) else None,
                        ),
                        position,
                    )
                )
        if parsed:
            return parsed
    single = parse_reasoning_content(msg)
    return [(single, 0)] if single is not None else []


def parse_reasoning_blocks(msg: dict) -> list[ThinkingBlock | RedactedThinkingBlock]:
    """Parse every reasoning segment from an OpenAI-format assistant message.

    Prefers the proxy's multi-segment carrier ``reasoning_segments`` — written
    by the OpenAI protocol formatter when an assistant turn interleaved several
    thinking / redacted segments, which the single ``reasoning_content`` field
    cannot represent — and otherwise falls back to ``parse_reasoning_content``.
    Returns blocks in their original order; callers that do not track tool-call
    positions prepend them (reasoning precedes the answer text).
    """
    return [block for block, _ in parse_reasoning_segments(msg)]


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


def unparseable_image_placeholder(part_type: str) -> TextBlock:
    """Text placeholder for an image part whose source cannot be parsed.

    Chosen over silently dropping the part so the client's intent (an image was
    sent, and of what kind) stays visible to the model downstream, matching the
    no-silent-loss behavior applied to other unrepresentable content.
    """
    return TextBlock(text=f"[{part_type}: unparseable image source]")


def extract_image_reference(part: dict) -> tuple[str, str | None]:
    """Extract ``(url, detail)`` from an OpenAI image content part.

    Accepts the documented object shape
    (``{"image_url": {"url": ..., "detail": ...}}``) and the bare-string
    shape some OpenAI-compatible clients emit (``{"image_url": "https://..."}``).
    Chat Completions carries ``detail`` inside the object; the Responses API
    carries it at the part level, so both locations are checked. A missing or
    non-string ``image_url`` yields an empty URL, which callers treat as
    unparseable.
    """
    raw = part.get("image_url")
    if isinstance(raw, dict):
        url = raw.get("url")
        detail = raw.get("detail") or part.get("detail")
    elif isinstance(raw, str):
        url = raw
        detail = part.get("detail")
    else:
        url = ""
        detail = part.get("detail")
    return (url if isinstance(url, str) else ""), detail


def parse_image_block_openai(part: dict) -> ImageBlock | None:
    if part.get("type") != "image_url":
        return None
    url, detail = extract_image_reference(part)
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
