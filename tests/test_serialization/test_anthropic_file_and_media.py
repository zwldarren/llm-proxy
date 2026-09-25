"""Anthropic file/audio mapping tests.

Anthropic Messages has no ``file`` and no ``audio`` content block: a ``file``
block is rejected upstream with ``invalid_request_error``, and audio is only
representable as text. These tests pin the mapping of Responses/OpenAI file
inputs onto the ``document``/``image`` blocks the API accepts, the degrade path
for audio, and the file/media ``cache_control`` round trip.
"""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
)
from llm_proxy.models.content_blocks import (
    AudioBlock,
    DocumentBlock,
    FileBlock,
    ImageBlock,
)
from llm_proxy.models.types import AudioSource, DocumentSource, ImageSource
from llm_proxy.serialization.anthropic.serializer import AnthropicProviderSerializer
from llm_proxy.serialization.context import BuildContext, UnsupportedBlockPolicy

_SERIALIZER = AnthropicProviderSerializer()

# base64("hello world")
_TEXT_B64 = "aGVsbG8gd29ybGQ="
# base64("%PDF1.") — a syntactically valid data URI payload; content is never
# parsed by the serializer.
_PDF_B64 = "JVBERi0xLg=="


def _content(blocks, policy: UnsupportedBlockPolicy = "drop") -> list[dict]:
    """Build an Anthropic message content list for the given internal blocks."""
    request = InternalRequest(
        model="claude-sonnet-4-6",
        conversation=ConversationContext(messages=[Message(role="user", content=list(blocks))]),
        params=GenerationParams(max_tokens=64),
    )
    context = BuildContext(
        provider_name="anthropic",
        unsupported_block_policy=policy,
        supported_content_blocks=_SERIALIZER.supported_content_blocks,
    )
    return _SERIALIZER.build_provider_request(request, context)["messages"][0]["content"]


class TestFileBlockMapping:
    """Every FileBlock source maps onto a block Anthropic actually accepts."""

    def test_pdf_data_uri_becomes_base64_document(self):
        content = _content(
            [FileBlock(file_data=f"data:application/pdf;base64,{_PDF_B64}", filename="r.pdf")]
        )
        assert content == [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": _PDF_B64,
                },
                "title": "r.pdf",
            }
        ]

    def test_image_data_uri_becomes_image_block(self):
        content = _content([FileBlock(file_data="data:image/png;base64,AAAA", filename="pic.png")])
        assert content == [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            }
        ]

    def test_text_data_uri_is_decoded_into_a_plain_text_document(self):
        """Anthropic's ``text`` source carries raw text, not base64."""
        content = _content(
            [FileBlock(file_data=f"data:text/plain;base64,{_TEXT_B64}", filename="notes.md")]
        )
        assert content == [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": "hello world"},
                "title": "notes.md",
            }
        ]

    def test_invalid_base64_text_degrades_under_drop_policy(self):
        content = _content(
            [FileBlock(file_data="data:text/plain;base64,!!!not-base64!!!", filename="notes.txt")]
        )
        assert content == [{"type": "text", "text": ""}]

    def test_file_url_becomes_document_url(self):
        content = _content([FileBlock(file_url="https://example.com/doc.pdf")])
        assert content == [
            {"type": "document", "source": {"type": "url", "url": "https://example.com/doc.pdf"}}
        ]

    def test_image_file_url_becomes_image_url(self):
        content = _content([FileBlock(file_url="https://example.com/a.png", filename="a.png")])
        assert content == [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}
        ]

    def test_file_id_becomes_document_file(self):
        content = _content([FileBlock(file_id="file_doc", filename="report.pdf")])
        assert content == [
            {
                "type": "document",
                "source": {"type": "file", "file_id": "file_doc"},
                "title": "report.pdf",
            }
        ]

    def test_image_file_id_becomes_image_file(self):
        content = _content([FileBlock(file_id="file_img", filename="shot.png")])
        assert content == [{"type": "image", "source": {"type": "file", "file_id": "file_img"}}]

    def test_url_carried_in_file_data_is_treated_as_a_url(self):
        """Chat Completions callers historically put the URL in ``file_data``."""
        content = _content([FileBlock(file_data="https://example.com/doc.pdf")])
        assert content == [
            {"type": "document", "source": {"type": "url", "url": "https://example.com/doc.pdf"}}
        ]

    def test_no_file_content_block_is_ever_emitted(self):
        content = _content(
            [
                FileBlock(file_data=f"data:application/pdf;base64,{_PDF_B64}"),
                FileBlock(file_url="https://example.com/doc.pdf"),
                FileBlock(file_id="file_doc"),
            ]
        )
        assert all(block["type"] != "file" for block in content)

    def test_malformed_data_uri_is_not_forwarded_as_base64(self):
        content = _content(
            [FileBlock(file_data="data:application/pdf,not-base64")], policy="degrade"
        )
        assert content == [{"type": "text", "text": "[File]"}]

    def test_file_block_is_not_declared_supported(self):
        """The set drives ``unsupported_block_policy``; FileBlock must not be in it."""
        assert FileBlock not in AnthropicProviderSerializer.supported_content_blocks


class TestUnsupportedFilePolicy:
    """Payloads Anthropic cannot represent follow the block policy."""

    _DOCX = FileBlock(
        file_data=(
            "data:application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document;base64,AAAA"
        ),
        filename="doc.docx",
    )

    def test_docx_is_dropped_under_drop_policy(self):
        assert _content([self._DOCX], policy="drop") == [{"type": "text", "text": ""}]

    def test_docx_degrades_to_a_placeholder_under_degrade_policy(self):
        assert _content([self._DOCX], policy="degrade") == [
            {"type": "text", "text": "[File: doc.docx]"}
        ]

    def test_docx_raises_under_error_policy(self):
        with pytest.raises(ProviderError):
            _content([self._DOCX], policy="error")


class TestAudioDegradation:
    """Anthropic has no audio content block; audio degrades to text."""

    def test_base64_audio_degrades_to_text(self):
        content = _content(
            [AudioBlock(source=AudioSource(type="base64", data="QUJD", media_type="audio/wav"))]
        )
        assert content == [{"type": "text", "text": "[Audio: audio/wav]"}]

    def test_url_audio_degrades_to_text(self):
        content = _content(
            [
                AudioBlock(
                    source=AudioSource(type="url", data="https://x/a.wav", media_type="audio/mpeg")
                )
            ]
        )
        assert content == [{"type": "text", "text": "[Audio: audio/mpeg]"}]

    def test_file_id_audio_names_the_id(self):
        content = _content(
            [
                AudioBlock(
                    source=AudioSource(type="file_id", data="file_abc", media_type="audio/mpeg")
                )
            ]
        )
        assert content == [{"type": "text", "text": "[Audio: file_id=file_abc]"}]

    def test_no_audio_content_block_is_ever_emitted(self):
        content = _content(
            [
                AudioBlock(source=AudioSource(type="base64", data="QUJD", media_type="audio/wav")),
                AudioBlock(source=AudioSource(type="url", data="https://x/a.wav", media_type=None)),
            ]
        )
        assert all(block["type"] != "audio" for block in content)


class TestMediaCacheControl:
    """``cache_control`` on media blocks must survive the round trip."""

    def test_document_cache_control_is_preserved(self):
        from llm_proxy.models.content_blocks.anthropic_builtin import CacheControl

        content = _content(
            [
                DocumentBlock(
                    source=DocumentSource(
                        type="base64", media_type="application/pdf", data=_PDF_B64
                    ),
                    cache_control=CacheControl(type="ephemeral"),
                )
            ]
        )
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_image_cache_control_is_preserved_with_ttl(self):
        from llm_proxy.models.content_blocks.anthropic_builtin import CacheControl

        content = _content(
            [
                ImageBlock(
                    source=ImageSource(type="base64", media_type="image/png", data="AAAA"),
                    cache_control=CacheControl(type="ephemeral", ttl="1h"),
                )
            ]
        )
        assert content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_file_cache_control_is_preserved(self):
        from llm_proxy.models.content_blocks.anthropic_builtin import CacheControl

        content = _content(
            [
                FileBlock(
                    file_data=f"data:application/pdf;base64,{_PDF_B64}",
                    cache_control=CacheControl(type="ephemeral"),
                )
            ]
        )
        assert content[0]["cache_control"] == {"type": "ephemeral"}


class TestImageMediaTypeValidation:
    """Anthropic accepts only jpeg/png/gif/webp in base64 image blocks.

    Forwarding any other media type (or a missing one) makes the upstream
    reject the whole request, so an unrepresentable payload follows the
    block policy instead of being sent as-is.
    """

    @staticmethod
    def _image(media_type: str | None, data: str = "AAAA") -> ImageBlock:
        return ImageBlock(source=ImageSource(type="base64", data=data, media_type=media_type))

    def test_jpg_alias_is_normalized_to_jpeg(self):
        content = _content([self._image("image/jpg")])
        assert content == [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
            }
        ]

    def test_media_type_case_and_parameters_are_normalized(self):
        content = _content([self._image("IMAGE/PNG;charset=utf-8")])
        assert content[0]["source"]["media_type"] == "image/png"

    @pytest.mark.parametrize(
        "media_type", ["image/svg+xml", "image/tiff", "image/bmp", "image/avif"]
    )
    def test_unsupported_media_type_degrades(self, media_type):
        content = _content([self._image(media_type)], policy="degrade")
        assert content == [{"type": "text", "text": f"[Image: {media_type}]"}]

    def test_missing_media_type_degrades(self):
        content = _content([self._image(None)], policy="degrade")
        assert content == [{"type": "text", "text": "[Image: image]"}]

    def test_unsupported_media_type_is_dropped_under_drop_policy(self):
        content = _content([self._image("image/tiff")])
        assert content == [{"type": "text", "text": ""}]

    def test_unsupported_media_type_raises_under_error_policy(self):
        with pytest.raises(ProviderError):
            _content([self._image("image/tiff")], policy="error")

    @pytest.mark.parametrize("media_type", ["image/jpeg", "image/png", "image/gif", "image/webp"])
    def test_supported_media_types_pass_through(self, media_type):
        content = _content([self._image(media_type)])
        assert content[0]["source"]["media_type"] == media_type

    def test_empty_url_image_degrades_instead_of_forwarding(self):
        content = _content(
            [ImageBlock(source=ImageSource(type="url", data="", media_type=None))],
            policy="degrade",
        )
        assert content == [{"type": "text", "text": "[Image: image]"}]

    def test_file_id_image_without_media_type_still_passes(self):
        content = _content(
            [ImageBlock(source=ImageSource(type="file_id", data="file_x", media_type=None))]
        )
        assert content == [{"type": "image", "source": {"type": "file", "file_id": "file_x"}}]

    def test_file_block_jpg_data_uri_is_normalized(self):
        content = _content([FileBlock(file_data="data:image/jpg;base64,AAAA")])
        assert content == [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
            }
        ]
