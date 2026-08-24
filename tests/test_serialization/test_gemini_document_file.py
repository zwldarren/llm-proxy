# tests/test_serialization/test_gemini_document_file.py
"""Regression tests for Gemini document/file conversion (P0 bug 3a/3d).

Verifies that Anthropic ``document`` blocks (url/base64/text sources) and
OpenAI ``file`` blocks (file_data data-URI/file_id/plain URL) are converted
into valid Gemini ``file_data``/``inline_data`` parts instead of being
silently dropped or producing malformed parts with the data: prefix intact.

Bug 3a: DocumentBlock url/base64 sources used to fall into the degrade branch
        and produced empty text parts, leaving ``contents: []`` (silent loss).
Bug 3d: OpenAI ``file: {file_data}`` with a ``data:...;base64,...`` URI used to
        emit ``inline_data`` with the raw data: URI as ``data`` and no
        ``mime_type`` (invalid Gemini part).
"""

import pytest

from llm_proxy.models import (
    ConversationContext,
    DocumentBlock,
    DocumentSource,
    FileBlock,
    GenerationParams,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.gemini.serializer import GeminiProviderSerializer


@pytest.fixture
def serializer():
    return GeminiProviderSerializer()


def _build_body(serializer, blocks):
    request = InternalRequest(
        model="gemini-2.5-flash",
        conversation=ConversationContext(messages=[Message(role="user", content=blocks)]),
        params=GenerationParams(),
    )
    ctx = BuildContext.from_request(
        request,
        provider_name="gemini",
        target_endpoint="chat_completions",
        unknown_fields_policy="ignore",
        unsupported_block_policy="drop",
        supported_content_blocks=serializer.supported_content_blocks,
    )
    return serializer.build_provider_request(request, ctx)


class TestDocumentBlockBase64:
    def test_base64_pdf_becomes_inline_data(self, serializer):
        blocks = [
            DocumentBlock(
                source=DocumentSource(
                    type="base64",
                    data="JVBERi0=",
                    media_type="application/pdf",
                )
            )
        ]
        body = _build_body(serializer, blocks)
        parts = body["contents"][0]["parts"]
        assert len(parts) == 1
        assert parts[0] == {
            "inline_data": {
                "mime_type": "application/pdf",
                "data": "JVBERi0=",
            }
        }

    def test_base64_without_media_type_defaults_to_pdf(self, serializer):
        blocks = [
            DocumentBlock(source=DocumentSource(type="base64", data="JVBERi0=", media_type=None))
        ]
        body = _build_body(serializer, blocks)
        parts = body["contents"][0]["parts"]
        assert parts[0]["inline_data"]["mime_type"] == "application/pdf"
        assert parts[0]["inline_data"]["data"] == "JVBERi0="


class TestDocumentBlockUrl:
    def test_url_pdf_becomes_file_data(self, serializer):
        blocks = [
            DocumentBlock(
                source=DocumentSource(
                    type="url",
                    data="https://x/a.pdf",
                    media_type="application/pdf",
                )
            )
        ]
        body = _build_body(serializer, blocks)
        parts = body["contents"][0]["parts"]
        assert len(parts) == 1
        assert parts[0] == {
            "file_data": {
                "mime_type": "application/pdf",
                "file_uri": "https://x/a.pdf",
            }
        }


class TestDocumentBlockText:
    def test_text_source_stays_text_part(self, serializer):
        blocks = [
            DocumentBlock(
                source=DocumentSource(type="text", data="hello doc", media_type="text/plain")
            )
        ]
        body = _build_body(serializer, blocks)
        parts = body["contents"][0]["parts"]
        assert parts == [{"text": "hello doc"}]

    def test_document_not_silently_dropped_when_mixed_with_text(self, serializer):
        # Regression: a base64 document followed by text must not yield empty contents.
        blocks = [
            TextBlock(text="intro"),
            DocumentBlock(
                source=DocumentSource(type="base64", data="JVBERi0=", media_type="application/pdf")
            ),
        ]
        body = _build_body(serializer, blocks)
        parts = body["contents"][0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "intro"}
        assert parts[1]["inline_data"]["data"] == "JVBERi0="


class TestFileBlockDataUri:
    def test_data_uri_strips_prefix_and_adds_mime(self, serializer):
        blocks = [
            FileBlock(
                file_data="data:application/pdf;base64,JVBERi0=",
                filename="a.pdf",
            )
        ]
        body = _build_body(serializer, blocks)
        parts = body["contents"][0]["parts"]
        assert len(parts) == 1
        assert parts[0] == {
            "inline_data": {
                "mime_type": "application/pdf",
                "data": "JVBERi0=",
            }
        }


class TestFileBlockUrl:
    def test_plain_url_becomes_file_data(self, serializer):
        blocks = [FileBlock(file_data="https://x/a.pdf")]
        body = _build_body(serializer, blocks)
        parts = body["contents"][0]["parts"]
        assert parts == [{"file_data": {"file_uri": "https://x/a.pdf"}}]


class TestFileBlockId:
    def test_file_id_becomes_file_data(self, serializer):
        blocks = [FileBlock(file_id="files/abc123")]
        body = _build_body(serializer, blocks)
        parts = body["contents"][0]["parts"]
        assert parts == [{"file_data": {"file_uri": "files/abc123"}}]


class TestEmptyFile:
    def test_empty_file_block_drops_message(self, serializer):
        # An empty FileBlock yields no part; the surrounding message is dropped
        # by the empty-placeholder filter, leaving contents empty. This matches
        # the prior behaviour and is unrelated to the data: prefix bug.
        blocks = [FileBlock()]
        body = _build_body(serializer, blocks)
        assert body.get("contents") == []
        # The helper itself returns None for an empty FileBlock.
        assert serializer._make_file_part(None, None) is None
