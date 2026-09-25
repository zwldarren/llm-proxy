"""Tests for Chat Completions URL download-and-inline and the GLM file/media split.

Two provider-specific behaviours:
- Zhipu / Z.AI GLM reject a ``file`` part in the same message as an
  ``image_url`` / ``video_url`` part, so ``format_conversation`` splits it.
- OpenAI and the generic ``openai-compatible`` adapter accept only inline base64
  on their ``file`` part, so URL document/file sources are downloaded and
  inlined before the body is built.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.models import (
    ConversationContext,
    DocumentBlock,
    DocumentSource,
    FileBlock,
    ImageBlock,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.models.types import ImageSource
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai import url_inline
from llm_proxy.serialization.openai.url_inline import inline_chat_document_urls

# ---------------------------------------------------------------------------
# GLM file/media split
# ---------------------------------------------------------------------------


def _pdf_block(title: str = "doc.pdf") -> DocumentBlock:
    return DocumentBlock(
        source=DocumentSource(type="base64", data="JVBERi0=", media_type="application/pdf"),
        title=title,
    )


def _image_block() -> ImageBlock:
    return ImageBlock(source=ImageSource(type="url", data="https://ex.com/a.png", media_type=None))


def _format(blocks, provider_name):
    from llm_proxy.serialization.openai.converter import format_conversation

    conv = ConversationContext(messages=[Message(role="user", content=list(blocks))])
    return format_conversation(conv, BuildContext(provider_name=provider_name))


class TestGlmFileMediaSplit:
    def test_zai_splits_file_and_image(self):
        messages = _format([TextBlock(text="look"), _pdf_block(), _image_block()], "zai")
        assert [m["role"] for m in messages] == ["user", "user"]
        first_types = [p["type"] for p in messages[0]["content"]]
        second_types = [p["type"] for p in messages[1]["content"]]
        assert "file" in first_types and "image_url" not in first_types
        assert second_types == ["image_url"]
        # Leading text stays with the file part (order preserved).
        assert first_types == ["text", "file"]

    def test_zhipu_splits_file_and_video(self):
        from llm_proxy.models import VideoBlock
        from llm_proxy.models.types import VideoSource

        video = VideoBlock(
            source=VideoSource(type="url", data="https://ex.com/v.mp4", media_type=None)
        )
        messages = _format([_pdf_block(), video], "zhipu")
        assert len(messages) == 2
        assert [p["type"] for p in messages[0]["content"]] == ["file"]
        assert [p["type"] for p in messages[1]["content"]] == ["video_url"]

    def test_text_between_media_stays_with_neighbour(self):
        messages = _format(
            [_pdf_block(), TextBlock(text="mid"), _image_block(), TextBlock(text="end")],
            "zai",
        )
        assert [[p["type"] for p in m["content"]] for m in messages] == [
            ["file", "text"],
            ["image_url", "text"],
        ]

    def test_non_glm_provider_not_split(self):
        messages = _format([_pdf_block(), _image_block()], "openai")
        assert len(messages) == 1
        assert [p["type"] for p in messages[0]["content"]] == ["file", "image_url"]

    def test_glm_message_without_file_not_split(self):
        messages = _format([TextBlock(text="hi"), _image_block()], "zai")
        assert len(messages) == 1
        assert [p["type"] for p in messages[0]["content"]] == ["text", "image_url"]


# ---------------------------------------------------------------------------
# URL download-and-inline
# ---------------------------------------------------------------------------


def _request_with_blocks(blocks) -> InternalRequest:
    return InternalRequest(
        model="m",
        conversation=ConversationContext(messages=[Message(role="user", content=list(blocks))]),
    )


@pytest.fixture
def fake_download(monkeypatch):
    """Patch the downloader to return a canned data URL per call."""
    calls: list[str] = []

    async def _download(_client, url, **_kwargs):
        calls.append(url)
        if "fail" in url:
            raise RuntimeError("boom")
        if url.endswith(".docx"):
            return (
                "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,AAAA",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        return "data:application/pdf;base64,JVBERi0=", "application/pdf"

    monkeypatch.setattr(url_inline, "download_image_as_base64", _download)
    return calls


class TestInlineChatDocumentUrls:
    async def test_inlines_anthropic_url_document_for_openai(self, fake_download):
        request = _request_with_blocks(
            [DocumentBlock(source=DocumentSource(type="url", data="https://ex.com/doc.pdf"))]
        )
        inlined = await inline_chat_document_urls(
            request, provider_type="openai", client=MagicMock()
        )
        assert inlined == 1
        source = request.conversation.messages[0].content[0].source
        assert source.type == "base64"
        assert source.data == "JVBERi0="
        assert source.media_type == "application/pdf"

    async def test_inlines_file_url_for_openai_compatible(self, fake_download):
        request = _request_with_blocks(
            [FileBlock(file_url="https://ex.com/doc.pdf", filename="doc.pdf")]
        )
        inlined = await inline_chat_document_urls(
            request, provider_type="openai-compatible", client=MagicMock()
        )
        assert inlined == 1
        block = request.conversation.messages[0].content[0]
        assert block.file_url is None
        assert block.file_data == "data:application/pdf;base64,JVBERi0="

    async def test_url_file_data_treated_as_url(self, fake_download):
        request = _request_with_blocks(
            [FileBlock(file_data="https://ex.com/doc.pdf", filename="doc.pdf")]
        )
        inlined = await inline_chat_document_urls(
            request, provider_type="openai", client=MagicMock()
        )
        assert inlined == 1
        assert (
            request.conversation.messages[0].content[0].file_data.startswith("data:application/pdf")
        )

    async def test_openrouter_keeps_url(self, fake_download):
        request = _request_with_blocks(
            [DocumentBlock(source=DocumentSource(type="url", data="https://ex.com/doc.pdf"))]
        )
        inlined = await inline_chat_document_urls(
            request, provider_type="openrouter", client=MagicMock()
        )
        assert inlined == 0
        assert fake_download == []  # no fetch at all
        assert request.conversation.messages[0].content[0].source.type == "url"

    async def test_non_pdf_not_inlined_for_openai(self, fake_download):
        request = _request_with_blocks(
            [DocumentBlock(source=DocumentSource(type="url", data="https://ex.com/doc.docx"))]
        )
        inlined = await inline_chat_document_urls(
            request, provider_type="openai", client=MagicMock()
        )
        assert inlined == 0
        # Left as a URL so the converter's normal degrade path handles it.
        assert request.conversation.messages[0].content[0].source.type == "url"

    async def test_download_failure_leaves_source(self, fake_download):
        request = _request_with_blocks(
            [DocumentBlock(source=DocumentSource(type="url", data="https://ex.com/fail.pdf"))]
        )
        inlined = await inline_chat_document_urls(
            request, provider_type="openai", client=MagicMock()
        )
        assert inlined == 0
        assert request.conversation.messages[0].content[0].source.type == "url"

    async def test_dedupes_repeated_urls(self, fake_download):
        doc = DocumentBlock(source=DocumentSource(type="url", data="https://ex.com/doc.pdf"))
        request = _request_with_blocks([doc, doc])
        inlined = await inline_chat_document_urls(
            request, provider_type="openai", client=MagicMock()
        )
        assert inlined == 2
        assert fake_download == ["https://ex.com/doc.pdf"]

    def test_gate_helper_matches_table(self):
        from llm_proxy.serialization.openai.converter import (
            chat_document_url_needs_download,
            chat_file_media_type_accepted,
        )

        assert chat_document_url_needs_download("openai") is True
        assert chat_document_url_needs_download("openai-compatible") is True
        assert chat_document_url_needs_download("openrouter") is False
        assert chat_document_url_needs_download("deepseek") is False
        assert chat_document_url_needs_download("qwen") is False
        assert chat_file_media_type_accepted("openai", "application/pdf") is True
        assert chat_file_media_type_accepted("openai", "text/csv") is False


# ---------------------------------------------------------------------------
# Adapter hook
# ---------------------------------------------------------------------------


class TestAdapterPrepareChatRequest:
    def _adapter(self, provider_type: str):
        from llm_proxy.core.adapter import get_adapter

        return get_adapter(provider_type, provider_name="Label", api_key="k")

    async def test_openai_compatible_adapter_inlines(self, monkeypatch, fake_download):
        adapter = self._adapter("openai-compatible")
        adapter._get_client = AsyncMock(return_value=MagicMock())
        request = _request_with_blocks(
            [DocumentBlock(source=DocumentSource(type="url", data="https://ex.com/doc.pdf"))]
        )
        await adapter._prepare_chat_request(request)
        assert request.conversation.messages[0].content[0].source.type == "base64"

    async def test_openrouter_adapter_skips(self, monkeypatch, fake_download):
        adapter = self._adapter("openrouter")
        adapter._get_client = AsyncMock(return_value=MagicMock())
        request = _request_with_blocks(
            [DocumentBlock(source=DocumentSource(type="url", data="https://ex.com/doc.pdf"))]
        )
        await adapter._prepare_chat_request(request)
        assert fake_download == []
        assert request.conversation.messages[0].content[0].source.type == "url"
