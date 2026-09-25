"""Tests for OpenAI converter formatting of image, file, audio blocks.

Validates that content_to_openai_parts correctly converts unified content
blocks to the OpenAI wire format for chat completions.
"""

from llm_proxy.models import (
    AudioBlock,
    DocumentBlock,
    DocumentSource,
    FileBlock,
    ImageBlock,
    VideoBlock,
)
from llm_proxy.models.types import AudioSource, ImageSource, VideoSource
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.converter import content_to_openai_parts


class TestImageBlockToOpenAI:
    def test_url_image(self):
        block = ImageBlock(
            source=ImageSource(type="url", data="https://example.com/img.png", media_type=None),
        )
        parts = content_to_openai_parts([block])
        assert isinstance(parts, list)
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"] == "https://example.com/img.png"

    def test_base64_image(self):
        block = ImageBlock(
            source=ImageSource(type="base64", data="iVBORw0KGgo=", media_type="image/png"),
        )
        parts = content_to_openai_parts([block])
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"] == "data:image/png;base64,iVBORw0KGgo="

    def test_image_with_detail(self):
        block = ImageBlock(
            source=ImageSource(type="url", data="https://example.com/img.png", media_type=None),
            detail="high",
        )
        parts = content_to_openai_parts([block])
        assert parts[0]["image_url"]["detail"] == "high"

    def test_file_id_image(self):
        # Chat Completions has no file_id image source, so the block degrades to
        # text rather than sending the raw id as an ``image_url``.
        block = ImageBlock(
            source=ImageSource(type="file_id", data="file_abc", media_type="image/png"),
        )
        result = content_to_openai_parts([block])
        assert isinstance(result, str)
        assert "[Image: image/png]" in result

    def test_file_id_image_for_responses_uses_file_id(self):
        block = ImageBlock(
            source=ImageSource(type="file_id", data="file_abc", media_type="image/png"),
        )
        ctx = BuildContext(provider_name="openai", target_endpoint="responses")
        parts = content_to_openai_parts([block], ctx)
        assert parts == [{"type": "input_image", "file_id": "file_abc"}]


class TestFileBlockToDeepSeek:
    """FileBlock should emit a ``file`` part for DeepSeek.

    DeepSeek's Chat Completions endpoint accepts file blocks (Files API
    references and inline base64) for vision models, but with top-level
    keys (file_id / file_data / filename) instead of the OpenAI nested
    ``file`` object (see DeepSeek vision docs).
    """

    def test_file_id_emitted_with_top_level_keys(self):
        block = FileBlock(file_id="file_abc", filename="doc.pdf")
        ctx = BuildContext(provider_name="deepseek")
        parts = content_to_openai_parts([block], ctx)
        assert len(parts) == 1
        assert parts[0]["type"] == "file"
        assert parts[0]["file_id"] == "file_abc"
        assert parts[0]["filename"] == "doc.pdf"
        assert "file" not in parts[0]

    def test_file_data_emitted_with_top_level_keys(self):
        block = FileBlock(
            file_data="data:image/jpeg;base64,AAAA",
            filename="image.jpg",
        )
        ctx = BuildContext(provider_name="deepseek")
        parts = content_to_openai_parts([block], ctx)
        assert len(parts) == 1
        assert parts[0]["type"] == "file"
        assert parts[0]["file_data"] == "data:image/jpeg;base64,AAAA"
        assert parts[0]["filename"] == "image.jpg"

    def test_empty_file_degrades_to_text_for_deepseek(self):
        block = FileBlock()
        ctx = BuildContext(provider_name="deepseek")
        result = content_to_openai_parts([block], ctx)
        assert isinstance(result, str)
        assert "File" in result

    def test_file_degrades_to_text_for_other_providers(self):
        block = FileBlock(file_id="file_abc", filename="doc.pdf")
        ctx = BuildContext(provider_name="ollama")
        result = content_to_openai_parts([block], ctx)
        assert isinstance(result, str)
        assert "doc.pdf" in result

    def test_file_kept_as_file_for_openrouter(self):
        block = FileBlock(file_id="file_123", filename="data.csv")
        ctx = BuildContext(provider_name="openrouter")
        parts = content_to_openai_parts([block], ctx)
        assert len(parts) == 1
        assert parts[0]["type"] == "file"
        assert parts[0]["file"]["file_id"] == "file_123"
        assert parts[0]["file"]["filename"] == "data.csv"


class TestFileBlockToOpenAI:
    def test_file_with_file_id(self):
        block = FileBlock(file_id="file_abc", filename="doc.pdf")
        parts = content_to_openai_parts([block])
        assert parts[0]["type"] == "file"
        assert parts[0]["file"]["file_id"] == "file_abc"
        assert parts[0]["file"]["filename"] == "doc.pdf"

    def test_file_with_base64_data(self):
        block = FileBlock(
            file_data="data:application/pdf;base64,AAAA",
            filename="report.pdf",
        )
        parts = content_to_openai_parts([block])
        assert parts[0]["type"] == "file"
        assert parts[0]["file"]["file_data"] == "data:application/pdf;base64,AAAA"
        assert parts[0]["file"]["filename"] == "report.pdf"

    def test_file_with_all_fields(self):
        block = FileBlock(
            file_data="data:application/pdf;base64,AAAA",
            file_id="file_abc",
            filename="doc.pdf",
        )
        parts = content_to_openai_parts([block])
        assert parts[0]["type"] == "file"
        assert parts[0]["file"]["file_data"] == "data:application/pdf;base64,AAAA"
        assert parts[0]["file"]["file_id"] == "file_abc"
        assert parts[0]["file"]["filename"] == "doc.pdf"

    def test_minimal_file(self):
        block = FileBlock(file_id="file_xyz")
        parts = content_to_openai_parts([block])
        assert parts[0]["type"] == "file"
        assert parts[0]["file"]["file_id"] == "file_xyz"


class TestDocumentBlockToDeepSeek:
    """DocumentBlock should degrade to text for non-OpenAI providers like DeepSeek."""

    def test_document_base64_degrades_for_deepseek(self):
        block = DocumentBlock(
            source=DocumentSource(type="base64", data="AAAA", media_type="application/pdf"),
            title="doc.pdf",
        )
        ctx = BuildContext(provider_name="deepseek")
        result = content_to_openai_parts([block], ctx)
        assert isinstance(result, str)
        assert "doc.pdf" in result

    def test_document_url_degrades_for_deepseek(self):
        block = DocumentBlock(
            source=DocumentSource(
                type="url", data="https://example.com/doc.pdf", media_type="application/pdf"
            ),
        )
        ctx = BuildContext(provider_name="deepseek")
        result = content_to_openai_parts([block], ctx)
        assert isinstance(result, str)
        assert "Document" in result

    def test_document_file_id_degrades_for_deepseek(self):
        block = DocumentBlock(
            source=DocumentSource(type="file_id", data="file_doc_1", media_type="application/pdf"),
            title="doc.pdf",
        )
        ctx = BuildContext(provider_name="deepseek")
        result = content_to_openai_parts([block], ctx)
        assert isinstance(result, str)
        assert "doc.pdf" in result

    def test_document_text_still_text_for_deepseek(self):
        block = DocumentBlock(
            source=DocumentSource(type="text", data="Hello world", media_type="text/plain"),
        )
        ctx = BuildContext(provider_name="deepseek")
        result = content_to_openai_parts([block], ctx)
        assert isinstance(result, str)
        assert result == "Hello world"


class TestDocumentBlockToOpenRouter:
    """DocumentBlock should emit type: "file" for OpenRouter."""

    def test_document_as_file_with_base64(self):
        block = DocumentBlock(
            source=DocumentSource(type="base64", data="AAAA", media_type="application/pdf"),
            title="doc.pdf",
        )
        ctx = BuildContext(provider_name="openrouter")
        parts = content_to_openai_parts([block], ctx)
        assert parts[0]["type"] == "file"
        assert "pdf" in parts[0]["file"]["file_data"]
        assert parts[0]["file"]["filename"] == "doc.pdf"

    def test_document_as_file_with_url(self):
        block = DocumentBlock(
            source=DocumentSource(
                type="url", data="https://example.com/doc.pdf", media_type="application/pdf"
            ),
        )
        ctx = BuildContext(provider_name="openrouter")
        parts = content_to_openai_parts([block], ctx)
        assert parts[0]["type"] == "file"
        assert parts[0]["file"]["file_data"] == "https://example.com/doc.pdf"

    def test_document_as_text(self):
        block = DocumentBlock(
            source=DocumentSource(type="text", data="Hello world", media_type="text/plain"),
        )
        ctx = BuildContext(provider_name="openrouter")
        result = content_to_openai_parts([block], ctx)
        assert isinstance(result, str)
        assert result == "Hello world"


class TestDocumentBlockToOpenAI:
    def test_document_as_file_with_base64(self):
        block = DocumentBlock(
            source=DocumentSource(type="base64", data="AAAA", media_type="application/pdf"),
            title="doc.pdf",
        )
        parts = content_to_openai_parts([block])
        assert parts[0]["type"] == "file"
        assert "pdf" in parts[0]["file"]["file_data"]
        assert parts[0]["file"]["filename"] == "doc.pdf"

    def test_document_as_file_with_url_degrades(self):
        # OpenAI Chat Completions does not support file URLs
        # (https://developers.openai.com/api/docs/guides/file-inputs), so the
        # block degrades instead of carrying the URL in ``file_data``.
        block = DocumentBlock(
            source=DocumentSource(
                type="url", data="https://example.com/doc.pdf", media_type="application/pdf"
            ),
            title="doc.pdf",
        )
        result = content_to_openai_parts([block])
        assert isinstance(result, str)
        assert "doc.pdf" in result

    def test_document_as_text(self):
        block = DocumentBlock(
            source=DocumentSource(type="text", data="Hello world", media_type="text/plain"),
        )
        result = content_to_openai_parts([block])
        assert isinstance(result, str)
        assert result == "Hello world"


class TestAudioBlockToOpenAI:
    def test_audio_url(self):
        block = AudioBlock(
            source=AudioSource(
                type="url",
                data="https://example.com/audio.mp3",
                media_type="audio/mpeg",
            ),
        )
        parts = content_to_openai_parts([block])
        assert parts[0]["type"] == "audio_url"
        assert parts[0]["audio_url"]["url"] == "https://example.com/audio.mp3"

    def test_audio_base64(self):
        block = AudioBlock(
            source=AudioSource(type="base64", data="AAAA", media_type="audio/wav"),
        )
        parts = content_to_openai_parts([block])
        assert parts[0]["type"] == "audio_url"
        assert "base64" in parts[0]["audio_url"]["url"]
        assert "audio/wav" in parts[0]["audio_url"]["url"]

    def test_audio_file_id(self):
        block = AudioBlock(
            source=AudioSource(type="file_id", data="file_audio_123", media_type="audio/wav"),
        )
        parts = content_to_openai_parts([block])
        assert parts[0]["type"] == "audio_url"
        assert parts[0]["audio_url"]["url"] == "file_audio_123"


class TestAudioBlockToOpenRouter:
    """AudioBlock should emit input_audio for OpenRouter."""

    def test_audio_base64_uses_input_audio(self):
        block = AudioBlock(
            source=AudioSource(type="base64", data="AAAA", media_type="audio/wav"),
        )
        ctx = BuildContext(provider_name="openrouter")
        parts = content_to_openai_parts([block], ctx)
        assert parts[0]["type"] == "input_audio"
        assert parts[0]["input_audio"]["data"] == "AAAA"
        assert parts[0]["input_audio"]["format"] == "wav"

    def test_audio_base64_mp3_format(self):
        block = AudioBlock(
            source=AudioSource(type="base64", data="BBBB", media_type="audio/mpeg"),
        )
        ctx = BuildContext(provider_name="openrouter")
        parts = content_to_openai_parts([block], ctx)
        assert parts[0]["type"] == "input_audio"
        assert parts[0]["input_audio"]["format"] == "mp3"

    def test_audio_url_degrades_to_text(self):
        block = AudioBlock(
            source=AudioSource(
                type="url",
                data="https://example.com/audio.mp3",
                media_type="audio/mpeg",
            ),
        )
        ctx = BuildContext(provider_name="openrouter")
        result = content_to_openai_parts([block], ctx)
        assert isinstance(result, str)
        assert "Audio" in result

    def test_audio_file_id_degrades_to_text(self):
        block = AudioBlock(
            source=AudioSource(type="file_id", data="file_audio_123", media_type="audio/wav"),
        )
        ctx = BuildContext(provider_name="openrouter")
        result = content_to_openai_parts([block], ctx)
        assert isinstance(result, str)
        assert "Audio" in result


class TestVideoBlockToOpenAI:
    """VideoBlock should emit video_url for OpenAI-format providers."""

    def test_video_url(self):
        block = VideoBlock(
            source=VideoSource(
                type="url",
                data="https://example.com/video.mp4",
                media_type="video/mp4",
            ),
        )
        parts = content_to_openai_parts([block])
        assert parts[0]["type"] == "video_url"
        assert parts[0]["video_url"]["url"] == "https://example.com/video.mp4"

    def test_video_base64(self):
        block = VideoBlock(
            source=VideoSource(type="base64", data="AAAA", media_type="video/mp4"),
        )
        parts = content_to_openai_parts([block])
        assert parts[0]["type"] == "video_url"
        assert parts[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        assert parts[0]["video_url"]["url"] == "data:video/mp4;base64,AAAA"

    def test_video_file_id(self):
        """VideoBlock with file_id source degrades to text (file_id is not a URL)."""
        block = VideoBlock(
            source=VideoSource(type="file_id", data="file_video_123", media_type="video/mp4"),
        )
        result = content_to_openai_parts([block])
        assert isinstance(result, str)
        assert "Video" in result

    def test_video_with_openrouter(self):
        """VideoBlock emits video_url for OpenRouter (same as OpenAI)."""
        block = VideoBlock(
            source=VideoSource(
                type="url",
                data="https://example.com/video.mp4",
                media_type="video/mp4",
            ),
        )
        ctx = BuildContext(provider_name="openrouter")
        parts = content_to_openai_parts([block], ctx)
        assert parts[0]["type"] == "video_url"
        assert parts[0]["video_url"]["url"] == "https://example.com/video.mp4"


class TestMixedContentToOpenAI:
    def test_text_and_image(self):
        from llm_proxy.models import TextBlock

        blocks = [
            TextBlock(text="What is this?"),
            ImageBlock(
                source=ImageSource(type="url", data="https://example.com/img.png", media_type=None),
            ),
        ]
        parts = content_to_openai_parts(blocks)
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"

    def test_text_image_file_mixed(self):
        from llm_proxy.models import TextBlock

        blocks = [
            TextBlock(text="Analyze these"),
            ImageBlock(
                source=ImageSource(type="url", data="https://example.com/img.png", media_type=None),
            ),
            FileBlock(file_id="file_123", filename="data.csv"),
        ]
        parts = content_to_openai_parts(blocks)
        assert len(parts) == 3
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"
        assert parts[2]["type"] == "file"

    def test_single_text_returns_string(self):
        from llm_proxy.models import TextBlock

        result = content_to_openai_parts([TextBlock(text="Just text")])
        assert isinstance(result, str)
        assert result == "Just text"


class TestPerProviderDocumentFileMapping:
    """Documents/files map per provider; only some accept URLs or non-PDF types.

    The provider shapes are verified against official docs; the source URLs are
    recorded on ``converter._CHAT_FILE_TARGETS``.
    """

    @staticmethod
    def _doc(source_type, *, media_type=None, data="AAAA", title=None):
        return DocumentBlock(
            source=DocumentSource(type=source_type, data=data, media_type=media_type),
            title=title,
        )

    @staticmethod
    def _parts(block, provider):
        return content_to_openai_parts([block], BuildContext(provider_name=provider))

    # -- OpenAI: PDF only, never a URL --------------------------------------
    def test_openai_pdf_base64_is_a_file(self):
        parts = self._parts(
            self._doc("base64", media_type="application/pdf", title="d.pdf"), "openai"
        )
        assert parts[0]["type"] == "file"
        assert parts[0]["file"]["file_data"].startswith("data:application/pdf;base64,")

    def test_openai_non_pdf_degrades(self):
        result = self._parts(
            self._doc(
                "base64",
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                title="d.docx",
            ),
            "openai",
        )
        assert isinstance(result, str)
        assert "d.docx" in result

    def test_openai_url_degrades(self):
        result = self._parts(self._doc("url", data="https://x/d.pdf", title="d.pdf"), "openai")
        assert isinstance(result, str)

    def test_openai_file_id_kept(self):
        parts = self._parts(self._doc("file_id", data="file_1", title="d.pdf"), "openai")
        assert parts[0]["file"]["file_id"] == "file_1"

    # -- OpenRouter: URL in file_data, broad document types -----------------
    def test_openrouter_url_uses_file_data(self):
        parts = self._parts(self._doc("url", data="https://x/d.pdf"), "openrouter")
        assert parts[0]["file"]["file_data"] == "https://x/d.pdf"

    def test_openrouter_non_pdf_allowed(self):
        parts = self._parts(self._doc("base64", media_type="text/csv", title="d.csv"), "openrouter")
        assert parts[0]["type"] == "file"

    # -- Zhipu / Z.AI: dedicated file_url -----------------------------------
    def test_zai_url_uses_file_url(self):
        parts = self._parts(self._doc("url", data="https://x/d.pdf"), "zai")
        assert parts[0]["file"]["file_url"] == "https://x/d.pdf"
        assert "file_data" not in parts[0]["file"]

    def test_zhipu_url_uses_file_url(self):
        parts = self._parts(self._doc("url", data="https://x/d.pdf"), "zhipu")
        assert parts[0]["file"]["file_url"] == "https://x/d.pdf"

    # -- Mistral: flat document_url -----------------------------------------
    def test_mistral_url_uses_document_url(self):
        parts = self._parts(self._doc("url", data="https://x/d.pdf"), "mistral")
        assert parts[0] == {"type": "document_url", "document_url": "https://x/d.pdf"}

    def test_mistral_base64_uses_document_url_data_uri(self):
        parts = self._parts(self._doc("base64", media_type="application/pdf"), "mistral")
        assert parts[0]["type"] == "document_url"
        assert parts[0]["document_url"].startswith("data:application/pdf;base64,")

    # -- DeepSeek: images only, no URL --------------------------------------
    def test_deepseek_document_degrades(self):
        result = self._parts(
            self._doc("base64", media_type="application/pdf", title="d.pdf"), "deepseek"
        )
        assert isinstance(result, str)

    def test_deepseek_image_file_is_top_level(self):
        parts = self._parts(
            FileBlock(file_data="data:image/png;base64,AAAA", filename="i.png"), "deepseek"
        )
        assert parts[0]["type"] == "file"
        assert parts[0]["file_data"] == "data:image/png;base64,AAAA"
        assert "file" not in parts[0]  # top-level keys, not nested under ``file``

    def test_deepseek_pdf_file_degrades(self):
        result = self._parts(
            FileBlock(file_data="data:application/pdf;base64,AAAA", filename="d.pdf"), "deepseek"
        )
        assert isinstance(result, str)

    def test_deepseek_file_url_degrades(self):
        result = self._parts(FileBlock(file_url="https://x/d.png", filename="d.png"), "deepseek")
        assert isinstance(result, str)

    # -- Unknown providers stay conservative ---------------------------------
    def test_unknown_provider_degrades(self):
        result = self._parts(
            self._doc("base64", media_type="application/pdf", title="d.pdf"), "qwen"
        )
        assert isinstance(result, str)


class TestImageDetailOmittedWhenAbsent:
    def test_no_detail_key(self):
        block = ImageBlock(source=ImageSource(type="url", data="https://x/i.png", media_type=None))
        parts = content_to_openai_parts([block])
        assert "detail" not in parts[0]["image_url"]

    def test_detail_preserved(self):
        block = ImageBlock(
            source=ImageSource(type="url", data="https://x/i.png", media_type=None),
            detail="low",
        )
        parts = content_to_openai_parts([block])
        assert parts[0]["image_url"]["detail"] == "low"
