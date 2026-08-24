"""Tests for OpenAI parse_content_blocks handling of file and image content.

Validates that the OpenAI protocol serializer correctly parses image_url,
file, input_audio, and other content types from chat completion requests.
"""

import pytest

from llm_proxy.models import AudioBlock, FileBlock, ImageBlock, TextBlock
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer


@pytest.fixture
def serializer():
    return OpenAIProtocolSerializer()


class TestParseImageUrl:
    def test_http_url_image(self, serializer):
        content = [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        images = [b for b in blocks if isinstance(b, ImageBlock)]
        assert len(images) == 1
        assert images[0].source.type == "url"
        assert images[0].source.data == "https://example.com/img.png"

    def test_base64_data_uri_image(self, serializer):
        content = [
            {"type": "text", "text": "What is this?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA"},
            },
        ]
        blocks = serializer.parse_content_blocks(content)
        images = [b for b in blocks if isinstance(b, ImageBlock)]
        assert len(images) == 1
        assert images[0].source.type == "base64"
        assert images[0].source.data == "iVBORw0KGgoAAAANSUhEUgAA"

    def test_multiple_images(self, serializer):
        content = [
            {"type": "text", "text": "Compare these:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/c.png"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        images = [b for b in blocks if isinstance(b, ImageBlock)]
        assert len(images) == 3

    def test_image_with_detail(self, serializer):
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/img.png", "detail": "low"},
            }
        ]
        blocks = serializer.parse_content_blocks(content)
        images = [b for b in blocks if isinstance(b, ImageBlock)]
        assert len(images) == 1
        assert images[0].detail == "low"


class TestParseFileBlock:
    def test_file_with_file_id(self, serializer):
        content = [
            {"type": "text", "text": "Analyze this file"},
            {"type": "file", "file": {"file_id": "file_abc123", "filename": "report.pdf"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        files = [b for b in blocks if isinstance(b, FileBlock)]
        assert len(files) == 1
        assert files[0].file_id == "file_abc123"
        assert files[0].filename == "report.pdf"

    def test_file_with_base64_data(self, serializer):
        content = [
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,AAAA"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        files = [b for b in blocks if isinstance(b, FileBlock)]
        assert len(files) == 1
        assert files[0].file_data == "data:application/pdf;base64,AAAA"

    def test_file_without_filename(self, serializer):
        content = [
            {"type": "file", "file": {"file_id": "file_xyz"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        files = [b for b in blocks if isinstance(b, FileBlock)]
        assert len(files) == 1
        assert files[0].file_id == "file_xyz"
        assert files[0].filename is None


class TestParseInputAudio:
    def test_input_audio_wav(self, serializer):
        content = [
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        audios = [b for b in blocks if isinstance(b, AudioBlock)]
        assert len(audios) == 1
        assert audios[0].source.type == "base64"
        assert audios[0].source.data == "AAAA"

    def test_input_audio_mp3(self, serializer):
        content = [
            {"type": "input_audio", "input_audio": {"data": "BBBB", "format": "mp3"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        audios = [b for b in blocks if isinstance(b, AudioBlock)]
        assert len(audios) == 1
        assert audios[0].source.media_type == "audio/mpeg"


class TestParseMixedContent:
    def test_text_image_file_mixed(self, serializer):
        content = [
            {"type": "text", "text": "Look at this image and file"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            {"type": "file", "file": {"file_id": "file_123", "filename": "data.csv"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        texts = [b for b in blocks if isinstance(b, TextBlock)]
        images = [b for b in blocks if isinstance(b, ImageBlock)]
        files = [b for b in blocks if isinstance(b, FileBlock)]
        assert len(texts) == 1
        assert len(images) == 1
        assert len(files) == 1

    def test_none_content(self, serializer):
        blocks = serializer.parse_content_blocks(None)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == ""

    def test_string_content(self, serializer):
        blocks = serializer.parse_content_blocks("Just text")
        assert len(blocks) == 1
        assert blocks[0].text == "Just text"

    def test_refusal_content(self, serializer):
        content = [{"type": "refusal", "refusal": "I cannot do that"}]
        blocks = serializer.parse_content_blocks(content)
        from llm_proxy.models import RefusalBlock

        refusals = [b for b in blocks if isinstance(b, RefusalBlock)]
        assert len(refusals) == 1
        assert refusals[0].refusal == "I cannot do that"


class TestParseMultipleFileBlocks:
    def test_multiple_files(self, serializer):
        content = [
            {"type": "text", "text": "Analyze these files"},
            {"type": "file", "file": {"file_id": "file_1", "filename": "doc1.pdf"}},
            {"type": "file", "file": {"file_id": "file_2", "filename": "doc2.pdf"}},
            {"type": "file", "file": {"file_id": "file_3", "filename": "doc3.csv"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        files = [b for b in blocks if isinstance(b, FileBlock)]
        assert len(files) == 3
        assert files[0].file_id == "file_1"
        assert files[1].file_id == "file_2"
        assert files[2].file_id == "file_3"

    def test_multiple_files_with_mixed_sources(self, serializer):
        content = [
            {"type": "file", "file": {"file_id": "file_abc", "filename": "uploaded.pdf"}},
            {
                "type": "file",
                "file": {"file_data": "data:application/pdf;base64,AAAA", "filename": "inline.pdf"},
            },
            {"type": "file", "file": {"file_data": "data:text/csv;base64,BBBB"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        files = [b for b in blocks if isinstance(b, FileBlock)]
        assert len(files) == 3
        assert files[0].file_id == "file_abc"
        assert files[1].file_data == "data:application/pdf;base64,AAAA"
        assert files[2].file_data == "data:text/csv;base64,BBBB"


class TestParseContentInUserMessage:
    """Full message-level parse through parse_conversation for image/file content."""

    def test_user_message_with_image(self, serializer):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        ctx = serializer.parse_request({"model": "gpt-4", "messages": messages})
        msg = ctx.conversation.messages[0]
        images = [b for b in msg.content if isinstance(b, ImageBlock)]
        assert len(images) == 1
        assert images[0].source.type == "url"

    def test_user_message_with_multiple_images_and_file(self, serializer):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}},
                    {"type": "file", "file": {"file_id": "file_ref", "filename": "notes.pdf"}},
                ],
            }
        ]
        ctx = serializer.parse_request({"model": "gpt-4", "messages": messages})
        msg = ctx.conversation.messages[0]
        images = [b for b in msg.content if isinstance(b, ImageBlock)]
        files = [b for b in msg.content if isinstance(b, FileBlock)]
        assert len(images) == 2
        assert len(files) == 1
        assert files[0].file_id == "file_ref"

    def test_developer_message_with_text_only(self, serializer):
        """Developer messages extract only text (images not expected in developer role)."""
        messages = [
            {
                "role": "developer",
                "content": [{"type": "text", "text": "You are a helpful assistant"}],
            }
        ]
        ctx = serializer.parse_request({"model": "gpt-4", "messages": messages})
        assert len(ctx.conversation.system_messages) == 1
        assert ctx.conversation.system_messages[0].text_content == "You are a helpful assistant"


class TestFullRoundTripParseAndConvert:
    """Parse then convert back to OpenAI format — ensures serialization cycle works."""

    def test_image_url_round_trip(self, serializer):
        from llm_proxy.serialization.openai.converter import content_to_openai_parts

        content = [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        parts = content_to_openai_parts(blocks)
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"] == "https://example.com/img.png"

    def test_base64_image_round_trip(self, serializer):
        from llm_proxy.serialization.openai.converter import content_to_openai_parts

        content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA"},
            }
        ]
        blocks = serializer.parse_content_blocks(content)
        parts = content_to_openai_parts(blocks)
        assert parts[0]["type"] == "image_url"
        url = parts[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert "iVBORw0KGgoAAAANSUhEUgAA" in url

    def test_image_with_detail_round_trip(self, serializer):
        from llm_proxy.serialization.openai.converter import content_to_openai_parts

        content = [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/img.png", "detail": "high"},
            }
        ]
        blocks = serializer.parse_content_blocks(content)
        parts = content_to_openai_parts(blocks)
        assert parts[0]["image_url"]["detail"] == "high"

    def test_file_round_trip(self, serializer):
        from llm_proxy.serialization.openai.converter import content_to_openai_parts

        content = [
            {"type": "file", "file": {"file_id": "file_abc", "filename": "doc.pdf"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        parts = content_to_openai_parts(blocks)
        assert parts[0]["type"] == "file"
        assert parts[0]["file"]["file_id"] == "file_abc"
        assert parts[0]["file"]["filename"] == "doc.pdf"

    def test_file_base64_round_trip(self, serializer):
        from llm_proxy.serialization.openai.converter import content_to_openai_parts

        content = [
            {
                "type": "file",
                "file": {"file_data": "data:application/pdf;base64,AAAA", "filename": "report.pdf"},
            },
        ]
        blocks = serializer.parse_content_blocks(content)
        parts = content_to_openai_parts(blocks)
        assert parts[0]["type"] == "file"
        assert parts[0]["file"]["file_data"] == "data:application/pdf;base64,AAAA"

    def test_mixed_content_round_trip(self, serializer):
        from llm_proxy.serialization.openai.converter import content_to_openai_parts

        content = [
            {"type": "text", "text": "Analyze these"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            {"type": "file", "file": {"file_id": "file_123", "filename": "data.csv"}},
        ]
        blocks = serializer.parse_content_blocks(content)
        parts = content_to_openai_parts(blocks)
        assert len(parts) == 3
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"
        assert parts[2]["type"] == "file"


class TestParseMarkdownImages:
    """Markdown image syntax is NOT parsed by the OpenAI protocol serializer.

    ``![alt](url)`` in string content is always treated as plain text.
    Only explicit ``image_url`` content blocks create ImageBlocks.
    Markdown image parsing for ``data:image/…`` URIs exists exclusively
    in the Gemini conversation converter (nano-banana support).
    """

    def test_markdown_data_uri_stays_as_text(self, serializer):
        """Even data:image URIs in markdown stay as plain text."""
        blocks = serializer.parse_content_blocks("![image](data:image/png;base64,AAAA)")
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "![image](data:image/png;base64,AAAA)"

    def test_markdown_http_url_stays_as_text(self, serializer):
        """HTTP URLs in markdown stay as plain text."""
        blocks = serializer.parse_content_blocks("![img](https://example.com/photo.png)")
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "![img](https://example.com/photo.png)"

    def test_plain_string_no_images(self, serializer):
        """Plain text without markdown syntax is unchanged."""
        blocks = serializer.parse_content_blocks("Just plain text")
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "Just plain text"

    def test_documentation_text_stays_as_text(self, serializer):
        """Documentation mentioning markdown syntax stays as a single text block."""
        blocks = serializer.parse_content_blocks(
            "Use the ![alt](https://example.com) syntax for images"
        )
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "Use the ![alt](https://example.com) syntax for images"

    def test_markdown_in_content_array_stays_as_text(self, serializer):
        """Markdown in array string elements stays as text."""
        blocks = serializer.parse_content_blocks(["Look: ![img](data:image/png;base64,AAAA)"])
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].text == "Look: ![img](data:image/png;base64,AAAA)"

    def test_explicit_image_url_still_works(self, serializer):
        """Explicit image_url content blocks are still parsed correctly."""
        blocks = serializer.parse_content_blocks(
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        )
        assert len(blocks) == 1
        assert isinstance(blocks[0], ImageBlock)
        assert blocks[0].source.data == "AAAA"
