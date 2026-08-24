"""Tests for shared content block parsers.

Validates that the shared parsing functions correctly handle all
valid OpenAI and Anthropic content block types for image, file,
and audio inputs.
"""

from llm_proxy.models import AudioBlock, FileBlock, ImageBlock, TextBlock
from llm_proxy.serialization.content_parsers import (
    parse_audio_block_openai,
    parse_file_block_anthropic,
    parse_file_block_openai,
    parse_image_block_anthropic,
    parse_image_block_openai,
    parse_text_block,
)


class TestParseTextBlock:
    def test_returns_text_block(self):
        block = parse_text_block({"type": "text", "text": "Hello"})
        assert isinstance(block, TextBlock)
        assert block.text == "Hello"

    def test_defaults_to_text_type(self):
        block = parse_text_block({"text": "No type"})
        assert isinstance(block, TextBlock)
        assert block.text == "No type"

    def test_non_text_type_returns_none(self):
        block = parse_text_block({"type": "image_url", "text": "ignored"})
        assert block is None


class TestParseImageBlockOpenAI:
    def test_non_image_type_returns_none(self):
        block = parse_image_block_openai({"type": "text", "text": "hello"})
        assert block is None

    def test_url_image(self):
        block = parse_image_block_openai(
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/img.png"},
            }
        )
        assert isinstance(block, ImageBlock)
        assert block.source.type == "url"
        assert block.source.data == "https://example.com/img.png"

    def test_base64_data_uri(self):
        block = parse_image_block_openai(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            }
        )
        assert isinstance(block, ImageBlock)
        assert block.source.type == "base64"
        assert block.source.data == "iVBORw0KGgo="
        assert block.source.media_type == "image/png"

    def test_with_detail(self):
        block = parse_image_block_openai(
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/img.png", "detail": "high"},
            }
        )
        assert block.detail == "high"

    def test_missing_image_url_dict(self):
        """When image_url key is missing, it still produces an ImageBlock with empty data."""
        block = parse_image_block_openai({"type": "image_url"})
        assert isinstance(block, ImageBlock)
        assert block.source.data == ""


class TestParseFileBlockOpenAI:
    def test_non_file_type_returns_none(self):
        block = parse_file_block_openai({"type": "text"})
        assert block is None

    def test_file_with_file_id(self):
        block = parse_file_block_openai(
            {
                "type": "file",
                "file": {"file_id": "file_abc123", "filename": "doc.pdf"},
            }
        )
        assert isinstance(block, FileBlock)
        assert block.file_id == "file_abc123"
        assert block.filename == "doc.pdf"
        assert block.file_data is None

    def test_file_with_base64_data(self):
        block = parse_file_block_openai(
            {
                "type": "file",
                "file": {"file_data": "data:application/pdf;base64,AAAA", "filename": "doc.pdf"},
            }
        )
        assert isinstance(block, FileBlock)
        assert block.file_data == "data:application/pdf;base64,AAAA"
        assert block.filename == "doc.pdf"
        assert block.file_id is None

    def test_file_data_only(self):
        block = parse_file_block_openai(
            {
                "type": "file",
                "file": {"file_data": "data:application/pdf;base64,AAAA"},
            }
        )
        assert isinstance(block, FileBlock)
        assert block.file_data == "data:application/pdf;base64,AAAA"
        assert block.filename is None
        assert block.file_id is None

    def test_missing_file_dict(self):
        block = parse_file_block_openai({"type": "file"})
        assert isinstance(block, FileBlock)
        assert block.file_data is None
        assert block.file_id is None
        assert block.filename is None

    def test_top_level_file_id(self):
        """DeepSeek-style file blocks carry file_id at the top level."""
        block = parse_file_block_openai({"type": "file", "file_id": "file-api-abc123"})
        assert isinstance(block, FileBlock)
        assert block.file_id == "file-api-abc123"
        assert block.file_data is None
        assert block.filename is None

    def test_top_level_file_data_and_filename(self):
        """DeepSeek-style inline file blocks carry file_data/filename at the top level."""
        block = parse_file_block_openai(
            {
                "type": "file",
                "file_data": "data:image/jpeg;base64,AAAA",
                "filename": "image.jpg",
            }
        )
        assert isinstance(block, FileBlock)
        assert block.file_data == "data:image/jpeg;base64,AAAA"
        assert block.filename == "image.jpg"
        assert block.file_id is None

    def test_nested_shape_wins_over_top_level(self):
        block = parse_file_block_openai(
            {
                "type": "file",
                "file": {"file_id": "file_nested"},
                "file_id": "file_top",
            }
        )
        assert isinstance(block, FileBlock)
        assert block.file_id == "file_nested"


class TestParseImageBlockAnthropic:
    def test_non_image_type_returns_none(self):
        block = parse_image_block_anthropic({"type": "text"})
        assert block is None

    def test_base64_image(self):
        block = parse_image_block_anthropic(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
            }
        )
        assert isinstance(block, ImageBlock)
        assert block.source.type == "base64"
        assert block.source.data == "AAAA"
        assert block.source.media_type == "image/jpeg"

    def test_url_image(self):
        block = parse_image_block_anthropic(
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/img.png"},
            }
        )
        assert isinstance(block, ImageBlock)
        assert block.source.type == "url"
        assert block.source.data == "https://example.com/img.png"

    def test_file_id_image(self):
        block = parse_image_block_anthropic(
            {
                "type": "image",
                "source": {"type": "file", "file_id": "file_abc", "media_type": "image/png"},
            }
        )
        assert isinstance(block, ImageBlock)
        assert block.source.type == "file_id"
        assert block.source.data == "file_abc"


class TestParseFileBlockAnthropic:
    def test_non_file_type_returns_none(self):
        block = parse_file_block_anthropic({"type": "text"})
        assert block is None

    def test_file_with_file_data(self):
        block = parse_file_block_anthropic(
            {
                "type": "file",
                "file_data": "data:application/pdf;base64,AAAA",
                "filename": "report.pdf",
            }
        )
        assert isinstance(block, FileBlock)
        assert block.file_data == "data:application/pdf;base64,AAAA"
        assert block.filename == "report.pdf"

    def test_file_with_file_id(self):
        block = parse_file_block_anthropic(
            {
                "type": "file",
                "file_id": "file_xyz",
                "filename": "data.csv",
            }
        )
        assert isinstance(block, FileBlock)
        assert block.file_id == "file_xyz"
        assert block.filename == "data.csv"

    def test_minimal_file_block(self):
        block = parse_file_block_anthropic({"type": "file"})
        assert isinstance(block, FileBlock)
        assert block.file_data is None
        assert block.file_id is None
        assert block.filename is None


class TestParseAudioBlockOpenAI:
    def test_non_audio_type_returns_none(self):
        block = parse_audio_block_openai({"type": "text"})
        assert block is None

    def test_wav_audio(self):
        block = parse_audio_block_openai(
            {
                "type": "input_audio",
                "input_audio": {"data": "AAAA", "format": "wav"},
            }
        )
        assert isinstance(block, AudioBlock)
        assert block.source.type == "base64"
        assert block.source.data == "AAAA"
        assert block.source.media_type == "audio/wav"

    def test_mp3_audio(self):
        block = parse_audio_block_openai(
            {
                "type": "input_audio",
                "input_audio": {"data": "BBBB", "format": "mp3"},
            }
        )
        assert isinstance(block, AudioBlock)
        assert block.source.media_type == "audio/mpeg"
