"""Tests for OpenAIResponseParser multimodal output handling.

Validates that message.images and message.audio are correctly parsed
into content blocks (ImageBlock, AudioBlock) for image generation and
audio output via chat completions (e.g., OpenRouter).
"""

from llm_proxy.models import AudioBlock, ImageBlock, TextBlock
from llm_proxy.serialization.openai.components.response_parser import (
    OpenAIResponseParser,
)


class TestOpenAIResponseParserImages:
    """Tests for message.images parsing in response parser."""

    def test_images_converted_to_image_blocks(self):
        """message.images produces ImageBlock entries."""
        parser = OpenAIResponseParser()
        message = {
            "role": "assistant",
            "content": "Here is your image.",
            "images": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgo=",
                    },
                }
            ],
        }
        output = parser._parse_message_content(message)
        image_blocks = [b for b in output if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 1
        assert image_blocks[0].source.type == "base64"
        assert "iVBORw0KGgo=" in image_blocks[0].source.data

    def test_multiple_images(self):
        """Multiple images in message.images all produce ImageBlocks."""
        parser = OpenAIResponseParser()
        message = {
            "role": "assistant",
            "content": "Here are two images.",
            "images": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,BBBB"},
                },
            ],
        }
        output = parser._parse_message_content(message)
        image_blocks = [b for b in output if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 2
        assert image_blocks[0].source.data == "AAAA"
        assert image_blocks[1].source.data == "BBBB"

    def test_images_with_url(self):
        """Image with URL source produces ImageBlock with url type."""
        parser = OpenAIResponseParser()
        message = {
            "role": "assistant",
            "content": "",
            "images": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.png",
                    },
                }
            ],
        }
        output = parser._parse_message_content(message)
        image_blocks = [b for b in output if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 1
        assert image_blocks[0].source.type == "url"
        assert image_blocks[0].source.data == "https://example.com/image.png"

    def test_images_without_text(self):
        """message.images without text content still produces ImageBlocks."""
        parser = OpenAIResponseParser()
        message = {
            "role": "assistant",
            "content": None,
            "images": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                }
            ],
        }
        output = parser._parse_message_content(message)
        image_blocks = [b for b in output if isinstance(b, ImageBlock)]
        assert len(image_blocks) == 1
        # TextBlock should not be created for None content
        text_blocks = [b for b in output if isinstance(b, TextBlock)]
        assert len(text_blocks) == 0

    def test_no_images_field(self):
        """Absence of images field should not affect normal parsing."""
        parser = OpenAIResponseParser()
        message = {
            "role": "assistant",
            "content": "Just text.",
        }
        output = parser._parse_message_content(message)
        text_blocks = [b for b in output if isinstance(b, TextBlock)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "Just text."


class TestOpenAIResponseParserAudio:
    """Tests for message.audio parsing in response parser."""

    def test_audio_converted_to_audio_block(self):
        """message.audio produces AudioBlock."""
        parser = OpenAIResponseParser()
        message = {
            "role": "assistant",
            "content": "Here is your audio.",
            "audio": {
                "data": "AAAA",
                "transcript": "Hello world",
                "id": "audio_123",
                "expires_at": 1234567890,
            },
        }
        output = parser._parse_message_content(message)
        audio_blocks = [b for b in output if isinstance(b, AudioBlock)]
        assert len(audio_blocks) == 1
        assert audio_blocks[0].source.data == "AAAA"
        assert audio_blocks[0].source.transcript == "Hello world"
        assert audio_blocks[0].source.id == "audio_123"
        # expires_at is intentionally not preserved in the AudioBlock source model;
        # it is available in the raw response for cache management if needed.

    def test_audio_without_transcript(self):
        """message.audio without transcript still produces AudioBlock."""
        parser = OpenAIResponseParser()
        message = {
            "role": "assistant",
            "audio": {
                "data": "BBBB",
            },
        }
        output = parser._parse_message_content(message)
        audio_blocks = [b for b in output if isinstance(b, AudioBlock)]
        assert len(audio_blocks) == 1
        assert audio_blocks[0].source.data == "BBBB"
        assert audio_blocks[0].source.transcript is None
        assert audio_blocks[0].source.id is None
