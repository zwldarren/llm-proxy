# tests/unit/models/test_content_blocks.py
"""Tests for ContentBlock hierarchy."""

from llm_proxy.models.content_blocks import (
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.types import (
    AudioSource,
    DocumentSource,
    ImageSource,
)


class TestContentBlock:
    """Test ContentBlock base class."""

    def test_is_dataclass(self):
        """ContentBlock should be a dataclass."""
        block = ContentBlock()
        assert isinstance(block, ContentBlock)

    def test_is_base_for_other_blocks(self):
        """ContentBlock should be the base for other block types."""
        text_block = TextBlock(text="Hello")
        assert isinstance(text_block, ContentBlock)

        img_src = ImageSource(type="url", data="http://example.com", media_type=None)
        image_block = ImageBlock(source=img_src)
        assert isinstance(image_block, ContentBlock)

        aud_src = AudioSource(type="url", data="http://example.com/audio.mp3", media_type=None)
        audio_block = AudioBlock(source=aud_src)
        assert isinstance(audio_block, ContentBlock)


class TestTextBlock:
    """Test TextBlock dataclass."""

    def test_create_basic_text(self):
        """Create TextBlock with just text."""
        block = TextBlock(text="Hello, world!")
        assert block.text == "Hello, world!"
        assert block.citations is None

    def test_create_text_with_citations(self):
        """Create TextBlock with citations."""
        citations = [
            {
                "type": "url_citation",
                "url": "http://example.com",
                "title": "Source 1",
                "start_index": 0,
                "end_index": 10,
            }
        ]
        block = TextBlock(text="According to source", citations=citations)
        assert block.text == "According to source"
        assert block.citations is not None
        assert len(block.citations) == 1
        assert block.citations[0]["type"] == "url_citation"


class TestImageBlock:
    """Test ImageBlock dataclass."""

    def test_create_with_base64(self):
        """Create ImageBlock with base64 source."""
        source = ImageSource(type="base64", data="SGVsbG8=", media_type="image/png")
        block = ImageBlock(source=source)
        assert block.source.type == "base64"
        assert block.source.data == "SGVsbG8="
        assert block.detail is None

    def test_create_with_url(self):
        """Create ImageBlock with URL source."""
        source = ImageSource(
            type="url", data="http://example.com/image.png", media_type="image/png"
        )
        block = ImageBlock(source=source, detail="low")
        assert block.source.type == "url"
        assert block.detail == "low"

    def test_create_with_file_id(self):
        """Create ImageBlock with file_id source."""
        source = ImageSource(type="file_id", data="file-123", media_type="image/png")
        block = ImageBlock(source=source, detail="high")
        assert block.source.type == "file_id"
        assert block.detail == "high"


class TestAudioBlock:
    """Test AudioBlock dataclass."""

    def test_create_with_base64(self):
        """Create AudioBlock with base64 source."""
        source = AudioSource(type="base64", data="U29uZUF1ZGlv", media_type="audio/mp3")
        block = AudioBlock(source=source)
        assert block.source.type == "base64"
        assert block.source.data == "U29uZUF1ZGlv"

    def test_create_with_url(self):
        """Create AudioBlock with URL source."""
        source = AudioSource(
            type="url", data="http://example.com/audio.mp3", media_type="audio/mp3"
        )
        block = AudioBlock(source=source)
        assert block.source.type == "url"
        assert block.source.data == "http://example.com/audio.mp3"


class TestDocumentBlock:
    """Test DocumentBlock dataclass."""

    def test_create_with_url(self):
        """Create DocumentBlock with URL source."""
        source = DocumentSource(
            type="url", data="http://example.com/doc.pdf", media_type="application/pdf"
        )
        block = DocumentBlock(source=source)
        assert block.source.type == "url"
        assert block.title is None

    def test_create_with_title(self):
        """Create DocumentBlock with title."""
        source = DocumentSource(
            type="url", data="http://example.com/doc.pdf", media_type="application/pdf"
        )
        block = DocumentBlock(source=source, title="My Document")
        assert block.title == "My Document"

    def test_create_with_file_id(self):
        """Create DocumentBlock with file_id source."""
        source = DocumentSource(type="file_id", data="doc-123", media_type="application/pdf")
        block = DocumentBlock(source=source, title="Report")
        assert block.source.type == "file_id"
        assert block.title == "Report"


class TestToolUseBlock:
    """Test ToolUseBlock dataclass."""

    def test_create(self):
        """Create ToolUseBlock."""
        block = ToolUseBlock(id="toolu_123", name="get_weather", input={"city": "San Francisco"})
        assert block.id == "toolu_123"
        assert block.name == "get_weather"
        assert block.input == {"city": "San Francisco"}

    def test_create_with_complex_input(self):
        """Create ToolUseBlock with complex input."""
        block = ToolUseBlock(
            id="toolu_456",
            name="search",
            input={"query": "test", "filters": {"type": "article", "date": "2024-01-01"}},
        )
        assert block.id == "toolu_456"
        assert block.input["filters"]["type"] == "article"

    def test_create_with_thought_signature(self):
        """Create ToolUseBlock with thought_signature in extra (Gemini)."""
        block = ToolUseBlock(
            id="call_123",
            name="get_weather",
            input={"city": "Boston"},
            extra={"thought_signature": "abc123sig"},
        )
        assert block.extra.get("thought_signature") == "abc123sig"

    def test_create_without_thought_signature_defaults_none(self):
        """ToolUseBlock thought_signature defaults to None in extra."""
        block = ToolUseBlock(id="call_123", name="get_weather", input={"city": "Boston"})
        assert block.extra.get("thought_signature") is None


class TestToolResultBlock:
    """Test ToolResultBlock dataclass."""

    def test_create_with_string_content(self):
        """Create ToolResultBlock with string content."""
        block = ToolResultBlock(tool_use_id="toolu_123", content="The weather is sunny")
        assert block.tool_use_id == "toolu_123"
        assert block.content == "The weather is sunny"
        assert block.is_error is False

    def test_create_with_list_content(self):
        """Create ToolResultBlock with list content."""
        content: list[ContentBlock] = [
            TextBlock(text="Result: 42"),
            TextBlock(text="Additional info"),
        ]
        block = ToolResultBlock(tool_use_id="toolu_123", content=content)
        assert block.tool_use_id == "toolu_123"
        assert isinstance(block.content, list)
        assert len(block.content) == 2

    def test_create_as_error(self):
        """Create ToolResultBlock as error."""
        block = ToolResultBlock(tool_use_id="toolu_123", content="Error: Not found", is_error=True)
        assert block.is_error is True


class TestThinkingBlock:
    """Test ThinkingBlock dataclass."""

    def test_create(self):
        """Create ThinkingBlock."""
        block = ThinkingBlock(thinking="Let me solve this step by step...")
        assert block.thinking == "Let me solve this step by step..."

    def test_create_with_signature(self):
        """Create ThinkingBlock with signature."""
        block = ThinkingBlock(thinking="Step-by-step...", signature="sig123")
        assert block.thinking == "Step-by-step..."
        assert block.signature == "sig123"


class TestRefusalBlock:
    """Test RefusalBlock dataclass."""

    def test_create(self):
        """Create RefusalBlock."""
        block = RefusalBlock(refusal="I cannot help with that request.")
        assert block.refusal == "I cannot help with that request."

    def test_create_with_detailed_message(self):
        """Create RefusalBlock with detailed refusal message."""
        block = RefusalBlock(refusal="I cannot provide information about sensitive topics.")
        assert "sensitive" in block.refusal
