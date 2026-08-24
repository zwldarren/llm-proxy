# tests/unit/models/test_types.py
from llm_proxy.models.types import (
    Annotation,
    AudioSource,
    DocumentSource,
    ImageSource,
    ResponseFormat,
    ResponseStatus,
    StreamOptions,
    ThinkingConfig,
    UrlCitation,
    Usage,
)


class TestImageSource:
    def test_create_with_base64(self):
        source = ImageSource(type="base64", data="SGVsbG8gV29ybGQ=", media_type="image/png")
        assert source.type == "base64"
        assert source.data == "SGVsbG8gV29ybGQ="
        assert source.media_type == "image/png"

    def test_create_with_url(self):
        source = ImageSource(type="url", data="https://example.com/image.png", media_type=None)
        assert source.type == "url"
        assert source.data == "https://example.com/image.png"
        assert source.media_type is None

    def test_create_with_file_id(self):
        source = ImageSource(type="file_id", data="file-123456", media_type="image/png")
        assert source.type == "file_id"
        assert source.data == "file-123456"
        assert source.media_type == "image/png"


class TestAudioSource:
    def test_create_with_base64(self):
        source = AudioSource(type="base64", data="U29tZUF1ZGlv", media_type="audio/mp3")
        assert source.type == "base64"
        assert source.data == "U29tZUF1ZGlv"
        assert source.media_type == "audio/mp3"

    def test_create_with_url(self):
        source = AudioSource(type="url", data="https://example.com/audio.mp3", media_type=None)
        assert source.type == "url"
        assert source.data == "https://example.com/audio.mp3"
        assert source.media_type is None

    def test_create_with_file_id(self):
        source = AudioSource(type="file_id", data="audio-123456", media_type="audio/wav")
        assert source.type == "file_id"
        assert source.data == "audio-123456"
        assert source.media_type == "audio/wav"


class TestDocumentSource:
    def test_create_with_base64(self):
        source = DocumentSource(
            type="base64", data="SGVsbG9Eb2N1bWVudA==", media_type="application/pdf"
        )
        assert source.type == "base64"
        assert source.data == "SGVsbG9Eb2N1bWVudA=="
        assert source.media_type == "application/pdf"

    def test_create_with_url(self):
        source = DocumentSource(type="url", data="https://example.com/doc.pdf", media_type=None)
        assert source.type == "url"
        assert source.data == "https://example.com/doc.pdf"
        assert source.media_type is None

    def test_create_with_file_id(self):
        source = DocumentSource(type="file_id", data="doc-123456", media_type="application/pdf")
        assert source.type == "file_id"
        assert source.data == "doc-123456"
        assert source.media_type == "application/pdf"


class TestAnnotation:
    def test_create_url_citation(self):
        annotation = Annotation(
            type="url_citation",
            url_citation=UrlCitation(
                url="https://example.com/source",
                title="Source at example.com",
                start_index=0,
                end_index=10,
            ),
        )
        assert annotation.type == "url_citation"
        assert annotation.url_citation is not None
        assert annotation.url_citation.url == "https://example.com/source"
        assert annotation.url_citation.title == "Source at example.com"

    def test_create_file_citation(self):
        annotation = Annotation(
            type="file_citation",
            file_citation={"file_id": "file-123456", "filename": "doc.pdf"},
        )
        assert annotation.type == "file_citation"
        assert annotation.file_citation is not None
        assert annotation.file_citation["file_id"] == "file-123456"


class TestUsage:
    def test_default_usage(self):
        usage = Usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0  # Computed

    def test_usage_with_all_fields(self):
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=20,
            reasoning_tokens=10,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_input_tokens == 20
        assert usage.reasoning_tokens == 10

    def test_total_tokens_computed(self):
        usage = Usage(input_tokens=100, output_tokens=50)
        # total_tokens should be computed if not provided
        assert usage.total_tokens == 150  # 100 + 50

    def test_total_tokens_explicit(self):
        usage = Usage(input_tokens=100, output_tokens=50, total_tokens=200)
        # Explicit total_tokens is preserved
        assert usage.total_tokens == 200


class TestResponseFormat:
    def test_create_text_format(self):
        fmt = ResponseFormat(type="text")
        assert fmt.type == "text"
        assert fmt.json_schema is None

    def test_create_json_object_format(self):
        fmt = ResponseFormat(type="json_object")
        assert fmt.type == "json_object"
        assert fmt.json_schema is None

    def test_create_json_schema_format(self):
        fmt = ResponseFormat(
            type="json_schema",
            json_schema={
                "name": "person",
                "description": "A person schema",
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                "strict": True,
            },
        )
        assert fmt.type == "json_schema"
        assert fmt.json_schema is not None
        assert fmt.json_schema.get("name") == "person"
        assert fmt.json_schema.get("strict") is True


class TestStreamOptions:
    def test_default_options(self):
        opts = StreamOptions()
        assert opts.include_usage is False

    def test_explicit_include_usage(self):
        opts = StreamOptions(include_usage=False)
        assert opts.include_usage is False


class TestThinkingConfig:
    def test_create_enabled(self):
        config = ThinkingConfig(type="enabled", budget_tokens=1000)
        assert config.type == "enabled"
        assert config.budget_tokens == 1000

    def test_create_disabled(self):
        config = ThinkingConfig(type="disabled", budget_tokens=None)
        assert config.type == "disabled"
        assert config.budget_tokens is None


class TestResponseStatus:
    def test_valid_status_values(self):
        completed: ResponseStatus = "completed"
        incomplete: ResponseStatus = "incomplete"
        error: ResponseStatus = "error"
        assert completed == "completed"
        assert incomplete == "incomplete"
        assert error == "error"

    def test_status_type_alias(self):
        def accept_status(status: ResponseStatus) -> ResponseStatus:
            return status

        assert accept_status("completed") == "completed"
        assert accept_status("incomplete") == "incomplete"
        assert accept_status("error") == "error"
