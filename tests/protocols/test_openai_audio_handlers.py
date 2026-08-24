"""Tests for OpenAI audio protocol handlers and serializers."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.datastructures import UploadFile

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models import (
    InternalSpeechRequest,
    InternalSpeechResponse,
    InternalTranscriptionRequest,
    InternalTranscriptionResponse,
    InternalTranslationRequest,
    InternalTranslationResponse,
)
from llm_proxy.protocols.openai.audio_serializer import OpenAIAudioSerializer
from llm_proxy.protocols.openai.audio_speech_handler import speech_protocol
from llm_proxy.protocols.openai.audio_transcription_handler import (
    _TranscriptionRequestWrapper,
    transcription_protocol,
)
from llm_proxy.protocols.openai.audio_translation_handler import (
    _TranslationRequestWrapper,
    translation_protocol,
)

_serializer = OpenAIAudioSerializer()


class TestSpeechProtocolEndpoint:
    """Tests for SpeechProtocolEndpoint."""

    def test_name(self):
        """Test handler name."""
        assert speech_protocol.name == "speech"

    def test_paths(self):
        """Test handler paths."""
        assert speech_protocol.paths == ["/v1/audio/speech"]

    def test_parse_request(self):
        """Test parsing speech request."""
        from llm_proxy.protocols.openai.schemas import SpeechRequestSchema

        schema = SpeechRequestSchema(
            model="tts-1",
            input="Hello",
            voice="alloy",
            response_format="mp3",
        )
        result = _serializer.parse_request(schema.model_dump(exclude_none=True))

        assert isinstance(result, InternalSpeechRequest)
        assert result.model == "tts-1"
        assert result.input == "Hello"
        assert result.voice == "alloy"
        assert result.response_format == "mp3"

    def test_format_response(self):
        """Test formatting speech response."""
        response = InternalSpeechResponse(
            content=b"audio-data",
            content_type="audio/mpeg",
        )
        result = _serializer.format_response(response)

        assert result["content"] == b"audio-data"
        assert result["content_type"] == "audio/mpeg"


class TestTranscriptionProtocolEndpoint:
    """Tests for TranscriptionProtocolEndpoint."""

    def test_name(self):
        """Test handler name."""
        assert transcription_protocol.name == "transcription"

    def test_paths(self):
        """Test handler paths."""
        assert transcription_protocol.paths == ["/v1/audio/transcriptions"]

    def test_request_model_is_none(self):
        """Test that request_model returns None for multipart."""
        assert transcription_protocol.request_model is None

    @pytest.mark.asyncio
    async def test_parse_http_request(self):
        """Test parsing multipart HTTP request."""
        mock_request = MagicMock()

        # Build a mock form with file and fields
        form_data = {
            "model": "whisper-1",
            "file": UploadFile(filename="audio.mp3", file=MagicMock()),
            "language": "en",
            "prompt": "Hello prompt",
            "response_format": "json",
            "temperature": "0.5",
            "stream": "true",
            "timestamp_granularities[]": "word",
            "include[]": "logprobs",
        }

        # Multi-items for list fields
        multi_items = [
            ("model", "whisper-1"),
            ("file", form_data["file"]),
            ("language", "en"),
            ("prompt", "Hello prompt"),
            ("response_format", "json"),
            ("temperature", "0.5"),
            ("stream", "true"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
            ("include[]", "logprobs"),
        ]

        mock_form = MagicMock()
        mock_form.get = lambda key, default="": form_data.get(key, default)
        mock_form.multi_items = lambda: multi_items
        mock_request.form = AsyncMock(return_value=mock_form)

        # Patch UploadFile.read to return bytes
        async def mock_read():
            return b"fake-audio-bytes"

        form_data["file"].read = mock_read

        result = await transcription_protocol.parse_http_request(mock_request)

        assert isinstance(result, _TranscriptionRequestWrapper)
        assert result.model == "whisper-1"
        assert result.file == b"fake-audio-bytes"
        assert result.filename == "audio.mp3"

        dumped = result.model_dump()
        assert dumped["model"] == "whisper-1"
        assert dumped["language"] == "en"
        assert dumped["prompt"] == "Hello prompt"
        assert dumped["response_format"] == "json"
        assert dumped["temperature"] == 0.5
        assert dumped["stream"] is True
        assert dumped["timestamp_granularities"] == ["word", "segment"]
        assert dumped["include"] == ["logprobs"]

    @pytest.mark.asyncio
    async def test_parse_http_request_missing_model(self):
        """Test parsing multipart HTTP request with missing model."""
        mock_request = MagicMock()

        multi_items = [("file", UploadFile(filename="audio.mp3", file=MagicMock()))]

        mock_form = MagicMock()
        mock_form.get = lambda key, default="": default
        mock_form.multi_items = lambda: multi_items
        mock_request.form = AsyncMock(return_value=mock_form)

        with pytest.raises(ValidationError, match="Missing required field 'model'"):
            await transcription_protocol.parse_http_request(mock_request)

    @pytest.mark.asyncio
    async def test_parse_http_request_missing_file(self):
        """Test parsing multipart HTTP request with missing file."""
        mock_request = MagicMock()

        multi_items = [("model", "whisper-1")]

        mock_form = MagicMock()
        mock_form.get = lambda key, default="": "whisper-1" if key == "model" else default
        mock_form.multi_items = lambda: multi_items
        mock_request.form = AsyncMock(return_value=mock_form)

        with pytest.raises(ValidationError, match="Missing required field 'file'"):
            await transcription_protocol.parse_http_request(mock_request)

    @pytest.mark.asyncio
    async def test_parse_http_request_invalid_temperature(self):
        """Test parsing multipart HTTP request with invalid temperature."""
        mock_request = MagicMock()

        async def mock_read():
            return b"fake-audio-bytes"

        file_field = UploadFile(filename="audio.mp3", file=MagicMock())
        file_field.read = mock_read

        multi_items = [
            ("model", "whisper-1"),
            ("file", file_field),
            ("temperature", "not-a-number"),
        ]

        mock_form = MagicMock()
        mock_form.get = lambda key, default="": (
            "whisper-1"
            if key == "model"
            else file_field
            if key == "file"
            else "not-a-number"
            if key == "temperature"
            else default
        )
        mock_form.multi_items = lambda: multi_items
        mock_request.form = AsyncMock(return_value=mock_form)

        with pytest.raises(ValidationError, match="Invalid temperature value"):
            await transcription_protocol.parse_http_request(mock_request)

    def test_parse_request_from_wrapper(self):
        """Test parsing from _TranscriptionRequestWrapper."""
        wrapper = _TranscriptionRequestWrapper(
            data={
                "model": "whisper-1",
                "language": "en",
                "response_format": "json",
            },
            file=b"fake-audio",
            filename="audio.mp3",
        )
        data = wrapper.model_dump(exclude_none=True)
        data["file"] = wrapper.file
        data["filename"] = wrapper.filename
        result = _serializer.parse_request(data)

        assert isinstance(result, InternalTranscriptionRequest)
        assert result.model == "whisper-1"
        assert result.file == b"fake-audio"
        assert result.filename == "audio.mp3"
        assert result.language == "en"

    def test_format_response_json(self):
        """Test formatting transcription JSON response."""
        response = InternalTranscriptionResponse(
            text="Hello world",
            _response_format="json",
        )
        result = _serializer.format_response(response)

        assert isinstance(result, dict)
        assert result["text"] == "Hello world"

    def test_format_response_text(self):
        """Test formatting transcription text response."""
        response = InternalTranscriptionResponse(
            text="Hello world",
            _response_format="text",
        )
        result = _serializer.format_response(response)

        assert result == {"text": "Hello world"}

    def test_format_response_verbose_json(self):
        """Test formatting transcription verbose_json response."""
        response = InternalTranscriptionResponse(
            text="Hello world",
            language="en",
            duration=5.0,
            segments=[{"start": 0.0, "end": 5.0, "text": "Hello world"}],
            _response_format="verbose_json",
        )
        result = _serializer.format_response(response)

        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        assert result["duration"] == 5.0
        assert result["segments"] == [{"start": 0.0, "end": 5.0, "text": "Hello world"}]


class TestTranslationProtocolEndpoint:
    """Tests for TranslationProtocolEndpoint."""

    def test_name(self):
        """Test handler name."""
        assert translation_protocol.name == "translation"

    def test_paths(self):
        """Test handler paths."""
        assert translation_protocol.paths == ["/v1/audio/translations"]

    def test_request_model_is_none(self):
        """Test that request_model returns None for multipart."""
        assert translation_protocol.request_model is None

    @pytest.mark.asyncio
    async def test_parse_http_request(self):
        """Test parsing multipart HTTP request."""
        mock_request = MagicMock()

        async def mock_read():
            return b"fake-audio-bytes"

        file_field = UploadFile(filename="audio.mp3", file=MagicMock())
        file_field.read = mock_read

        multi_items = [
            ("model", "whisper-1"),
            ("file", file_field),
            ("prompt", "Hello prompt"),
            ("response_format", "json"),
            ("temperature", "0.5"),
        ]

        mock_form = MagicMock()
        mock_form.get = lambda key, default="": (
            "whisper-1"
            if key == "model"
            else file_field
            if key == "file"
            else "Hello prompt"
            if key == "prompt"
            else "json"
            if key == "response_format"
            else "0.5"
            if key == "temperature"
            else default
        )
        mock_form.multi_items = lambda: multi_items
        mock_request.form = AsyncMock(return_value=mock_form)

        result = await translation_protocol.parse_http_request(mock_request)

        assert isinstance(result, _TranslationRequestWrapper)
        assert result.model == "whisper-1"
        assert result.file == b"fake-audio-bytes"
        assert result.filename == "audio.mp3"

        dumped = result.model_dump()
        assert dumped["model"] == "whisper-1"
        assert dumped["prompt"] == "Hello prompt"
        assert dumped["response_format"] == "json"
        assert dumped["temperature"] == 0.5

    def test_parse_request_from_wrapper(self):
        """Test parsing from _TranslationRequestWrapper."""
        wrapper = _TranslationRequestWrapper(
            data={
                "model": "whisper-1",
                "response_format": "json",
            },
            file=b"fake-audio",
            filename="audio.mp3",
        )
        data = wrapper.model_dump(exclude_none=True)
        data["file"] = wrapper.file
        data["filename"] = wrapper.filename
        result = _serializer.parse_request(data)

        assert isinstance(result, InternalTranslationRequest)
        assert result.model == "whisper-1"
        assert result.file == b"fake-audio"
        assert result.filename == "audio.mp3"

    def test_format_response_json(self):
        """Test formatting translation JSON response."""
        response = InternalTranslationResponse(
            text="Hello world",
            _response_format="json",
        )
        result = _serializer.format_response(response)

        assert isinstance(result, dict)
        assert result["text"] == "Hello world"

    def test_format_response_text(self):
        """Test formatting translation text response."""
        response = InternalTranslationResponse(
            text="Hello world",
            _response_format="text",
        )
        result = _serializer.format_response(response)

        assert result == {"text": "Hello world"}

    def test_format_response_verbose_json(self):
        """Test formatting translation verbose_json response."""
        response = InternalTranslationResponse(
            text="Hello world",
            language="english",
            duration=5.0,
            segments=[{"start": 0.0, "end": 5.0, "text": "Hello world"}],
            _response_format="verbose_json",
        )
        result = _serializer.format_response(response)

        assert result["text"] == "Hello world"
        assert result["language"] == "english"
        assert result["duration"] == 5.0
        assert result["segments"] == [{"start": 0.0, "end": 5.0, "text": "Hello world"}]


class TestOpenAIAudioSerializer:
    """Tests for OpenAIAudioSerializer."""

    def test_parse_speech_request(self):
        """Test parsing speech request dict."""
        serializer = OpenAIAudioSerializer()
        data = {
            "model": "tts-1",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "mp3",
            "speed": 1.5,
        }
        result = serializer.parse_speech_request(data)

        assert isinstance(result, InternalSpeechRequest)
        assert result.model == "tts-1"
        assert result.input == "Hello"
        assert result.voice == "alloy"
        assert result.response_format == "mp3"
        assert result.speed == 1.5

    def test_parse_speech_request_defaults(self):
        """Test parsing speech request with defaults."""
        serializer = OpenAIAudioSerializer()
        data = {
            "model": "tts-1",
            "input": "Hello",
            "voice": "alloy",
        }
        result = serializer.parse_speech_request(data)

        assert result.response_format == "mp3"
        assert result.speed == 1.0
        assert result.stream is False

    def test_format_speech_response(self):
        """Test formatting speech response."""
        serializer = OpenAIAudioSerializer()
        response = InternalSpeechResponse(
            content=b"audio-data",
            content_type="audio/mpeg",
        )
        result = serializer.format_speech_response(response)

        assert result == {"content": b"audio-data", "content_type": "audio/mpeg"}

    def test_parse_transcription_request(self):
        """Test parsing transcription request."""
        serializer = OpenAIAudioSerializer()
        data = {"model": "whisper-1", "language": "en"}
        result = serializer.parse_transcription_request(data, b"audio", "audio.mp3")

        assert isinstance(result, InternalTranscriptionRequest)
        assert result.model == "whisper-1"
        assert result.file == b"audio"
        assert result.filename == "audio.mp3"
        assert result.language == "en"
        assert result.response_format == "json"
        assert result.temperature == 0.0

    def test_parse_transcription_request_with_options(self):
        """Test parsing transcription request with optional fields."""
        serializer = OpenAIAudioSerializer()
        data = {
            "model": "whisper-1",
            "prompt": "Hello",
            "response_format": "verbose_json",
            "temperature": 0.5,
            "timestamp_granularities": ["word"],
            "include": ["logprobs"],
            "stream": True,
        }
        result = serializer.parse_transcription_request(data, b"audio", "audio.mp3")

        assert result.prompt == "Hello"
        assert result.response_format == "verbose_json"
        assert result.temperature == 0.5
        assert result.timestamp_granularities == ["word"]
        assert result.include == ["logprobs"]
        assert result.stream is True

    def test_format_transcription_response_text(self):
        """Test formatting transcription text response."""
        serializer = OpenAIAudioSerializer()
        response = InternalTranscriptionResponse(text="Hello", _response_format="text")
        result = serializer.format_transcription_response(response, "text")

        assert result == "Hello"

    def test_format_transcription_response_json(self):
        """Test formatting transcription JSON response."""
        serializer = OpenAIAudioSerializer()
        response = InternalTranscriptionResponse(
            text="Hello",
            _response_format="json",
        )
        result = serializer.format_transcription_response(response, "json")

        assert result == {"text": "Hello"}

    def test_format_transcription_response_verbose_json(self):
        """Test formatting transcription verbose_json response."""
        serializer = OpenAIAudioSerializer()
        response = InternalTranscriptionResponse(
            text="Hello",
            task="transcribe",
            language="en",
            duration=5.0,
            segments=[{"start": 0.0, "end": 5.0}],
            words=[{"word": "Hello", "start": 0.0}],
            _response_format="verbose_json",
        )
        result = serializer.format_transcription_response(response, "verbose_json")

        assert result["text"] == "Hello"
        assert result["task"] == "transcribe"
        assert result["language"] == "en"
        assert result["duration"] == 5.0
        assert result["segments"] == [{"start": 0.0, "end": 5.0}]
        assert result["words"] == [{"word": "Hello", "start": 0.0}]

    def test_format_transcription_response_diarized_json(self):
        """Test formatting transcription diarized_json response."""
        serializer = OpenAIAudioSerializer()
        response = InternalTranscriptionResponse(
            text="Hello",
            duration=5.0,
            segments=[{"speaker": "A", "text": "Hello"}],
            _response_format="diarized_json",
        )
        result = serializer.format_transcription_response(response, "diarized_json")

        assert result["text"] == "Hello"
        assert result["duration"] == 5.0
        assert result["task"] == "transcribe"

    def test_parse_translation_request(self):
        """Test parsing translation request."""
        serializer = OpenAIAudioSerializer()
        data = {"model": "whisper-1"}
        result = serializer.parse_translation_request(data, b"audio", "audio.mp3")

        assert isinstance(result, InternalTranslationRequest)
        assert result.model == "whisper-1"
        assert result.file == b"audio"
        assert result.filename == "audio.mp3"
        assert result.response_format == "json"
        assert result.temperature == 0.0

    def test_format_translation_response_text(self):
        """Test formatting translation text response."""
        serializer = OpenAIAudioSerializer()
        response = InternalTranslationResponse(text="Hello", _response_format="text")
        result = serializer.format_translation_response(response, "text")

        assert result == "Hello"

    def test_format_translation_response_srt(self):
        """Test formatting translation SRT response."""
        serializer = OpenAIAudioSerializer()
        response = InternalTranslationResponse(
            text="1\n00:00:00,000 --> ...",
            _response_format="srt",
        )
        result = serializer.format_translation_response(response, "srt")

        assert result == "1\n00:00:00,000 --> ..."

    def test_format_translation_response_vtt(self):
        """Test formatting translation VTT response."""
        serializer = OpenAIAudioSerializer()
        response = InternalTranslationResponse(text="WEBVTT\n\n...", _response_format="vtt")
        result = serializer.format_translation_response(response, "vtt")

        assert result == "WEBVTT\n\n..."

    def test_format_translation_response_verbose_json(self):
        """Test formatting translation verbose_json response."""
        serializer = OpenAIAudioSerializer()
        response = InternalTranslationResponse(
            text="Hello",
            language="english",
            duration=5.0,
            segments=[{"start": 0.0, "end": 5.0}],
            _response_format="verbose_json",
        )
        result = serializer.format_translation_response(response, "verbose_json")

        assert result["text"] == "Hello"
        assert result["language"] == "english"
        assert result["duration"] == 5.0
        assert result["segments"] == [{"start": 0.0, "end": 5.0}]

    def test_parse_request_dispatch_speech(self):
        """Test parse_request dispatches to speech parser."""
        serializer = OpenAIAudioSerializer()
        data = {
            "model": "tts-1",
            "input": "Hello",
            "voice": "alloy",
        }
        result = serializer.parse_request(data)

        assert isinstance(result, InternalSpeechRequest)

    def test_parse_request_dispatch_transcription(self):
        """Test parse_request dispatches to transcription parser."""
        serializer = OpenAIAudioSerializer()
        data = {
            "model": "whisper-1",
            "file": b"audio",
            "filename": "audio.mp3",
            "language": "en",
        }
        result = serializer.parse_request(data)

        assert isinstance(result, InternalTranscriptionRequest)

    def test_parse_request_dispatch_translation(self):
        """Test parse_request dispatches to translation parser."""
        serializer = OpenAIAudioSerializer()
        data = {
            "model": "whisper-1",
            "file": b"audio",
            "filename": "audio.mp3",
        }
        result = serializer.parse_request(data)

        assert isinstance(result, InternalTranslationRequest)

    def test_parse_request_missing_file_raises(self):
        """Test parse_request raises when file bytes missing for transcription."""
        serializer = OpenAIAudioSerializer()
        data = {"model": "whisper-1"}

        with pytest.raises(NotImplementedError):
            serializer.parse_request(data)
