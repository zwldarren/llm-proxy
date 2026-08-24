"""Tests for ParameterOverrideService bytes attribute preservation.

Verifies that all bytes/bytearray attributes on the original unified request
survive the re-parse cycle, not just audio file bytes. Covers image-edit
images/mask in addition to audio file/filename.
"""

from typing import Any

from llm_proxy.core.processing.stages.parameter_override import ParameterOverrideService

_UNSET = object()


def _make_metadata(protocol_name: str = "image_edits") -> object:
    return type("M", (), {"protocol_name": protocol_name})()


def _make_request(
    model: str = "gpt-4",
    file: bytes | None = None,
    filename: str | None | object = _UNSET,
    images: bytes | None = None,
    mask: bytes | None = None,
    protocol: str = "chat",
) -> object:
    attrs: dict[str, Any] = {
        "model": model,
        "file": file,
        "images": images,
        "mask": mask,
        "metadata": _make_metadata(protocol),
    }
    if filename is not _UNSET:
        attrs["filename"] = filename
    return type("Req", (), attrs)()


class _FakeSerializer:
    """A minimal serializer that round-trips dict keys as object attributes."""

    @property
    def protocol_name(self) -> str:
        return "image_edits"

    def parse_request(self, data: dict) -> object:
        class _R:
            def __init__(self, d):
                self.model = d.get("model")
                self.images = d.get("images")
                self.mask = d.get("mask")
                self.file = d.get("file")
                self.filename = d.get("filename")
                self.metadata = _make_metadata("image_edits")
                self._raw_protocol_data = d

        return _R(data)


class _FakeSerializerNoBytes:
    """Serializer whose parse_request only sets model/metadata, forcing the
    re-attach safety net in ParameterOverrideService.apply to restore bytes."""

    @property
    def protocol_name(self) -> str:
        return "image_edits"

    def parse_request(self, data: dict) -> object:
        class _R:
            def __init__(self, d):
                self.model = d.get("model")
                self.metadata = _make_metadata("image_edits")
                self._raw_protocol_data = d

        return _R(data)


class _FakeSerializerNoFilename:
    """Serializer that sets file bytes but intentionally omits filename, so
    the 'audio.mp3' fallback can be verified."""

    @property
    def protocol_name(self) -> str:
        return "audio"

    def parse_request(self, data: dict) -> object:
        class _R:
            def __init__(self, d):
                self.model = d.get("model")
                self.file = d.get("file")
                self.metadata = _make_metadata("audio")
                self._raw_protocol_data = d

        return _R(data)


class TestBytesPreservation:
    """ParameterOverrideService must preserve all bytes attributes across re-parse."""

    def test_preserves_image_edit_bytes(self):
        svc = ParameterOverrideService(_FakeSerializer())
        original = _make_request(
            model="dall-e-2",
            images=b"\x89PNG",
            mask=b"\x00",
            filename=None,
            protocol="image_edits",
        )
        _, new = svc.apply(
            raw_data={"model": "dall-e-2"},
            unified_request=original,
            parameter_overrides={"prompt": "cat"},
            provider_model_name="dall-e-2",
            request_id="r1",
        )
        assert new.images == b"\x89PNG"
        assert new.mask == b"\x00"

    def test_preserves_audio_file_bytes(self):
        svc = ParameterOverrideService(_FakeSerializer())
        original = _make_request(
            model="whisper-1",
            file=b"fake-audio-bytes",
            filename="audio.mp3",
            protocol="audio",
        )
        _, new = svc.apply(
            raw_data={"model": "whisper-1"},
            unified_request=original,
            parameter_overrides={"language": "en"},
            provider_model_name="whisper-1",
            request_id="r1",
        )
        assert new.file == b"fake-audio-bytes"
        assert new.filename == "audio.mp3"

    def test_preserves_both_audio_and_image_bytes(self):
        svc = ParameterOverrideService(_FakeSerializer())
        original = _make_request(
            model="some-model",
            file=b"audio-data",
            filename="audio.wav",
            images=b"\x89PNG",
            mask=b"\x00",
            protocol="mixed",
        )
        _, new = svc.apply(
            raw_data={"model": "some-model"},
            unified_request=original,
            parameter_overrides={"prompt": "test"},
            provider_model_name="some-model",
            request_id="r1",
        )
        assert new.file == b"audio-data"
        assert new.filename == "audio.wav"
        assert new.images == b"\x89PNG"
        assert new.mask == b"\x00"

    def test_no_bytes_attrs_does_not_break(self):
        svc = ParameterOverrideService(_FakeSerializer())
        original = _make_request(model="gpt-4")
        _, new = svc.apply(
            raw_data={"model": "gpt-4"},
            unified_request=original,
            parameter_overrides={"temperature": 0.8},
            provider_model_name="gpt-4",
            request_id="r1",
        )
        assert new.model == "gpt-4"
        assert new.file is None
        assert new.images is None

    def test_re_attach_when_serializer_skips_bytes(self):
        svc = ParameterOverrideService(_FakeSerializerNoBytes())
        original = _make_request(
            model="dall-e-2",
            images=b"\x89PNG",
            mask=b"\x00",
            filename=None,
            protocol="image_edits",
        )
        _, new = svc.apply(
            raw_data={"model": "dall-e-2"},
            unified_request=original,
            parameter_overrides={"prompt": "cat"},
            provider_model_name="dall-e-2",
            request_id="r1",
        )
        assert new.images == b"\x89PNG"
        assert new.mask == b"\x00"

    def test_filename_fallback_when_missing(self):
        svc = ParameterOverrideService(_FakeSerializerNoFilename())
        # No filename attribute at all — the fallback must supply it.
        original = type(
            "Req",
            (),
            {
                "model": "whisper-1",
                "file": b"fake-audio-bytes",
                "images": None,
                "mask": None,
                "metadata": _make_metadata("audio"),
            },
        )()
        _, new = svc.apply(
            raw_data={"model": "whisper-1"},
            unified_request=original,
            parameter_overrides={"language": "en"},
            provider_model_name="whisper-1",
            request_id="r1",
        )
        assert new.file == b"fake-audio-bytes"
        assert new.filename == "audio.mp3"
