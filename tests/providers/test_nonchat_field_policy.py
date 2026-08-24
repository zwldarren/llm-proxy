"""Tests for field policy enforcement on non-chat endpoints via _build_outbound_body."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    InternalEmbeddingRequest,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalSpeechRequest,
    InternalTranscriptionRequest,
    InternalTranslationRequest,
)
from llm_proxy.models.image import ImageEditSource
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase


def _adapter(policy):
    return OpenAICompatibleBase(
        api_key="k",
        base_url="https://api.openai.com/v1",
        unknown_fields_policy=policy,
    )


def _emb(extra):
    return InternalEmbeddingRequest(model="text-embedding-3-small", input="hi", extra=extra)


def _speech(extra):
    return InternalSpeechRequest(model="tts-1", input="Hello", voice="alloy", extra=extra)


def _transcription(extra):
    return InternalTranscriptionRequest(
        model="whisper-1", file=b"audio", filename="test.mp3", extra=extra
    )


def _translation(extra):
    return InternalTranslationRequest(
        model="whisper-1", file=b"audio", filename="test.mp3", extra=extra
    )


def _image_gen(extra):
    return InternalImageRequest(model="dall-e-3", prompt="a cat", extra=extra)


def _image_edit(extra):
    return InternalImageEditRequest(
        model="dall-e-2",
        prompt="a cat",
        images=[ImageEditSource(file_id="img_123")],
        extra=extra,
    )


# ── Embeddings ──────────────────────────────────────────────────────────


def _embedding_response():
    return {
        "data": [{"embedding": [0.0], "index": 0}],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


@pytest.mark.asyncio
async def test_embeddings_passthrough_keeps_extra(monkeypatch):
    a = _adapter("passthrough")
    captured = {}

    async def fake_post(url, headers, body):
        captured["body"] = body
        resp = MagicMock()
        resp.json.return_value = _embedding_response()
        resp.headers = {}
        return resp

    monkeypatch.setattr(a, "_post_json_response_with_retry", fake_post)
    await a.embeddings(_emb({"dimensions_override": 768}))
    assert captured["body"].get("dimensions_override") == 768


@pytest.mark.asyncio
async def test_embeddings_ignore_strips_extra(monkeypatch):
    a = _adapter("ignore")
    captured = {}

    async def fake_post(url, headers, body):
        captured["body"] = body
        resp = MagicMock()
        resp.json.return_value = _embedding_response()
        resp.headers = {}
        return resp

    monkeypatch.setattr(a, "_post_json_response_with_retry", fake_post)
    await a.embeddings(_emb({"dimensions_override": 768}))
    assert "dimensions_override" not in captured["body"]


@pytest.mark.asyncio
async def test_embeddings_error_raises(monkeypatch):
    a = _adapter("error")
    with pytest.raises(ProviderError, match="unknown request fields"):
        await a.embeddings(_emb({"dimensions_override": 768}))


# ── Speech (dispatch unit tests; speech() requires HTTP round-trip) ─────


def test_speech_dispatch_passthrough_keeps_extra():
    a = _adapter("passthrough")
    outbound = a._build_outbound_body(
        _speech({"voice_instructions": "whisper"}), request_type="speech"
    )
    assert outbound.json_body is not None
    assert outbound.json_body.get("voice_instructions") == "whisper"


def test_speech_dispatch_ignore_strips_extra():
    a = _adapter("ignore")
    outbound = a._build_outbound_body(
        _speech({"voice_instructions": "whisper"}), request_type="speech"
    )
    assert outbound.json_body is not None
    assert "voice_instructions" not in outbound.json_body


def test_speech_dispatch_error_raises():
    a = _adapter("error")
    with pytest.raises(ProviderError, match="unknown request fields"):
        a._build_outbound_body(_speech({"voice_instructions": "whisper"}), request_type="speech")


# ── Transcription ────────────────────────────────────────────────────────


def test_transcription_dispatch_passthrough_keeps_extra():
    a = _adapter("passthrough")
    outbound = a._build_outbound_body(
        _transcription({"diarize": "true"}), request_type="transcription"
    )
    assert outbound.form_data is not None
    assert outbound.form_data.get("diarize") == "true"


def test_transcription_dispatch_ignore_strips_extra():
    a = _adapter("ignore")
    outbound = a._build_outbound_body(
        _transcription({"diarize": "true"}), request_type="transcription"
    )
    assert outbound.form_data is not None
    assert "diarize" not in outbound.form_data


def test_transcription_dispatch_error_raises():
    a = _adapter("error")
    with pytest.raises(ProviderError, match="unknown request fields"):
        a._build_outbound_body(_transcription({"diarize": "true"}), request_type="transcription")


# ── Translation ──────────────────────────────────────────────────────────


def test_translation_dispatch_passthrough_keeps_extra():
    a = _adapter("passthrough")
    outbound = a._build_outbound_body(
        _translation({"some_field": "val"}), request_type="translation"
    )
    assert outbound.form_data is not None
    assert outbound.form_data.get("some_field") == "val"


def test_translation_dispatch_ignore_strips_extra():
    a = _adapter("ignore")
    outbound = a._build_outbound_body(
        _translation({"some_field": "val"}), request_type="translation"
    )
    assert outbound.form_data is not None
    assert "some_field" not in outbound.form_data


def test_translation_dispatch_error_raises():
    a = _adapter("error")
    with pytest.raises(ProviderError, match="unknown request fields"):
        a._build_outbound_body(_translation({"some_field": "val"}), request_type="translation")


# ── Image generation ────────────────────────────────────────────────────


def test_image_gen_dispatch_passthrough_keeps_extra():
    a = _adapter("passthrough")
    outbound = a._build_outbound_body(
        _image_gen({"custom_param": "value"}), request_type="image_generation"
    )
    assert outbound.json_body is not None
    assert outbound.json_body.get("custom_param") == "value"


def test_image_gen_dispatch_ignore_strips_extra():
    a = _adapter("ignore")
    outbound = a._build_outbound_body(
        _image_gen({"custom_param": "value"}), request_type="image_generation"
    )
    assert outbound.json_body is not None
    assert "custom_param" not in outbound.json_body


def test_image_gen_dispatch_error_raises():
    a = _adapter("error")
    with pytest.raises(ProviderError, match="unknown request fields"):
        a._build_outbound_body(
            _image_gen({"custom_param": "value"}), request_type="image_generation"
        )


# ── Image edit ───────────────────────────────────────────────────────────


def test_image_edit_dispatch_passthrough_keeps_extra():
    a = _adapter("passthrough")
    outbound = a._build_outbound_body(
        _image_edit({"custom_edit_param": "val"}), request_type="image_edit"
    )
    assert outbound.json_body is not None
    assert outbound.json_body.get("custom_edit_param") == "val"


def test_image_edit_dispatch_ignore_strips_extra():
    a = _adapter("ignore")
    outbound = a._build_outbound_body(
        _image_edit({"custom_edit_param": "val"}), request_type="image_edit"
    )
    assert outbound.json_body is not None
    assert "custom_edit_param" not in outbound.json_body


def test_image_edit_dispatch_error_raises():
    a = _adapter("error")
    with pytest.raises(ProviderError, match="unknown request fields"):
        a._build_outbound_body(_image_edit({"custom_edit_param": "val"}), request_type="image_edit")


# ════════════════════════════════════════════════════════════════════════
# Integration tests – call real endpoint methods, mock HTTP layer
# These MUST fail if an endpoint method bypasses _build_outbound_body.
# ════════════════════════════════════════════════════════════════════════


def _make_mock_client(status=200, json_data=None, content=b"", content_type=""):
    """Build a MagicMock client whose .post returns a mock response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.content = content
    resp.headers = MagicMock()
    resp.headers.get.return_value = content_type
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    return client


def _patch_for_audio_endpoint(adapter, monkeypatch, mock_client):
    """Patch _get_client, _check_response_status, and _retry.execute for audio endpoints."""

    async def _get_client_patch():
        return mock_client

    async def _check_response_status_patch(_resp):
        pass

    monkeypatch.setattr(adapter, "_get_client", _get_client_patch)
    monkeypatch.setattr(adapter, "_check_response_status", _check_response_status_patch)
    monkeypatch.setattr(adapter._retry, "execute", lambda op, *a, **kw: op())


# ── Speech (integration) ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_speech_integration_passthrough_keeps_extra(monkeypatch):
    a = _adapter("passthrough")
    mock_client = _make_mock_client(content=b"fake-audio", content_type="audio/mpeg")
    _patch_for_audio_endpoint(a, monkeypatch, mock_client)
    await a.speech(_speech({"voice_instructions": "whisper"}))
    posted_body = mock_client.post.call_args.kwargs["json"]
    assert posted_body.get("voice_instructions") == "whisper"


@pytest.mark.asyncio
async def test_speech_integration_ignore_strips_extra(monkeypatch):
    a = _adapter("ignore")
    mock_client = _make_mock_client(content=b"fake-audio", content_type="audio/mpeg")
    _patch_for_audio_endpoint(a, monkeypatch, mock_client)
    await a.speech(_speech({"voice_instructions": "whisper"}))
    posted_body = mock_client.post.call_args.kwargs["json"]
    assert "voice_instructions" not in posted_body


@pytest.mark.asyncio
async def test_speech_integration_error_raises(monkeypatch):
    a = _adapter("error")
    monkeypatch.setattr(a._retry, "execute", lambda op, *a, **kw: op())
    with pytest.raises(ProviderError, match="unknown request fields"):
        await a.speech(_speech({"voice_instructions": "whisper"}))


# ── Transcription (integration) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_transcription_integration_passthrough_keeps_extra(monkeypatch):
    a = _adapter("passthrough")
    mock_client = _make_mock_client(json_data={"text": "hello"}, content_type="application/json")
    _patch_for_audio_endpoint(a, monkeypatch, mock_client)
    await a.transcription(_transcription({"diarize": "true"}))
    data = mock_client.post.call_args.kwargs["data"]
    assert data.get("diarize") == "true"


@pytest.mark.asyncio
async def test_transcription_integration_ignore_strips_extra(monkeypatch):
    a = _adapter("ignore")
    mock_client = _make_mock_client(json_data={"text": "hello"}, content_type="application/json")
    _patch_for_audio_endpoint(a, monkeypatch, mock_client)
    await a.transcription(_transcription({"diarize": "true"}))
    data = mock_client.post.call_args.kwargs["data"]
    assert "diarize" not in data


@pytest.mark.asyncio
async def test_transcription_integration_error_raises(monkeypatch):
    a = _adapter("error")
    monkeypatch.setattr(a._retry, "execute", lambda op, *a, **kw: op())
    with pytest.raises(ProviderError, match="unknown request fields"):
        await a.transcription(_transcription({"diarize": "true"}))


# ── Translation (integration) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_translation_integration_passthrough_keeps_extra(monkeypatch):
    a = _adapter("passthrough")
    mock_client = _make_mock_client(json_data={"text": "hola"}, content_type="application/json")
    _patch_for_audio_endpoint(a, monkeypatch, mock_client)
    await a.translation(_translation({"some_field": "val"}))
    data = mock_client.post.call_args.kwargs["data"]
    assert data.get("some_field") == "val"


@pytest.mark.asyncio
async def test_translation_integration_ignore_strips_extra(monkeypatch):
    a = _adapter("ignore")
    mock_client = _make_mock_client(json_data={"text": "hola"}, content_type="application/json")
    _patch_for_audio_endpoint(a, monkeypatch, mock_client)
    await a.translation(_translation({"some_field": "val"}))
    data = mock_client.post.call_args.kwargs["data"]
    assert "some_field" not in data


@pytest.mark.asyncio
async def test_translation_integration_error_raises(monkeypatch):
    a = _adapter("error")
    monkeypatch.setattr(a._retry, "execute", lambda op, *a, **kw: op())
    with pytest.raises(ProviderError, match="unknown request fields"):
        await a.translation(_translation({"some_field": "val"}))


# ── Image generation (integration) ─────────────────────────────────────


def _image_gen_response():
    return {"data": [], "created": 1234567890}


@pytest.mark.asyncio
async def test_image_gen_integration_passthrough_keeps_extra(monkeypatch):
    a = _adapter("passthrough")
    captured = {}

    async def fake_post(url, headers, body):
        captured["body"] = body
        return _image_gen_response()

    monkeypatch.setattr(a, "_post_json_with_retry", fake_post)
    await a.image_generation(_image_gen({"custom_param": "value"}))
    assert captured["body"].get("custom_param") == "value"


@pytest.mark.asyncio
async def test_image_gen_integration_ignore_strips_extra(monkeypatch):
    a = _adapter("ignore")
    captured = {}

    async def fake_post(url, headers, body):
        captured["body"] = body
        return _image_gen_response()

    monkeypatch.setattr(a, "_post_json_with_retry", fake_post)
    await a.image_generation(_image_gen({"custom_param": "value"}))
    assert "custom_param" not in captured["body"]


@pytest.mark.asyncio
async def test_image_gen_integration_error_raises(monkeypatch):
    a = _adapter("error")
    with pytest.raises(ProviderError, match="unknown request fields"):
        await a.image_generation(_image_gen({"custom_param": "value"}))


# ── Image edit (integration) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_image_edit_integration_passthrough_keeps_extra(monkeypatch):
    a = _adapter("passthrough")
    captured = {}

    async def fake_post(url, headers, body):
        captured["body"] = body
        return _image_gen_response()

    monkeypatch.setattr(a, "_post_json_with_retry", fake_post)
    await a.image_edit(_image_edit({"custom_edit_param": "val"}))
    assert captured["body"].get("custom_edit_param") == "val"


@pytest.mark.asyncio
async def test_image_edit_integration_ignore_strips_extra(monkeypatch):
    a = _adapter("ignore")
    captured = {}

    async def fake_post(url, headers, body):
        captured["body"] = body
        return _image_gen_response()

    monkeypatch.setattr(a, "_post_json_with_retry", fake_post)
    await a.image_edit(_image_edit({"custom_edit_param": "val"}))
    assert "custom_edit_param" not in captured["body"]


@pytest.mark.asyncio
async def test_image_edit_integration_error_raises(monkeypatch):
    a = _adapter("error")
    with pytest.raises(ProviderError, match="unknown request fields"):
        await a.image_edit(_image_edit({"custom_edit_param": "val"}))
