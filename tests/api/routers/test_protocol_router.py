"""Regression tests for protocol router wiring.

Covers ``request_model=None`` endpoints (custom ``parse_http_request``): the
route handler must be wrapped exactly once. A double wrap makes the outer
wrapper call ``handler_func(None, request)`` against a single-argument
wrapper, so every request fails with ``TypeError`` before parsing.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.routers.protocol import create_protocol_router, require_any_auth
from llm_proxy.protocols.openai.audio_transcription_handler import transcription_protocol
from llm_proxy.protocols.openai.images_handler import image_generations_protocol


@pytest.fixture
def app():
    """Minimal FastAPI app with auth overridden and exception handlers on."""
    from llm_proxy.api.middleware.exceptions import register_exception_handlers

    app = FastAPI()
    app.dependency_overrides[require_any_auth] = lambda: None
    register_exception_handlers(app)
    return app


def _install_processor(app, name):
    processor = MagicMock()
    processor.process = AsyncMock(return_value={"ok": True})
    setattr(app.state, f"{name}_processor", processor)
    return processor


async def _dummy_context(_request, _fastapi_request):
    return SimpleNamespace(model="dummy")


def _patch_context_builder(protocol_name):
    return patch(
        "llm_proxy.api.routers.protocol._NON_CHAT_CONTEXT_BUILDERS",
        {protocol_name: _dummy_context},
    )


def test_images_generations_route_parses_json_body(app):
    """request_model=None endpoints must not double-wrap the route handler."""
    processor = _install_processor(app, "image_generations")
    app.include_router(create_protocol_router(image_generations_protocol))

    with _patch_context_builder("image_generations"), TestClient(app) as client:
        resp = client.post(
            "/v1/images/generations", json={"prompt": "a cat", "model": "gpt-image-1"}
        )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True}
    parsed = processor.process.call_args.kwargs["protocol_request"]
    assert parsed.prompt == "a cat"
    assert parsed.model == "gpt-image-1"


def test_transcription_route_parses_multipart(app):
    """The audio transcription endpoint (also request_model=None) works."""
    processor = _install_processor(app, "transcription")
    app.include_router(create_protocol_router(transcription_protocol))

    with _patch_context_builder("transcription"), TestClient(app) as client:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.mp3", b"fake-audio-bytes", "audio/mpeg")},
            data={"model": "whisper-1"},
        )

    assert resp.status_code == 200, resp.text
    parsed = processor.process.call_args.kwargs["protocol_request"]
    assert parsed.model == "whisper-1"
    assert parsed.file == b"fake-audio-bytes"
