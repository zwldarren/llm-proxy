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
from llm_proxy.protocols.anthropic.handler import anthropic_protocol
from llm_proxy.protocols.openai.audio_transcription_handler import transcription_protocol
from llm_proxy.protocols.openai.handler import openai_protocol
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


def test_additional_routes_run_protocol_middleware(app):
    """Additional routes (e.g. anthropic count_tokens) must run inside the
    protocol middleware chain: the anthropic capture middleware stores the
    client ``anthropic-beta``/fingerprint headers that count_tokens forwards.
    Regression: additional routes bypassed endpoint middleware entirely."""
    ran = {"middleware": False, "handler": False}

    async def capture_mw(_request, _fastapi_request):
        ran["middleware"] = True

    async def count_handler(_request, _fastapi_request):
        ran["handler"] = True
        return {"input_tokens": 7}

    from pydantic import BaseModel

    from llm_proxy.protocols.base import ProtocolEndpoint

    class CountBody(BaseModel):
        model: str

    endpoint = ProtocolEndpoint(
        name="mwprobe",
        paths=["/v1/mwprobe/messages"],
        request_model=None,
        middleware=[capture_mw],
        additional_routes=[("/v1/mwprobe/messages/count", CountBody, None, count_handler)],
    )
    app.include_router(create_protocol_router(endpoint))

    with TestClient(app) as client:
        resp = client.post("/v1/mwprobe/messages/count", json={"model": "m"})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"input_tokens": 7}
    assert ran == {"middleware": True, "handler": True}


class TestBaseUrlTolerantPathAliases:
    """Chat endpoints answer on aliases that tolerate a misconfigured base_url.

    A client whose base_url omits /v1 (requests ``/messages``) or double-writes
    it (``{base}/v1`` + ``/v1/messages``) must reach the same handler instead of
    the frontend catch-all's bare 405.
    """

    @pytest.mark.parametrize(
        ("endpoint", "expected_post_paths"),
        [
            (
                anthropic_protocol,
                {
                    "/v1/messages",
                    "/v1/messages/",
                    "/messages",
                    "/messages/",
                    "/v1/v1/messages",
                    "/v1/v1/messages/",
                    "/v1/messages/count_tokens",
                    "/v1/messages/count_tokens/",
                    "/messages/count_tokens",
                    "/messages/count_tokens/",
                    "/v1/v1/messages/count_tokens",
                    "/v1/v1/messages/count_tokens/",
                },
            ),
            (
                openai_protocol,
                {
                    "/v1/chat/completions",
                    "/v1/chat/completions/",
                    "/chat/completions",
                    "/chat/completions/",
                    "/v1/v1/chat/completions",
                    "/v1/v1/chat/completions/",
                },
            ),
        ],
    )
    def test_alias_routes_are_registered(self, endpoint, expected_post_paths):
        router = create_protocol_router(endpoint)
        post_paths = {route.path for route in router.routes if "POST" in (route.methods or set())}
        assert post_paths == expected_post_paths

    def test_alias_posts_reach_the_chat_handlers(self, app):
        openai_processor = _install_processor(app, "openai")
        anthropic_processor = _install_processor(app, "anthropic")
        app.include_router(create_protocol_router(openai_protocol))
        app.include_router(create_protocol_router(anthropic_protocol))

        chat_body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        messages_body = {
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }

        with (
            patch(
                "llm_proxy.api.routers.protocol._NON_CHAT_CONTEXT_BUILDERS",
                {"openai": _dummy_context, "anthropic": _dummy_context},
            ),
            TestClient(app) as client,
        ):
            for path in (
                "/v1/chat/completions",
                "/chat/completions",
                "/v1/v1/chat/completions",
                "/v1/chat/completions/",
                "/chat/completions/",
                "/v1/v1/chat/completions/",
            ):
                resp = client.post(path, json=chat_body)
                assert resp.status_code == 200, (path, resp.text)
            for path in (
                "/v1/messages",
                "/messages",
                "/v1/v1/messages",
                "/v1/messages/",
                "/messages/",
                "/v1/v1/messages/",
            ):
                resp = client.post(path, json=messages_body)
                assert resp.status_code == 200, (path, resp.text)

        assert openai_processor.process.await_count == 6
        assert anthropic_processor.process.await_count == 6

    def test_trailing_slash_posts_beat_the_spa_catch_all(self, app):
        """POST /v1/messages/ must not be answered by the frontend catch-all.

        The catch-all (``GET /{full_path:path}``) partial-matches ahead of
        Starlette's ``redirect_slashes``, so without a real trailing-slash
        route the request 405s; a 307 would also fail for SDKs that do not
        follow redirects.
        """
        anthropic_processor = _install_processor(app, "anthropic")
        app.include_router(create_protocol_router(anthropic_protocol))

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            return {"spa": full_path}

        messages_body = {
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }

        with (
            patch(
                "llm_proxy.api.routers.protocol._NON_CHAT_CONTEXT_BUILDERS",
                {"anthropic": _dummy_context},
            ),
            TestClient(app, follow_redirects=False) as client,
        ):
            for path in ("/v1/messages/", "/messages/", "/v1/v1/messages/"):
                resp = client.post(path, json=messages_body)
                assert resp.status_code == 200, (path, resp.text)

        assert anthropic_processor.process.await_count == 3
