"""Regression tests for streaming image edit routing and body building."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.models.image import ImageEditSource, InternalImageEditRequest
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase


def _image_edit_request(stream: bool = False) -> InternalImageEditRequest:
    """Build a minimal InternalImageEditRequest for tests."""
    return InternalImageEditRequest(
        model="gpt-image-1",
        prompt="add a hat",
        images=[ImageEditSource(file_id="file_123")],
        stream=stream,
        size=None,
    )


def test_build_image_edit_body_no_response_format_attribute_error():
    """Regression: building an image edit body must not access response_format.

    InternalImageEditRequest has no response_format field. A previous routing
    bug sent streaming image edits through the generation body builder, which
    accessed request.response_format and raised AttributeError.
    """
    adapter = OpenAICompatibleBase(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        unknown_fields_policy="passthrough",
    )
    request = _image_edit_request()

    outbound = adapter._build_outbound_body(request, request_type="image_edit")

    assert outbound.json_body is not None
    assert outbound.json_body["model"] == "gpt-image-1"
    assert outbound.json_body["prompt"] == "add a hat"
    assert outbound.json_body["images"] == [{"file_id": "file_123"}]


@pytest.mark.asyncio
async def test_image_edit_posts_to_edits_endpoint(monkeypatch):
    """Non-streaming image edit must POST to /images/edits."""
    adapter = OpenAICompatibleBase(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
    )
    captured: dict = {}

    async def fake_post(url: str, headers: dict, body: dict):
        captured["url"] = url
        captured["body"] = body
        return {"data": [], "created": 1234567890}

    monkeypatch.setattr(adapter, "_post_json_with_retry", fake_post)

    await adapter.image_edit(_image_edit_request())

    assert captured["url"] == "https://api.openai.com/v1/images/edits"
    assert captured["body"]["model"] == "gpt-image-1"
    assert captured["body"]["prompt"] == "add a hat"


@pytest.mark.asyncio
async def test_stream_image_edit_posts_to_edits_endpoint():
    """Streaming image edit must target the edits endpoint, not generations."""
    adapter = OpenAICompatibleBase(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
    )

    response = MagicMock()
    response.status_code = 200

    async def _iter_lines():
        yield b'data: {"type":"image_generation.progress"}\n\n'
        yield b"data: [DONE]\n\n"

    response.iter_lines.return_value = _iter_lines()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    async def _get_client():
        return client

    adapter._get_client = _get_client

    stream = await adapter.stream_image_edit(_image_edit_request(stream=True))
    chunks = [chunk async for chunk in stream]

    assert client.post.call_count == 1
    call_args, call_kwargs = client.post.call_args
    assert call_args[0] == "https://api.openai.com/v1/images/edits"
    assert call_kwargs["json"]["model"] == "gpt-image-1"
    assert call_kwargs["json"]["prompt"] == "add a hat"
    assert call_kwargs["json"]["stream"] is True
    assert any("image_generation.progress" in str(c) for c in chunks)
