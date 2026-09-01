"""Regression tests for the OpenAI Images HTTP contract."""

from tempfile import SpooledTemporaryFile
from types import SimpleNamespace

import orjson
import pytest
from pydantic import ValidationError as PydanticValidationError
from starlette.datastructures import FormData, UploadFile

from llm_proxy.models.image import ImageData, ImageEditSource, InternalImageResponse
from llm_proxy.models.types import PromptTokensDetails, Usage
from llm_proxy.protocols.openai.images_edits_handler import parse_image_edit_request
from llm_proxy.protocols.openai.images_serializer import ImageGenerationsSerializer
from llm_proxy.protocols.openai.schemas import (
    ImageEditRequestSchema,
    ImageGenerationRequestSchema,
)
from llm_proxy.providers.capabilities.image import normalize_image_stream_chunk
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase


def test_generation_accepts_auto_size_and_rejects_invalid_values():
    request = ImageGenerationRequestSchema.model_validate({"prompt": "cat", "size": "auto"})
    assert request.size == "auto"

    with pytest.raises(PydanticValidationError):
        ImageGenerationRequestSchema.model_validate({"prompt": "cat", "n": 0})
    with pytest.raises(PydanticValidationError):
        ImageGenerationRequestSchema.model_validate({"prompt": "cat", "unknown": True})


def test_edit_requires_at_least_one_json_image():
    with pytest.raises(PydanticValidationError):
        ImageEditRequestSchema.model_validate({"prompt": "edit", "images": []})


def test_edit_accepts_sdk_size_and_quality_values():
    # The official SDK allows 256x256/512x512 for edits but not 1792x1024.
    request = ImageEditRequestSchema.model_validate(
        {
            "prompt": "edit",
            "images": [{"image_url": "https://example.com/a.png"}],
            "size": "256x256",
            "quality": "standard",
        }
    )
    assert request.size == "256x256"
    assert request.quality == "standard"

    with pytest.raises(PydanticValidationError):
        ImageEditRequestSchema.model_validate(
            {
                "prompt": "edit",
                "images": [{"image_url": "https://example.com/a.png"}],
                "size": "1792x1024",
            }
        )


def test_edit_rejects_singular_image_json_field():
    # The JSON body uses the ``images`` array; a singular ``image`` object is
    # not part of the OpenAI contract and must be rejected.
    with pytest.raises(PydanticValidationError):
        ImageEditRequestSchema.model_validate(
            {"prompt": "edit", "image": {"image_url": "https://example.com/a.png"}}
        )


@pytest.mark.asyncio
async def test_edit_multipart_parser_preserves_repeated_image_files():
    with SpooledTemporaryFile() as file_a, SpooledTemporaryFile() as file_b:
        file_a.write(b"a")
        file_a.seek(0)
        file_b.write(b"b")
        file_b.seek(0)
        image_a = UploadFile(file_a, filename="a.png", headers={"content-type": "image/png"})
        image_b = UploadFile(file_b, filename="b.jpg", headers={"content-type": "image/jpeg"})
        request = SimpleNamespace(
            headers={"content-type": "multipart/form-data; boundary=test"},
            form=lambda: None,
        )

        async def form():
            return FormData(
                [
                    ("model", "gpt-image-1"),
                    ("prompt", "add a hat"),
                    ("image[]", image_a),
                    ("image[]", image_b),
                ]
            )

        request.form = form
        parsed = await parse_image_edit_request(request)
    assert parsed.prompt == "add a hat"
    assert len(parsed.images) == 2
    assert parsed.images[0]["file"] == b"a"
    assert parsed.images[1]["filename"] == "b.jpg"


def test_image_response_includes_required_input_token_details():
    serializer = ImageGenerationsSerializer()
    response = InternalImageResponse(
        created=123,
        data=[ImageData(url="https://example.com/a.png")],
        usage=Usage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            prompt_tokens_details=PromptTokensDetails(text_tokens=7, image_tokens=3),
        ),
    )
    body = serializer.format_response(response)
    assert body["usage"]["input_tokens_details"] == {"text_tokens": 7, "image_tokens": 3}


def test_image_edit_outbound_body_uses_multipart_for_uploaded_images():
    adapter = OpenAICompatibleBase(api_key="test-key", base_url="https://api.openai.com/v1")
    request = SimpleNamespace(
        model="gpt-image-1",
        prompt="edit",
        images=[ImageEditSource(file=b"png-data", filename="source.png", content_type="image/png")],
        mask=None,
        background=None,
        input_fidelity=None,
        moderation=None,
        n=1,
        output_compression=None,
        output_format=None,
        partial_images=None,
        quality=None,
        size=None,
        user=None,
        stream=False,
        extra={},
    )
    outbound = adapter._build_outbound_body(request, request_type="image_edit")
    assert outbound.json_body is None
    assert outbound.form_data is not None
    assert outbound.files == [("image[]", ("source.png", b"png-data", "image/png"))]


def test_gemini_image_stream_is_normalized_to_openai_events():
    chunk = (
        "data: "
        + orjson.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"inlineData": {"mimeType": "image/png", "data": "abc"}}]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 6,
                },
            }
        ).decode()
        + "\n\n"
    )
    normalized = normalize_image_stream_chunk(chunk, created_at=123)
    assert len(normalized) == 2
    assert normalized[0]["type"] == "image_generation.partial_image"
    assert normalized[0]["b64_json"] == "abc"
    assert normalized[1]["type"] == "image_generation.completed"
    assert "usageMetadata" not in normalized[1]
    assert normalized[1]["usage"]["input_tokens"] == 4
    # The completed event carries the final image per the OpenAI spec.
    assert normalized[1]["b64_json"] == "abc"


def test_gemini_image_stream_no_space_sse_is_normalized():
    """Kimi-style no-space ``data:{...}`` framing must parse identically."""
    payload = orjson.dumps(
        {
            "candidates": [
                {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "abc"}}]}}
            ],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
        }
    ).decode()
    normalized = normalize_image_stream_chunk(f"data:{payload}\n\n", created_at=123)
    assert len(normalized) == 2
    assert normalized[0]["type"] == "image_generation.partial_image"
    assert normalized[1]["type"] == "image_generation.completed"
    assert normalized[1]["usage"]["total_tokens"] == 6


def test_edit_stream_events_pass_through_unchanged():
    chunk = (
        "event: image_edit.partial_image\n"
        "data: "
        + orjson.dumps(
            {
                "type": "image_edit.partial_image",
                "b64_json": "partial",
                "partial_image_index": 0,
            }
        ).decode()
        + "\n\n"
        "event: image_edit.completed\n"
        "data: "
        + orjson.dumps(
            {
                "type": "image_edit.completed",
                "b64_json": "final",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 7,
                    "total_tokens": 12,
                    "input_tokens_details": {"text_tokens": 2, "image_tokens": 3},
                },
            }
        ).decode()
        + "\n\n"
    )
    normalized = normalize_image_stream_chunk(chunk, created_at=123)
    assert [event["type"] for event in normalized] == [
        "image_edit.partial_image",
        "image_edit.completed",
    ]
    assert normalized[0]["created_at"] == 123
    assert normalized[1]["b64_json"] == "final"
    assert normalized[1]["usage"]["input_tokens_details"] == {"text_tokens": 2, "image_tokens": 3}


def test_usage_tracker_observes_edit_completed():
    from llm_proxy.core.processing.streaming_processor import _ImageStreamUsageTracker

    tracker = _ImageStreamUsageTracker()
    tracker.observe(
        "event: image_edit.completed\n"
        "data: "
        + orjson.dumps(
            {
                "type": "image_edit.completed",
                "b64_json": "final",
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 7,
                    "total_tokens": 12,
                    "input_tokens_details": {"text_tokens": 2, "image_tokens": 3},
                },
            }
        ).decode()
        + "\n\n"
    )
    assert tracker.images_completed == 1
    assert tracker.captured_usage["input_tokens"] == 5


def test_image_provider_response_requires_created_timestamp():
    from llm_proxy.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        OpenAICompatibleBase(
            provider_name="test", api_key="test-key", base_url="https://example.com/v1"
        ).from_image_provider_format({"data": []})
