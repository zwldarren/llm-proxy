"""Tests for image-aware smart routing."""

import pytest

from llm_proxy.routing.selector import select_from_pool
from llm_proxy.routing.types import (
    CandidatePool,
    ModelPricing,
    RoutingFailureCode,
    RoutingInfeasibleError,
    RoutingMode,
    ServedQuality,
)


def _make_pool(
    models: list[str],
    supports_images: dict[str, bool] | None = None,
) -> CandidatePool:
    return CandidatePool(
        available_models=models,
        pricing={m: ModelPricing(input_price=0.0, output_price=0.0) for m in models},
        served_qualities={m: ServedQuality.ECONOMY for m in models},
        routing_assignments={},
        supports_images=supports_images or {},
    )


def test_require_images_filters_text_only_models():
    """When require_images=True, text-only models are excluded."""
    pool = _make_pool(
        models=["gpt-4o", "gpt-4o-mini", "llama-3-8b"],
        supports_images={"gpt-4o": True, "gpt-4o-mini": True, "llama-3-8b": False},
    )
    decision = select_from_pool(
        complexity=0.5,
        mode=RoutingMode.AUTO,
        confidence=0.8,
        reasoning_text="test",
        available_models=pool.available_models,
        estimated_input_tokens=100,
        max_output_tokens=1000,
        prompt="test",
        pricing=pool.pricing,
        served_qualities=pool.served_qualities,
        supports_images=pool.supports_images,
        require_images=True,
    )
    assert decision.model in ("gpt-4o", "gpt-4o-mini")
    assert decision.model != "llama-3-8b"


def test_require_images_raises_when_no_vision_models():
    """When require_images=True and no models support images, raise."""
    pool = _make_pool(
        models=["llama-3-8b"],
        supports_images={"llama-3-8b": False},
    )
    with pytest.raises(RoutingInfeasibleError) as exc:
        select_from_pool(
            complexity=0.5,
            mode=RoutingMode.AUTO,
            confidence=0.8,
            reasoning_text="test",
            available_models=pool.available_models,
            estimated_input_tokens=100,
            max_output_tokens=1000,
            prompt="test",
            pricing=pool.pricing,
            served_qualities=pool.served_qualities,
            supports_images=pool.supports_images,
            require_images=True,
        )
    assert exc.value.infeasibility.code == RoutingFailureCode.NO_AVAILABLE_MODELS


def test_require_images_false_includes_all():
    """When require_images=False, all models are candidates regardless of image support."""
    pool = _make_pool(
        models=["gpt-4o", "llama-3-8b"],
        supports_images={"gpt-4o": True, "llama-3-8b": False},
    )
    decision = select_from_pool(
        complexity=0.5,
        mode=RoutingMode.AUTO,
        confidence=0.8,
        reasoning_text="test",
        available_models=pool.available_models,
        estimated_input_tokens=100,
        max_output_tokens=1000,
        prompt="test",
        pricing=pool.pricing,
        served_qualities=pool.served_qualities,
        supports_images=pool.supports_images,
        require_images=False,
    )
    assert decision.model in ("gpt-4o", "llama-3-8b")


def test_require_images_without_supports_images_dict():
    """When supports_images is None, require_images is a no-op."""
    pool = _make_pool(models=["gpt-4o", "llama-3-8b"])
    decision = select_from_pool(
        complexity=0.5,
        mode=RoutingMode.AUTO,
        confidence=0.8,
        reasoning_text="test",
        available_models=pool.available_models,
        estimated_input_tokens=100,
        max_output_tokens=1000,
        prompt="test",
        pricing=pool.pricing,
        served_qualities=pool.served_qualities,
        supports_images=None,
        require_images=True,
    )
    assert decision.model in ("gpt-4o", "llama-3-8b")


def test_messages_contain_images_detection():
    """_messages_contain_images detects image_url and input_image types."""
    from llm_proxy.routing.api import _messages_contain_images

    # Text-only messages
    assert not _messages_contain_images([{"role": "user", "content": "hello"}])

    # Image URL in content array
    assert _messages_contain_images(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            }
        ]
    )

    # input_image type (Anthropic format)
    assert _messages_contain_images(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {
                        "type": "input_image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "...",
                        },
                    },
                ],
            },
        ]
    )

    # Empty messages
    assert not _messages_contain_images([])

    # inline_data type (Gemini format)
    assert _messages_contain_images(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {
                        "type": "inline_data",
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": "...",
                        },
                    },
                ],
            }
        ]
    )

    # image type (Anthropic format)
    assert _messages_contain_images(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "...",
                        },
                    },
                ],
            }
        ]
    )
