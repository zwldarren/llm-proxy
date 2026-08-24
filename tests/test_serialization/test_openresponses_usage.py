"""Regression tests for OpenResponses usage detail preservation."""

import pytest

from llm_proxy.models import InternalResponse, TextBlock
from llm_proxy.models.types import (
    CompletionTokensDetails,
    PromptTokensDetails,
    Usage,
)
from llm_proxy.protocols.openresponses.serializer import _convert_usage


class TestConvertUsage:
    """`_convert_usage` must preserve all detail fields and not zero them."""

    def test_preserves_cached_tokens_and_reasoning_tokens(self):
        usage = _convert_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            }
        )
        assert usage["input_tokens_details"] == {"cached_tokens": 3}
        assert usage["output_tokens_details"] == {"reasoning_tokens": 2}

    def test_preserves_all_detail_fields(self):
        usage = _convert_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {
                    "audio_tokens": 1,
                    "cached_tokens": 3,
                    "image_tokens": 4,
                    "text_tokens": 5,
                    "cache_write_tokens": 2,
                    "video_tokens": 7,
                },
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 1,
                    "audio_tokens": 1,
                    "reasoning_tokens": 4,
                    "rejected_prediction_tokens": 1,
                    "image_tokens": 2,
                },
            }
        )
        assert usage["input_tokens_details"]["cache_write_tokens"] == 2
        assert usage["input_tokens_details"]["video_tokens"] == 7
        assert usage["output_tokens_details"]["image_tokens"] == 2

    def test_omits_details_when_not_provided(self):
        usage = _convert_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        # Spec: ResponseResource.usage requires both details objects (with
        # their own required inner fields), so zero-value defaults are emitted
        # when the upstream response did not provide them.
        assert usage["input_tokens_details"] == {"cached_tokens": 0}
        assert usage["output_tokens_details"] == {"reasoning_tokens": 0}


class TestFormatResponseUsageDetails:
    """Response formatting must pass Usage details through to the wire format."""

    @pytest.fixture
    def serializer(self):
        from llm_proxy.protocols.openresponses.serializer import (
            OpenResponsesProtocolSerializer,
        )

        return OpenResponsesProtocolSerializer()

    def test_format_response_includes_token_details(self, serializer):
        response = InternalResponse(
            id="resp_1",
            model="gpt-4",
            output=[TextBlock(text="Hello")],
            usage=Usage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=3),
                completion_tokens_details=CompletionTokensDetails(reasoning_tokens=2),
            ),
            finish_reason="stop",
        )
        result = serializer.format_response(response)
        assert result["usage"]["input_tokens_details"] == {"cached_tokens": 3}
        assert result["usage"]["output_tokens_details"] == {"reasoning_tokens": 2}
