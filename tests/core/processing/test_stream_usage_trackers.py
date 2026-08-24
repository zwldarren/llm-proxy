"""Tests for the streaming usage trackers used during image generation and
transcription streaming.

These trackers parse provider SSE chunks to capture billing data (image counts,
token usage, audio duration) that is not available through the normal streaming
transformer path.
"""

from unittest.mock import MagicMock

import orjson

from llm_proxy.core.processing.streaming_processor import (
    _ImageStreamUsageTracker,
    _TranscriptionStreamUsageTracker,
)
from llm_proxy.observability.event_context import EventContext


def _sse(payload: dict) -> str:
    """Build a single ``data:`` SSE line for the given payload."""
    return f"data: {orjson.dumps(payload).decode()}\n\n"


class TestImageStreamUsageTrackerOpenAI:
    """_ImageStreamUsageTracker parsing OpenAI gpt-image streaming events."""

    def test_counts_completed_images_and_captures_usage(self):
        """Each image_generation.completed event increments the image count and
        captures the usage object (tokens + image_tokens)."""
        tracker = _ImageStreamUsageTracker()
        chunk = "event: image_generation.completed\n" + _sse(
            {
                "type": "image_generation.completed",
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                    "input_tokens_details": {"image_tokens": 400},
                },
            }
        )

        tracker.observe(chunk)

        assert tracker.images_completed == 1
        assert tracker.captured_usage == {
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "input_tokens_details": {"image_tokens": 400},
        }

    def test_multiple_completed_images(self):
        """Multiple completed events accumulate the image count."""
        tracker = _ImageStreamUsageTracker()
        for _ in range(3):
            tracker.observe(
                "event: image_generation.completed\n"
                + _sse({"type": "image_generation.completed", "usage": {"input_tokens": 1}})
            )
        assert tracker.images_completed == 3

    def test_apply_to_writes_usage_to_event_context(self):
        """apply_to overwrites the n-fallback with the actual count and writes
        token usage + image_input_tokens to the EventContext."""
        tracker = _ImageStreamUsageTracker()
        tracker.observe(
            "event: image_generation.completed\n"
            + _sse(
                {
                    "type": "image_generation.completed",
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 500,
                        "total_tokens": 1500,
                        "input_tokens_details": {"image_tokens": 400},
                    },
                }
            )
        )

        ctx = EventContext(request_id="req", trace_id="trace", model="gpt-image-1")
        ctx.images_generated = 2  # request-side n fallback
        tracker.apply_to(ctx)

        assert ctx.images_generated == 1
        assert ctx.prompt_tokens == 1000
        assert ctx.completion_tokens == 500
        assert ctx.total_tokens == 1500
        assert ctx.image_input_tokens == 400

    def test_done_marker_ignored(self):
        """The [DONE] marker is not parsed as JSON."""
        tracker = _ImageStreamUsageTracker()
        tracker.observe("data: [DONE]\n\n")
        assert tracker.images_completed == 0
        assert tracker.captured_usage is None

    def test_non_string_chunk_ignored(self):
        """Bytes chunks (e.g. audio) are ignored without error."""
        tracker = _ImageStreamUsageTracker()
        tracker.observe(b"bytes-not-relevant")
        assert tracker.images_completed == 0


class TestImageStreamUsageTrackerGemini:
    """_ImageStreamUsageTracker parsing Gemini image streaming responses."""

    def test_counts_inline_data_images_and_captures_usage_metadata(self):
        """Gemini candidates with inlineData image parts are counted, and
        usageMetadata is captured for token billing."""
        tracker = _ImageStreamUsageTracker()
        chunk = _sse(
            {
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 50,
                    "totalTokenCount": 150,
                },
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"inlineData": {"mimeType": "image/png", "data": "abc"}},
                                {"inlineData": {"mimeType": "image/jpeg", "data": "def"}},
                                {"text": "revised prompt"},
                            ]
                        }
                    }
                ],
            }
        )

        tracker.observe(chunk)

        assert tracker.images_completed == 2
        assert tracker.gemini_usage == {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "totalTokenCount": 150,
        }

    def test_apply_to_writes_gemini_usage(self):
        """apply_to writes Gemini usageMetadata token counts to the EventContext."""
        tracker = _ImageStreamUsageTracker()
        tracker.observe(
            _sse(
                {
                    "usageMetadata": {
                        "promptTokenCount": 100,
                        "candidatesTokenCount": 50,
                        "totalTokenCount": 150,
                    },
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"inlineData": {"mimeType": "image/png", "data": "x"}}]
                            }
                        }
                    ],
                }
            )
        )

        ctx = EventContext(request_id="req", trace_id="trace", model="imagen-3.0")
        tracker.apply_to(ctx)

        assert ctx.images_generated == 1
        assert ctx.prompt_tokens == 100
        assert ctx.completion_tokens == 50
        assert ctx.total_tokens == 150

    def test_apply_to_gemini_without_total_token_count(self):
        """When totalTokenCount is absent, total is derived from prompt + completion."""
        tracker = _ImageStreamUsageTracker()
        tracker.observe(
            _sse(
                {
                    "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50},
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"inlineData": {"mimeType": "image/png", "data": "x"}}]
                            }
                        }
                    ],
                }
            )
        )

        ctx = EventContext(request_id="req", trace_id="trace", model="imagen-3.0")
        tracker.apply_to(ctx)

        assert ctx.total_tokens == 150

    def test_non_image_inline_data_not_counted(self):
        """inlineData with a non-image mimeType is not counted as a generated image."""
        tracker = _ImageStreamUsageTracker()
        tracker.observe(
            _sse(
                {
                    "candidates": [
                        {"content": {"parts": [{"inlineData": {"mimeType": "audio/mp3"}}]}}
                    ]
                }
            )
        )
        assert tracker.images_completed == 0


class TestImageStreamUsageTrackerGeminiInteractions:
    """_ImageStreamUsageTracker parsing Gemini Interactions streaming events."""

    def test_counts_step_delta_image_content(self):
        """step.delta events with image content increment the image count."""
        tracker = _ImageStreamUsageTracker()
        tracker.observe(
            _sse({"type": "step.delta", "index": 0, "delta": {"type": "image", "data": "B1"}})
        )
        tracker.observe(
            _sse(
                {
                    "event_type": "step.delta",
                    "index": 1,
                    "delta": {"type": "image", "data": "B2"},
                }
            )
        )
        tracker.observe(
            _sse({"type": "step.delta", "index": 2, "delta": {"type": "text", "text": "x"}})
        )
        assert tracker.images_completed == 2

    def test_captures_interaction_completed_usage(self):
        """interaction.completed carries the new-vocabulary usage dict."""
        tracker = _ImageStreamUsageTracker()
        tracker.observe(
            _sse(
                {
                    "type": "interaction.completed",
                    "interaction": {
                        "id": "int_1",
                        "status": "completed",
                        "usage": {
                            "total_input_tokens": 10,
                            "total_output_tokens": 5,
                            "total_thought_tokens": 0,
                            "total_tool_use_tokens": 0,
                            "total_tokens": 15,
                        },
                    },
                }
            )
        )
        assert tracker.interactions_usage == {
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "total_thought_tokens": 0,
            "total_tool_use_tokens": 0,
            "total_tokens": 15,
        }

    def test_apply_to_writes_interactions_usage(self):
        """apply_to maps the new usage vocabulary onto the EventContext."""
        tracker = _ImageStreamUsageTracker()
        tracker.observe(
            _sse(
                {
                    "type": "step.delta",
                    "index": 0,
                    "delta": {"type": "image", "data": "B1"},
                }
            )
        )
        tracker.observe(
            _sse(
                {
                    "type": "interaction.completed",
                    "interaction": {
                        "id": "int_1",
                        "status": "completed",
                        "usage": {
                            "total_input_tokens": 10,
                            "total_output_tokens": 5,
                            "total_thought_tokens": 2,
                            "total_tool_use_tokens": 3,
                            "total_tokens": 15,
                        },
                    },
                }
            )
        )

        ctx = EventContext(request_id="req", trace_id="trace", model="gemini-3.1-flash-image")
        ctx.images_generated = 1  # request-side n fallback
        tracker.apply_to(ctx)

        assert ctx.images_generated == 1
        # tool use folds into input; thoughts fold into output
        assert ctx.prompt_tokens == 13
        assert ctx.completion_tokens == 7
        assert ctx.total_tokens == 15

    def test_apply_to_search_grounding_excludes_tool_use(self):
        """grounding_tool_count flips has_search_grounding like the chat path."""
        tracker = _ImageStreamUsageTracker()
        tracker.observe(
            _sse(
                {
                    "type": "interaction.completed",
                    "interaction": {
                        "id": "int_1",
                        "status": "completed",
                        "usage": {
                            "total_input_tokens": 100,
                            "total_output_tokens": 25,
                            "total_thought_tokens": 0,
                            "total_tool_use_tokens": 50,
                            "total_tokens": 125,
                            "grounding_tool_count": [{"type": "google_search", "count": 1}],
                        },
                    },
                }
            )
        )

        ctx = EventContext(request_id="req", trace_id="trace", model="gemini-3.1-flash-image")
        tracker.apply_to(ctx)

        # search-grounded tool-use tokens are NOT billed at the input rate
        assert ctx.prompt_tokens == 100
        assert ctx.completion_tokens == 25

    def test_apply_to_openai_style_usage_aliases(self):
        """The migration guide streams OpenAI-style usage on completed events."""
        tracker = _ImageStreamUsageTracker()
        tracker.observe(
            _sse(
                {
                    "type": "interaction.completed",
                    "interaction": {
                        "id": "int_1",
                        "status": "completed",
                        "usage": {"prompt_tokens": 256, "completion_tokens": 128},
                    },
                }
            )
        )

        ctx = EventContext(request_id="req", trace_id="trace", model="gemini-3.1-flash-image")
        tracker.apply_to(ctx)

        assert ctx.prompt_tokens == 256
        assert ctx.completion_tokens == 128
        assert ctx.total_tokens == 384


class TestTranscriptionStreamUsageTracker:
    """_TranscriptionStreamUsageTracker parsing STT streaming usage."""

    def test_captures_token_based_usage(self):
        """gpt-4o-transcribe streaming usage (token-based) is captured."""
        tracker = _TranscriptionStreamUsageTracker()
        tracker.observe(
            _sse(
                {
                    "type": "transcript.text.done",
                    "usage": {"input_tokens": 100, "output_tokens": 0, "total_tokens": 100},
                }
            )
        )
        assert tracker.captured_usage == {
            "input_tokens": 100,
            "output_tokens": 0,
            "total_tokens": 100,
        }

    def test_captures_duration_based_usage(self):
        """whisper duration-based usage is captured."""
        tracker = _TranscriptionStreamUsageTracker()
        tracker.observe(_sse({"usage": {"type": "duration", "seconds": 60}}))
        assert tracker.captured_usage == {"type": "duration", "seconds": 60}

    def test_apply_to_uses_adapter_parse_usage(self):
        """apply_to delegates to adapter._parse_usage and updates the context."""
        tracker = _TranscriptionStreamUsageTracker()
        tracker.observe(_sse({"usage": {"type": "duration", "seconds": 90}}))

        adapter = MagicMock()
        # Simulate the BaseHttpProvider._parse_usage duration handling
        from llm_proxy.models.types import Usage

        adapter._parse_usage.return_value = Usage(audio_duration_seconds=90)

        ctx = EventContext(request_id="req", trace_id="trace", model="whisper-1")
        tracker.apply_to(ctx, adapter)

        adapter._parse_usage.assert_called_once_with({"type": "duration", "seconds": 90})
        assert ctx.audio_duration_seconds == 90

    def test_apply_to_no_usage_is_noop(self):
        """apply_to with no captured usage does not call the adapter."""
        tracker = _TranscriptionStreamUsageTracker()
        adapter = MagicMock()
        ctx = EventContext(request_id="req", trace_id="trace", model="whisper-1")
        tracker.apply_to(ctx, adapter)
        adapter._parse_usage.assert_not_called()

    def test_apply_to_parse_returns_none(self):
        """apply_to is a no-op when _parse_usage returns None."""
        tracker = _TranscriptionStreamUsageTracker()
        tracker.observe(_sse({"usage": {"type": "duration", "seconds": 90}}))

        adapter = MagicMock()
        adapter._parse_usage.return_value = None

        ctx = EventContext(request_id="req", trace_id="trace", model="whisper-1")
        tracker.apply_to(ctx, adapter)

        assert ctx.audio_duration_seconds is None

    def test_done_marker_ignored(self):
        tracker = _TranscriptionStreamUsageTracker()
        tracker.observe("data: [DONE]\n\n")
        assert tracker.captured_usage is None


class TestStreamingTtsBillingDimension:
    """RequestExecutionStage._populate_unit_billing_dimensions sets tts_characters
    for streaming TTS so it is billable even though the response has no usage."""

    def test_populate_unit_billing_dimensions_sets_tts_characters(self):
        from types import SimpleNamespace

        from llm_proxy.core.processing.stages.request_execution import RequestExecutionStage
        from llm_proxy.models.audio import InternalSpeechRequest

        request = InternalSpeechRequest(
            model="tts-1",
            input="Hello, world!",
            voice="alloy",
            response_format="mp3",
            stream=True,
        )
        ctx = EventContext(request_id="req", trace_id="trace", model="tts-1")
        state = SimpleNamespace(unified_request=request, event_context=ctx)

        RequestExecutionStage._populate_unit_billing_dimensions(state)

        assert ctx.tts_characters == len("Hello, world!")

    def test_populate_unit_billing_dimensions_skips_non_speech(self):
        from types import SimpleNamespace

        from llm_proxy.core.processing.stages.request_execution import RequestExecutionStage
        from llm_proxy.models.conversation import ConversationContext
        from llm_proxy.models.internal import InternalRequest

        request = InternalRequest(
            model="gpt-4",
            conversation=ConversationContext(system_messages=[], messages=[]),
            stream=True,
        )
        ctx = EventContext(request_id="req", trace_id="trace", model="gpt-4")
        state = SimpleNamespace(unified_request=request, event_context=ctx)

        RequestExecutionStage._populate_unit_billing_dimensions(state)

        # tts_characters must not be set for non-speech requests
        assert ctx.tts_characters is None
