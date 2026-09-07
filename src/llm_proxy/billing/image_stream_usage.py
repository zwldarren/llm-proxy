"""Parse provider SSE chunks for image-generation billing data.

Extracted from ``core/processing/streaming_processor.py`` (issue LLMP-3):
billing is not a streaming-processor concern — this module is public so
tests and callers import the tracker by name instead of reaching for a
processor-private class.

Supported wire shapes:

- OpenAI Images streaming: ``data: {"type":"image_generation.completed","usage":{...}}``
  (and ``image_edit.completed`` for edit streaming)
- Gemini (generateContent): ``data: {...,"usageMetadata":{...}}``
  (inlineData images counted)
- Gemini (Interactions): ``data: {"type":"step.delta","delta":{"type":"image",…}}``
  partial images + ``data: {"type":"interaction.completed",
  "interaction":{"usage":{…}}}`` with the new usage vocabulary.
"""

from typing import Any

import orjson
from orjson import JSONDecodeError

from llm_proxy.observability.event_context import EventContext


class ImageStreamUsageTracker:
    """Parse provider SSE chunks for image-generation billing data."""

    def __init__(self) -> None:
        self._images_completed = 0
        self._usage: dict[str, Any] | None = None
        self._gemini_usage: dict[str, Any] | None = None
        # Interactions-API usage (new vocabulary: total_input_tokens, …)
        self._interactions_usage: dict[str, Any] | None = None

    @property
    def images_completed(self) -> int:
        return self._images_completed

    @property
    def captured_usage(self) -> dict[str, Any] | None:
        return self._usage

    @property
    def gemini_usage(self) -> dict[str, Any] | None:
        return self._gemini_usage

    @property
    def interactions_usage(self) -> dict[str, Any] | None:
        return self._interactions_usage

    def observe(self, chunk: Any) -> None:
        if not isinstance(chunk, str):
            return
        for line in chunk.split("\n"):
            line = line.strip()
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if not payload or payload == "[DONE]":
                continue
            try:
                data: dict[str, Any] = orjson.loads(payload)
            except JSONDecodeError:
                continue
            self._observe_json(data)

    def _observe_json(self, data: dict[str, Any]) -> None:
        # OpenAI Images API streaming: type field in payload
        event_type = data.get("type")
        if event_type in {"image_generation.completed", "image_edit.completed"}:
            self._images_completed += 1
            usage = data.get("usage")
            if isinstance(usage, dict):
                self._usage = usage
            return

        # Gemini Interactions: step.delta image content + interaction.completed
        # usage. Accept both the "type" and "event_type" discriminator keys
        # (the API reference uses event_type; the migration guide uses type).
        gemini_event = data.get("type") or data.get("event_type")
        if gemini_event == "step.delta":
            delta = data.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "image" and delta.get("data"):
                self._images_completed += 1
            return
        if gemini_event == "interaction.completed":
            interaction = data.get("interaction")
            usage = interaction.get("usage") if isinstance(interaction, dict) else None
            if isinstance(usage, dict):
                self._interactions_usage = usage
            return

        # Gemini (generateContent): usageMetadata + candidates with inlineData images
        gemini_usage = data.get("usageMetadata")
        if isinstance(gemini_usage, dict):
            self._gemini_usage = gemini_usage
        candidates = data.get("candidates")
        if isinstance(candidates, list):
            for c in candidates:
                if isinstance(c, dict):
                    parts = c.get("content", {}).get("parts", [])
                    if isinstance(parts, list):
                        for p in parts:
                            if isinstance(p, dict):
                                inline = p.get("inlineData")
                                if (
                                    isinstance(inline, dict)
                                    and isinstance(inline.get("mimeType", ""), str)
                                    and inline["mimeType"].startswith("image/")
                                ):
                                    self._images_completed += 1

    def apply_to(self, ctx: EventContext) -> None:
        """Write captured billing data into an EventContext."""
        if self._images_completed > 0:
            # Overwrite request-side n-based fallback with actual count.
            ctx.images_generated = self._images_completed

        usage = self._usage
        if isinstance(usage, dict):
            # OpenAI gpt-image usage: input_tokens, output_tokens,
            # input_tokens_details.image_tokens
            if usage.get("input_tokens") is not None:
                ctx.prompt_tokens = usage["input_tokens"]
            if usage.get("output_tokens") is not None:
                ctx.completion_tokens = usage["output_tokens"]
            if usage.get("total_tokens") is not None:
                ctx.total_tokens = usage["total_tokens"]
            itd = usage.get("input_tokens_details")
            if isinstance(itd, dict) and itd.get("image_tokens") is not None:
                ctx.image_input_tokens = itd["image_tokens"]
        elif self._interactions_usage is not None:
            # Gemini Interactions usage: total_input_tokens, total_output_tokens,
            # total_thought_tokens, total_tool_use_tokens, total_tokens.
            iu = self._interactions_usage
            from llm_proxy.serialization.gemini_interactions.usage import (
                interactions_billable_token_counts,
                interactions_web_search_requests,
            )

            has_search = interactions_web_search_requests(iu) > 0
            input_tokens, output_tokens = interactions_billable_token_counts(
                iu, has_search_grounding=has_search
            )
            ctx.prompt_tokens = input_tokens
            ctx.completion_tokens = output_tokens
            if "total_tokens" in iu:
                ctx.total_tokens = iu["total_tokens"]
            else:
                ctx.total_tokens = (input_tokens + output_tokens) or None
        elif self._gemini_usage is not None:
            # Gemini usageMetadata: promptTokenCount, candidatesTokenCount, etc.
            gu = self._gemini_usage
            ctx.prompt_tokens = gu.get("promptTokenCount", 0) or 0
            ctx.completion_tokens = gu.get("candidatesTokenCount", 0) or 0
            if "totalTokenCount" in gu:
                ctx.total_tokens = gu["totalTokenCount"]
            else:
                ctx.total_tokens = (ctx.prompt_tokens or 0) + (ctx.completion_tokens or 0) or None
