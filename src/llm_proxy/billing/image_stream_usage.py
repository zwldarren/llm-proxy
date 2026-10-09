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

from llm_proxy.models.types import PromptTokensDetails, Usage
from llm_proxy.observability.event_context import EventContext
from llm_proxy.serialization.usage import parse_usage as parse_wire_usage


class ImageStreamUsageTracker:
    """Parse provider SSE chunks for image-generation billing data."""

    def __init__(self) -> None:
        self._images_completed = 0
        self._usage: dict[str, Any] | None = None
        # The whole Gemini payload, not just its usageMetadata block: the intake
        # seam reads the candidates beside it for the grounding fact, which decides
        # whether tool-use prompt tokens are billed as input (ADR-0025).
        self._gemini_payload: dict[str, Any] | None = None
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
        if self._gemini_payload is None:
            return None
        usage = self._gemini_payload.get("usageMetadata")
        return usage if isinstance(usage, dict) else None

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
            self._gemini_payload = data
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
        """Write captured billing data into an EventContext.

        Each dialect is turned into a canonical record rather than into flat
        fields, so the merge rule that reconciles a turn's observations applies
        here too (ADR-0025). ``images_generated`` is the proxy's own count, not a
        provider fact, so it stays a dimension on the context.
        """
        if self._images_completed > 0:
            # Overwrite request-side n-based fallback with actual count.
            ctx.images_generated = self._images_completed

        usage = self._usage
        if isinstance(usage, dict):
            # OpenAI gpt-image usage: input_tokens, output_tokens,
            # input_tokens_details.image_tokens
            image_details = usage.get("input_tokens_details")
            image_tokens = (
                image_details.get("image_tokens") if isinstance(image_details, dict) else None
            )
            ctx.update_usage(
                Usage(
                    input_tokens=usage.get("input_tokens") or 0,
                    output_tokens=usage.get("output_tokens") or 0,
                    total_tokens=usage.get("total_tokens"),
                    prompt_tokens_details=PromptTokensDetails(image_tokens=image_tokens),
                )
            )
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
            ctx.update_usage(
                Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=iu.get("total_tokens"),
                )
            )
        elif self._gemini_payload is not None:
            # The gemini dialect, through the same seam the response parsers use, so
            # an image stream and a JSON body bill the same wire shape the same way:
            # tool-use prompt tokens, thoughts and cached content are the seam's
            # mapping, not this tracker's (ADR-0025).
            ctx.update_usage(parse_wire_usage("gemini", self._gemini_payload))
