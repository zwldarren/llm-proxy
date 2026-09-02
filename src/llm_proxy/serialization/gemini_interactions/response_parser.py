"""Response parser for the Gemini Interactions API serializer.

Parses an Interaction resource (``steps[]`` timeline) into
``InternalResponse``:

- ``model_output`` steps → text (with inline annotations), image, audio,
  video, document blocks.
- ``thought`` steps → ThinkingBlock (summary text + signature).
- ``function_call`` steps → ToolUseBlock.
- ``google_search_call`` steps → counted as web search requests (billed per
  search).

Finish reasons derive from ``interaction.status``: completed→stop,
requires_action→tool_calls, incomplete→length; failed/cancelled raise a
ProviderError (error propagation).
"""

from typing import Any

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.utils import generate_response_id
from llm_proxy.models import (
    AudioBlock,
    ContentBlock,
    ImageBlock,
    InternalResponse,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    VideoBlock,
)
from llm_proxy.models.types import AudioSource, ImageSource, Usage, VideoSource
from llm_proxy.serialization.gemini_interactions.annotations import (
    content_annotations_to_openai,
)
from llm_proxy.serialization.gemini_interactions.finish_reason import (
    FAILED_STATUSES,
    STATUS_TO_FINISH_REASON,
    interaction_error_message,
)
from llm_proxy.serialization.gemini_interactions.usage import (
    interactions_billable_token_counts,
    interactions_normalize_usage,
    interactions_web_search_requests,
)


def _item_annotations_to_openai(item: dict[str, Any], text: str) -> list[dict[str, Any]] | None:
    """Convert inline content annotations to OpenAI-style url_citation dicts."""
    raw_annotations = item.get("annotations") or []
    if not raw_annotations:
        return None
    return content_annotations_to_openai(raw_annotations, len(text))


class GeminiInteractionsResponseParserMixin:
    """Parse Interactions API responses into InternalResponse."""

    def parse_provider_response(
        self, response: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> InternalResponse:
        status = response.get("status")
        if status in FAILED_STATUSES:
            # The Interaction resource carries failure details in ``error``
            # (singular, per the SDK docs); ``errors`` is a defensive fallback.
            message = interaction_error_message(response)
            raise ProviderError(
                message=message or f"Gemini interaction ended with status={status!r}",
                error_type="api_error",
                status_code=502,
                provider_name="gemini-interactions",
                original_error=response,
            )

        output: list[ContentBlock] = []
        web_search_requests = 0
        # The Interactions API requires the thought signature to be replayed
        # with a function_call step in stateless multi-turn tool conversations
        # (the live API rejects function_call steps without it). The signature
        # lives on the preceding ``thought`` step, so track the most recent one
        # and attach it to subsequent function_call steps; the adapter caches it
        # keyed by tool call id and re-attaches on the next request.
        last_thought_signature: str | None = None

        for step in response.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            step_type = step.get("type")

            if step_type == "model_output":
                for item in step.get("content", []) or []:
                    block = self._content_item_to_block(item)
                    if block is not None:
                        output.append(block)
            elif step_type == "thought":
                signature = step.get("signature") or None
                if signature:
                    last_thought_signature = signature
                summary = step.get("summary") or []
                texts = [
                    c.get("text", "")
                    for c in summary
                    if isinstance(c, dict) and c.get("type") == "text" and c.get("text")
                ]
                if texts:
                    output.append(
                        ThinkingBlock(
                            thinking="\n".join(texts),
                            signature=signature,
                        )
                    )
            elif step_type == "function_call":
                block = ToolUseBlock(
                    id=step.get("id") or f"{step.get('name', '')}_call",
                    name=step.get("name", ""),
                    input=step.get("arguments") or {},
                )
                if last_thought_signature:
                    block.extra = {"thought_signature": last_thought_signature}
                output.append(block)
            elif step_type == "google_search_call":
                web_search_requests += 1

        usage = None
        raw_usage = response.get("usage")
        if isinstance(raw_usage, dict):
            raw_usage = interactions_normalize_usage(raw_usage)
            if web_search_requests == 0:
                web_search_requests = interactions_web_search_requests(raw_usage)
            input_tokens, output_tokens = interactions_billable_token_counts(
                raw_usage, has_search_grounding=web_search_requests > 0
            )
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=raw_usage.get("total_tokens"),
                cache_read_input_tokens=raw_usage.get("total_cached_tokens") or None,
                reasoning_tokens=raw_usage.get("total_thought_tokens") or None,
            )

        if web_search_requests > 0:
            if usage is None:
                usage = Usage()
            usage.web_search_requests = web_search_requests

        # The live API reports ``requires_action`` for function_call responses
        # in non-streaming mode but ``completed`` in streaming mode; derive the
        # finish reason from the accumulated output so both shapes map to
        # ``tool_calls`` consistently.
        finish_reason = STATUS_TO_FINISH_REASON.get(status)
        if finish_reason == "stop" and any(isinstance(block, ToolUseBlock) for block in output):
            finish_reason = "tool_calls"

        return InternalResponse(
            id=response.get("id") or generate_response_id(),
            model=model or "unknown",
            output=output,
            usage=usage,
            finish_reason=finish_reason,
            provider_info={"provider": "gemini-interactions"},
        )

    @staticmethod
    def _content_item_to_block(item: dict[str, Any]) -> ContentBlock | None:
        """Convert a Content item into an internal ContentBlock."""
        if not isinstance(item, dict):
            return None
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            annotations = _item_annotations_to_openai(item, text)
            return TextBlock(text=text, citations=annotations)
        if item_type == "image":
            data = item.get("data")
            if data:
                return ImageBlock(
                    source=ImageSource(
                        type="base64",
                        data=data,
                        media_type=item.get("mime_type") or "image/png",
                    )
                )
            uri = item.get("uri")
            if uri:
                return ImageBlock(
                    source=ImageSource(type="file_id", data=uri, media_type=item.get("mime_type"))
                )
            return None
        if item_type == "audio":
            data = item.get("data")
            if data:
                return AudioBlock(
                    source=AudioSource(
                        type="base64",
                        data=data,
                        media_type=item.get("mime_type") or "audio/l16",
                    )
                )
            uri = item.get("uri")
            if uri:
                return AudioBlock(
                    source=AudioSource(type="file_id", data=uri, media_type=item.get("mime_type"))
                )
            return None
        if item_type == "video":
            data = item.get("data")
            if data:
                return VideoBlock(
                    source=VideoSource(
                        type="base64",
                        data=data,
                        media_type=item.get("mime_type") or "video/mp4",
                    )
                )
            uri = item.get("uri")
            if uri:
                return VideoBlock(
                    source=VideoSource(type="file_id", data=uri, media_type=item.get("mime_type"))
                )
            return None
        # document and unknown item types have no dedicated internal block;
        # fall back to text when a text representation exists.
        text = item.get("text")
        if text:
            return TextBlock(text=text)
        return None
