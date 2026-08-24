"""Gemini response parser mixin."""

from typing import Any

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
from llm_proxy.models.finish_reasons import map_finish_reason
from llm_proxy.models.types import AudioSource, ImageSource, Usage, VideoSource
from llm_proxy.serialization.gemini.annotations import extract_gemini_annotations
from llm_proxy.serialization.gemini.usage import billable_token_counts


def _mime_of(part_data: Any) -> str:
    """Return the MIME type of an inlineData/fileData part.

    The Gemini REST API emits camelCase (``mimeType``) while some fixtures
    and SDKs use snake_case (``mime_type``); accept both.
    """
    if not isinstance(part_data, dict):
        return ""
    return part_data.get("mime_type") or part_data.get("mimeType") or ""


class GeminiResponseParserMixin:
    """Parse Gemini API responses into InternalResponse."""

    def parse_provider_response(
        self, response: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> InternalResponse:
        candidates = response.get("candidates", [])
        output: list[ContentBlock] = []
        finish_reason = None
        provider_info: dict[str, Any] = {"provider": "gemini"}

        if candidates:
            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            text_parts: list[str] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue

                match part:
                    case {"thought": True, "text": str(text)}:
                        sig = part.get("thoughtSignature") or part.get("thought_signature")
                        output.append(ThinkingBlock(thinking=text, signature=sig))
                    case {"text": str(text)}:
                        text_parts.append(text)
                    case {"functionCall": fc}:
                        sig = part.get("thoughtSignature") or part.get("thought_signature")
                        output.append(
                            ToolUseBlock(
                                id=f"{fc.get('name', '')}_call",
                                name=fc.get("name", ""),
                                input=fc.get("args", {}),
                                extra={"thought_signature": sig},
                            )
                        )
                    case {"inlineData": inline} if _mime_of(inline).startswith("audio/"):
                        output.append(
                            AudioBlock(
                                source=AudioSource(
                                    type="base64",
                                    data=inline.get("data", ""),
                                    media_type=_mime_of(inline),
                                )
                            )
                        )
                    case {"inlineData": inline} if inline.get("mime_type", "").startswith("video/"):
                        output.append(
                            VideoBlock(
                                source=VideoSource(
                                    type="base64",
                                    data=inline.get("data", ""),
                                    media_type=inline.get("mime_type", ""),
                                )
                            )
                        )
                    case {"inlineData": inline}:
                        output.append(
                            ImageBlock(
                                source=ImageSource(
                                    type="base64",
                                    data=inline.get("data", ""),
                                    media_type=inline.get("mime_type", ""),
                                )
                            )
                        )
                    case {"fileData": fd} if _mime_of(fd).startswith("audio/"):
                        output.append(
                            AudioBlock(
                                source=AudioSource(
                                    type="file_id",
                                    data=fd.get("file_uri") or fd.get("fileUri", ""),
                                    media_type=_mime_of(fd),
                                )
                            )
                        )
                    case {"fileData": fd} if fd.get("mime_type", "").startswith("video/"):
                        output.append(
                            VideoBlock(
                                source=VideoSource(
                                    type="file_id",
                                    data=fd.get("file_uri", ""),
                                    media_type=fd.get("mime_type", ""),
                                )
                            )
                        )
                    case {"fileData": fd}:
                        output.append(
                            ImageBlock(
                                source=ImageSource(
                                    type="file_id",
                                    data=fd.get("file_uri", ""),
                                    media_type=fd.get("mime_type", ""),
                                )
                            )
                        )
                    case _:
                        pass

            text = "".join(text_parts)
            annotations: list[dict[str, Any]] = []
            if text:
                annotations = extract_gemini_annotations(candidate, text)

            if text_parts:
                output.insert(0, TextBlock(text=text, citations=annotations or None))

            gemini_reason = candidate.get("finishReason")
            finish_reason = map_finish_reason(gemini_reason, "gemini", "openai")

            if annotations:
                provider_info["annotations"] = annotations

            if "citationMetadata" in candidate:
                provider_info["citationMetadata"] = candidate["citationMetadata"]
            if "groundingMetadata" in candidate:
                provider_info["groundingMetadata"] = candidate["groundingMetadata"]
            if "safetyRatings" in candidate:
                provider_info["safety_ratings"] = candidate["safetyRatings"]
            if "avgLogprobs" in candidate:
                provider_info["avgLogprobs"] = candidate["avgLogprobs"]
            if "logprobsResult" in candidate:
                provider_info["logprobsResult"] = candidate["logprobsResult"]

        if "promptFeedback" in response:
            provider_info["promptFeedback"] = response["promptFeedback"]

        usage = None
        # Native Google Search grounding is billed per search request;
        # webSearchQueries lists the queries the model actually issued.
        # Computed before usage construction because search-grounded tool-use
        # prompt tokens are excluded from input billing (Google charges the
        # per-request search fee instead).
        web_search_requests = 0
        grounding = provider_info.get("groundingMetadata")
        if isinstance(grounding, dict):
            queries = grounding.get("webSearchQueries")
            if isinstance(queries, list):
                web_search_requests = len(queries)

        if "usageMetadata" in response:
            meta = response["usageMetadata"]
            input_tokens, output_tokens = billable_token_counts(
                meta, has_search_grounding=web_search_requests > 0
            )
            usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=meta.get("totalTokenCount", 0),
                cache_read_input_tokens=meta.get("cachedContentTokenCount"),
                reasoning_tokens=meta.get("thoughtsTokenCount"),
            )

        if web_search_requests > 0:
            if usage is None:
                usage = Usage()
            usage.web_search_requests = web_search_requests

        return InternalResponse(
            id=response.get("id") or generate_response_id(),
            model=model or "unknown",
            output=output,
            usage=usage,
            finish_reason=finish_reason,
            provider_info=provider_info,
        )
