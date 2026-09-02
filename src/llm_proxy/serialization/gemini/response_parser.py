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
from llm_proxy.serialization.gemini.code_execution import extract_code_execution_text
from llm_proxy.serialization.gemini.usage import billable_token_counts


def _part_field(part_data: Any, camel: str, snake: str) -> str:
    """Return a part field by its REST API (camelCase) name, with a
    snake_case fallback for fixtures/SDKs that pre-normalize."""
    if not isinstance(part_data, dict):
        return ""
    return part_data.get(camel) or part_data.get(snake) or ""


def _mime_of(part_data: Any) -> str:
    """Return the MIME type of an inlineData/fileData part."""
    return _part_field(part_data, "mimeType", "mime_type")


def _uri_of(part_data: Any) -> str:
    """Return the URI of a fileData part (``fileUri`` in the REST API)."""
    return _part_field(part_data, "fileUri", "file_uri")


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
                        output.append(
                            ThinkingBlock(thinking=text, signature=sig, signature_origin="gemini")
                        )
                    case {"text": str(text)}:
                        text_parts.append(text)
                    case {"functionCall": fc}:
                        sig = part.get("thoughtSignature") or part.get("thought_signature")
                        output.append(
                            ToolUseBlock(
                                # FunctionCall.id is the correlation id the
                                # client must echo back in FunctionResponse.
                                id=fc.get("id") or f"{fc.get('name', '')}_call",
                                name=fc.get("name", ""),
                                input=fc.get("args", {}),
                                extra={"thought_signature": sig},
                            )
                        )
                    case {"executableCode": _} | {"codeExecutionResult": _}:
                        text = extract_code_execution_text(part)
                        if text:
                            text_parts.append(text)
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
                    case {"inlineData": inline} if _mime_of(inline).startswith("video/"):
                        output.append(
                            VideoBlock(
                                source=VideoSource(
                                    type="base64",
                                    data=inline.get("data", ""),
                                    media_type=_mime_of(inline),
                                )
                            )
                        )
                    case {"inlineData": inline}:
                        output.append(
                            ImageBlock(
                                source=ImageSource(
                                    type="base64",
                                    data=inline.get("data", ""),
                                    media_type=_mime_of(inline),
                                )
                            )
                        )
                    case {"fileData": fd} if _mime_of(fd).startswith("audio/"):
                        output.append(
                            AudioBlock(
                                source=AudioSource(
                                    type="file_id",
                                    data=_uri_of(fd),
                                    media_type=_mime_of(fd),
                                )
                            )
                        )
                    case {"fileData": fd} if _mime_of(fd).startswith("video/"):
                        output.append(
                            VideoBlock(
                                source=VideoSource(
                                    type="file_id",
                                    data=_uri_of(fd),
                                    media_type=_mime_of(fd),
                                )
                            )
                        )
                    case {"fileData": fd}:
                        output.append(
                            ImageBlock(
                                source=ImageSource(
                                    type="file_id",
                                    data=_uri_of(fd),
                                    media_type=_mime_of(fd),
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
            grounding_supports = candidate.get("groundingSupports")
            if isinstance(grounding_supports, list) and grounding_supports:
                provider_info["groundingSupports"] = grounding_supports

        if "promptFeedback" in response:
            provider_info["promptFeedback"] = response["promptFeedback"]
        if "modelVersion" in response:
            provider_info["modelVersion"] = response["modelVersion"]

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
            # GenerateContentResponse carries the correlation id as
            # ``responseId`` (there is no ``id`` field); keep ``id`` as a
            # defensive fallback for fixtures.
            id=response.get("responseId") or response.get("id") or generate_response_id(),
            model=model or "unknown",
            output=output,
            usage=usage,
            finish_reason=finish_reason,
            provider_info=provider_info,
        )
