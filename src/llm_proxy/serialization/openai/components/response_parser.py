"""OpenAI provider response parser component.

Responsible for parsing OpenAI-format provider responses into InternalResponse.
"""

import uuid
from typing import Any

import orjson
from orjson import JSONDecodeError

from llm_proxy.core.utils import create_image_source_from_url, generate_response_id
from llm_proxy.models import (
    AudioBlock,
    ContentBlock,
    CustomToolUseBlock,
    ImageBlock,
    InternalResponse,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llm_proxy.models.types import (
    AudioSource,
    ChoiceLogprobs,
    ChoiceMetadata,
    CompletionTokensDetails,
    PromptTokensDetails,
    TokenLogprob,
    Usage,
)
from llm_proxy.observability.logger import get_logger
from llm_proxy.serialization.openai.components.request_builder import (
    OpenAIRequestBuilder,
)

logger = get_logger(__name__)


def fold_deepseek_cache_hits(cached_tokens: int | None, cache_hits: Any) -> int | None:
    """Fold DeepSeek-style top-level ``prompt_cache_hit_tokens`` into ``cached_tokens``.

    DeepSeek reports ``prompt_tokens = hit + miss``, so hits are already
    included in ``prompt_tokens``; folding them into ``cached_tokens`` makes
    cache pricing apply instead of silently billing them at the full input
    rate. An explicit ``cached_tokens`` value wins — hits fold in only when
    the details carry no cached value.
    """
    if cached_tokens is not None:
        return cached_tokens
    if isinstance(cache_hits, int) and cache_hits > 0:
        return cache_hits
    return None


class OpenAIResponseParser:
    """Parses OpenAI Chat Completions response dicts into InternalResponse.

    Handles choices, content, tool calls, refusal, reasoning, audio, logprobs,
    annotations, and usage with token details.
    """

    def __init__(self, request_builder: OpenAIRequestBuilder | None = None) -> None:
        self._request_builder = request_builder or OpenAIRequestBuilder()

    def parse(
        self,
        response: dict[str, Any],
        model: str | None = None,
        **kwargs: Any,
    ) -> InternalResponse:
        """Parse OpenAI response to InternalResponse."""
        choices = response.get("choices", [])
        output: list[ContentBlock] = []
        finish_reason = None
        logprobs = None
        annotations = None
        choices_outputs: list[list[ContentBlock]] = []
        choices_metadata: list[ChoiceMetadata] = []

        provider_info: dict[str, Any] = {"provider": "openai"}

        if choices:
            # Parse all choices when n > 1
            native_finish_reasons: list[str] = []
            for idx, choice in enumerate(choices):
                if choice is None:
                    choice = {}
                message = choice.get("message") or {}

                # Detect and cache reasoning field preference from the first choice.
                # The convention belongs to the *model*, so it is cached under
                # the routed model id (the key future requests look up), plus
                # under the upstream-reported model when the response reports a
                # different one (model aliasing).
                if idx == 0:
                    base_url = kwargs.get("base_url")
                    # Function-level import: serialization modules must not
                    # import provider packages at module scope (the adapter
                    # registry pulls serialization back in — circular).
                    from llm_proxy.providers.reasoning import detect_reasoning_field_in_message

                    detected_field = detect_reasoning_field_in_message(message)
                    if detected_field and base_url is not None:
                        self._request_builder.record_reasoning_field_preference(
                            base_url,
                            detected_field,
                            model=model,
                            response_model=response.get("model"),
                        )

                # Preserve provider-specific per-choice finish reasons
                # (e.g. OpenRouter native_finish_reason) when present.
                native_finish = choice.get("native_finish_reason")
                if native_finish and isinstance(native_finish, str):
                    native_finish_reasons.append(native_finish)

                choice_output = self._parse_message_content(message)
                if idx == 0:
                    output = choice_output
                    finish_reason = choice.get("finish_reason")
                    logprobs = self._parse_logprobs(choice)
                    annotations = self._parse_annotations(message)
                else:
                    choices_outputs.append(choice_output)
                    choices_metadata.append(
                        ChoiceMetadata(
                            finish_reason=choice.get("finish_reason"),
                            logprobs=self._parse_logprobs(choice),
                            annotations=self._parse_annotations(message),
                        )
                    )

            if native_finish_reasons:
                provider_info["native_finish_reasons"] = native_finish_reasons

        usage = self.parse_usage(response)

        # Use the upstream-reported model when present (e.g. OpenRouter may
        # return the actual model that served the request).
        upstream_model = response.get("model")
        if upstream_model:
            model = upstream_model

        if response.get("system_fingerprint"):
            provider_info["system_fingerprint"] = response.get("system_fingerprint")
        if response.get("service_tier"):
            provider_info["service_tier"] = response.get("service_tier")
        if annotations:
            provider_info["annotations"] = annotations

        return InternalResponse(
            id=response.get("id") or generate_response_id(),
            model=model or "unknown",
            output=output,
            usage=usage,
            finish_reason=finish_reason,
            provider_info=provider_info,
            logprobs=logprobs,
            choices_outputs=choices_outputs,
            choices_metadata=choices_metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_message_content(self, message: dict[str, Any]) -> list[Any]:
        """Parse message content into ContentBlock list."""
        output: list[Any] = []

        # Text content (string or list of structured parts)
        content = message.get("content")
        if content:
            if isinstance(content, str):
                output.append(TextBlock(text=content))
            elif isinstance(content, list):
                texts = []
                for part in content:
                    match part:
                        case {"type": "text", "text": str(t)}:
                            texts.append(t)
                        case {"type": "image_url", "image_url": {"url": str(url)}}:
                            texts.append(f"![image]({url})")
                        case {"type": "image_url"}:
                            texts.append("[image]")
                        case {"type": "refusal", "refusal": str(r)}:
                            texts.append(r)
                        case dict():
                            texts.append(str(part))
                        case _:
                            texts.append(str(part))
                if texts:
                    output.append(TextBlock(text=" ".join(texts)))

        # Tool calls
        tool_calls = message.get("tool_calls")
        if tool_calls:
            output.extend(self._parse_tool_calls(tool_calls))

        # Refusal
        refusal = message.get("refusal")
        if refusal:
            output.append(RefusalBlock(refusal=refusal))

        # Reasoning/thinking
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is None:
            reasoning_content = message.get("reasoning")
        if reasoning_content:
            output.append(ThinkingBlock(thinking=reasoning_content))

        # Audio output
        audio = message.get("audio")
        if audio and isinstance(audio, dict):
            output.append(
                AudioBlock(
                    source=AudioSource(
                        type="base64",
                        data=audio.get("data", "") or "",
                        media_type="audio/wav",
                        id=audio.get("id"),
                        expires_at=audio.get("expires_at"),
                        transcript=audio.get("transcript"),
                    )
                )
            )

        # Image output (OpenRouter image generation via chat completions)
        images = message.get("images")
        if images and isinstance(images, list):
            for img in images:
                if isinstance(img, dict) and img.get("type") == "image_url":
                    image_url = img.get("image_url") or {}
                    url = image_url.get("url", "")
                    source = create_image_source_from_url(url)
                    if source:
                        output.append(ImageBlock(source=source))
                    else:
                        logger.warning("Failed to create image source from URL", extra={"url": url})
                else:
                    logger.debug("Skipping unexpected image format in response", extra={"img": img})

        return output

    @staticmethod
    def _parse_tool_calls(tool_calls: list[dict[str, Any]]) -> list[Any]:
        """Parse tool calls into ToolUseBlock / CustomToolUseBlock."""
        output: list[Any] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            match tc.get("type"):
                case "function":
                    func = tc.get("function") or {}
                    args = func.get("arguments", "{}")
                    try:
                        input_dict = orjson.loads(args) if isinstance(args, str) else args
                    except JSONDecodeError:
                        input_dict = {}
                    output.append(
                        ToolUseBlock(
                            id=tc.get("id", str(uuid.uuid4())),
                            name=func.get("name", "") if func else "",
                            input=input_dict,
                            extra={"thought_signature": tc.get("thought_signature")}
                            if tc.get("thought_signature")
                            else {},
                        )
                    )
                case "custom":
                    custom = tc.get("custom") or {}
                    output.append(
                        CustomToolUseBlock(
                            id=tc.get("id", str(uuid.uuid4())),
                            name=custom.get("name", ""),
                            input=custom.get("input", ""),
                        )
                    )
        return output

    @staticmethod
    def _parse_logprobs(choice: dict[str, Any]) -> ChoiceLogprobs | None:
        """Parse logprobs from a choice dict."""
        choice_logprobs = choice.get("logprobs")
        if not choice_logprobs or not isinstance(choice_logprobs, dict):
            return None

        content_logprobs = None
        refusal_logprobs = None

        if "content" in choice_logprobs:
            content_logprobs = [
                TokenLogprob(
                    token=t.get("token", ""),
                    logprob=t.get("logprob", 0.0),
                    bytes=t.get("bytes"),
                )
                for t in choice_logprobs["content"]
                if isinstance(t, dict)
            ]

        if "refusal" in choice_logprobs:
            refusal_logprobs = [
                TokenLogprob(
                    token=t.get("token", ""),
                    logprob=t.get("logprob", 0.0),
                    bytes=t.get("bytes"),
                )
                for t in choice_logprobs["refusal"]
                if isinstance(t, dict)
            ]

        return ChoiceLogprobs(content=content_logprobs, refusal=refusal_logprobs)

    @staticmethod
    def _parse_annotations(message: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Extract annotations from message."""
        raw = message.get("annotations")
        if raw and isinstance(raw, list):
            return raw
        return None

    @staticmethod
    def parse_usage(response: dict[str, Any]) -> Usage | None:
        """Parse usage from a Chat Completions response body.

        Public because the wire-reuse response tier reuses this routine for
        billing parity (``OpenAICompatibleBase._parse_passthrough_usage``).
        """
        if "usage" not in response:
            return None

        usage_data = response["usage"]

        completion_details = None
        ctd = usage_data.get("completion_tokens_details")
        if ctd is not None:
            completion_details = CompletionTokensDetails(
                accepted_prediction_tokens=ctd.get("accepted_prediction_tokens"),
                audio_tokens=ctd.get("audio_tokens"),
                reasoning_tokens=ctd.get("reasoning_tokens"),
                rejected_prediction_tokens=ctd.get("rejected_prediction_tokens"),
                image_tokens=ctd.get("image_tokens"),
            )

        prompt_details = None
        ptd = usage_data.get("prompt_tokens_details")
        if ptd is not None:
            prompt_details = PromptTokensDetails(
                audio_tokens=ptd.get("audio_tokens"),
                cached_tokens=ptd.get("cached_tokens"),
                image_tokens=ptd.get("image_tokens"),
                cache_write_tokens=ptd.get("cache_write_tokens"),
                video_tokens=ptd.get("video_tokens"),
            )

        # Fold DeepSeek-style top-level cache hits into cached_tokens
        # (see fold_deepseek_cache_hits).
        folded = fold_deepseek_cache_hits(
            prompt_details.cached_tokens if prompt_details is not None else None,
            usage_data.get("prompt_cache_hit_tokens"),
        )
        if folded is not None:
            if prompt_details is None:
                prompt_details = PromptTokensDetails()
            prompt_details.cached_tokens = folded

        # Web search request count (e.g. OpenRouter proxying Anthropic models
        # passes through server_tool_use.web_search_requests)
        web_search_requests: int | None = None
        server_tool_use = usage_data.get("server_tool_use")
        if isinstance(server_tool_use, dict):
            ws_count = server_tool_use.get("web_search_requests")
            if isinstance(ws_count, int) and ws_count > 0:
                web_search_requests = ws_count

        return Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            completion_tokens_details=completion_details,
            prompt_tokens_details=prompt_details,
            web_search_requests=web_search_requests,
        )

    @staticmethod
    def known_response_fields() -> set[str]:
        """Fields that are handled explicitly in responses."""
        return {
            "id",
            "model",
            "choices",
            "usage",
            "system_fingerprint",
            "service_tier",
            "object",
            "created",
        }


__all__ = ["OpenAIResponseParser"]
