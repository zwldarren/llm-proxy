"""Ollama response parser mixin."""

from typing import Any

import orjson

from llm_proxy.core.utils import generate_response_id
from llm_proxy.models import (
    ContentBlock,
    ImageBlock,
    InternalResponse,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llm_proxy.models.embedding import (
    EmbeddingData,
    InternalEmbeddingResponse,
)
from llm_proxy.models.types import ChoiceLogprobs, ImageSource, Usage
from llm_proxy.serialization.ollama.metrics import extract_ollama_metrics
from llm_proxy.serialization.ollama.tool_utils import (
    normalize_logprob_entries,
    normalize_tool_calls,
)


class OllamaResponseParserMixin:
    """Parse Ollama native API responses into InternalResponse."""

    def parse_provider_response(
        self,
        response: dict[str, Any],
        model: str | None = None,
        **kwargs: Any,
    ) -> InternalResponse:
        request_id: str | None = kwargs.get("request_id")
        logprobs: bool = kwargs.get("logprobs", False)
        message = response.get("message", {})
        content = message.get("content") or ""
        thinking = message.get("thinking")

        output: list[ContentBlock] = []

        # Thinking must precede answer text: Anthropic-protocol rendering
        # preserves block order (thinking blocks are only valid before text
        # blocks), and streaming emits thinking deltas first — non-streaming
        # must match that order.
        if thinking and isinstance(thinking, str) and thinking.strip():
            output.append(ThinkingBlock(thinking=thinking))

        if content:
            output.append(TextBlock(text=content))

        if isinstance(message, dict) and message.get("tool_calls"):
            tool_calls = normalize_tool_calls(
                message.get("tool_calls"),
                include_index=False,
                created_at=response.get("created_at"),
            )
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn_dict = tc.get("function", {})
                        args = fn_dict.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = orjson.loads(args)
                            except orjson.JSONDecodeError:
                                args = {}
                        output.append(
                            ToolUseBlock(
                                id=tc.get("id", generate_response_id()),
                                name=fn_dict.get("name", ""),
                                input=args,
                            )
                        )

        done_reason = response.get("done_reason", "stop")
        valid_reasons = ("stop", "length", "tool_calls")
        finish_reason = done_reason if done_reason in valid_reasons else "stop"

        # Ollama also reports "load" (empty messages load the model) and
        # "unload" (keep_alive=0) as done_reason; neither maps to an OpenAI
        # finish_reason, so the wire value falls back to "stop" while the
        # raw reason is preserved for observability.
        provider_info: dict[str, Any] = {"provider": "ollama"}
        if done_reason:
            provider_info["done_reason"] = done_reason

        usage = None
        if response.get("prompt_eval_count") is not None or response.get("eval_count") is not None:
            usage = Usage(
                input_tokens=response.get("prompt_eval_count") or 0,
                output_tokens=response.get("eval_count") or 0,
                total_tokens=(response.get("prompt_eval_count") or 0)
                + (response.get("eval_count") or 0),
            )

        response_id = response.get("id") or generate_response_id()

        # Preserve Ollama native duration metrics (nanoseconds) for observability.
        duration_metrics = extract_ollama_metrics(response)
        if duration_metrics:
            provider_info["ollama_metrics"] = duration_metrics

        if message.get("images"):
            for img_data in message["images"]:
                output.append(
                    ImageBlock(
                        source=ImageSource(
                            type="base64",
                            data=img_data,
                            media_type=None,
                        )
                    )
                )

        logprobs_obj: ChoiceLogprobs | None = None
        if logprobs and response.get("logprobs"):
            entries = normalize_logprob_entries(response.get("logprobs"))
            if entries:
                logprobs_obj = ChoiceLogprobs(content=entries)

        return InternalResponse(
            id=response_id,
            model=model or "unknown",
            output=output,
            usage=usage,
            finish_reason=finish_reason,
            request_id=request_id,
            provider_info=provider_info,
            logprobs=logprobs_obj,
        )

    def parse_provider_embedding_response(
        self, response: dict[str, Any], model: str = ""
    ) -> InternalEmbeddingResponse:
        embeddings = response.get("embeddings", [])
        data_list: list[EmbeddingData] = []

        for index, embedding in enumerate(embeddings):
            data_list.append(EmbeddingData(embedding=embedding, index=index))

        # /api/embed reports prompt_eval_count (no output tokens); surface it
        # as usage so billing/observability can account for it.
        usage = None
        prompt_eval_count = response.get("prompt_eval_count")
        if prompt_eval_count is not None:
            usage = Usage(
                input_tokens=prompt_eval_count,
                total_tokens=prompt_eval_count,
            )

        return InternalEmbeddingResponse(
            model=response.get("model") or model,
            data=data_list,
            usage=usage,
        )

    # _normalize_tool_calls and convert_logprobs are provided by
    # OllamaStreamingMixin via MRO - defined in serialization/ollama/streaming.py
