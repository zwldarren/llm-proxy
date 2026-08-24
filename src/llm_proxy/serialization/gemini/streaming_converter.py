"""Gemini streaming chunk converter (provider chunks -> unified)."""

import time
from typing import Any

import orjson
from orjson import JSONDecodeError

from llm_proxy.models.finish_reasons import map_finish_reason
from llm_proxy.serialization.gemini.annotations import extract_gemini_annotations
from llm_proxy.serialization.gemini.usage import billable_token_counts
from llm_proxy.streaming.transformer import StreamingTransformer, StreamingUsage


class GeminiStreamingTransformer(StreamingTransformer):
    """Transforms Gemini SSE chunks to OpenAI SSE format.

    This transformer converts streaming responses from Gemini's API format
    to OpenAI's chat.completion.chunk format for protocol compatibility.

    Example:
        transformer = GeminiStreamingTransformer(model="gemini-2.0-flash")
        for chunk in gemini_stream:
            openai_chunk = transformer.transform(chunk)
            if openai_chunk:
                yield openai_chunk
        yield transformer.finalize()
    """

    def __init__(
        self,
        model: str = "",
        request_id: str | None = None,
    ):
        super().__init__(
            model=model,
            request_id=request_id,
        )
        self._created = time.time_ns() // 1_000_000_000
        self._chunk_index = 0
        self._text_buffer: str = ""
        self._tool_calls_buffer: list[dict[str, Any]] = []
        self._reasoning_buffer: str = ""
        self._reasoning_signature_buffer: str = ""
        self._usage: StreamingUsage | None = None
        # Native Google Search grounding query count (billed per search)
        self._web_search_requests: int = 0
        # Accumulated base64 audio (TTS) output and its MIME type
        self._audio_buffer: str = ""
        self._audio_mime_type: str = ""

    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Transform a raw SSE chunk from Gemini to OpenAI format.

        Args:
            chunk: Raw SSE chunk string from Gemini API, or pre-parsed chunk dict

        Returns:
            OpenAI-formatted SSE chunk string, or None if chunk should be filtered
        """
        if isinstance(chunk, dict):
            return self.transform_chunk(chunk)

        if not chunk or not chunk.strip():
            return None

        try:
            data = orjson.loads(chunk)
        except JSONDecodeError:
            return None

        return self.transform_chunk(data)

    def transform_chunk(self, data: dict[str, Any]) -> str | None:
        """Transform already-parsed Gemini chunk dict to OpenAI format.

        Args:
            data: Parsed Gemini response chunk dictionary

        Returns:
            OpenAI-formatted SSE chunk string, or None if chunk should be filtered
        """
        if "error" in data:
            return self._make_chunk({"error": data["error"]})

        openai_chunk = self.convert_chunk(data)
        if openai_chunk is None:
            return None

        return self._make_chunk(openai_chunk)

    def convert_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Convert Gemini chunk format to OpenAI chunk format.

        Args:
            chunk: Gemini response chunk

        Returns:
            OpenAI chat.completion.chunk dictionary, or None if no valid content
        """
        candidates = chunk.get("candidates", [])
        if not candidates:
            return None

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Native Google Search grounding is billed per search request;
        # webSearchQueries lists the queries the model actually issued.
        grounding = candidate.get("groundingMetadata")
        if isinstance(grounding, dict):
            queries = grounding.get("webSearchQueries")
            if isinstance(queries, list) and queries:
                self._web_search_requests = len(queries)

        delta: dict[str, Any] = {}
        finish_reason = None

        gemini_finish = candidate.get("finishReason")
        if gemini_finish:
            finish_reason = map_finish_reason(gemini_finish, "gemini", "openai")

        text_parts = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        image_parts: list[dict[str, Any]] = []
        audio_parts: list[str] = []

        for part in parts:
            match part:
                case {"thought": True, "text": str(text)}:
                    reasoning_parts.append(text)
                    sig = part.get("thoughtSignature") or part.get("thought_signature")
                    if sig and not self._reasoning_signature_buffer:
                        self._reasoning_signature_buffer = sig
                case {"text": str(text)}:
                    text_parts.append(text)
                case {"inlineData": inline} if str(
                    inline.get("mime_type") or inline.get("mimeType") or ""
                ).startswith("audio/"):
                    # TTS audio output: stream as OpenAI audio deltas
                    mime_type = inline.get("mime_type") or inline.get("mimeType") or ""
                    data = inline.get("data", "")
                    if data:
                        audio_parts.append(data)
                        if not self._audio_mime_type:
                            self._audio_mime_type = mime_type
                case {"inlineData": inline}:
                    mime_type = inline.get("mime_type") or inline.get("mimeType") or "image/png"
                    data = inline.get("data", "")
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{data}"},
                        }
                    )
                case {"functionCall": func_call}:
                    tool_calls.append(
                        {
                            "index": len(tool_calls),
                            "id": f"call_{self._chunk_index}_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": func_call.get("name", ""),
                                "arguments": orjson.dumps(func_call.get("args", {})).decode(),
                            },
                            "thought_signature": part.get("thoughtSignature")
                            or part.get("thought_signature"),
                        }
                    )
                case _:
                    pass

        text = "".join(text_parts)
        if text_parts:
            delta["content"] = text
            self._text_buffer += text

        # Map citation/grounding metadata to OpenAI-style annotations in the delta.
        # Gemini's startIndex/endIndex are absolute offsets into the full response
        # text, so resolve them against the accumulated buffer rather than the
        # current delta (whose length would be too short when endIndex is absent).
        annotations = extract_gemini_annotations(candidate, self._text_buffer)
        if annotations:
            delta["annotations"] = annotations

        if reasoning_parts:
            delta["reasoning_content"] = "".join(reasoning_parts)
            self._reasoning_buffer += "".join(reasoning_parts)

        if tool_calls:
            delta["tool_calls"] = tool_calls
            self._tool_calls_buffer.extend(tool_calls)

        if audio_parts:
            # OpenAI chat.completion.chunk audio delta shape:
            # delta.audio = {id, data, transcript?, expires_at?}
            delta["audio"] = {
                "id": f"audio_{self.response_id or self._chunk_index}",
                "data": "".join(audio_parts),
            }
            self._audio_buffer += "".join(audio_parts)

        if image_parts:
            # Degrade images to markdown syntax like new-api does,
            # since OpenAI Chat Completions streaming response does not
            # support image_url in assistant delta content.
            markdown_images = []
            for img in image_parts:
                url = img.get("image_url", {}).get("url", "")
                if url:
                    markdown_images.append(f"![image]({url})")
            if markdown_images:
                # Append markdown images to text content
                existing_content = delta.get("content", "")
                if existing_content:
                    delta["content"] = existing_content + "\n" + "\n".join(markdown_images)
                else:
                    delta["content"] = "\n".join(markdown_images)
                if self._text_buffer:
                    self._text_buffer += "\n" + "\n".join(markdown_images)
                else:
                    self._text_buffer += "\n".join(markdown_images)

        if not delta and not finish_reason:
            return None

        openai_chunk: dict[str, Any] = {
            "id": self.response_id or f"chatcmpl-{self._chunk_index}",
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }

        # Preserve candidate-level safety ratings when present.
        safety_ratings = candidate.get("safetyRatings")
        if isinstance(safety_ratings, list) and safety_ratings:
            openai_chunk["safety_ratings"] = safety_ratings

        usage_metadata = chunk.get("usageMetadata")
        if usage_metadata:
            prompt_tokens, completion_tokens = billable_token_counts(
                usage_metadata, has_search_grounding=self._web_search_requests > 0
            )
            usage: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            # Canonical keys only: cache_read_input_tokens / reasoning_tokens
            # are the keys billing/logging read (via the protocol-side
            # transformer's get_usage); the Gemini-flavored key names would
            # not be picked up there.
            if "cachedContentTokenCount" in usage_metadata:
                usage["cache_read_input_tokens"] = usage_metadata["cachedContentTokenCount"]
            if "thoughtsTokenCount" in usage_metadata:
                usage["reasoning_tokens"] = usage_metadata["thoughtsTokenCount"]
            if "totalTokenCount" in usage_metadata:
                usage["total_tokens"] = usage_metadata["totalTokenCount"]
            openai_chunk["usage"] = usage

            # Store usage for get_usage(). Cached tokens go ONLY into
            # cache_read_input_tokens: setting prompt_tokens_details too would
            # make billing apply the cache-rate adjustment twice for the same
            # tokens (the canonical Usage record expresses each fact once).
            self._usage = StreamingUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=usage_metadata.get("totalTokenCount", 0),
                cache_read_input_tokens=usage_metadata.get("cachedContentTokenCount") or None,
                web_search_requests=self._web_search_requests or None,
            )

        if "promptFeedback" in chunk:
            openai_chunk["prompt_feedback"] = chunk["promptFeedback"]

        if finish_reason:
            self._finalize_accumulation()

        self._chunk_index += 1
        return openai_chunk

    def _finalize_accumulation(self) -> None:
        """Convert accumulated buffers to ContentBlocks."""
        from llm_proxy.models import AudioBlock, TextBlock, ThinkingBlock, ToolUseBlock
        from llm_proxy.models.types import AudioSource

        if self._audio_buffer:
            self._accumulated_output.append(
                AudioBlock(
                    source=AudioSource(
                        type="base64",
                        data=self._audio_buffer,
                        media_type=self._audio_mime_type or None,
                    )
                )
            )

        if self._reasoning_buffer:
            self._accumulated_output.append(
                ThinkingBlock(
                    thinking=self._reasoning_buffer,
                    signature=self._reasoning_signature_buffer or None,
                )
            )

        if self._text_buffer:
            self._accumulated_output.append(TextBlock(text=self._text_buffer))

        for tc in self._tool_calls_buffer:
            try:
                tool_input = (
                    orjson.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                )
            except JSONDecodeError, TypeError:
                tool_input = {}

            self._accumulated_output.append(
                ToolUseBlock(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    input=tool_input,
                    extra={"thought_signature": tc.get("thought_signature")}
                    if tc.get("thought_signature")
                    else {},
                )
            )

        self._text_buffer = ""
        self._tool_calls_buffer = []
        self._reasoning_buffer = ""
        self._reasoning_signature_buffer = ""
        self._audio_buffer = ""
        self._audio_mime_type = ""

    def finalize(self) -> str:
        """Generate stream end marker.

        Returns:
            OpenAI-style stream termination: 'data: [DONE]\\n\\n'
        """
        if (
            self._text_buffer
            or self._tool_calls_buffer
            or self._audio_buffer
            or self._reasoning_buffer
        ):
            self._finalize_accumulation()
        return "data: [DONE]\n\n"

    def get_usage(self) -> StreamingUsage | None:
        """Get accumulated usage information from streaming response.

        Returns:
            StreamingUsage object if usage data is available from Gemini.
        """
        if self._usage is not None:
            # groundingMetadata may arrive after usageMetadata was captured.
            if self._usage.web_search_requests is None and self._web_search_requests > 0:
                self._usage.web_search_requests = self._web_search_requests
            return self._usage
        if self._web_search_requests > 0:
            # No token usage, but native search grounding is still billable.
            return StreamingUsage(web_search_requests=self._web_search_requests)
        return None
