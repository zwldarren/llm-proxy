"""OpenAI streaming transformer.

Converts canonical ``chat.completion.chunk`` dicts into OpenAI SSE wire
format. Protocol-side transformer — lives next to the OpenAI protocol
module; the provider-side chunk converters stay in
``serialization/<family>/streaming_converter.py``.
"""

import secrets
import string
import time
from typing import Any

import orjson
from orjson import JSONDecodeError

from llm_proxy.core.utils import create_image_source_from_url
from llm_proxy.models import AudioBlock, ImageBlock
from llm_proxy.models.content_blocks import (
    CustomToolUseBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llm_proxy.models.types import AudioSource
from llm_proxy.serialization.openai.components.response_parser import (
    fold_deepseek_cache_hits,
)
from llm_proxy.streaming.transformer import StreamingTransformer, StreamingUsage

# Keys prefixed with ``_`` are proxy-internal and are stripped at every level
# the transformer traverses (chunk, choice, delta, tool call, usage). Unknown
# pass-through structures are not walked: only the proxy introduces ``_``
# keys, and never there. Everything else passes through: the transformer is
# fidelity-first (unknown provider fields reach the client verbatim) and only
# applies load-bearing transforms — user-facing model aliasing,
# custom→function tool-call normalization, and obfuscation injection.
_INTERNAL_KEY_PREFIX = "_"


def _strip_internal_keys(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return ``mapping`` without None values and proxy-internal keys.

    Keys prefixed with ``_`` are proxy bookkeeping, stripped at every level
    the transformer traverses; ``None`` values are dropped to match the
    canonical Chat Completions shape.
    """
    return {
        key: value
        for key, value in mapping.items()
        if value is not None and not key.startswith(_INTERNAL_KEY_PREFIX)
    }


# Standard chunk-envelope fields: present on every chunk, so a chunk reduced
# to only these (no choices, no usage, no extension fields) carries nothing
# the client can act on and is skipped.
_CHUNK_ENVELOPE_FIELDS = frozenset(
    {"id", "object", "created", "model", "system_fingerprint", "service_tier"}
)


class OpenAIStreamingTransformer(StreamingTransformer):
    """Transform StreamEvents to OpenAI SSE format."""

    def __init__(
        self,
        model: str = "",
        request_id: str | None = None,
        include_obfuscation: bool | None = None,
    ):
        """Initialize the streaming transformer.

        Args:
            model: Model name for this response
            request_id: Request ID for this response
            include_obfuscation: Whether to include obfuscation strings in delta chunks
        """
        super().__init__(
            model=model,
            request_id=request_id,
        )
        self._pending_usage: dict[str, Any] | None = None
        self._emitted_usage_chunk: bool = False
        self._text_buffer: str = ""
        self._reasoning_buffer: str = ""
        self._reasoning_signature: str = ""
        self._reasoning_is_redacted: bool = False
        self._reasoning_encrypted_content: str | None = None
        self._tool_calls_buffer: dict[int, dict[str, Any]] = {}
        self._images_buffer: list[dict[str, Any]] = []
        self._audio_data_buffer: str = ""
        self._audio_transcript_buffer: str = ""
        self._include_obfuscation: bool | None = include_obfuscation

    @classmethod
    def continuation(
        cls, model: str, request_id: str, start_index: int
    ) -> OpenAIStreamingTransformer:
        """Create a transformer that continues an existing stream.

        OpenAI chat-completion chunks are self-contained (each carries its
        own id/object/created/model), so a fresh transformer instance is
        sufficient for the web-search continuation loop; ``start_index`` is
        accepted for interface compatibility with the continuation seam
        (Anthropic uses it for content-block indices, OpenAI has none).
        """
        return cls(model=model, request_id=request_id)

    def transform(self, chunk: str | dict) -> str | None:
        """Transform chunks to OpenAI SSE format.

        Fidelity-first: unknown provider fields (e.g. OpenRouter's
        ``reasoning_details``) pass through verbatim — the proxy is a
        faithful wire, consistent with the native passthrough tiers. Only
        load-bearing transforms run (model aliasing, custom→function
        tool-call normalization, obfuscation injection); ``_``-prefixed
        proxy-internal keys and null values are stripped at every level.

        Args:
            chunk: Dict input from adapter (single serialization path),
                   or pre-encoded SSE string from adapters using native
                   streaming (e.g. OpenAI Responses provider).

        Returns:
            Normalized chunk string, or None if chunk should be filtered
        """
        if isinstance(chunk, dict):
            normalized = self._normalize_chunk(chunk)
            if normalized is None:
                return None
            return f"data: {orjson.dumps(normalized).decode()}\n\n"

        # Handle SSE string input from adapters that yield pre-encoded SSE
        # (e.g. OpenAI Responses provider's stream_chat_completion)
        if isinstance(chunk, str):
            chunk = chunk.strip()
            if not chunk or chunk == "[DONE]":
                return None
            # Extract data payload from SSE format
            if chunk.startswith("data:"):
                payload = chunk[5:].strip()
                if payload == "[DONE]":
                    return None
                try:
                    data = orjson.loads(payload)
                    return self.transform(data)
                except JSONDecodeError:
                    return None
            # Non-data SSE lines (e.g. "event: ...") are filtered
            return None

        return None

    def _normalize_chunk(self, chunk: dict | None) -> dict | None:
        """Normalize a parsed chunk for the wire, passing unknown fields through.

        Args:
            chunk: Parsed JSON chunk dictionary (may be None if provider sent 'null')

        Returns:
            Normalized chunk dictionary, or None if chunk should be skipped
        """
        if chunk is None or not isinstance(chunk, dict):
            return None

        result = _strip_internal_keys(chunk)
        # choices/usage need per-level normalization; re-added below.
        result.pop("choices", None)
        result.pop("usage", None)

        # Override model with the user-facing model name (self.model),
        # not the provider-specific model from the raw chunk.
        if self.model:
            result["model"] = self.model

        if "choices" in chunk and chunk["choices"] is not None:
            normalized_choices = []
            for choice in chunk["choices"]:
                normalized_choice = self._normalize_choice(choice)
                if normalized_choice is not None:
                    normalized_choices.append(normalized_choice)
            if normalized_choices:
                result["choices"] = normalized_choices
            elif "usage" in chunk:
                # Usage-only chunk: keep the (empty) choices array, matching
                # the canonical usage-chunk shape.
                result["choices"] = []

        has_choices = bool(result.get("choices"))
        if "usage" in chunk and chunk["usage"] is not None:
            normalized_usage = self._normalize_usage(chunk["usage"])
            if normalized_usage:
                self._pending_usage = normalized_usage
                if not has_choices:
                    result["usage"] = normalized_usage
                    self._emitted_usage_chunk = True

        # Bookkeeping (accumulation for tracing/logging) reads the ORIGINAL
        # chunk, so it is unaffected by wire normalization.
        self._accumulate_from_chunk(chunk)

        # Fidelity-first: any surviving field beyond the standard envelope
        # (provider extension fields, encrypted_content) is deliverable
        # content; a chunk reduced to envelope-only metadata is skipped.
        has_content = (
            result.get("choices")
            or "usage" in result
            or any(key not in _CHUNK_ENVELOPE_FIELDS for key in result)
        )
        if not has_content:
            return None

        return result if result else None

    def _accumulate_from_chunk(self, original_chunk: dict | None) -> None:
        """Accumulate content from a streaming chunk.

        This method extracts and buffers content from streaming chunks to build
        the final accumulated output for tracing/logging purposes.

        Args:
            original_chunk: The original parsed chunk dictionary
        """
        if not original_chunk:
            return

        # Capture encrypted reasoning state from OpenAI Responses provider.
        # This can arrive as a top-level field (response.completed fallback)
        # or inside a delta; either way it must survive to the OpenResponses
        # streaming transformer so it can emit reasoning.encrypted_content.
        encrypted = original_chunk.get("encrypted_content")
        if encrypted is not None:
            self._reasoning_encrypted_content = encrypted

        choices = original_chunk.get("choices", [])
        if not choices:
            return

        for choice in choices:
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            if not isinstance(delta, dict):
                continue

            if "content" in delta and delta["content"] is not None:
                content = delta["content"]
                if isinstance(content, str):
                    self._text_buffer += content
                elif isinstance(content, list):
                    # Handle content array (e.g., text + images from image generation)
                    text_parts = []
                    image_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text = part.get("text", "")
                                if text:
                                    text_parts.append(text)
                            elif part.get("type") == "image_url":
                                url = part.get("image_url", {}).get("url", "")
                                if url:
                                    image_parts.append(f"![image]({url})")
                    if text_parts:
                        self._text_buffer += "".join(text_parts)
                    if image_parts:
                        if self._text_buffer:
                            self._text_buffer += "\n" + "\n".join(image_parts)
                        else:
                            self._text_buffer += "\n".join(image_parts)

            if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                self._reasoning_buffer += delta["reasoning_content"]
            elif "reasoning" in delta and delta["reasoning"] is not None:
                self._reasoning_buffer += delta["reasoning"]

            if "reasoning_signature" in delta and delta["reasoning_signature"] is not None:
                self._reasoning_signature += delta["reasoning_signature"]
            if "reasoning_is_redacted" in delta and delta["reasoning_is_redacted"]:
                self._reasoning_is_redacted = True

            if "encrypted_content" in delta and delta["encrypted_content"] is not None:
                self._reasoning_encrypted_content = delta["encrypted_content"]

            if "tool_calls" in delta and isinstance(delta["tool_calls"], list):
                for tc in delta["tool_calls"]:
                    if not isinstance(tc, dict):
                        continue
                    idx = tc.get("index", 0)
                    if idx not in self._tool_calls_buffer:
                        tc_type = tc.get("type", "function")
                        if tc_type == "custom":
                            self._tool_calls_buffer[idx] = {
                                "id": "",
                                "type": "custom",
                                "custom": {"name": "", "input": ""},
                            }
                        else:
                            self._tool_calls_buffer[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                                "thought_signature": None,
                            }

                    if "id" in tc and tc["id"]:
                        self._tool_calls_buffer[idx]["id"] = tc["id"]
                    if "type" in tc and tc["type"]:
                        self._tool_calls_buffer[idx]["type"] = tc["type"]
                    if "function" in tc and isinstance(tc["function"], dict):
                        func = tc["function"]
                        if "name" in func and func["name"]:
                            self._tool_calls_buffer[idx]["function"]["name"] = func["name"]
                        if "arguments" in func and func["arguments"]:
                            self._tool_calls_buffer[idx]["function"]["arguments"] += func[
                                "arguments"
                            ]
                    if "thought_signature" in tc and tc["thought_signature"]:
                        self._tool_calls_buffer[idx]["thought_signature"] = tc["thought_signature"]
                    if "custom" in tc and isinstance(tc["custom"], dict):
                        custom = tc["custom"]
                        if "name" in custom and custom["name"]:
                            self._tool_calls_buffer[idx]["custom"]["name"] = custom["name"]
                        if "input" in custom and custom["input"]:
                            self._tool_calls_buffer[idx]["custom"]["input"] += custom["input"]

            # Accumulate image output (OpenRouter image generation via chat completions)
            if "images" in delta and isinstance(delta["images"], list):
                for img in delta["images"]:
                    if isinstance(img, dict) and img not in self._images_buffer:
                        self._images_buffer.append(img)

            # Accumulate audio output (OpenRouter audio output via chat completions)
            if "audio" in delta and isinstance(delta["audio"], dict):
                audio = delta["audio"]
                if "data" in audio and audio["data"]:
                    self._audio_data_buffer += audio["data"]
                if "transcript" in audio and audio["transcript"]:
                    self._audio_transcript_buffer += audio["transcript"]

            if finish_reason:
                self._finalize_accumulation()

    def _finalize_accumulation(self) -> None:
        """Finalize and add accumulated content to _accumulated_output."""
        if self._reasoning_buffer:
            if self._reasoning_is_redacted:
                self._accumulated_output.append(RedactedThinkingBlock(data=self._reasoning_buffer))
            else:
                self._accumulated_output.append(
                    ThinkingBlock(
                        thinking=self._reasoning_buffer,
                        signature=self._reasoning_signature or None,
                        encrypted_content=self._reasoning_encrypted_content,
                    )
                )
            self._reasoning_buffer = ""
            self._reasoning_signature = ""
            self._reasoning_is_redacted = False
            self._reasoning_encrypted_content = None

        if self._text_buffer:
            self._accumulated_output.append(TextBlock(text=self._text_buffer))
            self._text_buffer = ""

        if self._tool_calls_buffer:
            for idx in sorted(self._tool_calls_buffer.keys()):
                tc = self._tool_calls_buffer[idx]
                tc_type = tc.get("type", "function")
                if tc_type == "custom":
                    if tc["id"] and tc["custom"]["name"]:
                        self._accumulated_output.append(
                            CustomToolUseBlock(
                                id=tc["id"],
                                name=tc["custom"]["name"],
                                input=tc["custom"]["input"],
                            )
                        )
                else:
                    if tc["id"] and tc["function"]["name"]:
                        try:
                            tool_input = (
                                orjson.loads(tc["function"]["arguments"])
                                if tc["function"]["arguments"]
                                else {}
                            )
                        except JSONDecodeError:
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
            self._tool_calls_buffer = {}

        # Accumulate image output blocks
        if self._images_buffer:
            for img in self._images_buffer:
                if isinstance(img, dict) and img.get("type") == "image_url":
                    image_url = img.get("image_url") or {}
                    url = image_url.get("url", "")
                    source = create_image_source_from_url(url)
                    if source:
                        self._accumulated_output.append(ImageBlock(source=source))
            self._images_buffer = []

        # Accumulate audio output block
        if self._audio_data_buffer:
            self._accumulated_output.append(
                AudioBlock(
                    source=AudioSource(
                        type="base64",
                        data=self._audio_data_buffer,
                        media_type="audio/wav",
                        transcript=self._audio_transcript_buffer or None,
                    )
                )
            )
            self._audio_data_buffer = ""
            self._audio_transcript_buffer = ""

    def _normalize_choice(self, choice: dict) -> dict | None:
        """Normalize a choice object, passing unknown fields through."""
        if choice is None or not isinstance(choice, dict):
            return None
        result = _strip_internal_keys(choice)
        result.pop("delta", None)

        if "delta" in choice and choice["delta"] is not None:
            normalized_delta = self._normalize_delta(choice["delta"])
            if normalized_delta:
                result["delta"] = normalized_delta
            elif "finish_reason" in choice:
                result["delta"] = {}

        return result if result else None

    def _normalize_delta(self, delta: dict) -> dict:
        """Normalize a delta object: unknown fields pass through, tool calls
        get their own normalization, obfuscation is injected when enabled."""
        result = _strip_internal_keys(delta)
        tool_calls = result.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_tool_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    normalized_tc = self._normalize_tool_call(tc)
                    if normalized_tc:
                        normalized_tool_calls.append(normalized_tc)
                else:
                    normalized_tool_calls.append(tc)
            if normalized_tool_calls:
                result["tool_calls"] = normalized_tool_calls
            else:
                del result["tool_calls"]
        # Inject obfuscation when enabled to mitigate timing-based inference attacks.
        # Only applied to content/reasoning deltas where token-timing inference applies;
        # structured fields (tool_calls, refusal) have less timing variance.
        if self._include_obfuscation and (result.get("content") or result.get("reasoning_content")):
            length = secrets.randbelow(5) + 2  # 2..6 inclusive
            result["obfuscation"] = "".join(
                secrets.choice(string.ascii_letters + string.digits) for _ in range(length)
            )
        return result

    def _normalize_tool_call(self, tool_call: dict) -> dict | None:
        """Normalize a tool call object, passing unknown fields through.

        Custom tool calls (a Responses-API concept) are normalized to function
        tool calls because Chat Completions clients reject ``type: "custom"``
        with "unknown variant `custom`, expected `function`". The freeform
        ``custom.input`` delta fragments map onto ``function.arguments``; the
        ``{"content": ...}`` bridge envelope is applied by the producer when it
        emits complete arguments. Function-typed calls without a ``function``
        payload are invalid mid-stream and dropped, to prevent client-side
        validation errors.
        """
        # Normalize custom tool call deltas to function shape before cleaning.
        if tool_call.get("type") == "custom" or "custom" in tool_call:
            custom = tool_call.get("custom") or {}
            function = dict(tool_call.get("function") or {})
            if custom.get("name"):
                function["name"] = custom["name"]
            if custom.get("input"):
                function["arguments"] = custom["input"]
            tool_call = {k: v for k, v in tool_call.items() if k != "custom"}
            tool_call["type"] = "function"
            tool_call["function"] = function

        tool_type = tool_call.get("type", "function")
        result = _strip_internal_keys(tool_call)
        if isinstance(result.get("function"), dict):
            function = _strip_internal_keys(result["function"])
            if function:
                result["function"] = function
            else:
                del result["function"]

        if tool_type == "function" and "function" not in result:
            return None

        return result if result else None

    def _normalize_usage(self, usage: dict) -> dict:
        """Normalize a usage object, passing unknown fields through.

        Provider-reported cost fields (OpenRouter ``cost``, NanoGPT
        ``nanogpt_cost``) and ``server_tool_use`` (web search billing) flow
        through like any other field; billing reads the known keys via
        :meth:`get_usage` regardless.
        """
        result = _strip_internal_keys(usage)
        for details_key in ("prompt_tokens_details", "completion_tokens_details"):
            details = result.get(details_key)
            if isinstance(details, dict):
                normalized_details = _strip_internal_keys(details)
                if normalized_details:
                    result[details_key] = normalized_details
                else:
                    del result[details_key]
        return result

    def _make_chunk(self, data: dict[str, Any]) -> str:
        """Create an SSE chunk string."""
        chunk = {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": time.time_ns() // 1_000_000_000,
            "model": self.model,
            **data,
        }
        return f"data: {orjson.dumps(chunk).decode()}\n\n"

    def finalize(self) -> str:
        """Generate stream end marker."""
        if self._text_buffer or self._reasoning_buffer or self._tool_calls_buffer:
            self._finalize_accumulation()
        chunks: list[str] = []
        if self._pending_usage is not None and not self._emitted_usage_chunk:
            chunks.append(self._make_chunk({"choices": [], "usage": self._pending_usage}))
        chunks.append("data: [DONE]\n\n")
        return "".join(chunks)

    def get_usage(self) -> StreamingUsage | None:
        """Get accumulated usage information from streaming response.

        Returns:
            StreamingUsage object if usage data is available from provider.
        """
        if self._pending_usage and isinstance(self._pending_usage, dict):
            # Extract provider-reported cost (NanoGPT's nanogpt_cost, OpenRouter's cost)
            provider_cost = self._pending_usage.get("nanogpt_cost")
            if provider_cost is None:
                provider_cost = self._pending_usage.get("cost")
            if provider_cost is not None and not isinstance(provider_cost, int | float):
                provider_cost = None

            # Extract nested details (OpenAI-style cached_tokens, audio_tokens)
            prompt_details: dict[str, int] | None = None
            ptd = self._pending_usage.get("prompt_tokens_details")
            if isinstance(ptd, dict) and any(v is not None for v in ptd.values()):
                prompt_details = {k: v for k, v in ptd.items() if v is not None}

            # Fold DeepSeek-style top-level cache hits into cached_tokens
            # (see fold_deepseek_cache_hits); matches the non-streaming parser.
            folded = fold_deepseek_cache_hits(
                (prompt_details or {}).get("cached_tokens"),
                self._pending_usage.get("prompt_cache_hit_tokens"),
            )
            if folded is not None:
                prompt_details = {**(prompt_details or {}), "cached_tokens": folded}

            completion_details: dict[str, int] | None = None
            ctd = self._pending_usage.get("completion_tokens_details")
            if isinstance(ctd, dict) and any(v is not None for v in ctd.values()):
                completion_details = {k: v for k, v in ctd.items() if v is not None}

            # Web search request count (server_tool_use.web_search_requests)
            web_search_requests: int | None = None
            server_tool_use = self._pending_usage.get("server_tool_use")
            if isinstance(server_tool_use, dict):
                ws_count = server_tool_use.get("web_search_requests")
                if isinstance(ws_count, int) and ws_count > 0:
                    web_search_requests = ws_count

            return StreamingUsage(
                input_tokens=self._pending_usage.get("prompt_tokens", 0),
                output_tokens=self._pending_usage.get("completion_tokens", 0),
                total_tokens=self._pending_usage.get("total_tokens", 0),
                cache_read_input_tokens=self._pending_usage.get("cache_read_input_tokens", 0),
                cache_creation_input_tokens=self._pending_usage.get(
                    "cache_creation_input_tokens", 0
                ),
                prompt_tokens_details=prompt_details,
                completion_tokens_details=completion_details,
                provider_reported_cost=provider_cost,
                web_search_requests=web_search_requests,
            )
        return None

    def _web_search_result_block(
        self,
        index: int,
        tool_use_id: str,
        content: list[dict[str, Any]] | str,
        is_error: bool = False,
        query: str = "",
    ) -> str:
        """Emit a web search result block in OpenAI Chat Completions SSE format.

        For Chat Completions the search results are consumed by the
        continuation mechanism — the proxy sends a second request with
        the results as tool messages and the model's text response is
        what the user sees.  We therefore return an empty string so that
        no intermediate SSE chunks are emitted to the client.
        """
        return ""

    def _message_delta_with_usage(self, _usage: dict[str, Any]) -> str:
        """Emit a message delta with server-side tool usage.

        For Chat Completions the usage delta emitted by :meth:`finalize`
        already covers the accumulated usage across the original request
        and any continuation turns, so we return an empty string here.
        """
        return ""
