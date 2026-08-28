"""Ollama streaming conversion mixin and chunk converter."""

import time
from typing import Any

from llm_proxy.serialization.ollama.metrics import extract_ollama_metrics
from llm_proxy.serialization.ollama.tool_utils import convert_logprobs, normalize_tool_calls
from llm_proxy.streaming.transformer import StreamingTransformer, StreamingUsage

_DONE_REASON_MAP: dict[str, str] = {
    "length": "length",
    "tool_calls": "tool_calls",
}


class OllamaStreamingMixin:
    """Ollama native streaming chunk conversion and normalization."""

    def _normalize_tool_calls(
        self,
        tool_calls: Any,
        *,
        include_index: bool,
        created_at: str | None = None,
    ) -> list[dict[str, Any]] | None:
        return normalize_tool_calls(tool_calls, include_index=include_index, created_at=created_at)

    def convert_logprobs(self, ollama_logprobs: list[Any] | None) -> dict[str, Any] | None:
        return convert_logprobs(ollama_logprobs)

    def convert_native_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """Convert a provider-native streaming chunk to OpenAI streaming format.

        Called internally by ``OllamaChunkConverter.convert_chunk()``.
        """
        message = chunk.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking", "")

        delta: dict[str, Any] = {"content": content}
        if thinking:
            delta["reasoning_content"] = thinking
        if message.get("role"):
            delta["role"] = message["role"]
        if message.get("images"):
            delta["images"] = message["images"]

        if message.get("tool_calls"):
            normalized_tool_calls = self._normalize_tool_calls(
                message.get("tool_calls"),
                include_index=True,
                created_at=chunk.get("created_at"),
            )
            if normalized_tool_calls:
                delta["tool_calls"] = normalized_tool_calls

        finish_reason = None
        if chunk.get("done"):
            finish_reason = _DONE_REASON_MAP.get(chunk.get("done_reason"), "stop")

        openai_chunk: dict[str, Any] = {
            # Stable per-stream id: ``created_at`` is an ISO timestamp (with
            # colons/dots) that would leak into the id; the converter's fixed
            # ``_created_at`` epoch keeps it clean and identical across chunks.
            "id": f"chatcmpl-{getattr(self, '_created_at', None) or int(time.time())}",
            "object": "chat.completion.chunk",
            "created": getattr(self, "_created_at", None) or int(time.time()),
            "model": chunk.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }

        if chunk.get("done"):
            prompt_eval_count = chunk.get("prompt_eval_count")
            eval_count = chunk.get("eval_count")
            if prompt_eval_count is not None or eval_count is not None:
                openai_chunk["usage"] = {
                    "prompt_tokens": prompt_eval_count or 0,
                    "completion_tokens": eval_count or 0,
                    "total_tokens": (prompt_eval_count or 0) + (eval_count or 0),
                }
                # Preserve Ollama native duration metrics (nanoseconds) in the
                # final chunk so observability/billing can access them.
                duration_metrics = extract_ollama_metrics(chunk)
                if duration_metrics:
                    openai_chunk["usage"]["ollama_metrics"] = duration_metrics

        if chunk.get("logprobs"):
            converted = self.convert_logprobs(chunk.get("logprobs"))
            if converted:
                openai_chunk["choices"][0]["logprobs"] = converted

        return openai_chunk


class OllamaChunkConverter(OllamaStreamingMixin, StreamingTransformer):
    """Convert Ollama native streaming chunks to canonical OpenAI chunk dicts.

    Wraps the conversion logic from ``OllamaStreamingMixin.convert_native_chunk()``
    into a ``StreamingTransformer``-compatible converter so that the Ollama
    adapter can follow the same ``get_chunk_converter()`` pattern as other
    providers.

    Inherits from ``OllamaStreamingMixin`` to reuse the shared chunk-building
    logic and avoid duplication between the mixin and the converter.
    """

    def __init__(
        self,
        model: str = "",
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            request_id=request_id,
        )
        # Tool call index tracking (was in the adapter, now in the converter).
        self._tool_call_index: int = 0
        # Fixed per-stream timestamp so all chunks share the same ``created``.
        self._created_at: int = int(time.time())
        # OpenAI convention: only the first delta carries ``role``. Ollama
        # repeats ``role`` on every streamed chunk, so track whether the
        # first delta has been emitted and strip it afterwards.
        self._role_sent: bool = False
        # Usage captured from the terminal ``done`` chunk (Ollama reports
        # prompt_eval_count/eval_count only there).
        self._usage: StreamingUsage | None = None

    # ------------------------------------------------------------------
    # Provider-side API  (called by adapter)
    # ------------------------------------------------------------------

    def convert_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """Convert a single Ollama JSON-line chunk to canonical OpenAI format.

        Delegates the common chunk-building logic to
        ``OllamaStreamingMixin.convert_native_chunk()``, then overrides
        tool call indices with sequential values (Ollama can emit duplicate
        indices across chunks).

        Args:
            chunk: Parsed JSON dict from an Ollama streaming response line.

        Returns:
            Canonical OpenAI ``chat.completion.chunk`` dict.
        """
        openai_chunk = self.convert_native_chunk(chunk)

        # Capture usage from the terminal done chunk so get_usage() can serve
        # billing even when the protocol-side transformer did not observe the
        # final usage chunk.
        if chunk.get("done"):
            prompt_eval_count = chunk.get("prompt_eval_count")
            eval_count = chunk.get("eval_count")
            if prompt_eval_count is not None or eval_count is not None:
                self._usage = StreamingUsage(
                    input_tokens=prompt_eval_count or 0,
                    output_tokens=eval_count or 0,
                    total_tokens=(prompt_eval_count or 0) + (eval_count or 0),
                )

        # Override tool call indices with sequential values (Ollama can emit
        # duplicate indices across chunks, which breaks the protocol transformer).
        choices = openai_chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})

            # Keep ``role`` only on the first delta; Ollama emits it on every
            # chunk, but the OpenAI wire convention is role-once.
            if "role" in delta:
                if self._role_sent:
                    delta.pop("role")
                else:
                    self._role_sent = True

            tool_calls = delta.get("tool_calls")
            if tool_calls:
                for call in tool_calls:
                    call["index"] = self._tool_call_index
                    self._tool_call_index += 1

        return openai_chunk

    def get_usage(self) -> StreamingUsage | None:
        """Return the usage captured from the terminal done chunk."""
        return self._usage

    # ------------------------------------------------------------------
    # StreamingTransformer protocol
    # ------------------------------------------------------------------

    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Protocol-side transform — not used for provider-side conversion.

        This method exists only to satisfy the ``StreamingTransformer`` ABC.
        """
        raise NotImplementedError("OllamaChunkConverter is a provider-side converter.")

    def finalize(self) -> str:
        """Stream end marker (protocol-side requirement).

        This method exists only to satisfy the ``StreamingTransformer`` ABC.
        The adapter yields ``[DONE]`` directly; this is a fallback that is
        never called in the provider-side role.
        """
        return "[DONE]"
