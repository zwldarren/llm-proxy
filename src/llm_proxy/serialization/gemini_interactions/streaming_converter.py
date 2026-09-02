"""Interactions API streaming converter (provider chunks -> unified).

The Interactions API streams typed SSE events instead of candidate chunks:

``interaction.created`` → (``step.start`` → ``step.delta``* → ``step.stop``)+
→ ``interaction.completed``

This transformer maps that event timeline onto the canonical OpenAI
``chat.completion.chunk`` format:

- ``step.delta`` text → ``delta.content``
- ``step.delta`` thought_summary / thought → ``delta.reasoning_content``
  (signature captured from ``thought_signature`` deltas)
- ``step.delta`` arguments → incremental ``delta.tool_calls`` (argument
  fragments, matching OpenAI's streaming tool-call shape)
- ``step.delta`` audio / image → ``delta.audio`` / markdown image content
- ``interaction.requires_action`` / ``interaction.completed`` → finish_reason
  (``tool_calls`` / ``stop`` / ``length``) + usage
- failed/cancelled status → ProviderError (error propagation)
"""

import time
from typing import Any

import orjson
from orjson import JSONDecodeError

from llm_proxy.core.exceptions import ProviderError
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
from llm_proxy.streaming.transformer import StreamingTransformer, StreamingUsage


class InteractionsStreamingTransformer(StreamingTransformer):
    """Transforms Interactions SSE events to OpenAI SSE chunk dicts."""

    def __init__(
        self,
        model: str = "",
        request_id: str | None = None,
    ):
        super().__init__(model=model, request_id=request_id)
        self._created = time.time_ns() // 1_000_000_000
        self._chunk_index = 0
        self._interaction_id: str | None = None
        self._text_buffer: str = ""
        self._reasoning_buffer: str = ""
        self._reasoning_signature: str | None = None
        self._audio_buffer: str = ""
        self._audio_mime_type: str | None = None
        # Tool call state per step index: {"id", "name", "arguments", "emitted"}
        self._tool_calls: dict[int, dict[str, Any]] = {}
        # web search requests seen as google_search_call steps
        self._web_search_requests: int = 0
        self._usage: StreamingUsage | None = None
        self._finished = False

    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Transform a raw SSE payload to an OpenAI SSE chunk string.

        Mirrors the legacy Gemini converter: accepts either a pre-parsed
        event dict or a JSON string; events with no client-visible output
        return None.
        """
        if isinstance(chunk, dict):
            converted = self.convert_chunk(chunk)
        else:
            if not chunk or not chunk.strip():
                return None
            try:
                data = orjson.loads(chunk)
            except JSONDecodeError:
                return None
            converted = self.convert_chunk(data)
        if converted is None:
            return None
        return self._make_chunk(converted)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def convert_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a parsed Interactions SSE event dict to an OpenAI chunk dict.

        Returns None for events that produce no client-visible output
        (interaction.created, step.start without content, …).
        """
        if not isinstance(chunk, dict):
            return None

        event_type = chunk.get("type") or chunk.get("event_type")

        if event_type == "error":
            error = chunk.get("error") or {}
            raise ProviderError(
                message=error.get("message") or "Gemini interaction stream error",
                error_type="api_error",
                status_code=502,
                provider_name="gemini-interactions",
                original_error=chunk,
            )

        if event_type == "interaction.created":
            interaction = chunk.get("interaction")
            if isinstance(interaction, dict):
                self._interaction_id = interaction.get("id")
            return None

        if event_type in ("interaction.in_progress", "interaction.status_update"):
            status = chunk.get("status")
            if status in FAILED_STATUSES:
                self._raise_status_error(status, chunk)
            if status in STATUS_TO_FINISH_REASON:
                # Terminal status on a status_update event. The migration
                # guide ends tool-call streams at requires_action (no
                # interaction.completed follows); the API reference models
                # the same states as status_update statuses. Emit the finish
                # chunk for either shape; _finished guards against double
                # emission when interaction.completed follows.
                return self._finish_chunk(self._terminal_finish_reason(status), usage=None)
            return None

        if event_type == "interaction.requires_action":
            return self._finish_chunk("tool_calls", usage=None)

        if event_type == "interaction.completed":
            interaction = chunk.get("interaction") or {}
            status = interaction.get("status") or chunk.get("status")
            if status in FAILED_STATUSES:
                self._raise_status_error(status, chunk)
            finish = self._terminal_finish_reason(status)
            usage = interaction.get("usage")
            if self._finished:
                # A status_update already emitted the finish chunk; keep the
                # usage so get_usage() reflects the real bill even though the
                # client-side chunk was emitted without it.
                if isinstance(usage, dict):
                    self._usage = self._usage_to_streaming(usage)
                return None
            return self._finish_chunk(finish, usage=usage)

        if event_type == "step.start":
            return self._on_step_start(chunk)

        if event_type == "step.delta":
            return self._on_step_delta(chunk)

        if event_type == "step.stop":
            return self._on_step_stop(chunk)

        return None

    def _raise_status_error(self, status: str, event: dict[str, Any]) -> None:
        # The Interaction resource carries failure details in ``error``
        # (singular); ``errors`` is accepted as a defensive fallback. The
        # interaction-level error wins over the event-level one.
        interaction = event.get("interaction") or {}
        message = interaction_error_message(interaction) or interaction_error_message(event)
        raise ProviderError(
            message=message or f"Gemini interaction ended with status={status!r}",
            error_type="api_error",
            status_code=502,
            provider_name="gemini-interactions",
            original_error=event,
        )

    # ------------------------------------------------------------------
    # Step handlers
    # ------------------------------------------------------------------

    def _on_step_start(self, event: dict[str, Any]) -> dict[str, Any] | None:
        step = event.get("step") or {}
        if not isinstance(step, dict):
            return None
        step_type = step.get("type")

        if step_type == "model_output":
            # The step.start event may carry initial content (the May-2026
            # breaking-changes guide shows model_output steps with content);
            # seed the text buffer so it is not lost when no text deltas
            # follow.
            seed = self._text_seed(step.get("content") or [])
            if seed:
                self._text_buffer += seed
                return self._make_openai_chunk({"content": seed})
            return None

        if step_type == "function_call":
            index = event.get("index", len(self._tool_calls))
            self._tool_calls[index] = {
                "id": step.get("id") or f"call_{index}",
                "name": step.get("name", ""),
                "arguments": "",
                "emitted": False,
                # Snapshot the signature of the most recent thought step so
                # each call replays with the signature that preceded it (a
                # turn may interleave several thought/function_call pairs).
                "signature": self._reasoning_signature,
            }
            return None

        if step_type == "thought":
            # A thought step may already carry its (partial) summary in the
            # step object; seed the reasoning buffer so the text is not lost
            # when no thought_summary deltas follow.
            self._reasoning_buffer = ""
            signature = step.get("signature")
            if signature:
                self._reasoning_signature = signature
            seed = self._text_seed(step.get("summary") or [])
            return self._emit_reasoning(seed) if seed else None

        if step_type == "google_search_call":
            self._web_search_requests += 1
            return None

        return None

    @staticmethod
    def _text_seed(content: Any) -> str:
        """Join the text pieces of a Content array (step.start seeding)."""
        return "".join(
            c.get("text", "")
            for c in content or []
            if isinstance(c, dict) and c.get("type") == "text" and c.get("text")
        )

    def _on_step_delta(self, event: dict[str, Any]) -> dict[str, Any] | None:
        delta = event.get("delta")
        if not isinstance(delta, dict):
            return None
        delta_type = delta.get("type")

        if delta_type == "text":
            text = delta.get("text") or ""
            if not text:
                return None
            self._text_buffer += text
            return self._make_openai_chunk({"content": text})

        if delta_type in ("thought", "thought_summary"):
            if delta_type == "thought_summary":
                content = delta.get("content")
                if isinstance(content, dict):
                    text = content.get("text") or ""
                else:
                    text = delta.get("text") or ""
            else:
                text = delta.get("text") or ""
            return self._emit_reasoning(text) if text else None

        if delta_type == "thought_signature":
            signature = delta.get("signature") or delta.get("thought_signature")
            if signature:
                # Track the most recent signature: a turn may contain several
                # thought steps, and each function_call must replay with the
                # signature of the thought step that preceded it.
                self._reasoning_signature = signature
            return None

        if delta_type in ("arguments", "arguments_delta"):
            return self._emit_arguments_delta(event, delta)

        if delta_type == "audio":
            data = delta.get("data") or ""
            mime_type = delta.get("mime_type") or delta.get("mimeType") or ""
            if not data:
                return None
            self._audio_buffer += data
            if mime_type and self._audio_mime_type is None:
                self._audio_mime_type = mime_type
            return self._make_openai_chunk(
                {
                    "audio": {
                        "id": f"audio_{self.response_id or self._chunk_index}",
                        "data": data,
                    }
                }
            )

        if delta_type == "image":
            data = delta.get("data") or ""
            if not data:
                return None
            mime_type = delta.get("mime_type") or delta.get("mimeType") or "image/png"
            markdown = f"![image](data:{mime_type};base64,{data})"
            self._text_buffer += markdown
            return self._make_openai_chunk({"content": markdown})

        if delta_type == "text_annotation_delta":
            # Annotations reference the accumulated text; emit them as-is.
            annotations = delta.get("annotations") or []
            if not annotations:
                return None
            converted = content_annotations_to_openai(annotations, len(self._text_buffer))
            if not converted:
                return None
            return self._make_openai_chunk({"annotations": converted})

        return None

    def _emit_reasoning(self, text: str) -> dict[str, Any] | None:
        """Emit a reasoning delta.

        ``thought``/``thought_summary`` deltas carry incremental text (like
        ``text`` deltas). A cumulative snapshot shape (the whole summary so
        far, supersets of what we buffered) is handled defensively: only the
        new tail is emitted. Genuine increments are appended verbatim.
        """
        buffer = self._reasoning_buffer
        if text.startswith(buffer):
            if len(text) == len(buffer):
                return None
            tail = text[len(buffer) :]
            self._reasoning_buffer = text
            return self._make_openai_chunk({"reasoning_content": tail})
        # Not a cumulative snapshot: an incremental fragment — append it.
        self._reasoning_buffer = buffer + text
        return self._make_openai_chunk({"reasoning_content": text})

    def _emit_arguments_delta(
        self, event: dict[str, Any], delta: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Emit an incremental tool-call arguments fragment."""
        index = event.get("index")
        if index is None:
            return None
        call = self._tool_calls.get(index)
        if call is None:
            # arguments delta without a recorded step.start — create the call.
            call = {"id": f"call_{index}", "name": "", "arguments": "", "emitted": False}
            self._tool_calls[index] = call

        # Interactions streams arguments as partial JSON fragments that "must
        # be accumulated across deltas" (breaking-changes guide); a cumulative
        # snapshot shape is handled defensively for older revisions.
        partial = delta.get("partial_arguments")
        if partial is None:
            partial = delta.get("arguments")
        if not isinstance(partial, str):
            partial = str(partial or "")

        previous = call["arguments"]
        assert isinstance(previous, str)  # always a str: initialized and updated below
        if previous and partial.startswith(previous) and len(partial) > len(previous):
            # Cumulative snapshot: emit only the newly appended tail.
            tail = partial[len(previous) :]
            call["arguments"] = partial
        else:
            # Fragment: append to the accumulated arguments.
            tail = partial
            call["arguments"] = previous + partial

        if not tail:
            return None

        tool_delta: dict[str, Any] = {"index": index}
        if not call["emitted"]:
            call["emitted"] = True
            tool_delta["id"] = call["id"]
            tool_delta["type"] = "function"
            tool_delta["function"] = {"name": call["name"], "arguments": tail}
        else:
            tool_delta["function"] = {"arguments": tail}
        return self._make_openai_chunk({"tool_calls": [tool_delta]})

    def _on_step_stop(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Handle step completion; flush tool calls that had no deltas."""
        index = event.get("index")
        if index is None:
            return None
        call = self._tool_calls.get(index)
        if call is None:
            return None
        if call["emitted"]:
            return None
        # No arguments deltas arrived; emit the (possibly empty) call once.
        call["emitted"] = True
        arguments = call["arguments"] or "{}"
        return self._make_openai_chunk(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": arguments},
                    }
                ]
            }
        )

    # ------------------------------------------------------------------
    # Finish / usage / accumulation
    # ------------------------------------------------------------------

    def _terminal_finish_reason(self, status: str | None) -> str:
        """Map a terminal interaction status to an OpenAI finish reason.

        The live streaming API reports ``status: "completed"`` even for
        interactions whose only output is a function_call step (the
        non-streaming API reports ``requires_action`` for the same response).
        When tool calls were accumulated, the finish reason must be
        ``tool_calls`` regardless of the reported status.
        """
        if self._tool_calls:
            return "tool_calls"
        return STATUS_TO_FINISH_REASON.get(status or "", "stop")

    def _finish_chunk(
        self,
        finish_reason: str,
        *,
        usage: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Emit the terminal chunk: finish_reason (+ usage when present)."""
        if self._finished:
            return None
        self._finished = True

        chunk = self._make_openai_chunk({}, finish_reason=finish_reason)

        if isinstance(usage, dict):
            chunk["usage"] = self._usage_to_openai(usage)
            self._usage = self._usage_to_streaming(usage)

        self._finalize_accumulation()
        return chunk

    def _usage_to_streaming(self, usage: dict[str, Any]) -> StreamingUsage:
        """Map an Interactions usage dict onto the canonical StreamingUsage."""
        search_requests = interactions_web_search_requests(usage)
        prompt_tokens, completion_tokens = interactions_billable_token_counts(
            usage, has_search_grounding=search_requests > 0
        )
        web_search = search_requests or self._web_search_requests
        total = usage.get("total_tokens")
        # Cached tokens go ONLY into cache_read_input_tokens: setting
        # prompt_tokens_details too would make billing apply the
        # cache-rate adjustment twice for the same tokens.
        return StreamingUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total if total is not None else prompt_tokens + completion_tokens,
            cache_read_input_tokens=usage.get("total_cached_tokens"),
            web_search_requests=web_search or None,
        )

    def _usage_to_openai(self, usage: dict[str, Any]) -> dict[str, Any]:
        usage = interactions_normalize_usage(usage)
        has_search = interactions_web_search_requests(usage) > 0
        prompt_tokens, completion_tokens = interactions_billable_token_counts(
            usage, has_search_grounding=has_search
        )
        openai_usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        if usage.get("total_cached_tokens") is not None:
            openai_usage["cache_read_input_tokens"] = usage["total_cached_tokens"]
        if usage.get("total_thought_tokens") is not None:
            openai_usage["reasoning_tokens"] = usage["total_thought_tokens"]
        total = usage.get("total_tokens")
        openai_usage["total_tokens"] = (
            total if total is not None else prompt_tokens + completion_tokens
        )
        return openai_usage

    def _make_openai_chunk(
        self,
        delta: dict[str, Any],
        finish_reason: str | None = None,
    ) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "id": self.response_id or f"chatcmpl-{self._chunk_index}",
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        self._chunk_index += 1
        return chunk

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
                        media_type=self._audio_mime_type,
                    )
                )
            )

        if self._reasoning_buffer:
            self._accumulated_output.append(
                ThinkingBlock(
                    thinking=self._reasoning_buffer,
                    signature=self._reasoning_signature,
                    signature_origin="gemini",
                )
            )

        if self._text_buffer:
            self._accumulated_output.append(TextBlock(text=self._text_buffer))

        for call in self._tool_calls.values():
            try:
                tool_input = orjson.loads(call["arguments"]) if call["arguments"] else {}
            except JSONDecodeError, TypeError:
                tool_input = {}
            # The Interactions API requires the thought signature to be
            # replayed with a function_call step in stateless multi-turn tool
            # conversations; the signature arrives in a thought_signature delta
            # before the function_call step, so attach the per-call snapshot
            # here. The adapter caches it keyed by tool call id and re-attaches
            # on the next request.
            block = ToolUseBlock(id=call["id"], name=call["name"], input=tool_input)
            signature = call.get("signature")
            if signature:
                block.extra = {"thought_signature": signature}
            self._accumulated_output.append(block)

        self._text_buffer = ""
        self._reasoning_buffer = ""
        self._reasoning_signature = None
        self._audio_buffer = ""
        self._audio_mime_type = None
        self._tool_calls = {}

    def finalize(self) -> str:
        """Generate stream end marker."""
        if not self._finished:
            self._finalize_accumulation()
        return "data: [DONE]\n\n"

    def get_usage(self) -> StreamingUsage | None:
        """Return accumulated usage from the interactions stream."""
        if self._usage is not None:
            return self._usage
        if self._web_search_requests > 0:
            return StreamingUsage(web_search_requests=self._web_search_requests)
        return None
