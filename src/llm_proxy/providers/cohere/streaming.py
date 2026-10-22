"""Cohere v2 chat SSE events -> canonical ``chat.completion.chunk`` dicts.

Cohere streams named events rather than OpenAI-shaped chunks, and it never sends
``[DONE]``: the stream ends after the single ``message-end`` event
(docs.cohere.com/docs/streaming). Each event carries its payload under
``delta``:

==================  ==========================================================
``message-start``   the response id and the assistant role
``content-start``   opens a content block
``content-delta``   one token of ``delta.message.content.text`` or
                    ``…content.thinking`` (reasoning models stream thinking
                    through the same event)
``content-end``     closes the content block
``tool-plan-delta`` one token of the chain-of-thought plan that precedes a tool
                    call (``delta.message.tool_plan``)
``tool-call-start`` a tool call's id and function name (``index`` selects it)
``tool-call-delta`` one fragment of that call's JSON ``arguments``
``tool-call-end``   closes the call
``citation-start``  / ``citation-end``  a citation of a RAG answer
``message-end``     ``delta.finish_reason`` and ``delta.usage``
==================  ==========================================================

So the converter owns two translations: Cohere's event vocabulary becomes an
OpenAI delta, and the terminal event becomes the chunk carrying
``finish_reason`` plus usage. Citation events have no slot on the canonical
chunk channel; the non-streaming parser attaches citations to the text block
they cite, and this converter drops them the same way the Chat Completions
dialects drop their annotation events.
"""

import time
from typing import Any

from llm_proxy.core.exceptions import ProviderError

# The stream and the non-streaming body must agree on a turn's finish reason, so
# both read the parser's single table.
from llm_proxy.providers.cohere.response_parser import COHERE_TO_OPENAI_FINISH_REASONS
from llm_proxy.serialization.usage import UsageAccumulator
from llm_proxy.serialization.usage import parse_usage as parse_wire_usage
from llm_proxy.streaming.transformer import StreamingTransformer


class CohereChunkConverter(StreamingTransformer):
    """Convert Cohere's named stream events to canonical OpenAI chunk dicts."""

    def __init__(self, model: str = "", request_id: str | None = None) -> None:
        super().__init__(model=model, request_id=request_id)
        # One timestamp/id per stream: every chunk of a turn shares them, which
        # is what clients key their accumulator on.
        self._created_at: int = int(time.time())
        self._stream_id: str = request_id or f"chatcmpl-{self._created_at}"
        self._role_sent: bool = False
        self._terminal_sent: bool = False
        # Usage is reported once, on ``message-end``; the intake seam owns the
        # parse so billing never depends on the protocol transformer.
        self._usage = UsageAccumulator("cohere")

    # ------------------------------------------------------------------
    # Provider-side API (called by the adapter)
    # ------------------------------------------------------------------

    def convert_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Convert one Cohere stream event into a canonical chunk.

        Returns ``None`` for events that carry no client-visible delta
        (content/tool-call/citation delimiters), which is how the canonical
        channel expresses "nothing to forward".
        """
        self._usage.observe(chunk)
        event_type = chunk.get("type")

        match event_type:
            case "message-start":
                event_id = chunk.get("id")
                if isinstance(event_id, str) and event_id:
                    self._stream_id = event_id
                return self._delta_chunk({"role": "assistant", "content": ""})

            case "content-delta":
                content = _nested(chunk, "delta", "message", "content")
                text = content.get("text") if isinstance(content, dict) else None
                thinking = content.get("thinking") if isinstance(content, dict) else None
                delta: dict[str, Any] = {}
                if isinstance(thinking, str) and thinking:
                    delta["reasoning_content"] = thinking
                if isinstance(text, str) and text:
                    delta["content"] = text
                return self._delta_chunk(delta) if delta else None

            case "tool-plan-delta":
                # Cohere reports its pre-call plan as its own token stream. It is
                # the model's reasoning about which call to make, so it rides the
                # canonical reasoning channel beside ``thinking``.
                plan = _nested(chunk, "delta", "message").get("tool_plan")
                if isinstance(plan, str) and plan:
                    return self._delta_chunk({"reasoning_content": plan})
                return None

            case "tool-call-start":
                calls = _nested(chunk, "delta", "message").get("tool_calls")
                call = _first(calls)
                if call is None:
                    return None
                function = call.get("function") or {}
                return self._delta_chunk(
                    {
                        "tool_calls": [
                            {
                                "index": self._event_index(chunk),
                                "id": call.get("id") or "",
                                "type": call.get("type") or "function",
                                "function": {
                                    "name": function.get("name") or "",
                                    "arguments": function.get("arguments") or "",
                                },
                            }
                        ]
                    }
                )

            case "tool-call-delta":
                call = _first(_nested(chunk, "delta", "message").get("tool_calls"))
                if call is None:
                    return None
                fragments = (call.get("function") or {}).get("arguments") or ""
                fragment = _first(fragments) if isinstance(fragments, list) else fragments
                if not isinstance(fragment, str) or not fragment:
                    return None
                return self._delta_chunk(
                    {
                        "tool_calls": [
                            {
                                "index": self._event_index(chunk),
                                "function": {"arguments": fragment},
                            }
                        ]
                    }
                )

            case "message-end":
                return self._terminal_chunk(chunk)

            case _:
                # content-start / content-end / tool-call-end / citation-start /
                # citation-end and any future delimiter carry no delta.
                return None

    def finalize_chunks(self) -> list[dict[str, Any]]:
        """Close a stream that ended without its ``message-end`` event.

        An upstream cut (or a proxy-side cancel) leaves the turn without a
        ``finish_reason``; a spec-strict client would then wait forever. One
        terminal chunk is emitted so the client sees a stop.
        """
        if self._terminal_sent or not self._role_sent:
            return []
        return [self._terminal_chunk({})]

    def _own_usage(self):
        """The turn's canonical usage record, as reported by ``message-end``."""
        return self._usage.result()

    # ------------------------------------------------------------------
    # Chunk building
    # ------------------------------------------------------------------

    def _delta_chunk(self, delta: dict[str, Any]) -> dict[str, Any]:
        self._role_sent = True
        return {
            "id": self._stream_id,
            "object": "chat.completion.chunk",
            "created": self._created_at,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }

    def _terminal_chunk(self, event: dict[str, Any]) -> dict[str, Any]:
        """Build the chunk carrying ``finish_reason`` and the turn's usage."""
        self._terminal_sent = True
        raw_delta = event.get("delta")
        delta: dict[str, Any] = raw_delta if isinstance(raw_delta, dict) else {}
        error = delta.get("error")
        if isinstance(error, str) and error:
            # v2 reports a generation failure on the terminal event. Surfacing
            # it as a truncated "stop" would hand the client a silently broken
            # answer, so it becomes a real provider error.
            raise ProviderError(
                message=error,
                error_type="api_error",
                status_code=500,
                provider_name="cohere",
            )

        raw_reason = delta.get("finish_reason")
        chunk: dict[str, Any] = {
            "id": self._stream_id,
            "object": "chat.completion.chunk",
            "created": self._created_at,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": COHERE_TO_OPENAI_FINISH_REASONS.get(raw_reason or "", "stop"),
                }
            ],
        }
        usage = _openai_usage(event)
        if usage is not None:
            chunk["usage"] = usage
        return chunk

    @staticmethod
    def _event_index(event: dict[str, Any]) -> int:
        index = event.get("index")
        return index if isinstance(index, int) else 0

    # ------------------------------------------------------------------
    # StreamingTransformer protocol (provider-side role)
    # ------------------------------------------------------------------

    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Protocol-side transform — unused for provider-side conversion."""
        raise NotImplementedError("CohereChunkConverter is a provider-side converter.")

    def finalize(self) -> str:
        """Stream end marker — the adapter emits ``[DONE]`` itself."""
        return "[DONE]"


def _nested(payload: Any, *keys: str) -> dict[str, Any]:
    """Walk nested dicts, returning ``{}`` as soon as a level is not a dict."""
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _first(value: Any) -> dict[str, Any] | None:
    """Cohere sends single-element lists and bare objects interchangeably."""
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, dict) else None


def _openai_usage(event: dict[str, Any]) -> dict[str, Any] | None:
    """The terminal event's usage, in the OpenAI dialect the chunk carries.

    Cohere reports ``usage.tokens.{input_tokens, output_tokens}`` plus a cache-hit
    count; the canonical chunk speaks OpenAI's counter names. Cache-read tokens
    are a subset of the prompt count, so they ride their own key (and the nested
    details object) instead of being added to it.
    """
    parsed = parse_wire_usage("cohere", event)
    if parsed is None:
        return None
    usage: dict[str, Any] = {
        "prompt_tokens": parsed.input_tokens,
        "completion_tokens": parsed.output_tokens,
        "total_tokens": parsed.total_tokens,
    }
    cached_tokens = parsed.cache_read_input_tokens
    if cached_tokens is not None:
        usage["cache_read_input_tokens"] = cached_tokens
        if cached_tokens:
            usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return usage


__all__ = ["CohereChunkConverter"]
