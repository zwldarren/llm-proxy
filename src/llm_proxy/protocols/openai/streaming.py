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

from llm_proxy.models.content_blocks import ContentBlock
from llm_proxy.models.finish_reasons import to_openai_finish_reason
from llm_proxy.models.types import Usage
from llm_proxy.serialization.usage import UsageAccumulator
from llm_proxy.streaming.accumulation import SegmentAccumulator
from llm_proxy.streaming.sse_parse import iter_sse_data_events
from llm_proxy.streaming.transformer import StreamingTransformer

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

    # Native Chat Completions frames carry the same ``choices[].delta`` payloads
    # the canonical path accumulates, so the native-passthrough tier can be
    # logged as a reassembled body too (see accumulate_native_frame).
    native_frame_accumulation = True

    # Chat Completions participates in the continuation but renders neither the
    # result blocks nor the usage delta on the client wire: the search results
    # reach the client through the continuation request, and ``finalize``
    # already reports the accumulated usage.
    supports_web_search_continuation = True

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
        # The canonical parse of the provider's usage frames (ADR-0025), kept
        # beside the verbatim copy above: that one is served to the client and
        # stays byte-faithful, this one is what the proxy bills from.
        self._usage = UsageAccumulator("chat")
        self._emitted_usage_chunk: bool = False
        # The segment engine assembles the canonical blocks (ADR-0031); the
        # Chat Completions policy is the default one: parallel tool lanes,
        # no cursor, structural tool classification. The events it returns
        # are ignored — this protocol's wire is a fidelity-first passthrough
        # and derives nothing from the assembly decisions.
        self._segments = SegmentAccumulator()
        self._include_obfuscation: bool | None = include_obfuscation
        self._finish_reason: str | None = None
        self._system_fingerprint: str | None = None
        self._service_tier: str | None = None

    def get_finish_reason(self) -> str | None:
        """Return the finish_reason seen on the terminal chunk, if any."""
        return self._finish_reason

    def get_terminal_provider_info(self) -> dict[str, Any] | None:
        """Envelope fields for the reassembled request-log body.

        Mirrors the non-streaming OpenAI response's ``provider_info`` (see
        ``response_parser.parse_provider_response``) so a streamed and a
        non-streamed call log the same fields. Read before ``finalize``,
        which flushes the pending terminal state.
        """
        info: dict[str, Any] = {}
        if self._system_fingerprint:
            info["system_fingerprint"] = self._system_fingerprint
        if self._service_tier:
            info["service_tier"] = self._service_tier
        return info or None

    def accumulate_native_frame(self, frame: str | dict[str, Any]) -> None:
        """Accumulate a protocol-native chat-completions frame for logging.

        The native-passthrough tier forwards upstream frames verbatim (see
        ``NativePassthroughHandler.handle_native_openai_chunk``), so the
        accumulation that :meth:`transform` performs as a side effect never runs
        for those requests. The frames are ordinary ``chat.completion.chunk``
        payloads, so they feed the same bookkeeping — letting the request log
        store the reassembled body instead of the raw SSE text.

        Args:
            frame: One native SSE frame (``data: {...}\n\n``, possibly carrying
                several data lines) or an already-parsed chunk dict.
        """
        if isinstance(frame, dict):
            self._accumulate_from_chunk(frame)
            return
        if not isinstance(frame, str):
            return
        for _event_type, payload in iter_sse_data_events(frame):
            if isinstance(payload, dict):
                self._accumulate_from_chunk(payload)

    def flush_pending_accumulation(self) -> None:
        """Finalize content still buffered when the stream ended early.

        Called before the response log is reassembled: a native stream that
        never delivered a terminal ``finish_reason`` (client abort, upstream
        cut) would otherwise log nothing. ``finalize`` flushes the same
        buffers on the converted path, which is skipped for native streams.
        """
        self._segments.finish()

    @classmethod
    def continuation(
        cls,
        model: str,
        request_id: str,
        start_index: int,
        *,
        intercept_web_search: bool = True,
    ) -> OpenAIStreamingTransformer:
        """Create a transformer that continues an existing stream.

        OpenAI chat-completion chunks are self-contained (each carries its
        own id/object/created/model), so a fresh transformer instance is
        sufficient for the web-search continuation loop; ``start_index`` is
        accepted for interface compatibility with the continuation seam
        (Anthropic uses it for content-block indices, OpenAI has none), and so
        is ``intercept_web_search`` (Chat Completions has no proxy-side search
        interception to carry).
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
                # Read the ORIGINAL chunk, like the accumulation below: the wire
                # copy above is normalized for the client, not for billing.
                self._usage.observe(chunk)
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
        """Feed one canonical chunk's segments to the accumulator.

        Envelope facts (``system_fingerprint``, ``service_tier``) and the
        terminal ``finish_reason`` stay here — they are chunk-envelope
        knowledge, not segment assembly. The encrypted reasoning blob the
        OpenAI Responses bridge attaches at the chunk's top level feeds as a
        synthetic delta; the engine holds it for the next reasoning segment
        instead of opening an empty one (ADR-0031).
        """
        if not original_chunk:
            return

        # Captured for the reassembled log body: the OpenAI formatter emits
        # system_fingerprint/service_tier from provider_info, which reassembly
        # has no provider response to fill — the transformer supplies the
        # values seen on the chunk envelope instead. First non-empty wins.
        fingerprint = original_chunk.get("system_fingerprint")
        if fingerprint and self._system_fingerprint is None:
            self._system_fingerprint = fingerprint
        service_tier = original_chunk.get("service_tier")
        if service_tier and self._service_tier is None:
            self._service_tier = service_tier

        # Capture encrypted reasoning state from OpenAI Responses provider.
        # This can arrive as a top-level field (response.completed fallback)
        # or inside a delta; either way it must survive to the reasoning
        # segment it belongs to.
        encrypted = original_chunk.get("encrypted_content")
        if isinstance(encrypted, str) and encrypted:
            self._segments.feed({"encrypted_content": encrypted})

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

            # The engine assembles the canonical blocks (ADR-0031); the
            # returned segment events are ignored — this protocol's wire is
            # a fidelity-first passthrough and derives nothing from the
            # assembly decisions.
            self._segments.feed(delta)

            if finish_reason:
                # Kept for the reassembled log body: the wire finish_reason is
                # otherwise discarded, and the formatter would default it to
                # "stop" even for a truncated (``length``) stream.
                self._finish_reason = finish_reason
                self._segments.finish()

    def get_accumulated_output(self) -> list[ContentBlock]:
        """The assembled canonical blocks, from the segment engine (ADR-0031)."""
        return self._segments.blocks()

    def supersede(self, index: int, block: ContentBlock) -> None:
        """Replace a closed block in the engine's canonical list (ADR-0031).

        The web-search interception's ``ToolUseBlock → ServerToolUseBlock``
        substitution; the base verb, delegating to the same engine this
        transformer reads through.
        """
        self._segments.supersede(index, block)

    def _normalize_choice(self, choice: dict) -> dict | None:
        """Normalize a choice object, passing unknown fields through."""
        if choice is None or not isinstance(choice, dict):
            return None
        result = _strip_internal_keys(choice)
        result.pop("delta", None)

        # Canonical-channel extras (Anthropic ``compaction``/``stop_sequence``,
        # the internal ``context_length`` sentinel) are not members of the
        # official OpenAI enum; clamp at the wire. Legal values — including the
        # legacy ``function_call`` — pass untouched.
        if "finish_reason" in result:
            result["finish_reason"] = to_openai_finish_reason(result["finish_reason"])

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
        # Flush every non-empty lane — media included. The old gate checked
        # only the text/reasoning/tool buffers, so a turn that produced only
        # images or audio lost them from the reassembled log (ADR-0031).
        self._segments.finish()
        chunks: list[str] = []
        if self._pending_usage is not None and not self._emitted_usage_chunk:
            chunks.append(self._make_chunk({"choices": [], "usage": self._pending_usage}))
        chunks.append("data: [DONE]\n\n")
        return "".join(chunks)

    def _own_usage(self) -> Usage | None:
        """The turn's canonical usage record, from the shared intake accumulator.

        The provider's frames are parsed by the seam (ADR-0025), so the dialect
        decisions — the cache-alias lift, the DeepSeek fold, the cost precedence —
        live once, in the chat adapter, instead of here.
        """
        return self._usage.result()

    def web_search_result_block(
        self,
        index: int,
        tool_use_id: str,
        results: list[dict[str, Any]] | str,
        is_error: bool = False,
        query: str = "",
        error_message: str | None = None,
    ) -> str:
        """Emit a web search result block in OpenAI Chat Completions SSE format.

        For Chat Completions the search results are consumed by the
        continuation mechanism — the proxy sends a second request with
        the results as tool messages and the model's text response is
        what the user sees.  We therefore return an empty string so that
        no intermediate SSE chunks are emitted to the client.
        """
        return ""

    def usage_delta(self, usage: dict[str, Any]) -> str:  # noqa: ARG002
        """Emit a message delta with server-side tool usage.

        For Chat Completions the usage delta emitted by :meth:`finalize`
        already covers the accumulated usage across the original request
        and any continuation turns, so we return an empty string here.
        """
        return ""
