"""OpenAI provider request builder component.

Responsible for building OpenAI-format request bodies from InternalRequest.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from llm_proxy.models import InternalRequest
from llm_proxy.models.tools import ToolChoiceAllowedTools
from llm_proxy.serialization.openai.converter import format_conversation

if TYPE_CHECKING:
    from llm_proxy.serialization.context import BuildContext

logger = logging.getLogger(__name__)

# Default reasoning field name for standard OpenAI format
_DEFAULT_REASONING_FIELD = "reasoning_content"
_CACHE_MAX_SIZE = 100
_CACHE_TTL_SECONDS = 3600


# OpenAI Responses API fields that have no equivalent in the Chat Completions
# API. When a /v1/responses request is proxied through an openai-compatible
# Chat Completions provider, these must be stripped from the upstream body.
_RESPONSES_ONLY_EXTRA_KEYS = frozenset(
    {
        "previous_response_id",
        "background",
        "context_management",
        "conversation",
        "prompt",
        "max_tool_calls",
        "truncation",
        "include",
        "reasoning",
        "responses_tools",
        "text",
        # Carrier for raw /v1/responses fields the schema does not model
        # (context_management, prompt_cache_options, ...); merged back into
        # the body only by the native Responses provider serializer.
        "responses_raw_fields",
    }
)


class OpenAIRequestBuilder:
    """Builds OpenAI Chat Completions request bodies from InternalRequest.

    Handles parameter mapping, tools conversion, thinking/reasoning fields,
    and provider-specific reasoning field name detection with TTL-based caching.
    """

    def __init__(self) -> None:
        # Reasoning field cache: (base_url, model) -> (field_name, timestamp).
        # The field convention belongs to the *model* (one gateway can serve
        # models with different conventions), so entries are keyed per model
        # with a TTL; the base_url-only key (model=None) remains the fallback
        # for lookups that carry no model.
        self._reasoning_field_cache: dict[tuple[str | None, str | None], tuple[str, float]] = {}
        self._reasoning_cache_lock = threading.Lock()

    def build(self, request: InternalRequest, context: BuildContext) -> dict[str, Any]:
        """Build OpenAI request body from InternalRequest."""
        messages = format_conversation(request.conversation, context)

        body: dict[str, Any] = {
            "model": context.model or request.model,
            "messages": messages,
            "stream": context.stream,
        }

        body = self._build_stream_options(body, request)
        body = self._build_params(body, request)
        body = self._build_response_format(body, request)
        body = self._build_tools(body, request, context)
        body = self._build_thinking(body, request)
        body = self._build_openai_params(body, request, context)
        body = self._build_extra(body, request)
        body = self.normalize_reasoning_for_request(body, context.base_url, model=body.get("model"))

        return body

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_stream_options(body: dict[str, Any], request: InternalRequest) -> dict[str, Any]:
        if request.stream_options is not None:
            opts: dict[str, Any] = {
                "include_usage": request.stream_options.include_usage,
            }
            if request.stream_options.include_obfuscation is not None:
                opts["include_obfuscation"] = request.stream_options.include_obfuscation
            body["stream_options"] = opts
        return body

    @staticmethod
    def _build_params(body: dict[str, Any], request: InternalRequest) -> dict[str, Any]:
        # Token-limit field selection for the Chat Completions wire.
        # ``max_completion_tokens`` is OpenAI's non-deprecated, o-series-compatible
        # replacement for ``max_tokens``; the protocol parser mirrors an isolated
        # ``max_completion_tokens`` into the common ``max_tokens`` field (so
        # non-OpenAI providers that only read ``max_tokens`` still get the limit).
        # Precedence when both are present: an explicit ``max_tokens`` whose value
        # *differs* from ``max_completion_tokens`` wins; otherwise emit
        # ``max_completion_tokens`` (o-series safe, and a no-op when the values
        # are equal because the mirror already copied it into ``max_tokens``).
        max_completion_tokens = (
            request.params.openai.max_completion_tokens
            if request.params.openai is not None
            else None
        )
        max_tokens = request.params.max_tokens
        if (
            max_tokens is not None
            and max_completion_tokens is not None
            and max_tokens != max_completion_tokens
        ):
            # Both sent with different values: explicit max_tokens wins.
            body["max_tokens"] = max_tokens
        elif max_completion_tokens is not None:
            # Only max_completion_tokens (mirrored), or both equal: emit the
            # o-series-compatible field.
            body["max_completion_tokens"] = max_completion_tokens
        elif max_tokens is not None:
            body["max_tokens"] = max_tokens
        for attr in (
            "temperature",
            "top_p",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "seed",
        ):
            val = getattr(request.params, attr, None)
            if val is not None:
                body[attr] = val
        return body

    @staticmethod
    def _build_response_format(body: dict[str, Any], request: InternalRequest) -> dict[str, Any]:
        if request.params.response_format is not None:
            rf = request.params.response_format
            rf_dict: dict[str, Any] = {"type": rf.type}
            if rf.type == "json_schema" and rf.json_schema is not None:
                rf_dict["json_schema"] = rf.json_schema
            body["response_format"] = rf_dict
        return body

    def _build_tools(
        self, body: dict[str, Any], request: InternalRequest, context: BuildContext
    ) -> dict[str, Any]:
        if not request.tools and not request.tool_choice:
            return body

        from llm_proxy.serialization.openai.components.tools_handler import (
            OpenAIToolsHandler,
        )

        handler = OpenAIToolsHandler()
        if request.tools:
            tools = list(request.tools)
            # Enforce the allowed_tools hard constraint: Chat Completions has no
            # allowed_tools concept, so the tool set is filtered to the allowed
            # subset and the mode is used as tool_choice.
            if isinstance(request.tool_choice, ToolChoiceAllowedTools):
                allowed_names = {
                    t.get("name", "")
                    for t in request.tool_choice.allowed_tools.tools
                    if isinstance(t, dict)
                }
                if allowed_names:
                    tools = [t for t in tools if getattr(t, "name", None) in allowed_names]
            body["tools"] = handler.build_tools(tools, target_endpoint=context.target_endpoint)
        if request.tool_choice:
            body["tool_choice"] = handler.build_tool_choice(
                request.tool_choice, target_endpoint=context.target_endpoint
            )
        return body

    @staticmethod
    def _build_thinking(body: dict[str, Any], request: InternalRequest) -> dict[str, Any]:
        from llm_proxy.core.thinking import convert_to_openai, resolve_thinking

        thinking = resolve_thinking(request)
        if thinking is not None:
            thinking_params = convert_to_openai(thinking)
            if thinking_params:
                body.update(thinking_params)
        return body

    @staticmethod
    def _should_send_prompt_cache_key(base_url: str | None) -> bool:
        """Whether ``prompt_cache_key`` may be sent to a Chat Completions upstream.

        ``prompt_cache_key`` is a Responses-API-originated field that strict
        Chat Completions gateways reject with HTTP 400 ("unknown parameter").
        Only known-compatible upstreams accept it: api.openai.com, and
        api.kimi.com under the ``/coding`` base path.
        """
        if not base_url:
            return False
        try:
            parsed = urlparse(base_url)
            host = (parsed.hostname or "").lower()
        except ValueError:
            return False
        if host == "api.openai.com":
            return True
        if host == "api.kimi.com":
            path = (parsed.path or "").rstrip("/")
            return path == "/coding" or path.startswith("/coding/")
        return False

    @staticmethod
    def _build_openai_params(
        body: dict[str, Any], request: InternalRequest, context: BuildContext
    ) -> dict[str, Any]:
        if request.params.openai is None:
            return body

        op = request.params.openai
        for attr in (
            "service_tier",
            "verbosity",
            "store",
            "metadata",
            "prompt_cache_retention",
            "safety_identifier",
            "logprobs",
            "top_logprobs",
            "audio",
            "modalities",
            "prediction",
            "web_search_options",
            "parallel_tool_calls",
            "logit_bias",
        ):
            val = getattr(op, attr, None)
            if val is not None:
                body[attr] = val
        # Gated: only forward prompt_cache_key to upstreams known to accept it
        # (strict gateways 400 on unknown fields). See
        # _should_send_prompt_cache_key.
        if op.prompt_cache_key is not None and OpenAIRequestBuilder._should_send_prompt_cache_key(
            context.base_url
        ):
            body["prompt_cache_key"] = op.prompt_cache_key
        return body

    @staticmethod
    def _build_extra(body: dict[str, Any], request: InternalRequest) -> dict[str, Any]:
        if request.n is not None:
            body["n"] = request.n
        if request.user is not None:
            body["user"] = request.user
        if request.extra:
            # Responses-API-only keys have no Chat Completions equivalent and
            # must not leak into the upstream body.
            dropped = [
                k
                for k in request.extra
                if k in _RESPONSES_ONLY_EXTRA_KEYS and request.extra[k] is not None
            ]
            if dropped:
                # Expected on every Responses -> Chat Completions translation
                # (Responses clients routinely send include/reasoning); not
                # actionable, so keep it out of the default log level.
                logger.debug(
                    "Dropping Responses-API-only extra keys %r from Chat Completions request body",
                    dropped,
                )
            body.update(
                {
                    k: v
                    for k, v in request.extra.items()
                    if v is not None and k not in _RESPONSES_ONLY_EXTRA_KEYS
                }
            )
        return body

    # ------------------------------------------------------------------
    # Reasoning field handling
    # ------------------------------------------------------------------

    def get_reasoning_field_preference(self, base_url: str | None, model: str | None = None) -> str:
        """Get cached reasoning field preference for a base URL and model.

        Looks up the exact ``(base_url, model)`` entry first, then falls back
        to the model-less ``(base_url, None)`` entry so callers that do not
        know the model still benefit from a detected convention.
        """
        if base_url is None:
            return _DEFAULT_REASONING_FIELD
        with self._reasoning_cache_lock:
            for key in ((base_url, model), (base_url, None)):
                entry = self._reasoning_field_cache.get(key)
                if entry is None:
                    continue
                field, timestamp = entry
                if time.time() - timestamp > _CACHE_TTL_SECONDS:
                    del self._reasoning_field_cache[key]
                    continue
                return field
            return _DEFAULT_REASONING_FIELD

    def set_reasoning_field_preference(
        self,
        base_url: str | None,
        field: str,
        model: str | None = None,
    ) -> None:
        """Cache reasoning field preference for a base URL and model.

        Stored under the exact ``(base_url, model)`` key only — one model's
        convention never bleeds into another model's lookups (a gateway can
        mix ``reasoning`` and ``reasoning_content`` models on one base URL).
        """
        if base_url is None or field not in ("reasoning", "reasoning_content"):
            return
        key = (base_url, model)
        with self._reasoning_cache_lock:
            if (
                len(self._reasoning_field_cache) >= _CACHE_MAX_SIZE
                and key not in self._reasoning_field_cache
            ):
                oldest_key = min(
                    self._reasoning_field_cache, key=lambda k: self._reasoning_field_cache[k][1]
                )
                del self._reasoning_field_cache[oldest_key]
            self._reasoning_field_cache[key] = (field, time.time())

    def record_reasoning_field_preference(
        self,
        base_url: str | None,
        detected: str,
        *,
        model: str | None = None,
        response_model: Any = None,
    ) -> None:
        """Cache a detected reasoning field, under the routed model and its alias.

        Single write pattern shared by every response-side detection site
        (parsed responses, wire-reuse bodies, streaming chunks): the
        preference is stored under the routed ``model`` (the key future
        requests look up) and, when the upstream reported a different id,
        under the upstream-reported ``response_model`` too (model aliasing).
        """
        self.set_reasoning_field_preference(base_url, detected, model=model)
        if response_model and response_model != model:
            self.set_reasoning_field_preference(base_url, detected, model=response_model)

    def clear_reasoning_field_preference(self, base_url: str | None) -> None:
        """Drop every cached reasoning field entry for a base URL.

        Used by tests to reset learned state; the TTL would self-heal stale
        entries in production anyway.
        """
        if base_url is None:
            return
        with self._reasoning_cache_lock:
            for key in list(self._reasoning_field_cache):
                if key[0] == base_url:
                    del self._reasoning_field_cache[key]

    def normalize_reasoning_for_request(
        self,
        body: dict[str, Any],
        base_url: str | None,
        preferred: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Normalize assistant-message reasoning fields for a provider.

        Uses the cached per-model preference (``reasoning`` or
        ``reasoning_content``) so the request body matches what the upstream
        model expects. If no preference has been detected yet, defaults to
        ``reasoning_content``.

        Args:
            body: Request body containing ``messages``.
            base_url: Upstream base URL used for preference caching.
            preferred: Optional explicit field name to use, overriding the cache.
            model: The routed model id, used as the preference cache key.

        Public entry point used by OpenAICompatibleBase adapters.
        """
        if base_url is None and preferred is None:
            return body
        field = preferred or self.get_reasoning_field_preference(base_url, model)
        if field == "reasoning_content":
            return body
        messages = body.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant" and "reasoning_content" in msg:
                msg["reasoning"] = msg.pop("reasoning_content")
        return body


__all__ = ["OpenAIRequestBuilder"]
