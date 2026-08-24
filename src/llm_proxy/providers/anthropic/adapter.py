"""Anthropic provider implementation using native HTTP calls."""

import copy
import time
from collections.abc import AsyncIterator
from typing import Any

import orjson

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.core.conversion import plan_conversion
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    ConversionTier,
    InternalRequest,
    InternalResponse,
    Usage,
)
from llm_proxy.models.provider import ProviderModelInfo
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.anthropic.client_headers import (
    ensure_claude_code_beta,
    get_client_headers,
)
from llm_proxy.providers.base import BaseHttpProvider, _extract_rate_limit_headers
from llm_proxy.providers.capabilities import ChatCapabilityMixin
from llm_proxy.serialization.anthropic.serializer import (
    _normalize_anthropic_messages,
    parse_usage_and_provider_extras,
)
from llm_proxy.serialization.providers import get_provider_serializer

logger = get_logger(__name__)

_serializer = get_provider_serializer("anthropic")


@register_adapter("anthropic")
class AnthropicAdapter(ChatCapabilityMixin, BaseHttpProvider):
    _DEFAULT_PROVIDER_NAME = "anthropic"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Anthropic"
    DISPLAY_NAME_ZH = "Anthropic"
    LOBE_ICON_ID = "anthropic"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://api.anthropic.com"
    CHAT_ENDPOINT = "/v1/messages"

    EXTRA_HEADERS = {"anthropic-version": "2023-06-01"}
    AUTH_HEADER = "x-api-key"
    AUTH_PREFIX = ""

    def __init__(
        self,
        *,
        config: AdapterConfig | None = None,
        **kwargs: Any,
    ):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "anthropic")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)

    def _build_headers(
        self,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
    ) -> dict[str, str]:
        """Build upstream headers, merging captured Claude Code client headers.

        On the native Anthropic path, client fingerprint headers captured by
        the anthropic protocol layer are forwarded verbatim without overriding
        provider/auth headers. ``anthropic-beta`` is rebuilt to guarantee it
        carries the ``claude-code-20250219`` marker so the upstream enables
        Claude Code features.
        """
        headers = super()._build_headers(auth_header, auth_prefix)
        client_headers = get_client_headers()
        if client_headers:
            existing = {k.lower() for k in headers}
            for key, value in client_headers.items():
                if key.lower() not in existing:
                    headers[key] = value
            headers["anthropic-beta"] = ensure_claude_code_beta(headers.get("anthropic-beta"))
        return headers

    def _build_chat_raw(self, request, context):
        return _serializer.build_provider_request(request, context)

    def native_body_hook(self, body: dict[str, Any]) -> dict[str, Any]:
        """Structural message repairs (cc-switch parity) for native bodies.

        Runs on a deep copy of the messages list — the list inside the body is
        still shared by reference with the stashed raw protocol body, which
        must never be mutated in place.
        """
        messages = body.get("messages")
        if isinstance(messages, list):
            body["messages"] = _normalize_anthropic_messages(copy.deepcopy(messages))
        return body

    def _stream_body(self, request: InternalRequest) -> dict[str, Any]:
        # Bodies arrive fully prepared: the passthrough seam handles native
        # bodies, rebuilt bodies pass through unchanged.
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("outbound.json_body must not be None for chat requests")
        return outbound.json_body

    async def chat_completion(self, request: InternalRequest, **_kwargs: Any) -> InternalResponse:
        url = self._stream_url(request)
        headers = self._build_headers()
        body = self._stream_body(request)

        response = await self._post_json_response_with_retry(url, headers, body)
        raw = response.json()
        if plan_conversion(self, request).response_mode == ConversionTier.NATIVE_PASSTHROUGH:
            result = self._build_passthrough_response(raw, request)
        else:
            result = self._parse_response(
                _serializer,
                raw,
                model=request.model,
                request_id=request.request_id,
                request=request,
            )
        result.provider_info["_rate_limit_headers"] = _extract_rate_limit_headers(
            getattr(response, "headers", None)
        )
        return result

    native_protocols = frozenset({"anthropic"})

    def _parse_passthrough_usage(
        self, body: dict[str, Any], request: InternalRequest
    ) -> tuple[Usage | None, dict[str, Any]]:
        """Anthropic usage parsing: cache-token folding + billing extras."""
        return parse_usage_and_provider_extras(body)

    def _native_stream_request(self, request: InternalRequest) -> tuple[str, dict[str, Any]]:
        """URL + body for the native Anthropic Messages stream."""
        url = self._stream_url(request)
        body = self._stream_body(request)
        body["stream"] = True
        return url, body

    async def stream_chat_completion_native(
        self,
        request: InternalRequest,
        cancel_token=None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        url, body = self._native_stream_request(request)
        return self._with_retry_generator(
            lambda: self._stream_raw_sse(url, body, cancel_token),
            cancel_token=cancel_token,
        )

    async def stream_chat_completion(
        self,
        request: InternalRequest,
        cancel_token=None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Stream a chat completion, converting Anthropic SSE to OpenAI chunks.

        The adapter reads raw SSE frames from Anthropic and delegates event-to-chunk
        conversion to ``AnthropicChunkConverter``.  This keeps SSE framing logic in
        the adapter layer while the provider-specific format conversion lives in the
        serialization layer.
        """
        response_id = request.request_id or f"msg_{int(time.time())}"
        model = request.model or ""

        converter = _serializer.get_chunk_converter(model=model, request_id=response_id)
        url, body = self._native_stream_request(request)

        async def _stream_generator():
            try:
                async for frame in self._stream_raw_sse(url, body, cancel_token):
                    data_str = _extract_data_from_sse_frame(frame)
                    if data_str is None:
                        continue

                    try:
                        event = orjson.loads(data_str)
                    except orjson.JSONDecodeError:
                        continue

                    chunk = converter.convert_chunk(event)
                    if chunk is not None:
                        yield chunk

                    # Check if this was the final message_stop event by
                    # looking at the event type.
                    event_type = event.get("type", "")
                    if event_type == "message_stop":
                        yield "[DONE]"
                        return

                # End of stream without message_stop — flush any pending
                # stop_reason + usage as final chunks.
                for final_chunk in converter.finalize_chunks():
                    yield final_chunk
                yield "[DONE]"

            except ProviderError:
                raise
            except Exception as e:
                error = await self._handle_http_error(e)
                raise error from e

        return self._with_retry_generator(_stream_generator, cancel_token=cancel_token)

    def _parse_model(self, raw: dict[str, Any]) -> ProviderModelInfo:
        return ProviderModelInfo(
            id=raw.get("id", ""),
            name=raw.get("display_name", raw.get("id", "")),
            description=(f"Created: {raw.get('created_at')}" if raw.get("created_at") else None),
            owned_by=raw.get("owner"),
        )


def _extract_data_from_sse_frame(frame: str) -> str | None:
    for line in frame.split("\n"):
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str:
                return data_str
    return None


__all__ = ["AnthropicAdapter"]
