"""Chat Completions base with optional native Anthropic/Responses passthrough.

Generalizes the DeepSeek adapter's passthrough pattern (ADR-0002) for
providers whose upstreams expose native Anthropic Messages and/or OpenAI
Responses endpoints alongside Chat Completions. A concrete adapter declares
its native endpoints as data:

- ``native_protocols`` — the client protocols served verbatim (subset of
  ``{"anthropic", "openresponses"}``);
- ``ANTHROPIC_MESSAGES_URL`` / ``RESPONSES_URL`` — full default endpoint URLs
  (native endpoints often live on a different root than the chat base URL);
- when a URL constant is None the endpoint is derived as
  ``{native_root}{PATH}`` from ``_native_root_base_url()`` (which defaults to
  the configured base URL — xAI, MiniMax — and can be overridden by
  adapters whose native endpoints live on a different root, e.g. DeepSeek
  strips the ``/v1`` compatibility alias); ``None`` on both the URL and the
  path means the protocol is not natively supported.

``endpoint_base_urls`` overrides (keys ``anthropic_messages`` / ``responses``)
always win over the class constants, and the ``native_passthrough: false``
provider-metadata flag disables passthrough entirely. The Chat Completions
client protocol never takes the native tier (native_protocols covers only
Anthropic/OpenResponses), so the reasoning-echo guarantee keeps applying.
"""

import copy
from collections.abc import AsyncIterator
from typing import Any

from llm_proxy.core.conversion import plan_conversion, prepare_native_body
from llm_proxy.models import ConversionTier, InternalRequest, InternalResponse, Usage
from llm_proxy.providers.anthropic.client_headers import (
    ensure_claude_code_beta,
)
from llm_proxy.providers.anthropic.client_headers import (
    get_client_headers as get_anthropic_client_headers,
)
from llm_proxy.providers.base import _extract_rate_limit_headers
from llm_proxy.providers.openai.client_headers import (
    get_client_headers as get_openai_client_headers,
)
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase
from llm_proxy.serialization.anthropic.serializer import (
    _normalize_anthropic_messages,
    parse_usage_and_provider_extras,
)
from llm_proxy.serialization.openai.serializer import parse_usage_from_response


class NativePassthroughChatBase(OpenAICompatibleBase):
    """OpenAI-compatible chat base + declared native Anthropic/Responses endpoints."""

    #: Full default URL of the native Anthropic Messages endpoint. None falls
    #: back to ``{native_root}{ANTHROPIC_MESSAGES_PATH}``.
    ANTHROPIC_MESSAGES_URL: str | None = None
    #: Anthropic path appended to ``_native_root_base_url()`` when
    #: ANTHROPIC_MESSAGES_URL is None. None means the provider has no native
    #: Anthropic endpoint.
    ANTHROPIC_MESSAGES_PATH: str | None = None
    #: Full default URL of the native OpenAI Responses endpoint. None falls
    #: back to ``{native_root}{RESPONSES_PATH}``.
    RESPONSES_URL: str | None = None
    #: Responses path appended to ``_native_root_base_url()`` when
    #: RESPONSES_URL is None (see above).
    RESPONSES_PATH = "/responses"

    # ------------------------------------------------------------------
    # Passthrough gate & veto
    # ------------------------------------------------------------------

    def _native_passthrough_enabled(self) -> bool:
        """Provider-metadata kill switch (``native_passthrough: false``).

        Relays that mirror only the Chat Completions endpoint can opt out;
        both the request and the stream side honor the flag.
        """
        return bool(self._extra_config.get("native_passthrough", True))

    def supports_native_request(
        self, protocol_name: str | None, request: InternalRequest | None = None
    ) -> bool:
        if not self._native_passthrough_enabled():
            return False
        return super().supports_native_request(protocol_name, request)

    def supports_native_streaming(self, protocol_name: str) -> bool:
        if not self._native_passthrough_enabled():
            return False
        return super().supports_native_streaming(protocol_name)

    def _parse_passthrough_usage(
        self, body: dict[str, Any], request: InternalRequest
    ) -> tuple[Usage | None, dict[str, Any]]:
        """Usage parsing per protocol: Anthropic cache folding, Responses, or chat."""
        if request.protocol_name == "anthropic":
            return parse_usage_and_provider_extras(body)
        if request.protocol_name == "openresponses":
            return parse_usage_from_response(body), {}
        # Chat Completions (response wire-reuse tier): defer to the chat
        # parser via the OpenAICompatibleBase override for billing parity
        # with the fully parsed path.
        return super()._parse_passthrough_usage(body, request)

    # ------------------------------------------------------------------
    # Endpoint routing
    # ------------------------------------------------------------------

    def _native_root_base_url(self) -> str:
        """Root that path-derived native endpoints hang off.

        Defaults to the configured Chat Completions base URL (xAI, MiniMax —
        their Responses endpoint shares the chat root). Adapters whose native
        endpoints live on a different root than the chat base URL override
        this (DeepSeek strips the ``/v1`` compatibility alias).
        """
        return self._base_url or self.DEFAULT_BASE_URL

    def _anthropic_messages_url(self, model: str | None = None) -> str:
        if "anthropic_messages" in self._endpoint_base_urls:
            return self._resolve_endpoint_url("anthropic_messages", "", model=model)
        if self.ANTHROPIC_MESSAGES_URL:
            return self.ANTHROPIC_MESSAGES_URL
        if self.ANTHROPIC_MESSAGES_PATH is not None:
            return f"{self._native_root_base_url()}{self.ANTHROPIC_MESSAGES_PATH}"
        raise NotImplementedError(
            f"{self.provider_name} has no native Anthropic Messages endpoint configured"
        )

    def _responses_url(self, model: str | None = None) -> str:
        if "responses" in self._endpoint_base_urls:
            return self._resolve_endpoint_url("responses", "", model=model)
        if self.RESPONSES_URL:
            return self.RESPONSES_URL
        return f"{self._native_root_base_url()}{self.RESPONSES_PATH}"

    # ------------------------------------------------------------------
    # Upstream headers
    # ------------------------------------------------------------------

    def _build_headers(
        self,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
    ) -> dict[str, str]:
        """Build upstream headers, merging captured client fingerprint headers.

        On the native Anthropic/Responses passthrough paths, client fingerprint
        headers captured by the protocol layers (Claude Code, Codex, ...) are
        forwarded verbatim without overriding provider/auth headers.
        ``anthropic-beta`` is rebuilt to guarantee it carries the
        ``claude-code-20250219`` marker so the upstream enables Claude Code
        features. Only the protocol layer of the current request has captured
        anything, so the other merge — and every merge on the Chat Completions
        translation path, which has no capture middleware — is a no-op.

        Adapters with their own ``_build_headers`` (e.g. KimiCodeAdapter,
        GLMBase's ``x-api-key``) call ``super()`` and inherit this merge.
        """
        headers = super()._build_headers(auth_header, auth_prefix)
        anthropic_headers = get_anthropic_client_headers()
        if anthropic_headers:
            existing = {k.lower() for k in headers}
            for key, value in anthropic_headers.items():
                if key.lower() not in existing:
                    headers[key] = value
            headers["anthropic-beta"] = ensure_claude_code_beta(headers.get("anthropic-beta"))
        openai_headers = get_openai_client_headers()
        if openai_headers:
            existing = {k.lower() for k in headers}
            for key, value in openai_headers.items():
                if key.lower() not in existing:
                    headers[key] = value
        return headers

    # ------------------------------------------------------------------
    # Native request bodies
    # ------------------------------------------------------------------

    def native_body_hook(self, body: dict[str, Any]) -> dict[str, Any]:
        """Anthropic-shaped native bodies get the structural message repairs.

        Responses-shaped bodies carry ``input`` items instead of ``messages``
        and pass through untouched. The repairs run on a deep copy — the
        messages list is still shared with the stashed raw protocol body,
        which must never be mutated in place.
        """
        messages = body.get("messages")
        if isinstance(messages, list):
            body["messages"] = _normalize_anthropic_messages(copy.deepcopy(messages))
        return body

    # ------------------------------------------------------------------
    # Native request dispatch
    # ------------------------------------------------------------------

    def _native_request_parts(
        self, request: InternalRequest, *, stream: bool
    ) -> tuple[str, dict[str, Any]]:
        """(url, body) for the native endpoint matching the client protocol.

        Single dispatch site for the protocol → (endpoint, body) pairing used
        by both the non-streaming and streaming passthrough paths. Body
        preparation (copy, None-strip, routed model, stream flag, message
        repairs) is owned by the passthrough seam.
        """
        body = prepare_native_body(self, request, stream=stream)
        protocol = request.protocol_name
        if protocol == "anthropic":
            return (
                self._anthropic_messages_url(model=request.model),
                body,
            )
        if protocol == "openresponses":
            return (
                self._responses_url(model=request.model),
                body,
            )
        raise NotImplementedError(
            f"{self.provider_name} does not support native requests for '{protocol}'"
        )

    # ------------------------------------------------------------------
    # Non-streaming chat completion
    # ------------------------------------------------------------------

    async def chat_completion(self, request: InternalRequest, **kwargs: Any) -> InternalResponse:
        # response_mode is NATIVE_PASSTHROUGH exactly when the request body
        # goes out verbatim (capability + veto + stash folded in by the
        # seam), making the completion native end-to-end.
        if plan_conversion(self, request).response_mode == ConversionTier.NATIVE_PASSTHROUGH:
            return await self._native_completion(request)
        return await super().chat_completion(request, **kwargs)

    async def _native_completion(self, request: InternalRequest) -> InternalResponse:
        """POST the raw body to the native endpoint and carry the response verbatim."""
        url, body = self._native_request_parts(request, stream=False)
        # Header customization (client fingerprint merge, Anthropic-style
        # ``x-api-key``) happens in ``_build_headers`` so the streaming path
        # (``_stream_raw_sse`` builds its own headers from the same method)
        # stays consistent.
        response = await self._post_json_response_with_retry(url, self._build_headers(), body)
        result = self._build_passthrough_response(response.json(), request)
        result.provider_info["_rate_limit_headers"] = _extract_rate_limit_headers(
            getattr(response, "headers", None)
        )
        return result

    # ------------------------------------------------------------------
    # Native streaming
    # ------------------------------------------------------------------

    async def stream_chat_completion_native(
        self,
        request: InternalRequest,
        cancel_token=None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream protocol-native SSE blocks verbatim from the native endpoint.

        Only called when ``plan_conversion`` (stream_mode) accepted the request.
        """
        url, body = self._native_request_parts(request, stream=True)
        return self._with_retry_generator(
            lambda: self._stream_raw_sse(url, body, cancel_token),
            cancel_token=cancel_token,
        )


__all__ = ["NativePassthroughChatBase"]
