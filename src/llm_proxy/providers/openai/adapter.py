"""OpenAI provider implementation using native HTTP calls (Responses API)."""

from collections.abc import AsyncIterator
from typing import Any

import orjson

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.core.conversion import plan_conversion
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.reasoning_cache import try_cache_reasoning_from_responses_output
from llm_proxy.models import (
    ConversionTier,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalImageResponse,
    InternalRequest,
    InternalResponse,
)
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.base import (
    STREAM_READ_TIMEOUT,
    BaseHttpProvider,
    extract_rate_limit_headers,
)
from llm_proxy.providers.capabilities import (
    AudioCapabilityMixin,
    ChatCapabilityMixin,
    EmbeddingCapabilityMixin,
    ImageCapabilityMixin,
)
from llm_proxy.providers.openai.client_headers import get_client_headers
from llm_proxy.providers.openai.sse_fallback import parse_json_or_sse
from llm_proxy.serialization.providers import get_provider_serializer

logger = get_logger(__name__)
_serializer = get_provider_serializer("openai")

# Input item types whose ``id`` is a lookup key rather than a client-supplied
# label. ``item_reference.id`` points at an output item of a previous response
# and must be forwarded verbatim; every other input item's id is dropped (see
# ``_strip_input_item_id``).
_REFERENCE_TYPES: frozenset[str] = frozenset({"item_reference"})


def _strip_input_item_id(item: Any) -> Any:
    """Drop a client-supplied input item id.

    The Responses API validates per-type id prefixes (``msg_``, ``fc_``,
    ``ctc_``, ...) and rejects arbitrary ids — e.g. Codex's ``item_...``
    history ids — with 400 "Expected an ID that begins with X". No request
    field references these ids: ``call_id`` pairs function calls with their
    outputs, and ``item_reference`` points at output ids of a previous
    response. So instead of maintaining a per-type prefix table (which goes
    stale as OpenAI adds item types), the id is simply omitted and the API
    generates one itself.
    """
    if not isinstance(item, dict) or "id" not in item:
        return item
    item_type = item.get("type")
    if not isinstance(item_type, str) or item_type in _REFERENCE_TYPES:
        # item_reference.id is a lookup key into a previous response's
        # outputs (stripping it would break the reference); items without a
        # type are left untouched as unknown structures.
        return item
    normalized = dict(item)
    del normalized["id"]
    return normalized


@register_adapter("openai")
class OpenAIAdapter(
    ChatCapabilityMixin,
    EmbeddingCapabilityMixin,
    ImageCapabilityMixin,
    AudioCapabilityMixin,
    BaseHttpProvider,
):
    _DEFAULT_PROVIDER_NAME = "openai"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "OpenAI"
    DISPLAY_NAME_ZH = "OpenAI"
    LOBE_ICON_ID = "openai"
    LOBE_ICON_VARIANT = "mono"

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    # The upstream speaks the Responses API natively; the raw SSE stream is
    # forwarded to the client instead of round-tripping through the canonical
    # chat-chunk intermediate format (which cannot represent Codex item types
    # like ``custom_tool_call`` / ``local_shell_call`` / ``agent_message`` and
    # would silently drop them).
    native_protocols = frozenset({"openresponses"})
    RESPONSES_ENDPOINT = "/responses"
    COMPACT_ENDPOINT = "/responses/compact"
    EMBEDDINGS_ENDPOINT = "/embeddings"
    IMAGES_ENDPOINT = "/images/generations"
    IMAGES_EDITS_ENDPOINT = "/images/edits"
    SPEECH_ENDPOINT = "/audio/speech"
    TRANSCRIPTION_ENDPOINT = "/audio/transcriptions"
    TRANSLATION_ENDPOINT = "/audio/translations"

    def __init__(
        self,
        *,
        config: AdapterConfig | None = None,
        **kwargs: Any,
    ):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "openai")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)

    def _target_endpoint(self) -> str:
        """OpenAI adapter targets the Responses API endpoint."""
        return "responses"

    def _build_headers(
        self,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
    ) -> dict[str, str]:
        """Build upstream headers, merging captured Codex client headers.

        On the native Responses passthrough path, client fingerprint headers
        captured by the openresponses protocol layer are forwarded verbatim
        without overriding provider/auth headers.
        """
        headers = super()._build_headers(auth_header, auth_prefix)
        client_headers = get_client_headers()
        if client_headers:
            existing = {k.lower() for k in headers}
            for key, value in client_headers.items():
                if key.lower() not in existing:
                    headers[key] = value
        return headers

    def native_body_hook(self, body: dict[str, Any]) -> dict[str, Any]:
        """Responses-API repairs for native passthrough bodies.

        * legacy flat ``text`` shape ``{"type": ...}`` (old Codex) is wrapped
          under ``format`` — the API rejects ``text.type`` with "Unknown
          parameter" (a ``text`` dict without ``format`` but also without
          ``type``, e.g. ``{"verbosity": "low"}``, is already a valid
          TextConfig and passes through);
        * client-supplied input item ids are stripped (see
          ``_strip_input_item_id``): the API validates per-type id prefixes
          that Codex's ``item_...`` ids do not satisfy.

        Idempotent by design: chat paths also run the hook via
        ``_build_responses_passthrough_body`` after the passthrough seam has
        already applied it.
        """
        text = body.get("text")
        # Only the legacy flat shape {"type": ...} needs wrapping.
        if isinstance(text, dict) and "format" not in text and "type" in text:
            body["text"] = {"format": text}
        input_items = body.get("input")
        if isinstance(input_items, list):
            body["input"] = [_strip_input_item_id(item) for item in input_items]
        return body

    def _build_responses_passthrough_body(
        self, request: dict[str, Any], stream: bool = False, model: str | None = None
    ) -> dict[str, Any]:
        """Copy the request dict, add stream flag, and substitute the routed model.

        Used by create_response when invoked with a raw dict
        from non-InternalRequest paths.  Chat paths go through
        _build_outbound_body first — whose passthrough branch already applies
        the seam's shared preparation (copy, None-strip, routed model,
        ``native_body_hook``); the hook is idempotent, so re-applying it here
        is safe and keeps this function correct for raw-dict callers.

        ``model`` is the routed upstream model id (``InternalRequest.model``
        after ProviderSelectionStage); the raw body still carries the client's
        original alias, so without substitution model routing would silently
        be ignored on the passthrough path.

        Top-level ``None`` values are stripped: a parameter override set to
        None means "delete the field", and the API rejects explicit nulls
        for several fields.
        """
        body = {k: v for k, v in request.items() if v is not None}
        body["stream"] = stream
        if model:
            body["model"] = model
        return self.native_body_hook(body)

    def _build_chat_raw(self, request, context):
        # Fallback path: used when native request passthrough is not applicable
        # (conversation materialized from the proxy's response store, proxy-side
        # web search interception, or a non-OpenResponses protocol).
        return _serializer.build_provider_request(request, context)

    def _responses_url(self, model: str | None = None) -> str:
        return self._resolve_endpoint_url("chat_completion", self.RESPONSES_ENDPOINT, model=model)

    async def compact_response(self, request: dict[str, Any], **_kwargs: Any) -> tuple[int, Any]:
        """Forward a /v1/responses/compact body verbatim to the native upstream.

        Returns ``(status_code, parsed_body)`` without raising on upstream
        error statuses, so the router can pass the upstream response through
        as-is: the upstream performs real model-driven compaction instead of
        the proxy's local lossless packing.
        """
        url = self._resolve_endpoint_url(
            "chat_completion", self.COMPACT_ENDPOINT, model=request.get("model")
        )
        headers = self._build_headers()
        client = await self._get_client()
        response = await client.post(url, headers=headers, json=request)
        try:
            return response.status_code, response.json()
        except Exception:
            return response.status_code, {
                "error": {
                    "message": response.text[:500] or "Upstream returned a non-JSON body",
                    "type": "server_error",
                    "code": "upstream_invalid_response",
                }
            }

    async def chat_completion(self, request: InternalRequest, **_kwargs: Any) -> InternalResponse:
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for request_type='chat'")

        response = await self._post_json_response_with_retry(
            self._responses_url(model=request.model),
            self._build_headers(),
            self._build_responses_passthrough_body(
                outbound.json_body, stream=False, model=request.model
            ),
        )
        body = parse_json_or_sse(response)
        if plan_conversion(self, request).response_mode == ConversionTier.NATIVE_PASSTHROUGH:
            # Verbatim passthrough: return the upstream body untouched (the
            # protocol formatter emits it as-is). Only usage is parsed, for
            # billing; output items are never re-parsed into blocks, so
            # Codex-specific item types cannot be dropped or mangled.
            result = self._build_passthrough_response(body, request)
            # Verbatim bodies never become parsed blocks, so the reasoning
            # cache (next-turn reasoning restoration) is written straight
            # from the raw output items. Never fatal.
            if isinstance(body, dict):
                try_cache_reasoning_from_responses_output(
                    body.get("output"),
                    body.get("id", "") or "",
                    logger_prefix="OpenAINative",
                )
        else:
            result = self._parse_response(
                _serializer,
                body,
                model=request.model,
                request_id=request.request_id,
                request=request,
            )
            from llm_proxy.core.reasoning_cache import try_cache_reasoning_from_response

            try_cache_reasoning_from_response(result)
        result.provider_info["_rate_limit_headers"] = extract_rate_limit_headers(
            getattr(response, "headers", None)
        )
        return result

    async def stream_chat_completion(
        self,
        request: InternalRequest,
        cancel_token=None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Stream a chat completion, converting Responses API events to OpenAI chunks.

        Uses ``OpenAIResponsesChunkConverter`` to convert Responses API SSE events
        into canonical OpenAI ``chat.completion.chunk`` dicts.
        """
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for request_type='chat'")
        json_body: dict[str, Any] = self._build_responses_passthrough_body(
            outbound.json_body, stream=True, model=request.model
        )

        url = self._responses_url(model=json_body.get("model"))
        headers = self._build_headers()
        stream_timeout = (self._connect_timeout, STREAM_READ_TIMEOUT)

        async def _stream_generator():
            # Created per attempt so a retry starts from a fresh converter
            # (stateful converters must not leak accumulated state across retries).
            converter = _serializer.get_chunk_converter(
                model=request.model, request_id=request.request_id or ""
            )
            client = await self._get_client()
            try:
                async with self._streaming_post(
                    client,
                    url,
                    headers=headers,
                    json=json_body,
                    timeout=stream_timeout,
                ) as response:
                    await self._raise_for_stream_status(response)
                    self._stash_stream_response_headers(response)

                    current_event_type = None

                    async for line_str in self._iter_stream_lines(response, cancel_token):
                        if line_str.startswith("event:"):
                            current_event_type = line_str[6:].strip()
                            continue
                        if line_str.startswith("data:"):
                            data_str = line_str[5:].strip()
                            if data_str == "[DONE]":
                                # Flush pending final chunks before [DONE].
                                for final_chunk in converter.finalize_chunks():
                                    yield final_chunk
                                yield "[DONE]"
                                return
                            try:
                                data = orjson.loads(data_str)
                                event_type = current_event_type or data.get("type")
                                current_event_type = None
                                data["event_type"] = event_type
                                chunk = converter.convert_chunk(data)
                                if chunk is not None:
                                    yield chunk
                            except orjson.JSONDecodeError:
                                logger.warning(
                                    f"Failed to decode SSE data payload: {data_str[:200]}"
                                )
                                continue
                        # Non-data, non-event lines are skipped.

                    # Stream ended without [DONE] — flush any pending final chunks.
                    for final_chunk in converter.finalize_chunks():
                        yield final_chunk
                    yield "[DONE]"

            except ProviderError:
                raise
            except Exception as e:
                error = await self._handle_http_error(e)
                raise error from e

        return self._with_retry_generator(_stream_generator, cancel_token=cancel_token)

    async def stream_chat_completion_native(
        self,
        request: InternalRequest,
        cancel_token=None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding native Responses API SSE blocks.

        Verbatim passthrough for the OpenResponses protocol: the upstream's
        own events (``response.output_item.added``, ``response.output_text.delta``,
        ``response.function_call_arguments.delta``, ``response.completed``, ...)
        are forwarded unchanged so Codex-specific item types that the chat-chunk
        intermediate format cannot represent are never dropped. Each yielded
        chunk is one complete SSE block (``event: X\ndata: {...}\n\n``).
        """
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for request_type='chat'")
        json_body: dict[str, Any] = self._build_responses_passthrough_body(
            outbound.json_body, stream=True, model=request.model
        )

        url = self._responses_url(model=json_body.get("model"))
        headers = self._build_headers()
        stream_timeout = (self._connect_timeout, STREAM_READ_TIMEOUT)

        async def _stream_generator():
            client = await self._get_client()
            try:
                async with self._streaming_post(
                    client,
                    url,
                    headers=headers,
                    json=json_body,
                    timeout=stream_timeout,
                ) as response:
                    await self._raise_for_stream_status(response)
                    self._stash_stream_response_headers(response)

                    buf: list[str] = []
                    # Use response.iter_lines() directly to preserve the empty
                    # line that terminates each SSE block (mirrors the
                    # Anthropic adapter's native streaming).
                    async for raw_line in response.iter_lines():
                        if cancel_token and cancel_token.is_set():
                            logger.debug(f"{self.provider_name} stream cancelled")
                            break
                        line_str = (
                            raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                        )
                        for part in line_str.split("\n"):
                            if part == "":
                                if buf:
                                    yield "\n".join(buf) + "\n\n"
                                    buf = []
                            else:
                                buf.append(part)
                    if buf and not (cancel_token and cancel_token.is_set()):
                        yield "\n".join(buf) + "\n\n"

            except ProviderError:
                raise
            except Exception as e:
                error = await self._handle_http_error(e)
                raise error from e

        return self._with_retry_generator(_stream_generator, cancel_token=cancel_token)

    async def image_generation(
        self, request: InternalImageRequest, **kwargs: Any
    ) -> InternalImageResponse:
        url = self._image_generation_url(model=request.model)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="image_generation")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for image_generation")
        response = await self._post_json_with_retry(url, headers, outbound.json_body)
        return self.from_image_provider_format(response)

    async def image_edit(
        self, request: InternalImageEditRequest, **kwargs: Any
    ) -> InternalImageResponse:
        url = self._image_edit_url(model=request.model)
        headers = self._build_headers()
        outbound = self._build_outbound_body(request, request_type="image_edit")
        if outbound.form_data is not None:
            # Uploaded image/mask files are sent as multipart/form-data; the
            # httpx client sets the boundary itself, so drop the JSON
            # Content-Type header (mirrors OpenAICompatibleBase.image_edit).
            headers.pop("Content-Type", None)

            async def _make_form_request():
                new_client = await self._get_client()
                response = await new_client.post(
                    url,
                    headers=headers,
                    data=outbound.form_data,
                    files=outbound.files,
                )
                await self._check_response_status(response)
                return response.json()

            response = await self._with_retry(_make_form_request)
        else:
            if outbound.json_body is None:
                raise ValueError("Expected JSON or multipart body for image edit request")
            response = await self._post_json_with_retry(url, headers, outbound.json_body)
        return self.from_image_edit_provider_format(response)


__all__ = ["OpenAIAdapter"]
