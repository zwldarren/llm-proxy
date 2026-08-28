"""Ollama provider implementation."""

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import orjson

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.http.client import AsyncSession
from llm_proxy.models import (
    InternalEmbeddingRequest,
    InternalRequest,
    InternalResponse,
)
from llm_proxy.models.content_blocks import ImageBlock
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.base import BaseHttpProvider, extract_rate_limit_headers
from llm_proxy.providers.capabilities import ChatCapabilityMixin, EmbeddingCapabilityMixin
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.ollama.request_builder import (
    OLLAMA_NATIVE_OPTIONS,
    OLLAMA_NATIVE_TOP_LEVEL_KEYS,
    OLLAMA_RESPONSES_ONLY_KEYS,
)
from llm_proxy.serialization.providers import get_provider_serializer

if TYPE_CHECKING:
    from llm_proxy.models.provider import ProviderModelInfo

logger = get_logger(__name__)

_serializer = get_provider_serializer("ollama")


@register_adapter("ollama")
class OllamaAdapter(ChatCapabilityMixin, EmbeddingCapabilityMixin, BaseHttpProvider):
    """Provider adapter for Ollama native API."""

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Ollama"
    DISPLAY_NAME_ZH = "Ollama"
    LOBE_ICON_ID = "ollama"
    LOBE_ICON_VARIANT = "mono"

    DEFAULT_BASE_URL = "http://localhost:11434"
    CHAT_ENDPOINT = "/api/chat"
    EMBEDDINGS_ENDPOINT = "/api/embed"

    #: Extra keys that are native /api/embed parameters; exempt from the
    #: unknown-fields policy so they survive the merge into the body.
    _EMBEDDING_EXEMPT_EXTRA_KEYS: frozenset[str] = frozenset({"keep_alive", "truncate", "options"})

    def __init__(
        self,
        *,
        config: AdapterConfig | None = None,
        **kwargs: Any,
    ):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "ollama")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)

    async def _download_images_in_conversation(
        self,
        request: InternalRequest,
        client: AsyncSession,
    ) -> None:
        """Download HTTP(S) image URLs in ConversationContext, converting to base64."""
        import dataclasses

        from llm_proxy.http.client import download_image_as_base64
        from llm_proxy.models.types import ImageSource

        download_tasks: list[tuple[int, int, str]] = []

        for msg_idx, msg in enumerate(request.conversation.messages):
            for block_idx, block in enumerate(msg.content):
                if isinstance(block, ImageBlock) and block.source.type == "url":
                    url = block.source.data
                    if url and (url.startswith("http://") or url.startswith("https://")):
                        download_tasks.append((msg_idx, block_idx, url))

        if not download_tasks:
            return

        async def _download_one(url: str) -> tuple[str | None, str | None]:
            try:
                result = await download_image_as_base64(client, url)
                if result:
                    data_url, media_type = result
                    # Extract base64 data from data URL (data:media_type;base64,DATA)
                    base64_data = data_url.split(",", 1)[1] if "," in data_url else data_url
                    return base64_data, media_type or "image/png"
            except Exception:
                logger.debug("Failed to download image URL for Ollama", exc_info=True)
            return None, None

        results = await asyncio.gather(*[_download_one(url) for _, _, url in download_tasks])
        url_to_result: dict[str, tuple[str, str]] = {}
        for (_, _, url), result in zip(download_tasks, results, strict=True):
            base64_data, media_type = result
            if base64_data is not None:
                url_to_result[url] = (base64_data, media_type or "image/png")

        for msg_idx, block_idx, url in download_tasks:
            if url in url_to_result:
                base64_data, media_type = url_to_result[url]
                old_block = request.conversation.messages[msg_idx].content[block_idx]
                new_source = ImageSource(
                    type="base64",
                    data=base64_data,
                    media_type=media_type,
                )
                request.conversation.messages[msg_idx].content[block_idx] = dataclasses.replace(
                    old_block, source=new_source
                )

    #: Extra keys the request builder explicitly handles (native options,
    #: native top-level params, and responses-only keys it deliberately
    #: drops). All of them are exempt from the unknown-fields policy so a
    #: strict ``unknown_fields_policy: error`` config does not reject keys
    #: the builder already decided the fate of.
    _OLLAMA_HANDLED_EXTRA_KEYS: set[str] = (
        set(OLLAMA_NATIVE_OPTIONS)
        | set(OLLAMA_NATIVE_TOP_LEVEL_KEYS)
        | set(OLLAMA_RESPONSES_ONLY_KEYS)
    )

    def _build_chat_raw(self, request: InternalRequest, context: BuildContext) -> dict[str, Any]:
        return _serializer.build_provider_request(request, context)

    def _build_request_body(self, request: InternalRequest) -> dict[str, Any]:
        outbound = self._build_outbound_body(
            request, request_type="chat", exempt_keys=self._OLLAMA_HANDLED_EXTRA_KEYS
        )
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for Ollama request")
        return outbound.json_body

    def _extract_error_from_body(self, error_body: dict, fallback: str) -> tuple[str, str]:
        error_value = error_body.get("error", "")
        if isinstance(error_value, dict):
            msg = error_value.get("message", str(error_value))
            return msg, error_value.get("type", "api_error")
        return str(error_value) if error_value else fallback, "api_error"

    async def chat_completion(self, request: InternalRequest, **_kwargs: Any) -> InternalResponse:
        client = await self._get_client()
        url = self._resolve_endpoint_url("chat_completion", self.CHAT_ENDPOINT)
        headers = self._build_headers()

        await self._download_images_in_conversation(request, client)
        body = self._build_request_body(request)

        async def _make_request():
            new_client = await self._get_client()
            response = await new_client.post(url, headers=headers, json=body)
            await self._check_response_status(response)
            return response.json()

        response_data = await self._with_retry(_make_request)
        result = self._parse_response(
            _serializer,
            response_data,
            model=request.model,
            request_id=request.request_id,
            request=request,
            logprobs=bool(request.params.openai and request.params.openai.logprobs),
        )
        from llm_proxy.core.reasoning_cache import try_cache_reasoning_from_response

        try_cache_reasoning_from_response(result)
        return result

    async def stream_chat_completion(
        self,
        request: InternalRequest,
        cancel_token=None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Process a streaming chat completion request.

        Uses ``OllamaChunkConverter`` to convert Ollama native JSON-line chunks
        into canonical OpenAI ``chat.completion.chunk`` dicts.
        """
        client = await self._get_client()
        url = self._resolve_endpoint_url("chat_completion", self.CHAT_ENDPOINT)
        headers = self._build_headers()

        await self._download_images_in_conversation(request, client)
        request.stream = True
        body = self._build_request_body(request)

        stream_timeout = self._get_stream_timeout()

        async def _stream_generator():
            # Created per attempt so a retry starts from a fresh converter
            # (OllamaChunkConverter tracks tool-call indices per stream).
            converter = _serializer.get_chunk_converter(
                model=request.model, request_id=request.request_id or ""
            )
            new_client = await self._get_client()
            done_sent = False

            try:
                async with self._streaming_post(
                    new_client,
                    url,
                    headers=headers,
                    json=body,
                    timeout=stream_timeout,
                ) as response:
                    assert response.status_code is not None
                    # Stash upstream response headers so the API layer can
                    # forward them (request-id, ratelimit-*, ...) once the
                    # client StreamingResponse is created — same as the shared
                    # _stream_raw_sse path.
                    self._last_stream_response_headers = extract_rate_limit_headers(
                        getattr(response, "headers", None)
                    )
                    if response.status_code >= 400:
                        try:
                            await response.aread()
                            error_body = response.json()
                            error_message, error_type = self._extract_error_from_body(
                                error_body, f"HTTP {response.status_code}"
                            )
                        except Exception:
                            logger.debug(
                                "Failed to parse error response body as JSON", exc_info=True
                            )
                            error_text = response.text
                            error_message = error_text or f"HTTP {response.status_code}"
                            error_type = "api_error"
                        raise ProviderError(
                            message=error_message,
                            error_type=error_type,
                            status_code=response.status_code,
                            provider_name=self._provider_name,
                        )

                    async for line in cast(
                        AsyncIterator[bytes],
                        response.iter_lines(),
                    ):
                        if cancel_token and cancel_token.is_set():
                            break

                        if not line:
                            continue

                        line_str = line.decode("utf-8") if isinstance(line, bytes) else line

                        try:
                            chunk_data = orjson.loads(line_str)

                            # Ollama reports mid-stream failures as an
                            # {"error": ...} JSON line inside an HTTP 200
                            # stream: its server can only set the HTTP status
                            # for errors that precede any streamed content.
                            # Without this check the error is silently
                            # swallowed and the client sees an empty or
                            # truncated response ending in a normal [DONE].
                            if "error" in chunk_data:
                                error_message, error_type = self._extract_error_from_body(
                                    chunk_data, "Ollama stream error"
                                )
                                # Ollama's stream error lines may carry the
                                # original HTTP status (server/routes.go sends
                                # {"error": ..., "status": ...} for
                                # api.StatusError); fall back to 500 when absent.
                                status = chunk_data.get("status")
                                raise ProviderError(
                                    message=error_message,
                                    error_type=error_type,
                                    status_code=status if isinstance(status, int) else 500,
                                    provider_name=self._provider_name,
                                )

                            openai_chunk = converter.convert_chunk(chunk_data)

                            if chunk_data.get("done"):
                                yield openai_chunk
                                yield "[DONE]"
                                done_sent = True
                                break

                            yield openai_chunk

                        except orjson.JSONDecodeError:
                            continue

                if not done_sent:
                    yield "[DONE]"

            except ProviderError:
                raise
            except Exception as e:
                error = await self._handle_http_error(e)
                raise error from e

        return self._with_retry_generator(_stream_generator, cancel_token=cancel_token)

    def _embeddings_url(self, request: InternalEmbeddingRequest) -> str:
        return self._resolve_endpoint_url("embeddings", self.EMBEDDINGS_ENDPOINT)

    _models_endpoint = "/api/tags"
    _models_data_key = "models"

    def _models_url(self) -> str:
        return f"{self._base_url}{self._models_endpoint}"

    def _parse_model(self, raw: dict[str, Any]) -> ProviderModelInfo:
        from llm_proxy.models.provider import ProviderModelInfo

        size = raw.get("size") or 0
        return ProviderModelInfo(
            id=raw.get("name", ""),
            name=raw.get("name", ""),
            description=f"Size: {size:,} bytes",
            owned_by=None,
        )


__all__ = ["OllamaAdapter"]
