"""Cohere provider adapter — the native v2 API.

Cohere serves one native surface for every capability, all from the API root
``https://api.cohere.com``:

- **Chat** ``POST /v2/chat`` — one chronological ``messages`` array, function
  tools in JSON Schema, ``thinking``/``documents``/``citation_options`` fields
  and a named-event SSE stream (docs.cohere.com/reference/chat). Not
  OpenAI-shaped, so the Cohere serializer translates both directions.
- **Embeddings** ``POST /v2/embed`` (docs.cohere.com/reference/embed).
- **Rerank** ``POST /v2/rerank`` (docs.cohere.com/reference/rerank).
- **Audio transcription** ``POST /v2/audio/transcriptions`` — multipart with
  ``model``, ``language`` and ``file`` required, answering ``{"text": ...}``
  (docs.cohere.com/reference/create-audio-transcription).

Cohere publishes no native Anthropic Messages or OpenAI Responses endpoint, so
no native passthrough protocols are declared: every turn is translated.

Cohere's *compatibility* surface (``https://api.cohere.ai/compatibility/v1``) is
deliberately not used. It is an OpenAI-shaped projection that drops the fields
this adapter exists to keep (``documents``, ``citation_options``, a
``thinking.token_budget``) and rejects OpenAI fields the proxy would otherwise
forward, so it would be a second, lossier wire on the same provider.
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import orjson
from orjson import JSONDecodeError

from llm_proxy.core.adapter import register_adapter
from llm_proxy.core.errors import get_error_type_for_status
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.request_type import RequestType
from llm_proxy.models import (
    InternalRequest,
    InternalRerankRequest,
    InternalRerankResponse,
    InternalResponse,
)
from llm_proxy.models.rerank import RerankDocument
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.base import BaseHttpProvider, extract_rate_limit_headers
from llm_proxy.providers.capabilities import (
    AudioCapabilityMixin,
    ChatCapabilityMixin,
    EmbeddingCapabilityMixin,
    RerankCapabilityMixin,
)
from llm_proxy.providers.cohere.serializer import COHERE_CHAT_EXTRA_KEYS
from llm_proxy.streaming.sse_parse import parse_sse_data_line

if TYPE_CHECKING:
    from llm_proxy.models.provider import ProviderModelInfo

logger = get_logger(__name__)

#: Cohere-only rerank fields a client may send through ``extra``; v2 documents
#: all three (docs.cohere.com/reference/rerank).
COHERE_RERANK_EXTRA_KEYS: frozenset[str] = frozenset(
    {"max_tokens_per_doc", "relevant_spans", "priority"}
)


@register_adapter("cohere")
class CohereAdapter(
    ChatCapabilityMixin,
    EmbeddingCapabilityMixin,
    RerankCapabilityMixin,
    AudioCapabilityMixin,
    BaseHttpProvider,
):
    """Provider adapter for Cohere's native v2 API."""

    _DEFAULT_PROVIDER_NAME = "cohere"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Cohere"
    DISPLAY_NAME_ZH = "Cohere"
    LOBE_ICON_ID = "cohere"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://api.cohere.com"

    CHAT_ENDPOINT = "/v2/chat"
    EMBEDDINGS_ENDPOINT = "/v2/embed"
    RERANK_ENDPOINT = "/v2/rerank"
    TRANSCRIPTION_ENDPOINT = "/v2/audio/transcriptions"
    #: Cohere's native API has no speech-synthesis and no audio-translation
    #: endpoint; leaving the constants empty keeps the capability mixin's
    #: ``NotImplementedError`` as the honest answer.
    SPEECH_ENDPOINT = ""
    TRANSLATION_ENDPOINT = ""

    #: Model listing stays on the v1 route even under the v2 API: Cohere
    #: documents ``GET /v1/models`` (docs.cohere.com/reference/list-models) and
    #: serves no ``/v2/models``. Its catalog paginates with ``page_size`` /
    #: ``page_token`` and answers ``models`` + ``next_page_token``.
    _models_endpoint = "/v1/models"
    _models_data_key = "models"
    _MODELS_PAGE_SIZE = 1000

    #: Extra keys the upstream documents as supported, per request type; exempt
    #: from the unknown-fields policy so they survive the merge onto the body.
    EXEMPT_EXTRA_KEYS = {
        RequestType.CHAT: COHERE_CHAT_EXTRA_KEYS,
        # ``input_type`` is required by Cohere and has no OpenAI embeddings
        # equivalent, so a client states it through ``extra``.
        RequestType.EMBEDDING: frozenset({"input_type"}),
        RequestType.RERANK: COHERE_RERANK_EXTRA_KEYS,
    }

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _build_request_body(self, request: InternalRequest) -> dict[str, Any]:
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("Expected JSON body for Cohere chat request")
        return outbound.json_body

    async def chat_completion(self, request: InternalRequest, **_kwargs: Any) -> InternalResponse:
        url = self._resolve_endpoint_url("chat_completion", self.CHAT_ENDPOINT, model=request.model)
        body = self._build_request_body(request)
        response = await self._post_json_response_with_retry(url, self._build_headers(), body)

        result = self._parse_response(
            self._get_serializer(),
            response.json(),
            model=request.model,
            request_id=request.request_id,
            request=request,
        )
        result.provider_info["_rate_limit_headers"] = extract_rate_limit_headers(
            getattr(response, "headers", None)
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
        """Stream a chat completion, converting Cohere's events to canonical chunks.

        Cohere streams named events rather than OpenAI chunks and never sends
        ``[DONE]`` (docs.cohere.com/docs/streaming): the stream ends after the
        single ``message-end`` event, so the terminal chunk is flushed and
        ``[DONE]`` appended here instead of by the shared template.
        """
        url = self._resolve_endpoint_url("chat_completion", self.CHAT_ENDPOINT, model=request.model)
        headers = self._build_headers()
        request.stream = True
        body = self._build_request_body(request)
        stream_timeout = self._get_stream_timeout()

        async def _stream_generator():
            # Fresh per attempt: the converter carries the turn's usage and
            # terminal state, which must not leak across a retry.
            converter = self._get_serializer().get_chunk_converter(
                model=request.model, request_id=request.request_id or ""
            )
            client = await self._get_client()
            try:
                async with self._streaming_post(
                    client,
                    url,
                    headers=headers,
                    json=body,
                    timeout=stream_timeout,
                ) as response:
                    await self._raise_for_stream_status(response)
                    self._stash_stream_response_headers(response)

                    async for line in self._iter_stream_lines(response, cancel_token):
                        payload = parse_sse_data_line(line)
                        if payload is None:
                            continue
                        try:
                            event = orjson.loads(payload)
                        except JSONDecodeError:
                            continue

                        chunk = converter.convert_chunk(event)
                        if chunk is not None:
                            yield chunk
                        if event.get("type") == "message-end":
                            break

                    for chunk in converter.finalize_chunks():
                        yield chunk
                    yield "[DONE]"

            except ProviderError:
                raise
            except Exception as e:
                error = await self._handle_http_error(e)
                raise error from e

        return self._with_retry_generator(_stream_generator, cancel_token=cancel_token)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def _parse_error_response(self, status_code: int, error_body: dict[str, Any]) -> ProviderError:
        """Parse Cohere's error body.

        Cohere answers ``{"message": ..., "id": ...}`` at the top level rather
        than OpenAI's ``{"error": {...}}`` envelope (docs.cohere.com/reference/
        chat, "Errors"), so the shared parser would hand the client a stringified
        dict as the message.
        """
        body = error_body.get("error")
        container = body if isinstance(body, dict) else error_body
        message = container.get("message") if isinstance(container, dict) else None
        return ProviderError(
            message=str(message) if message else str(error_body),
            error_type=get_error_type_for_status(status_code),
            status_code=status_code,
            provider_name=self.provider_name,
            original_error=error_body,
        )

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    async def rerank(
        self, request: InternalRerankRequest, **_kwargs: Any
    ) -> InternalRerankResponse:
        """Rerank, re-attaching the documents v2 does not echo.

        v2 answers ranked indexes and scores only, while the proxy's rerank
        protocol reports each ranked document back to the client (the v1 behaviour
        every other upstream in this family keeps). The request's own documents are
        the same texts the upstream scored, so they are projected back by index.
        """
        result = await super().rerank(request)
        if request.return_documents is False:
            # The client asked for ranked indexes and scores only; the mixin has
            # already dropped the documents, so nothing is projected back.
            return result
        for ranked in result.results:
            if ranked.document is not None:
                continue
            if not 0 <= ranked.index < len(request.documents):
                continue
            document = request.documents[ranked.index]
            ranked.document = (
                document if isinstance(document, RerankDocument) else RerankDocument(text=document)
            )
        return result

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _build_transcription_data(
        self, request: Any, stream: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build Cohere's transcription multipart form.

        Cohere requires ``model``, ``language`` (ISO-639-1) and ``file``, and
        documents no other OpenAI transcription field: ``prompt``,
        ``response_format``, ``timestamp_granularities`` and ``include`` are
        dropped, so the client's own response format is honoured by the protocol
        formatter instead of being promised upstream.
        """
        if not request.language:
            raise ProviderError(
                message=(
                    "Cohere transcription requires 'language' (ISO-639-1); "
                    "the endpoint has no auto-detect."
                ),
                error_type="invalid_request_error",
                provider_name=self.provider_name,
            )
        data: dict[str, Any] = {
            "model": request.model,
            "language": request.language,
        }
        if request.temperature:
            data["temperature"] = str(request.temperature)
        # ``file`` must be the last part of the multipart body (Cohere's
        # compatibility docs state the same for the OpenAI envelope); the
        # transport sends data fields before files, so this stays last.
        files = {"file": (request.filename, request.file)}
        return data, files

    async def stream_transcription(self, request: Any, **_kwargs: Any) -> AsyncIterator[str]:
        """Reject streaming transcription: Cohere has no streaming variant."""
        raise ProviderError(
            message="Cohere transcription does not support streaming.",
            error_type="invalid_request_error",
            provider_name=self.provider_name,
        )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    async def list_models(self, client: Any = None) -> list[ProviderModelInfo]:
        """List Cohere's catalog, following its own pagination vocabulary."""
        from llm_proxy.http.client import fetch_json

        http_client = client or await self._get_client()
        url = self._models_url()
        headers = self._build_headers()

        models: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": self._MODELS_PAGE_SIZE}
            if page_token:
                params["page_token"] = page_token
            data = await fetch_json(http_client, url, headers=headers, params=params)
            models.extend(data.get(self._models_data_key) or [])
            page_token = data.get("next_page_token")
            if not page_token:
                break

        return sorted([self._parse_model(raw) for raw in models], key=lambda info: info.id)

    def _parse_model(self, raw: dict[str, Any]) -> ProviderModelInfo:
        from llm_proxy.models.provider import ProviderModelInfo

        model_id = raw.get("name") or ""
        context_length = raw.get("context_length")
        return ProviderModelInfo(
            id=model_id,
            name=model_id,
            # Cohere's catalog describes its models by endpoint family and context
            # window, not by a display name.
            context_length=(
                context_length
                if isinstance(context_length, int) and not isinstance(context_length, bool)
                else None
            ),
        )


__all__ = ["COHERE_RERANK_EXTRA_KEYS", "CohereAdapter"]
