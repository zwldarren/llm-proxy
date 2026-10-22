"""ProviderSerializer base class.

Converts between InternalRequest/InternalResponse and provider API format.
Provider adapters use this to build provider-specific request bodies and parse
provider responses.
"""

from abc import ABC, abstractmethod
from typing import Any

from llm_proxy.models import ContentBlock, ConversionTier, InternalRequest, InternalResponse
from llm_proxy.models.decisions import (
    InternalDecisionRequest,
    InternalDecisionResponse,
)
from llm_proxy.models.embedding import (
    EmbeddingData,
    InternalEmbeddingRequest,
    InternalEmbeddingResponse,
)
from llm_proxy.models.rerank import (
    InternalRerankRequest,
    InternalRerankResponse,
    RerankDocument,
    RerankResult,
)
from llm_proxy.models.systemone import (
    InternalSystemOneRequest,
    InternalSystemOneResponse,
)
from llm_proxy.models.types import Usage
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.components.request_builder import OpenAIRequestBuilder
from llm_proxy.serialization.providers.field_utils import (
    extract_unknown_response_fields as _extract_unknown_fields,
)
from llm_proxy.serialization.providers.field_utils import is_int, reported_cost
from llm_proxy.serialization.usage import parse_usage as parse_wire_usage
from llm_proxy.streaming.transformer import StreamingTransformer


class IdentityChunkConverter(StreamingTransformer):
    """A no-op chunk converter for providers that already emit OpenAI-format chunks.

    When a provider's streaming response is already in canonical OpenAI
    ``chat.completion.chunk`` dict format (e.g., DeepSeek, OpenRouter, NanoGPT,
    Chutes), no conversion is needed.  This converter implements the
    ``convert_chunk()`` interface as an identity function.

    It extends ``StreamingTransformer`` purely so it satisfies the
    ``ProviderSerializer.get_chunk_converter()`` return type; it has no
    accumulation, no usage tracking, and no SSE formatting.  Adapters that use
    identity converters handle ``[DONE]`` and other stream-level concerns
    themselves, so the protocol-side ``transform()``/``finalize()`` methods are
    unused no-ops.

    Despite the old name, this is NOT the native passthrough seam
    (``llm_proxy.core.conversion``): chunks still round-trip through dict
    form and the protocol-side transformer; nothing is forwarded verbatim.
    """

    def convert_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Return the chunk unchanged (identity).

        Args:
            chunk: A provider-native chunk that is already in canonical
                   OpenAI ``chat.completion.chunk`` format.

        Returns:
            The same dict, or None if the chunk is empty/null.
        """
        if not chunk:
            return None
        return chunk

    def finalize_chunks(self) -> list[dict[str, Any]]:
        """Return any pending chunks after the stream ends.

        Identity converters have no accumulated state, so this
        always returns an empty list.
        """
        return []

    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Protocol-side transform — unused for provider-side identity."""
        return None

    def finalize(self) -> str:
        """Stream end marker — unused; adapters emit ``[DONE]`` directly."""
        return "[DONE]"


__all__ = ["ProviderSerializer", "IdentityChunkConverter"]


class ProviderSerializer(ABC):
    """Convert between Internal* models and a specific provider's API format.

    Each provider (OpenAI, Anthropic, Gemini, Ollama, etc.) has its own
    API format. ProviderSerializer handles the conversion in both directions.

    Subclasses can list compatible wire-format protocols via ``compatible_protocols``.
    """

    _DEFAULT_PROVIDER_NAME: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._registered_provider_name: str = ""
        self._request_builder: OpenAIRequestBuilder | None = None

    @property
    def provider_name(self) -> str:
        """Provider name as registered, falling back to the class default."""
        return self._registered_provider_name or self._DEFAULT_PROVIDER_NAME

    @property
    def request_builder(self) -> OpenAIRequestBuilder:
        """The OpenAI request builder this serializer composes.

        Adapters that speak the Chat Completions request shape read it from
        here rather than reaching for a private attribute: the builder is the
        one instance that carries the per-model reasoning-field preference
        cache (ADR-0013), so a caller holding its own copy would never learn
        the upstream's convention. Serializers that do not build OpenAI
        request bodies never touch it.
        """
        if self._request_builder is None:
            self._request_builder = OpenAIRequestBuilder()
        return self._request_builder

    # ------------------------------------------------------------------
    # Content block support — subclasses override with the set of block
    # types they can natively serialize.  Used by ``should_degrade_block``
    # for capability-based degradation.
    # ------------------------------------------------------------------

    supported_content_blocks: frozenset[type[ContentBlock]] = frozenset()

    @property
    def compatible_protocols(self) -> frozenset[str]:
        """Protocol names whose wire format this provider can reuse directly.

        Pure capability data read by the conversion seam
        (``llm_proxy.core.conversion.plan_conversion``) via
        ``BuildContext.compatible_protocols``: when the provider natively
        speaks the client protocol's wire format (e.g. OpenAI protocol →
        OpenAI-compatible provider), the seam may prepare the outbound body
        from the stashed raw body (WIRE_REUSE) instead of calling
        ``build_provider_request``.
        """
        return frozenset()

    def filter_extra_for_body(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Return the subset of ``request.extra`` this provider accepts on the body.

        The default accepts every key (the legacy open merge). Dialects that
        reject unknown top-level fields override this with their whitelist so
        the adapter's generic body-merge path (“_merge_extra”) applies the
        same policy the request builder does.
        """
        return extra

    def build_provider_request(
        self,
        request: InternalRequest,
        context: BuildContext | None = None,
    ) -> dict[str, Any]:
        """Build a provider-specific request body from InternalRequest.

        Always a full conversion. The tier decision and all raw-reuse body
        preparation live in the conversion seam
        (``llm_proxy.core.conversion.plan_conversion`` /
        ``prepare_wire_reuse_body``), which runs before the serializer is
        called; this method only rebuilds from the parsed request.
        """
        if context is None:
            context = BuildContext.from_request(
                request,
                provider_name=self.provider_name,
                provider_type=self.provider_name,
                supported_content_blocks=self.supported_content_blocks,
            )

        body = self._build_provider_request(request, context)
        request.conversion_tier = ConversionTier.FULL_CONVERSION
        return body

    @abstractmethod
    def _build_provider_request(
        self,
        request: InternalRequest,
        context: BuildContext,
    ) -> dict[str, Any]:
        """Build provider-specific request body (subclass implementation)."""
        ...

    @abstractmethod
    def parse_provider_response(
        self, response: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> InternalResponse:
        """Parse a provider response into InternalResponse.

        Args:
            response: Raw response dict from the provider
            model: Optional model name for the response
            **kwargs: Provider-specific keyword arguments (e.g., request_id, logprobs)

        Returns:
            InternalResponse instance
        """
        ...

    @abstractmethod
    def get_chunk_converter(self, model: str = "", request_id: str = "") -> StreamingTransformer:
        """Get the provider-side chunk converter for streaming responses.

        The converter transforms provider-native streaming chunks into the
        canonical OpenAI ``chat.completion.chunk`` dict format used by the
        protocol-side streaming transformer.

        This is the streaming equivalent of ``parse_provider_response()``:
        just as non-streaming responses go through the serializer to become
        ``InternalResponse``, streaming chunks go through the converter to
        become canonical OpenAI chunk dicts.

        Args:
            model: Model name for this response
            request_id: Request ID for correlation

        Returns:
            StreamingTransformer instance whose ``convert_chunk()`` method
            accepts provider-native chunk dicts and returns canonical OpenAI
            chunk dicts.
        """
        ...

    @staticmethod
    def _known_response_fields() -> set[str]:
        """Return the set of response fields that this serializer handles explicitly.

        Fields not in this set are candidates for provider_info passthrough.
        Override in subclasses to declare explicitly handled response fields.
        """
        return set()

    def extract_unknown_response_fields(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract fields not in _known_response_fields() for storage in provider_info.

        Override in subclasses to capture provider-specific metadata
        (e.g., OpenRouter's id_provider, cost fields).

        Args:
            response: Raw provider response dict

        Returns:
            Dict of unknown fields (empty by default)
        """
        return _extract_unknown_fields(response, self._known_response_fields())

    def build_provider_embedding_request(self, request: InternalEmbeddingRequest) -> dict[str, Any]:
        """Build a provider-specific embedding request body.

        Default implementation produces OpenAI-compatible format.
        Override for providers with different embedding request formats.

        Args:
            request: The unified embedding request

        Returns:
            Provider-specific request body dict
        """
        body: dict[str, Any] = {
            "model": request.model,
            "input": request.input,
        }
        if request.dimensions is not None:
            body["dimensions"] = request.dimensions
        if request.encoding_format is not None:
            body["encoding_format"] = request.encoding_format
        return body

    def parse_provider_embedding_response(
        self, response: dict[str, Any], model: str = ""
    ) -> InternalEmbeddingResponse:
        """Parse a provider embedding response into InternalEmbeddingResponse.

        Default implementation handles OpenAI-compatible format.
        Override for providers with different embedding response formats.

        Args:
            response: Raw response dict from the provider
            model: Model name for the response

        Returns:
            InternalEmbeddingResponse instance
        """
        data_list: list[EmbeddingData] = []
        for item in response.get("data", []):
            if isinstance(item, dict):
                data_list.append(
                    EmbeddingData(
                        embedding=item.get("embedding", []),
                        index=item.get("index", 0),
                    )
                )

        usage = None
        if "usage" in response:
            usage_data = response["usage"]
            usage = Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return InternalEmbeddingResponse(
            model=response.get("model") or model,
            data=data_list,
            usage=usage,
        )

    def build_provider_rerank_request(self, request: InternalRerankRequest) -> dict[str, Any]:
        """Build a provider-specific rerank request body.

        Default implementation produces the shared ``/v1/rerank`` format
        (Cohere's, also served by OpenRouter, Jina and the vLLM/SGLang
        ``/rerank`` route): ``{model, query, documents[, top_n]}``.

        ``return_documents`` is a response projection the proxy applies itself and
        is never forwarded. ``user`` and OpenRouter's ``session_id``/``trace``
        ride ``request.extra`` (not this body) and are merged in by the outbound
        chokepoint, so upstreams that document them still receive them.
        """
        documents: list[str | dict[str, Any]] = [
            document if isinstance(document, str) else document.to_wire()
            for document in request.documents
        ]
        body: dict[str, Any] = {
            "model": request.model,
            "query": request.query,
            "documents": documents,
        }
        if request.top_n is not None:
            body["top_n"] = request.top_n
        return body

    def parse_provider_rerank_response(
        self, response: dict[str, Any], model: str = ""
    ) -> InternalRerankResponse:
        """Parse a provider rerank response into InternalRerankResponse.

        The shared response shape is ``{results: [{index, relevance_score,
        document?}], model?, id?, provider?, usage?}``. ``usage`` carries
        ``total_tokens`` plus, on OpenRouter/Cohere, ``search_units`` and the
        billed ``cost``; the dialect is :func:`parse_usage`'s ``"rerank"`` and the
        raw object is re-shaped — not echoed verbatim — onto the three keys the
        client wire reports, so a provider's private usage fields stay upstream.
        """
        raw_usage = response.get("usage")
        usage_echo = _rerank_usage_echo(raw_usage) if isinstance(raw_usage, dict) else {}

        return InternalRerankResponse(
            model=response.get("model") or model,
            results=self._rerank_results(response),
            id=response.get("id"),
            provider=response.get("provider"),
            usage=parse_wire_usage("rerank", response),
            provider_info=self._rerank_provider_info(usage_echo),
        )

    @staticmethod
    def _rerank_results(response: dict[str, Any]) -> list[RerankResult]:
        """The ranked documents of a rerank response, in the shared shape.

        A score the wire did not report — or reported as something other than a
        number — reads as ``0.0`` rather than failing the whole response.
        ``relevant_spans`` (Cohere Rerank 5+, requested through ``extra``) is
        carried through verbatim: the client asked for the spans, so dropping
        them would make the request field a no-op.
        """
        results: list[RerankResult] = []
        for item in response.get("results") or []:
            if not isinstance(item, dict):
                continue
            score = item.get("relevance_score", 0.0)
            results.append(
                RerankResult(
                    index=item.get("index", 0),
                    relevance_score=float(score) if isinstance(score, int | float) else 0.0,
                    document=RerankDocument.from_wire(item.get("document")),
                    relevant_spans=_reported_spans(item.get("relevant_spans")),
                )
            )
        return results

    @staticmethod
    def _rerank_provider_info(usage_echo: dict[str, Any]) -> dict[str, Any]:
        """The provider-info extra a rerank response carries.

        ``rerank_usage`` is the client-facing echo of the upstream's usage block;
        the billed cost is mirrored onto ``openrouter_cost``, the extra the
        billing pipeline reads in preference to a local estimate (the same
        second route ``nanogpt_cost`` has).
        """
        provider_info: dict[str, Any] = {}
        if usage_echo:
            provider_info["rerank_usage"] = usage_echo
        cost = usage_echo.get("cost")
        if cost is not None:
            provider_info["openrouter_cost"] = cost
        return provider_info

    def build_provider_systemone_request(self, request: InternalSystemOneRequest) -> dict[str, Any]:
        """Build a provider-specific System One request body.

        Every upstream shares the same wire format, so the default is the whole
        body. Optional non-shared fields (OpenRouter's ``provider`` routing,
        ``session_id``, ``trace``, ``user``; Ollama's ``images`` and
        ``keep_alive``) ride ``request.extra`` and are merged by the adapter's
        field-policy chokepoint, not here.
        """
        return {
            "model": request.model,
            "state": request.state,
            "questions": request.questions,
        }

    def parse_provider_systemone_response(
        self, response: dict[str, Any], model: str = ""
    ) -> InternalSystemOneResponse:
        """Parse a provider System One response into InternalSystemOneResponse.

        Shared by every System One provider. The upstream response shape is
        ``{model, answers, usage, [id], [provider]}``; OpenRouter additionally
        reports ``usage.cost`` and the upstream ``usage`` object is preserved
        verbatim so the protocol formatter can echo it losslessly.
        """
        raw_usage = response.get("usage")
        usage = None
        provider_info: dict[str, Any] = {}
        if isinstance(raw_usage, dict):
            usage = Usage(
                input_tokens=raw_usage.get("input_tokens", 0) or 0,
                output_tokens=raw_usage.get("output_tokens", 0) or 0,
            )
            provider_info["systemone_usage"] = raw_usage
            cost = reported_cost(raw_usage)
            if cost is not None:
                # Provider-reported billed cost; the billing pipeline reads it
                # in preference to a local estimate (same key the chat and
                # media endpoints use for an OpenRouter-reported charge).
                provider_info["openrouter_cost"] = cost

        return InternalSystemOneResponse(
            model=response.get("model") or model,
            answers=response.get("answers", {}) or {},
            usage=usage,
            id=response.get("id"),
            provider=response.get("provider"),
            provider_info=provider_info,
        )

    def build_provider_decisions_request(self, request: InternalDecisionRequest) -> dict[str, Any]:
        """Build a provider-specific Decisions request body.

        Every upstream that serves Decisions natively shares the same wire
        format, so the default is the whole body. An optional non-shared field
        (``safety_identifier``) rides ``request.extra`` and is merged by the
        adapter's field-policy chokepoint, not here. A provider that speaks the
        System One envelope instead is bridged in
        :mod:`llm_proxy.models.decisions_bridge`, not here.
        """
        return {
            "model": request.model,
            "input": request.input,
            "questions": request.questions,
        }

    def parse_provider_decisions_response(
        self, response: dict[str, Any], model: str = ""
    ) -> InternalDecisionResponse:
        """Parse a provider Decisions response into InternalDecisionResponse.

        The upstream shape is ``{answers, model, usage}``; ``usage`` reports
        cache and reasoning tokens as nested details, which are mapped onto the
        canonical flat fields. The upstream ``usage`` object is preserved
        verbatim so the protocol formatter can echo it losslessly.
        """
        raw_usage = response.get("usage")
        # Decisions' usage object is the Responses dialect's: ``input_tokens`` /
        # ``output_tokens`` with the cache and thinking counts nested under
        # ``input_tokens_details`` / ``output_tokens_details``. It declares that
        # dialect rather than growing a second adapter for one shape (ADR-0025).
        usage = parse_wire_usage("responses", response)
        provider_info: dict[str, Any] = {}
        if isinstance(raw_usage, dict):
            provider_info["decisions_usage"] = raw_usage
            cost = reported_cost(raw_usage)
            if cost is not None:
                # Provider-reported billed cost; the billing pipeline reads it
                # in preference to a local estimate.
                provider_info["openrouter_cost"] = cost

        answers = response.get("answers")
        return InternalDecisionResponse(
            model=response.get("model") or model,
            answers=answers if isinstance(answers, list) else [],
            usage=usage,
            id=response.get("id"),
            provider=response.get("provider"),
            provider_info=provider_info,
        )


def _reported_spans(value: Any) -> list[dict[str, Any]] | None:
    """The span objects a rerank upstream reported, when it reported any.

    Spans are the upstream's own ``{start, end, score}`` objects (Cohere's
    ``RerankRelevantSpanResponse``), so they are passed through verbatim rather
    than re-modelled — this proxy has no opinion on their contents. A payload
    that is not a list of objects reads as "not reported".
    """
    if not isinstance(value, list):
        return None
    spans = [span for span in value if isinstance(span, dict)]
    return spans or None


def _rerank_usage_echo(raw_usage: dict[str, Any]) -> dict[str, Any]:
    """Project a shared ``/rerank`` usage block onto the client-facing keys.

    The rerank wire reports ``total_tokens``, ``search_units`` and the billed
    ``cost`` (OpenRouter states all three; Cohere's ``ApiMeta`` is re-shaped onto
    them by its own serializer). Anything else a provider puts in its usage block
    is provider-private and never reaches the client, so the echo is built field by
    field rather than handed through.
    """
    echo: dict[str, Any] = {}
    total_tokens = raw_usage.get("total_tokens")
    if is_int(total_tokens):
        echo["total_tokens"] = total_tokens
    search_units = raw_usage.get("search_units")
    if is_int(search_units):
        echo["search_units"] = search_units
    cost = reported_cost(raw_usage)
    if cost is not None:
        echo["cost"] = cost
    return echo


__all__ = ["ProviderSerializer"]
