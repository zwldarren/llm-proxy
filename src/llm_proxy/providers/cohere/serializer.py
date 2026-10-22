"""Cohere v2 provider serializer.

Cohere's native API is not OpenAI-shaped, so this serializer owns both
directions for every endpoint the proxy serves from it:

==========  ===================================================================
chat        ``POST /v2/chat`` — ``messages``/``tools``/``thinking``/
            ``documents``/``citation_options``; the response is an assistant
            message of ``text``/``thinking`` content items plus ``tool_calls``
            (docs.cohere.com/reference/chat).
embed       ``POST /v2/embed`` — ``texts`` plus a required ``input_type`` and
            ``embedding_types``, answering ``embeddings: {<type>: [[...]]}``
            (docs.cohere.com/reference/embed).
rerank      ``POST /v2/rerank`` — plain-string ``documents``, answering
            ``results[].relevance_score`` with no echoed document
            (docs.cohere.com/reference/rerank).
==========  ===================================================================

Nothing here is wire-compatible with a client protocol, so
``compatible_protocols`` stays empty: every Cohere turn is a full conversion.
"""

from typing import Any

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    CustomTool,
    FunctionTool,
    InternalEmbeddingRequest,
    InternalEmbeddingResponse,
    InternalRequest,
    InternalRerankRequest,
    InternalRerankResponse,
    ToolChoice,
    ToolChoiceAllowedTools,
    ToolChoiceCustom,
    ToolChoiceFunction,
    ToolChoiceNamed,
)
from llm_proxy.models.embedding import EmbeddingData
from llm_proxy.models.tools import custom_tool_bridge_description
from llm_proxy.models.types import unwrap_json_schema_wrapper
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.cohere.conversation import CohereConversationMixin
from llm_proxy.providers.cohere.response_parser import CohereResponseParserMixin
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.providers.base import ProviderSerializer
from llm_proxy.serialization.providers.field_utils import is_int, reported_cost
from llm_proxy.serialization.providers.registry import register_provider_serializer
from llm_proxy.serialization.usage import parse_usage as parse_wire_usage

logger = get_logger(__name__)

#: Cohere-only chat body fields a client may pass through ``extra``. They are
#: exempt from the unknown-fields policy so declaring them is not the same as
#: sending an unknown field (``EXEMPT_EXTRA_KEYS`` on the adapter), and they are
#: the only keys this serializer merges onto the body.
COHERE_CHAT_EXTRA_KEYS: frozenset[str] = frozenset(
    {
        "documents",  # strings or {id, data} objects for RAG and citations
        "citation_options",  # {"mode": "ENABLED"|"DISABLED"|"FAST"|"ACCURATE"|"OFF"}
        "safety_mode",  # CONTEXTUAL | STRICT | OFF
        "strict_tools",  # force tool calls to follow their schema
        "priority",  # admission priority, lower is earlier
        "k",  # top-k sampling
        "logprobs",  # per-token log probabilities
    }
)

#: Cohere's ``input_type`` enum. Required from Embed v3 on, and the OpenAI
#: embeddings wire has no such field, so the proxy states its default (how the
#: stored-document side of a search index embeds) unless the caller names one.
COHERE_INPUT_TYPES: frozenset[str] = frozenset(
    {"search_document", "search_query", "classification", "clustering", "image"}
)

#: Cohere accepts at most 96 inputs per embed call (docs.cohere.com/reference/
#: embed); a larger batch is rejected rather than split, because a split batch is
#: billed and rate-limited as several requests.
COHERE_MAX_EMBED_INPUTS = 96

#: Cohere accepts at most five stop sequences (docs.cohere.com/reference/chat).
COHERE_MAX_STOP_SEQUENCES = 5


@register_provider_serializer("cohere")
class CohereProviderSerializer(
    CohereConversationMixin,
    CohereResponseParserMixin,
    ProviderSerializer,
):
    """Convert between the internal models and Cohere's v2 API."""

    _DEFAULT_PROVIDER_NAME = "cohere"

    @property
    def compatible_protocols(self) -> frozenset[str]:
        """Cohere v2 has no compatible client wire format: every turn translates."""
        return frozenset()

    def filter_extra_for_body(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Return the Cohere-only chat fields from ``request.extra``."""
        return {key: value for key, value in extra.items() if key in COHERE_CHAT_EXTRA_KEYS}

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _build_provider_request(
        self,
        request: InternalRequest,
        context: BuildContext,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": self._build_cohere_messages(request, context),
            # v2 documents the field as required with a false default; state it
            # so a streaming turn is never ambiguous.
            "stream": bool(request.stream),
        }
        self._apply_sampling(body, request)
        self._apply_thinking(body, request)
        self._apply_response_format(body, request)
        self._apply_tools(body, request)
        # ``_finalize_body`` merges extras only for the non-chat endpoints, so the
        # chat body merges its own — through the same whitelist the field policy
        # sees, which keeps "ignore" and "error" behaving as configured.
        body.update(self.filter_extra_for_body(request.extra or {}))
        return body

    @staticmethod
    def _apply_sampling(body: dict[str, Any], request: InternalRequest) -> None:
        """Map the unified sampling parameters onto Cohere's spellings."""
        params = request.params
        common = params.common
        if common.temperature is not None:
            body["temperature"] = common.temperature
        if common.top_p is not None:
            # Cohere names nucleus sampling ``p``; the value is the same 0.01–0.99.
            body["p"] = common.top_p
        max_tokens = common.max_tokens
        if max_tokens is None and params.openai is not None:
            max_tokens = params.openai.max_completion_tokens
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if common.stop:
            # More than five is an upstream 400 on every request; v2 honours up to
            # five sequences, so the cap is applied at the proxy edge.
            if len(common.stop) > COHERE_MAX_STOP_SEQUENCES:
                logger.warning(
                    f"Cohere accepts at most {COHERE_MAX_STOP_SEQUENCES} stop "
                    "sequences; dropping the rest"
                )
            body["stop_sequences"] = list(common.stop[:COHERE_MAX_STOP_SEQUENCES])
        if common.frequency_penalty is not None:
            body["frequency_penalty"] = common.frequency_penalty
        if common.presence_penalty is not None:
            body["presence_penalty"] = common.presence_penalty
        if common.seed is not None:
            body["seed"] = common.seed
        if common.n is not None and common.n > 1:
            # v2 has no ``n``: one request is one completion. Fanning out would
            # multiply the billed request count, so extra choices are refused
            # rather than silently collapsed.
            raise ProviderError(
                message=(
                    "Cohere v2 chat returns a single completion; "
                    "n > 1 is not supported by this provider."
                ),
                error_type="invalid_request_error",
                provider_name="cohere",
            )

    @staticmethod
    def _apply_thinking(body: dict[str, Any], request: InternalRequest) -> None:
        """Map the unified thinking config onto v2's ``thinking`` object.

        v2 spells it ``{"type": "enabled"|"disabled", "token_budget": N}`` and has
        reasoning on by default for models that support it — so only an explicit
        "off" or an explicit budget needs to travel.
        """
        thinking = request.params.thinking
        if thinking is None:
            return
        effort = (thinking.effort or "").lower()
        if thinking.type == "disabled" or effort in {"none", "minimal"}:
            body["thinking"] = {"type": "disabled"}
            return
        config: dict[str, Any] = {"type": "enabled"}
        if thinking.budget_tokens is not None and thinking.budget_tokens > 0:
            config["token_budget"] = thinking.budget_tokens
        body["thinking"] = config

    @staticmethod
    def _apply_response_format(body: dict[str, Any], request: InternalRequest) -> None:
        """Map the unified response format onto v2's ``text``/``json_object`` pair."""
        response_format = request.params.common.response_format
        if response_format is None or response_format.type == "text":
            # Text is v2's default; stating it adds nothing.
            return
        if response_format.type == "json_object":
            body["response_format"] = {"type": "json_object"}
            return
        if response_format.type == "json_schema":
            # v2 takes the plain JSON Schema itself, not OpenAI's
            # ``{name, description, schema, strict}`` wrapper, and has no
            # strict-mode flag.
            schema = unwrap_json_schema_wrapper(response_format.json_schema)
            body["response_format"] = {"type": "json_object", "json_schema": schema or {}}

    def _apply_tools(self, body: dict[str, Any], request: InternalRequest) -> None:
        """Render function tools and the tool-choice mode Cohere understands."""
        tools = self._cohere_tools(request)
        if tools:
            body["tools"] = tools
        tool_choice = self._cohere_tool_choice(request.tool_choice)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

    @staticmethod
    def _cohere_tools(request: InternalRequest) -> list[dict[str, Any]]:
        """Render the request's tools as v2 ``function`` entries.

        v2 declares ``parameters`` as required, so a tool without a schema still
        gets the empty object schema. A freeform (custom) tool bridges to a
        single-string function — the same bridge the Chat Completions dialects
        use — because v2 has no grammar tool type. A hosted tool has no slot on
        this wire: it is dropped with the reason, never turned into a function
        the upstream would then "call" itself.
        """
        rendered: list[dict[str, Any]] = []
        for tool in request.tools or []:
            if isinstance(tool, FunctionTool):
                function: dict[str, Any] = {
                    "name": tool.name,
                    "parameters": tool.parameters or {"type": "object", "properties": {}},
                }
                if tool.description:
                    function["description"] = tool.description
                rendered.append({"type": "function", "function": function})
            elif isinstance(tool, CustomTool):
                rendered.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": custom_tool_bridge_description(tool),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "The raw tool input.",
                                    }
                                },
                                "required": ["content"],
                            },
                        },
                    }
                )
            else:
                logger.warning(
                    f"Cohere has no wire support for hosted tool {tool.name!r}; dropping it"
                )
        return rendered

    @staticmethod
    def _cohere_tool_choice(tool_choice: Any) -> str | None:
        """Map the unified tool choice onto v2's ``REQUIRED`` / ``NONE`` pair.

        v2 has no "call *this* tool" mode and no "auto" spelling (omitting the
        field is auto). A named choice therefore collapses onto ``REQUIRED``: the
        model must call some tool, which is the closest documented behaviour.
        """
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            mode = tool_choice
        elif isinstance(tool_choice, ToolChoice):
            mode = tool_choice.mode
        elif isinstance(tool_choice, ToolChoiceAllowedTools):
            mode = tool_choice.allowed_tools.mode
        elif isinstance(tool_choice, ToolChoiceFunction | ToolChoiceNamed | ToolChoiceCustom):
            logger.debug("Cohere v2 cannot force a specific tool; asking for any tool instead")
            return "REQUIRED"
        else:
            return None
        match mode:
            case "required" | "any":
                return "REQUIRED"
            case "none":
                return "NONE"
            case _:
                # "auto" is expressed by leaving the field out.
                return None

    def get_chunk_converter(self, model: str = "", request_id: str = ""):
        """Return the Cohere stream-event converter for this turn."""
        from llm_proxy.providers.cohere.streaming import CohereChunkConverter

        return CohereChunkConverter(model=model, request_id=request_id)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def build_provider_embedding_request(self, request: InternalEmbeddingRequest) -> dict[str, Any]:
        """Build a v2 embed body.

        v2 requires ``input_type`` (from Embed v3 on) and answers in the embedding
        type asked for, so the OpenAI fields map as: ``input`` -> ``texts``,
        ``dimensions`` -> ``output_dimension``, and ``encoding_format:
        "base64"`` -> ``embedding_types: ["base64"]``.
        """
        inputs = request.input if isinstance(request.input, list) else [request.input]
        if len(inputs) > COHERE_MAX_EMBED_INPUTS:
            raise ProviderError(
                message=(
                    f"Cohere embeds at most {COHERE_MAX_EMBED_INPUTS} inputs per "
                    f"request; got {len(inputs)}."
                ),
                error_type="invalid_request_error",
                provider_name="cohere",
            )

        requested_type = (request.extra or {}).get("input_type")
        if requested_type is not None and requested_type not in COHERE_INPUT_TYPES:
            logger.warning(
                f"Unsupported Cohere input_type {requested_type!r}; using search_document"
            )
            requested_type = None

        body: dict[str, Any] = {
            "model": request.model,
            "input_type": requested_type or "search_document",
            "texts": list(inputs),
            "embedding_types": ["base64" if request.encoding_format == "base64" else "float"],
        }
        if request.dimensions is not None:
            body["output_dimension"] = request.dimensions
        return body

    def parse_provider_embedding_response(
        self, response: dict[str, Any], model: str = ""
    ) -> InternalEmbeddingResponse:
        """Parse a v2 embed response, whose vectors are grouped by embedding type."""
        embeddings = response.get("embeddings") or {}
        vectors = embeddings.get("float")
        if vectors is None:
            vectors = embeddings.get("base64")
        data = [
            EmbeddingData(embedding=vector, index=index)
            for index, vector in enumerate(vectors if isinstance(vectors, list) else [])
        ]
        provider_info: dict[str, Any] = {}
        meta = response.get("meta")
        if meta:
            provider_info["cohere_meta"] = meta
        return InternalEmbeddingResponse(
            model=response.get("model") or model,
            data=data,
            usage=parse_wire_usage("cohere", response),
            provider_info=provider_info,
        )

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def build_provider_rerank_request(self, request: InternalRerankRequest) -> dict[str, Any]:
        """Build a v2 rerank body.

        v2 takes plain-string documents only (no ``{text, image}`` objects), so a
        document that carries an image and no text has no representation on this
        wire and is refused rather than flattened to an empty string.
        ``return_documents`` is a response projection the proxy applies itself;
        the other v2 fields (``max_tokens_per_doc``, ``relevant_spans``,
        ``priority``) ride ``request.extra`` through the adapter's chokepoint.
        """
        documents: list[str] = []
        for document in request.documents:
            if isinstance(document, str):
                documents.append(document)
                continue
            if document.text is None:
                raise ProviderError(
                    message=(
                        "Cohere v2 rerank accepts text documents only; an image "
                        "document has no representation on this endpoint."
                    ),
                    error_type="invalid_request_error",
                    provider_name="cohere",
                )
            documents.append(document.text)

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
        """Parse a v2 rerank response.

        v2 answers ``results[].{index, relevance_score, relevant_spans}`` with no
        echoed document; the adapter re-attaches the request's own document so the
        client sees the projection the other rerank upstreams return.
        """
        meta = response.get("meta")
        provider_info = self._rerank_provider_info(
            _cohere_usage_echo(meta) if isinstance(meta, dict) else {}
        )
        if isinstance(meta, dict):
            provider_info["cohere_meta"] = meta
        return InternalRerankResponse(
            model=model,
            results=self._rerank_results(response),
            id=response.get("id"),
            usage=parse_wire_usage("cohere", response),
            provider_info=provider_info,
        )


def _cohere_usage_echo(meta: dict[str, Any]) -> dict[str, Any]:
    """Project Cohere's ``ApiMeta`` onto the shared rerank usage echo.

    The client-facing rerank response reports ``total_tokens``,
    ``search_units`` and the billed ``cost`` (the keys OpenRouter states on
    its own ``usage``), so the ``tokens``/``billed_units``/``cost`` fields of
    Cohere's meta are re-shaped into that one vocabulary instead of shipping
    the vendor envelope. A meta block with none of them yields nothing, and
    the protocol formatter then falls back to the canonical usage record.
    """
    echo: dict[str, Any] = {}
    tokens = meta.get("tokens")
    if isinstance(tokens, dict):
        input_tokens = tokens.get("input_tokens")
        output_tokens = tokens.get("output_tokens")
        if is_int(input_tokens) and is_int(output_tokens):
            echo["total_tokens"] = input_tokens + output_tokens
        elif is_int(input_tokens):
            echo["total_tokens"] = input_tokens
    billed = meta.get("billed_units")
    if isinstance(billed, dict) and is_int(billed.get("search_units")):
        echo["search_units"] = billed["search_units"]
    cost = reported_cost(meta)
    if cost is not None:
        echo["cost"] = cost
    return echo


__all__ = [
    "COHERE_CHAT_EXTRA_KEYS",
    "COHERE_INPUT_TYPES",
    "COHERE_MAX_EMBED_INPUTS",
    "COHERE_MAX_STOP_SEQUENCES",
    "CohereProviderSerializer",
]
