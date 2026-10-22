"""Pydantic request schemas for the Rerank protocol.

The wire format is Cohere's ``POST /v1/rerank``, which OpenRouter
(``openrouter.ai/api/v1/rerank``) and the ``openai-compatible`` family's
``/rerank`` route also serve. OpenRouter accepts a superset: the optional
``provider`` routing block, ``session_id``, ``trace`` and ``user``. Those fields
are declared here so the endpoint validates and documents them, and they ride
``InternalRerankRequest.extra`` so a provider that documents them receives them
and one that does not can strip them under its field policy.

The top-level body stays open (``extra="allow"``, as on every other
client-facing wire schema in this package): a field this module does not model
is not a client error, it lands in ``extra`` and the provider's
``unknown_fields_policy`` decides what happens to it. A structured *document*
is open for the same reason: OpenRouter's document schema sets
``additionalProperties: false`` nowhere, so a key this proxy does not model rides
``RerankDocument.extra`` to an upstream that documents it.

Cohere's own v2-only knobs (``max_tokens_per_doc``, ``relevant_spans``,
``priority``) are declared here as well, so the endpoint validates, documents and
forwards them instead of treating them as opaque extras.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RerankRelevantSpansSchema(BaseModel):
    """Cohere's ``relevant_spans`` request block (Rerank 5 and newer).

    Turned on, the upstream also reports, per result, the spans of the document
    that drove its score; the protocol formatter passes them through on
    ``results[].relevant_spans``.
    """

    model_config = ConfigDict(extra="allow")

    mode: str = Field(
        "enabled",
        pattern=r"^(enabled|disabled)$",
        description="Whether spans are returned for each ranked document.",
    )
    threshold: float | None = Field(
        None, description="Only spans scoring above this threshold are returned (default 0.1)."
    )
    max_spans: int | None = Field(
        None, ge=1, description="Maximum spans per document; all when omitted."
    )


class RerankDocumentSchema(BaseModel):
    """A structured rerank document (multimodal models only).

    At least one of ``text`` or ``image`` must be present; any other key is
    carried through unchanged (see the module docstring).
    """

    model_config = ConfigDict(extra="allow")

    text: str | None = Field(None, description="The document text.")
    image: str | None = Field(
        None,
        description="A remote URL (http/https) or base64 data URI for the document image.",
    )

    @model_validator(mode="after")
    def _require_content(self) -> RerankDocumentSchema:
        if self.text is None and self.image is None:
            raise ValueError("a document object must carry 'text', 'image', or both")
        return self


class RerankRequestSchema(BaseModel):
    """Top-level rerank request body."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The rerank model, e.g. 'cohere/rerank-v3.5'.")
    query: str = Field(..., description="The search query to rank the documents against.")
    documents: list[str | RerankDocumentSchema] = Field(
        ...,
        min_length=1,
        description=(
            "The candidate documents: plain strings, or structured objects with "
            "'text' and/or 'image' for multimodal rerankers."
        ),
    )
    top_n: int | None = Field(
        None,
        ge=1,
        description="Number of most relevant documents to return (default: all).",
    )
    return_documents: bool | None = Field(
        None,
        description=(
            "Whether the response echoes each document. Defaults to true; when "
            "false the 'document' field is omitted from every result."
        ),
    )
    user: str | None = Field(
        None, max_length=256, description="End-user identifier (OpenRouter only)."
    )

    # OpenRouter-only optional fields. Declared so they are validated and
    # documented; forwarded upstream only where documented (see
    # EXEMPT_EXTRA_KEYS on the OpenRouter adapter).
    provider: dict[str, Any] | None = Field(
        None, description="Provider routing preferences (OpenRouter only)."
    )
    session_id: str | None = Field(
        None,
        max_length=256,
        description=(
            "Observability grouping id, forwarded only to upstreams that document "
            "it (OpenRouter only)."
        ),
    )
    trace: dict[str, Any] | None = Field(
        None, description="Observability/tracing metadata (OpenRouter only)."
    )

    # Cohere v2 rerank fields: declared so they validate and are documented, and
    # forwarded through ``extra`` to the providers that document them (Cohere's
    # ``EXEMPT_EXTRA_KEYS`` whitelist).
    max_tokens_per_doc: int | None = Field(
        None,
        ge=1,
        description="Long documents are truncated to this many tokens (Cohere only, default 4096).",
    )
    relevant_spans: RerankRelevantSpansSchema | None = Field(
        None,
        description=(
            "Ask the upstream for the document spans that drove each score "
            "(Cohere Rerank 5+ only). Returned on `results[].relevant_spans`."
        ),
    )
    priority: int | None = Field(
        None, description="Admission priority; lower is handled sooner (Cohere only)."
    )

    @model_validator(mode="after")
    def _clamp_top_n(self) -> RerankRequestSchema:
        # Cohere and the vLLM/SGLang ``/rerank`` route clamp a ``top_n`` larger
        # than the document list instead of rejecting it, so a drop-in client
        # that relies on that keeps working here.
        if self.top_n is not None and self.top_n > len(self.documents):
            self.top_n = len(self.documents)
        return self


__all__ = ["RerankDocumentSchema", "RerankRelevantSpansSchema", "RerankRequestSchema"]
