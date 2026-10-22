"""Rerank protocol serializer.

Converts between the Cohere-compatible ``/v1/rerank`` wire format and
:class:`InternalRerankRequest` / :class:`InternalRerankResponse`.
"""

from typing import Any

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models.rerank import (
    InternalRerankRequest,
    InternalRerankResponse,
    RerankDocument,
)
from llm_proxy.models.types import Usage
from llm_proxy.protocols.registry import register_protocol_serializer
from llm_proxy.protocols.serializer_base import ProtocolSerializer


@register_protocol_serializer("rerank")
class RerankProtocolSerializer(ProtocolSerializer):
    """Rerank protocol serializer."""

    @property
    def protocol_name(self) -> str:
        return "rerank"

    def parse_request(self, data: dict[str, Any]) -> InternalRerankRequest:
        """Parse a rerank request from the wire format."""
        model = data.get("model")
        if not model:
            raise ValidationError(
                message="Rerank request is missing required field 'model'.",
                code="invalid_request_error",
                status_code=422,
            )
        if "query" not in data:
            raise ValidationError(
                message="Rerank request is missing required field 'query'.",
                code="invalid_request_error",
                status_code=422,
            )

        raw_documents = data.get("documents")
        if not isinstance(raw_documents, list) or not raw_documents:
            raise ValidationError(
                message="Rerank request requires a non-empty 'documents' list.",
                code="invalid_request_error",
                status_code=422,
            )

        documents: list[str | RerankDocument] = []
        for index, raw in enumerate(raw_documents):
            if isinstance(raw, str):
                documents.append(raw)
                continue
            document = RerankDocument.from_wire(raw)
            if document is None:
                raise ValidationError(
                    message=(
                        f"documents[{index}] must be a string or an object with "
                        "'text' and/or 'image'."
                    ),
                    code="invalid_request_error",
                    status_code=422,
                )
            documents.append(document)

        extra = {
            k: v for k, v in data.items() if k not in self._known_request_fields() and v is not None
        }
        return InternalRerankRequest(
            model=model,
            query=data["query"],
            documents=documents,
            top_n=data.get("top_n"),
            return_documents=data.get("return_documents"),
            # ``user`` is mapped here *and* left in ``extra``: the internal field
            # is the attribution slot, while the extra ride is what lets an
            # upstream that documents it (OpenRouter) receive it.
            user=data.get("user"),
            extra=extra,
        )

    def format_response(self, response: object, context: object | None = None) -> dict[str, Any]:
        """Format a rerank response for the client wire format.

        ``results`` always carries ``index``, ``relevance_score`` and the
        upstream's echoed ``document`` (dropped earlier by the adapter when the
        client asked for ``return_documents: false``). ``usage`` is echoed when the
        upstream reports one, in the normalized vocabulary every rerank provider is
        projected onto: ``total_tokens``, ``search_units`` and the billed ``cost``.
        """
        if isinstance(response, InternalRerankResponse):
            results: list[dict[str, Any]] = []
            for item in response.results:
                result: dict[str, Any] = {
                    "index": item.index,
                    "relevance_score": item.relevance_score,
                }
                if item.document is not None:
                    result["document"] = item.document.to_wire()
                results.append(result)

            body: dict[str, Any] = {"model": response.model, "results": results}
            if response.id is not None:
                body["id"] = response.id
            if response.provider is not None:
                body["provider"] = response.provider
            usage = response.provider_info.get("rerank_usage")
            if usage is not None:
                body["usage"] = usage
            elif response.usage is not None:
                body["usage"] = {"total_tokens": _total_tokens(response.usage)}
            return body
        if isinstance(response, dict):
            return response
        return {"error": "Invalid response type"}

    @staticmethod
    def _known_request_fields() -> set[str]:
        """Fields handled explicitly; every other top-level key rides ``extra``.

        ``extra`` is forwarded to the upstreams that document it and stripped
        from those that do not. ``user`` is deliberately absent: it is mapped
        onto the internal field *and* rides ``extra`` so that forwarding keeps
        working (see :meth:`parse_request`).
        """
        return {"model", "query", "documents", "top_n", "return_documents"}


def _total_tokens(usage: Usage) -> int:
    """The turn's token count, as the rerank wire reports it.

    The rerank dialect states the count directly; the fallback covers a
    provider whose usage record only carries the input/output split.
    """
    if usage.total_tokens is not None:
        return usage.total_tokens
    return (usage.input_tokens or 0) + (usage.output_tokens or 0)


__all__ = ["RerankProtocolSerializer"]
