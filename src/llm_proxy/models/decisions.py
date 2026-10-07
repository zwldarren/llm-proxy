"""Unified Decisions request and response models.

Decisions is OpenAI's evaluation endpoint (``POST /v1/decisions``, model
``gpt-6-luna``): it takes shared evidence — a string or user messages holding
text and inline images — plus a list of typed questions and returns exactly one
typed answer per question. There is no conversation, no tools and no streaming,
so it gets its own internal models instead of being squeezed into
:class:`InternalRequest`.

It answers the same three primitives as System One (see
:mod:`llm_proxy.models.systemone`) under different names — ``predicate`` for
``noul``, and explicit ``choices``/``levels`` arrays instead of a ``criteria``
map or array — and it names questions in the question object instead of keying
a map. The two wire formats are bridged by
:mod:`llm_proxy.models.decisions_bridge`.

The upstream response is ``{model, answers, usage}`` where ``usage`` carries
``input_tokens_details``/``output_tokens_details``. The raw upstream usage is
preserved verbatim so the protocol formatter can echo it losslessly.
"""

from dataclasses import dataclass, field
from typing import Any

from llm_proxy.models.conversation import ConversationContext
from llm_proxy.models.internal import RequestMetadata
from llm_proxy.models.params import GenerationParams
from llm_proxy.models.tools import ToolDefinition
from llm_proxy.models.types import Usage


@dataclass
class InternalDecisionRequest:
    """Unified Decisions request.

    This is the protocol-agnostic request format for Decisions evaluation.
    Every decisions protocol serializer parses its wire format into this model.

    Attributes:
        request_type: Always ``"decisions"``; routes the request to the
            Decisions strategy.
        model: The Decisions model to evaluate with (e.g. ``"gpt-6-luna"``).
        input: Shared evidence for every question — a plain string, or a list
            of user messages holding ``input_text``/``input_image`` parts. Kept
            as raw JSON so both shapes round-trip without loss.
        questions: The typed questions, in request order. Kept as raw JSON so
            every question shape the upstreams accept round-trips without loss;
            ``name`` is the answer key and is optional on the wire.
        extra: Optional provider fields (``safety_identifier``) plus any
            overrides injected by the pipeline. Merged into the outbound body
            subject to the provider's unknown-fields policy.
    """

    request_type: str = field(default="decisions", init=False)
    model: str
    input: Any
    questions: list[dict[str, Any]]
    # Shared-pipeline fields kept for parity with the other non-chat internal
    # requests (see ``InternalSystemOneRequest``): stages such as WebSearchStage
    # read ``params``/``tools`` unconditionally when web search is enabled.
    # They carry no Decisions meaning.
    stream: bool = False
    tools: list[ToolDefinition] | None = None
    conversation: ConversationContext | None = None
    params: GenerationParams = field(default_factory=GenerationParams)
    metadata: RequestMetadata = field(default_factory=RequestMetadata)
    extra: dict[str, Any] = field(default_factory=dict)
    _override_injected_keys: set[str] = field(default_factory=set, repr=False)
    _raw_protocol_data: dict[str, Any] | None = field(default=None, repr=False)
    user_facing_model: str | None = field(default=None, repr=False)


@dataclass
class InternalDecisionResponse:
    """Unified Decisions response.

    Attributes:
        model: The model that performed the evaluation.
        answers: The typed answers, in question order. Each entry is the raw
            upstream answer object (``predicate``/``choice``/``score``/
            ``refusal``), so a field the proxy does not model still reaches the
            client.
        usage: Token usage for the request (input/output tokens).
        id: Provider generation id, when the upstream reports one.
        provider: Provider name that served the request, when the upstream
            reports one.
        provider_info: Provider-specific metadata (upstream usage for an exact
            echo, reported cost, rate-limit headers).
    """

    model: str
    answers: list[dict[str, Any]]
    usage: Usage | None = None
    id: str | None = None
    provider: str | None = None
    request_id: str | None = None
    provider_info: dict[str, Any] = field(default_factory=dict)


__all__ = ["InternalDecisionRequest", "InternalDecisionResponse"]
