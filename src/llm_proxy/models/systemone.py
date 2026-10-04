"""Unified System One request and response models.

System One (TypeSafe's Jev model, also resold by OpenRouter and served locally
by Ollama v0.35+) is an evaluation endpoint rather than a chat completion: it
takes a ``state`` plus a map of typed ``questions`` and returns exactly one
typed ``answer`` per question. There is no conversation, no tools and no
streaming, so it gets its own internal models instead of being squeezed into
:class:`InternalRequest`.

All upstreams share the same wire format. OpenRouter accepts a few optional
fields on top (``provider`` routing, ``session_id``, ``trace``, ``user``) and
returns a few extra response fields (``id``, ``provider``, ``usage.cost``);
Ollama adds its own optional ``images`` and ``keep_alive``. Those ride
``extra``/``provider_info`` so the request and response stay lossless for
either provider.
"""

from dataclasses import dataclass, field
from typing import Any

from llm_proxy.models.conversation import ConversationContext
from llm_proxy.models.internal import RequestMetadata
from llm_proxy.models.params import GenerationParams
from llm_proxy.models.tools import ToolDefinition
from llm_proxy.models.types import Usage


@dataclass
class InternalSystemOneRequest:
    """Unified System One request.

    This is the protocol-agnostic request format for System One evaluation.
    Every systemone protocol serializer parses its wire format into this model.

    Attributes:
        request_type: Always ``"systemone"``; routes the request to the
            System One strategy.
        model: The System One model to evaluate with (e.g., ``"jev-latest"``).
        state: The content to evaluate — a plain string, or a JSON object or
            array of related context.
        questions: Map of question id to a typed question object. Kept as raw
            JSON so every question shape the upstreams accept round-trips
            without loss.
        extra: Optional provider fields (OpenRouter's ``provider`` routing,
            ``session_id``, ``trace``, ``user``; Ollama's ``images`` and
            ``keep_alive``) plus any overrides injected by the pipeline. Merged
            into the outbound body subject to the provider's unknown-fields
            policy.
    """

    request_type: str = field(default="systemone", init=False)
    model: str
    state: Any
    questions: dict[str, Any]
    # Shared-pipeline fields kept for parity with the other non-chat internal
    # requests (see ``InternalEmbeddingRequest``): stages such as WebSearchStage
    # read ``params``/``tools`` unconditionally when web search is enabled, and
    # embeddings set them to empty defaults. They carry no System One meaning.
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
class InternalSystemOneResponse:
    """Unified System One response.

    Attributes:
        model: The model that performed the evaluation.
        answers: Map of question id to the typed answer, keyed by the same
            ids the client used in ``questions``.
        usage: Token usage for the request (input/output tokens).
        id: Provider generation id (OpenRouter only).
        provider: Provider name that served the request (OpenRouter only).
        provider_info: Provider-specific metadata (upstream usage for an exact
            echo, reported cost, rate-limit headers).
    """

    model: str
    answers: dict[str, Any]
    usage: Usage | None = None
    id: str | None = None
    provider: str | None = None
    request_id: str | None = None
    provider_info: dict[str, Any] = field(default_factory=dict)


__all__ = ["InternalSystemOneRequest", "InternalSystemOneResponse"]
