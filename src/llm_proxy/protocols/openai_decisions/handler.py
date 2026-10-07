"""OpenAI Decisions protocol endpoint configuration."""

from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openai_decisions.schemas import DecisionsRequestSchema
from llm_proxy.protocols.openai_decisions.serializer import (  # noqa: F401
    OpenAIDecisionsProtocolSerializer,
)

openai_decisions_protocol = ProtocolEndpoint(
    name="openai_decisions",
    # Path aliases: clients whose base_url is missing /v1 or double-writes it
    # ("{base}/v1" + "/v1/decisions") still reach the endpoint, matching the
    # openai and openresponses protocols.
    paths=["/v1/decisions", "/decisions", "/v1/v1/decisions"],
    request_model=DecisionsRequestSchema,
    tags=["decisions"],
    description=(
        "OpenAI Decisions endpoint: answer typed questions (predicate, choice, score) "
        "about shared text and image evidence. Served natively by OpenAI and, through "
        "the System One bridge, by TypeSafe, OpenRouter and Ollama decision models."
    ),
)


__all__ = ["openai_decisions_protocol"]
