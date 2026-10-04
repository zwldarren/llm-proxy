"""System One protocol endpoint configuration."""

from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.systemone.schemas import SystemOneRequestSchema
from llm_proxy.protocols.systemone.serializer import SystemOneProtocolSerializer  # noqa: F401

systemone_protocol = ProtocolEndpoint(
    name="systemone",
    paths=["/v1/systemone"],
    request_model=SystemOneRequestSchema,
    tags=["systemone"],
    description=(
        "System One evaluation endpoint: evaluate a state against a map of typed "
        "questions. Shared by TypeSafe's Jev, OpenRouter and Ollama."
    ),
)


__all__ = ["systemone_protocol"]
