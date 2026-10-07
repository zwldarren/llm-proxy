"""OpenAI Decisions protocol package.

Registers the ``/v1/decisions`` endpoint and its protocol serializer. Upstream
support lives in the ``openai`` adapter (native Decisions) and, through the
System One bridge, in the ``typesafe``, ``openrouter`` and ``ollama`` adapters.
"""

from llm_proxy.protocols.openai_decisions.handler import openai_decisions_protocol
from llm_proxy.protocols.registry import register_protocol

__all__ = ["openai_decisions_protocol"]

register_protocol(openai_decisions_protocol)
