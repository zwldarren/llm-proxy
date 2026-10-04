"""System One protocol package.

Registers the ``/v1/systemone`` endpoint and its protocol serializer. Upstream
support lives in the ``typesafe`` provider, the ``openrouter`` adapter and the
``ollama`` adapter (local models, v0.35+).
"""

from llm_proxy.protocols.registry import register_protocol
from llm_proxy.protocols.systemone.handler import systemone_protocol

__all__ = ["systemone_protocol"]

register_protocol(systemone_protocol)
