"""Rerank protocol package.

Registers the ``/v1/rerank`` endpoint and its protocol serializer. The wire
format is Cohere's, which OpenRouter's ``/api/v1/rerank`` and the
``openai-compatible`` family's ``/rerank`` route also speak.
"""

from llm_proxy.protocols.registry import register_protocol
from llm_proxy.protocols.rerank.handler import rerank_protocol

__all__ = ["rerank_protocol"]

register_protocol(rerank_protocol)
