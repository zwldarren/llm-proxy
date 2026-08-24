# src/llm_proxy/protocols/anthropic/__init__.py
"""Anthropic protocol endpoint using unified format."""

from llm_proxy.protocols.anthropic.handler import anthropic_protocol
from llm_proxy.protocols.registry import register_protocol

__all__ = ["anthropic_protocol"]

register_protocol(anthropic_protocol)
