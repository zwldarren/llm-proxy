"""Provider-hosted tools that have no protocol-agnostic equivalent.

OpenAI's ``web_search`` / ``file_search`` and Anthropic's ``web_search_20250305``
family are modeled individually because several wires consume them. The rest —
``google_search``, ``url_context``, ``google_maps``, ``x_search``,
``web_extractor``, ``retrieval``, ``builtin_function`` — are *provider-hosted*
tools: the upstream executes them, so the only honest handling is to forward the
declaration to a wire that understands it and to say so loudly when none does.

``HostedTool`` is that carrier. It keeps the wire fields verbatim (``params``)
and lets the provider dialect decide, per ``type``, whether to emit or drop it
(``llm_proxy.serialization.hosted_tools`` holds those tables).
"""

from dataclasses import dataclass, field
from typing import Any

from llm_proxy.models.tools.core import ToolDefinition

#: Hosted-tool types a client may declare that are executed by an upstream
#: rather than by the caller. ``web_search``-family, ``file_search`` and
#: ``code_interpreter`` are excluded: they already have typed models.
#: ``builtin_function`` is Kimi's wrapper for ``$web_search`` (deprecated by the
#: upstream, still forwarded when a client asks for it explicitly).
PROVIDER_HOSTED_TOOL_TYPES: frozenset[str] = frozenset(
    {
        "google_search",
        "google_search_retrieval",
        "url_context",
        "google_maps",
        "x_search",
        "web_extractor",
        "retrieval",
        "builtin_function",
    }
)

#: Hosted types the upstream has deprecated or never executes on a chat wire.
#: Forwarding still happens (the client asked for it); the log records why the
#: proxy does not generate them on its own.
DEPRECATED_HOSTED_TOOL_TYPES: frozendict[str, str] = frozendict(
    {
        "builtin_function": (
            "Kimi's $web_search built-in function is deprecated upstream "
            "(retires 2026-10-20); prefer the Responses web_search tool"
        ),
    }
)


@dataclass
class HostedTool(ToolDefinition):
    """A provider-hosted tool carried verbatim.

    Attributes:
        name: Tool name; hosted tools use their ``type`` (``x_search`` has no
            separate name on the wire, but the response side needs one).
        type: Wire ``type`` value, exactly as the client declared it.
        params: The remaining wire fields, minus ``type``.
    """

    name: str = ""
    type: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, wire: dict[str, Any]) -> HostedTool:
        """Carry a raw ``tools[]`` entry verbatim: ``type`` lifted, rest kept.

        The one home of the "strip ``type``, keep everything else" rule, so the
        intakes never re-implement it (and never drift apart on coercing a
        non-string ``type``).
        """
        tool_type = wire.get("type")
        tool_type = tool_type if isinstance(tool_type, str) else str(tool_type)
        return cls(
            name=tool_type,
            type=tool_type,
            params={k: v for k, v in wire.items() if k != "type"},
        )


def is_provider_hosted_tool_type(tool_type: Any) -> bool:
    """Whether ``tool_type`` names a provider-hosted tool ``HostedTool`` carries."""
    return isinstance(tool_type, str) and tool_type in PROVIDER_HOSTED_TOOL_TYPES


__all__ = [
    "DEPRECATED_HOSTED_TOOL_TYPES",
    "PROVIDER_HOSTED_TOOL_TYPES",
    "HostedTool",
    "is_provider_hosted_tool_type",
]
