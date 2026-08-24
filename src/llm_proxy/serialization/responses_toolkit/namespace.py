"""Namespace mapping utility for OpenResponses protocol.

Provides NamespaceMapping to flatten and restore MCP namespace-prefixed tool names,
so that nested namespace structures can round-trip through flat tool name lists.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any

MAX_FN = 64


@dataclass
class NamespaceMapping:
    """Bidirectional mapping between (namespace, tool_name) tuples and flat tool names.

    Used when converting namespace-structured tool definitions (e.g., MCP namespaces
    like "mcp__github") into flat tool names for providers that don't support
    hierarchical tool naming, and back again.
    """

    flat_to_original: dict[str, tuple[str, str]] = field(default_factory=dict)
    original_to_flat: dict[tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def build(cls, namespaces: list[dict[str, Any]]) -> NamespaceMapping:
        """Build a mapping from a list of namespace dicts.

        Each namespace dict has 'type': 'namespace', 'name': '<ns_name>', and
        'tools': [{'type': 'function', 'name': '<tool_name>'}, ...].
        """
        m = cls()
        for ns in namespaces:
            ns_name = ns.get("name", "")
            if not ns_name:
                continue
            for child in ns.get("tools") or []:
                cn = child.get("name", "")
                if cn:
                    m.flatten(ns_name, cn)
        return m

    def flatten(self, ns: str, name: str) -> str:
        """Flatten a namespace and name into a single flat identifier.

        If the combined name exceeds MAX_FN (64) characters, the name is
        truncated and a short hash is appended to ensure uniqueness.
        """
        flat = f"{ns}__{name}"
        if len(flat) > MAX_FN:
            h = hashlib.sha256(flat.encode()).hexdigest()[:8]
            # Preserve fragments of both ns and name for readability
            # and lower collision probability within the same namespace.
            name_budget = max(1, (MAX_FN - len(h) - 4) // 2)
            ns_budget = MAX_FN - len(h) - 4 - name_budget
            flat = f"{ns[:ns_budget]}__{name[:name_budget]}__{h}"
        # Detect hash collisions: if a different (ns, name) pair already
        # maps to the same flat key, append a disambiguating suffix.
        if flat in self.flat_to_original and self.flat_to_original[flat] != (ns, name):
            base = flat
            i = 1
            while flat in self.flat_to_original and self.flat_to_original[flat] != (ns, name):
                suffix = f"_{i}"
                flat = base[: MAX_FN - len(suffix)] + suffix
                i += 1
        self.flat_to_original[flat] = (ns, name)
        self.original_to_flat[(ns, name)] = flat
        return flat

    def to_dict(self) -> dict[str, list[str]]:
        """Serialize the mapping to a JSON-compatible dict."""
        return {k: list(v) for k, v in self.flat_to_original.items()}

    @classmethod
    def from_dict(cls, d: dict | None) -> NamespaceMapping:
        """Deserialize a NamespaceMapping from a dict."""
        m = cls()
        if not d:
            return m
        for k, v in d.items():
            if isinstance(v, list) and len(v) == 2:
                key = (v[0], v[1])
                m.flat_to_original[k] = key
                m.original_to_flat[key] = k
        return m


def flatten_history_tool_name(namespace_map: dict[str, list[str]] | None, name: str) -> str:
    """Map a history tool-call name to its flattened tool-definition name.

    Conversation history items carry the original (short) tool name while the
    tool definitions sent upstream are flattened (``functions__exec``). Models
    echo the history name, so provider serializers rewrite history call names
    to the flattened form to keep the conversation internally consistent.

    Names already flattened (present as keys) are returned unchanged. A short
    name is flattened only when exactly one mapping ends with it — ambiguous
    or unknown names are left as-is.
    """
    if not namespace_map or not name or name in namespace_map:
        return name
    matches = [
        flat for flat, parts in namespace_map.items() if len(parts) == 2 and parts[1] == name
    ]
    return matches[0] if len(matches) == 1 else name


def restore_tool_name(
    namespace_map: dict[str, list[str]] | None, name: str
) -> tuple[str, str | None]:
    """Restore a (possibly short) tool name to its (name, namespace) pair.

    Response-side counterpart of ``flatten_history_tool_name``: providers may
    echo either the flattened definition name (``mcp__github__create_issue``)
    or the original short name (``create_issue``) when they emit a tool call.
    Exact matches on the flattened key are restored first; otherwise a short
    name is restored only when exactly one mapping ends with it (ambiguous or
    unknown names are returned unchanged with no namespace).
    """
    if not namespace_map or not name:
        return name, None
    if name in namespace_map:
        parts = namespace_map[name]
        if parts:
            return parts[-1], (parts[0] if len(parts) > 1 else None)
    else:
        matches = [
            parts for parts in namespace_map.values() if len(parts) == 2 and parts[1] == name
        ]
        if len(matches) == 1:
            return matches[0][1], matches[0][0]
    return name, None
