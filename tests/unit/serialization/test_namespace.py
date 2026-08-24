"""Tests for NamespaceMapping utility."""

from llm_proxy.serialization.responses_toolkit.namespace import (
    NamespaceMapping,
    restore_tool_name,
)


class TestNamespaceMapping:
    def test_flatten_records_bidirectional(self):
        nm = NamespaceMapping()
        flat = nm.flatten("mcp", "create_pr")
        assert flat == "mcp__create_pr"
        assert restore_tool_name(nm.to_dict(), flat) == ("create_pr", "mcp")

    def test_restore_unknown_returns_none(self):
        assert restore_tool_name(NamespaceMapping().to_dict(), "x") == ("x", None)

    def test_truncation_long_name(self):
        nm = NamespaceMapping()
        flat = nm.flatten("a" * 60, "tool")
        assert len(flat) <= 64

    def test_build_from_namespace_dicts(self):
        nm = NamespaceMapping.build(
            [
                {
                    "type": "namespace",
                    "name": "mcp__gh",
                    "tools": [
                        {"type": "function", "name": "list_issues"},
                        {"type": "function", "name": "create_pr"},
                    ],
                }
            ]
        )
        assert len(nm.flat_to_original) == 2
        assert restore_tool_name(nm.to_dict(), "mcp__gh__list_issues") == (
            "list_issues",
            "mcp__gh",
        )

    def test_serialization_roundtrip(self):
        nm = NamespaceMapping()
        nm.flatten("mcp", "a")
        d = nm.to_dict()
        r = NamespaceMapping.from_dict(d)
        assert restore_tool_name(r.to_dict(), "mcp__a") == ("a", "mcp")


class TestRestoreToolName:
    _MAP = {
        "functions__exec": ["functions", "exec"],
        "mcp__github__create_issue": ["mcp__github", "create_issue"],
    }

    def test_exact_flat_name_restored(self):
        assert restore_tool_name(self._MAP, "mcp__github__create_issue") == (
            "create_issue",
            "mcp__github",
        )

    def test_short_name_restored_when_unique(self):
        # Models often echo the short history name instead of the flattened
        # definition name; the single matching namespace restores it.
        assert restore_tool_name(self._MAP, "create_issue") == (
            "create_issue",
            "mcp__github",
        )

    def test_default_namespace_short_name_restored(self):
        assert restore_tool_name(self._MAP, "exec") == ("exec", "functions")

    def test_ambiguous_short_name_untouched(self):
        amb = {"a__exec": ["a", "exec"], "b__exec": ["b", "exec"]}
        assert restore_tool_name(amb, "exec") == ("exec", None)

    def test_unknown_name_untouched(self):
        assert restore_tool_name(self._MAP, "other_tool") == ("other_tool", None)

    def test_none_map_untouched(self):
        assert restore_tool_name(None, "exec") == ("exec", None)

    def test_empty_name_untouched(self):
        assert restore_tool_name(self._MAP, "") == ("", None)
