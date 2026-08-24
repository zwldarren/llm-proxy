"""Tests for McpSecurityValidator."""

from __future__ import annotations

import pytest

from llm_proxy.core.exceptions import MCPSecurityError
from llm_proxy.mcp.security.policy import McpSecurityPolicy
from llm_proxy.mcp.security.validator import McpSecurityValidator


def test_rejects_blocked_command() -> None:
    policy = McpSecurityPolicy(allowed_commands=["npx"])
    validator = McpSecurityValidator(policy)
    with pytest.raises(MCPSecurityError, match="not allowed"):
        validator.validate_stdio_command("bash", ["-c", "evil"])


def test_rejects_command_not_in_allowlist() -> None:
    policy = McpSecurityPolicy(allowed_commands=["npx"])
    validator = McpSecurityValidator(policy)
    with pytest.raises(MCPSecurityError, match="not allowed"):
        validator.validate_stdio_command("uvx", ["-y", "foo"])


def test_rejects_wrong_exact_invocation() -> None:
    policy = McpSecurityPolicy(allowed_commands=["npx mcp-searxng"])
    validator = McpSecurityValidator(policy)
    with pytest.raises(MCPSecurityError, match="not allowed"):
        validator.validate_stdio_command("npx", ["-y", "other-pkg"])


def test_filters_env_to_allowlist() -> None:
    policy = McpSecurityPolicy(allowed_commands=["npx"], allowed_env_keys=["GITHUB_TOKEN"])
    validator = McpSecurityValidator(policy)
    filtered = validator.validate_stdio_env({"GITHUB_TOKEN": "abc", "PATH": "/usr/bin"})
    assert filtered == {"GITHUB_TOKEN": "abc"}


def test_rejects_blocked_url() -> None:
    policy = McpSecurityPolicy()
    validator = McpSecurityValidator(policy)
    with pytest.raises(MCPSecurityError, match="blocked"):
        validator.validate_streamable_http_url("http://127.0.0.1:8080/mcp")


def test_accepts_allowed_command() -> None:
    policy = McpSecurityPolicy(allowed_commands=["npx"])
    validator = McpSecurityValidator(policy)
    validator.validate_stdio_command("npx", ["-y", "@modelcontextprotocol/server-filesystem"])


def test_accepts_exact_invocation() -> None:
    policy = McpSecurityPolicy(allowed_commands=["npx mcp-searxng"])
    validator = McpSecurityValidator(policy)
    validator.validate_stdio_command("npx", ["-y", "mcp-searxng"])


def test_rejects_dangerous_shell_metacharacters_in_args() -> None:
    policy = McpSecurityPolicy(allowed_commands=["npx"])
    validator = McpSecurityValidator(policy)
    with pytest.raises(MCPSecurityError, match="shell metacharacters"):
        validator.validate_stdio_command("npx", ["-c", "foo; bar"])


def test_accepts_allowed_url() -> None:
    policy = McpSecurityPolicy()
    validator = McpSecurityValidator(policy)
    validator.validate_streamable_http_url("https://example.com/mcp")
