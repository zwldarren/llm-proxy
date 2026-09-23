"""Tests for McpSecurityValidator."""

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


def test_unlisted_command_error_points_to_settings() -> None:
    """The error should tell the operator where to allowlist the command."""
    policy = McpSecurityPolicy(allowed_commands=["npx"])
    validator = McpSecurityValidator(policy)
    with pytest.raises(MCPSecurityError) as exc_info:
        validator.validate_stdio_command("uvx", ["-y", "foo"])
    message = str(exc_info.value)
    assert "uvx" in message
    assert "Allowed Commands" in message
    assert "npx" in message


def test_empty_allowlist_error_explains_deny_default() -> None:
    """With no allowlist, the error should explain the deny-by-default posture."""
    policy = McpSecurityPolicy()
    validator = McpSecurityValidator(policy)
    with pytest.raises(MCPSecurityError) as exc_info:
        validator.validate_stdio_command("uvx", ["-y", "foo"])
    message = str(exc_info.value)
    assert "not allowed" in message
    assert "No stdio commands are allowed yet" in message


def test_blocked_command_error_mentions_blocked() -> None:
    """Hard-blocked commands should not be described as merely unlisted."""
    policy = McpSecurityPolicy(allowed_commands=["bash"])
    validator = McpSecurityValidator(policy)
    with pytest.raises(MCPSecurityError) as exc_info:
        validator.validate_stdio_command("bash", ["-c", "echo"])
    message = str(exc_info.value).lower()
    assert "blocked" in message
    assert "not allowed" in message


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
