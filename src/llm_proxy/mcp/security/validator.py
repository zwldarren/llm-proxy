"""MCP security validation logic."""

from llm_proxy.core.exceptions import MCPSecurityError
from llm_proxy.mcp.security.policy import McpSecurityPolicy


class McpSecurityValidator:
    """Validates MCP server configurations against a security policy."""

    def __init__(self, policy: McpSecurityPolicy) -> None:
        self._policy = policy

    def validate_stdio_command(self, command: str, args: list[str]) -> None:
        """Validate the command for a stdio MCP server.

        The allowlist is always enforced. An empty ``allowed_commands`` list
        means no stdio commands are permitted.
        """
        if not command:
            raise MCPSecurityError("stdio command cannot be empty")
        if not self._policy.is_allowed_command(command, args):
            raise MCPSecurityError(f"command '{command}' is not allowed by the MCP security policy")
        # Reject obvious shell escapes in args (belt-and-suspenders).
        combined = " ".join(args).lower()
        dangerous = [";", "&&", "||", "|", "$(", "`", ">", "<"]
        if any(token in combined for token in dangerous):
            raise MCPSecurityError("command args contain disallowed shell metacharacters")

    def validate_stdio_env(self, env: dict[str, str] | None) -> dict[str, str]:
        """Validate and return the allowed subset of environment variables.

        The allowlist is always enforced. An empty ``allowed_env_keys`` list
        means no custom environment variables are permitted.
        """
        if not env:
            return {}
        allowed: dict[str, str] = {}
        for key, value in env.items():
            if self._policy.is_allowed_env_key(key):
                allowed[key] = value
        return allowed

    def validate_streamable_http_url(self, url: str) -> None:
        """Validate the URL for an HTTP MCP server."""
        if not url:
            raise MCPSecurityError("streamableHttp URL cannot be empty")
        if self._policy.is_blocked_url(url):
            raise MCPSecurityError(f"URL '{url}' is blocked by the MCP security policy")
