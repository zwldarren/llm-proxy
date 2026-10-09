"""MCP security policy model for the admin server-settings endpoint.

Canonical model for the ``mcp_security_policy`` server_config key: the
UI-managed list-based policy fields (allowed/blocked commands, env keys, URL
rules) plus the permission switch. The runtime policy object
(``llm_proxy.mcp.security.policy.McpSecurityPolicy``) subclasses this model to
add runtime-only fields (e.g. ``allowed_url_schemes``), so the shared fields
have exactly one source of defaults (see ADR-0026).
"""

from pydantic import BaseModel, Field


class McpSecurityPolicyConfig(BaseModel):
    """MCP security policy configuration stored in the database."""

    require_key_mcp_permissions: bool = Field(
        default=True,
        description="Require explicit MCP permissions on API keys for MCP access",
    )

    allowed_commands: list[str] = Field(
        default_factory=list,
        description="Commands permitted for stdio MCP servers",
    )
    blocked_commands: list[str] = Field(
        default_factory=lambda: [
            "bash",
            "sh",
            "zsh",
            "cmd.exe",
            "powershell.exe",
            "python",
            "python3",
            "node",
            "perl",
            "ruby",
        ],
        description="Commands always blocked even if in allowed list",
    )
    allowed_env_keys: list[str] = Field(
        default_factory=list,
        description="Environment variable keys permitted for MCP servers",
    )
    blocked_env_keys: list[str] = Field(
        default_factory=lambda: [
            "PATH",
            "LD_PRELOAD",
            "DYLD_INSERT_LIBRARIES",
            "PYTHONPATH",
            "NODE_OPTIONS",
            "SHELL",
            "HOME",
            "USER",
        ],
        description="Environment variable keys always blocked",
    )
    blocked_url_hosts: list[str] = Field(
        default_factory=list,
        description="URL hosts blocked for streamableHttp MCP servers",
    )
    blocked_url_ips: list[str] = Field(
        default_factory=lambda: [
            "127.0.0.0/8",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "169.254.169.254/32",
            "100.64.0.0/10",
            "::1/128",
            "fc00::/7",
            "fe80::/10",
            "::ffff:0:0/96",
        ],
        description="IP ranges blocked for streamableHttp MCP servers",
    )
