"""MCP security policy definition and loading."""

import ipaddress
import logging
import socket
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class McpSecurityPolicy(BaseModel):
    """Central policy object for MCP hardening.

    Defaults implement a secure-by-default posture:
    - No stdio command is allowed until explicitly listed.
    - No custom environment variable is allowed until explicitly listed.
    - Dangerous commands and environment keys are always blocked.
    - Only http/https URLs are allowed, and private/reserved IP ranges are blocked.

    The ``allowed_commands`` list supports two forms:
    - Broad command name, e.g. ``npx`` allows any invocation starting with ``npx``.
    - Exact invocation, e.g. ``npx mcp-searxng`` allows only ``npx`` followed by the
      given first positional argument. This is useful for package runners that can
      execute arbitrary remote code: admins can whitelist a specific package instead
      of the whole runner.
    """

    # stdio
    allowed_commands: list[str] = Field(default_factory=list)
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
        ]
    )
    allowed_env_keys: list[str] = Field(default_factory=list)
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
        ]
    )

    # streamableHttp
    allowed_url_schemes: list[str] = Field(default_factory=lambda: ["http", "https"])
    blocked_url_hosts: list[str] = Field(default_factory=list)
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
        ]
    )

    # global switches
    require_key_mcp_permissions: bool = True

    model_config = ConfigDict(extra="ignore")

    def is_allowed_command(self, command: str, args: list[str] | None = None) -> bool:
        """Return True if the command is permitted by the policy.

        The allowlist supports two forms:

        - Broad: ``npx`` permits any invocation whose base command is ``npx``.
        - Exact: ``npx mcp-searxng`` permits only ``npx`` followed by the given
          first positional argument. Leading ``-`` or ``--`` flags in ``args``
          are ignored when building the invocation key for exact matching.
        """
        base = command.strip().lower()
        name = base.split("/")[-1]
        if name in {c.strip().lower() for c in self.blocked_commands}:
            return False
        if not self.allowed_commands:
            return False

        allowed_broad = {c.strip().lower() for c in self.allowed_commands if " " not in c.strip()}
        allowed_exact = {c.strip().lower() for c in self.allowed_commands if " " in c.strip()}

        if name in allowed_broad:
            return True

        if not args or not allowed_exact:
            return False

        # Build invocation key from base command and first positional arg.
        positional = [a.strip().lower() for a in args if not a.startswith("-")]
        if not positional:
            return False
        invocation = f"{name} {positional[0]}"
        return invocation in allowed_exact

    def is_allowed_env_key(self, key: str) -> bool:
        """Return True if the env key may be set by a server config."""
        upper = key.upper()
        if self.is_blocked_env_key(key):
            return False
        if not self.allowed_env_keys:
            return False
        return upper in {k.upper() for k in self.allowed_env_keys}

    def is_blocked_env_key(self, key: str) -> bool:
        """Return True if the env key is explicitly blocked."""
        return key.upper() in {k.upper() for k in self.blocked_env_keys}

    def _resolve_and_check_blocked(self, host: str) -> bool:
        """Resolve hostname and check if any resolved IP is in blocked ranges.

        This mitigates DNS-rebinding attacks where a public-looking hostname
        is later mapped to an internal address.
        """
        try:
            addrinfo = socket.getaddrinfo(host, None)
        except OSError:
            return False

        seen: set[str] = set()
        for _, _, _, _, sockaddr in addrinfo:
            ip = str(sockaddr[0])
            if ip in seen:
                continue
            seen.add(ip)
            try:
                resolved_addr = ipaddress.ip_address(ip)
            except ValueError:
                continue
            for network in self.blocked_url_ips:
                if resolved_addr in ipaddress.ip_network(network, strict=False):
                    return True
        return False

    def is_blocked_url(self, url: str) -> bool:
        """Return True if the URL is blocked by scheme, host, or IP rules."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme not in {s.lower() for s in self.allowed_url_schemes}:
            return True
        if not parsed.hostname:
            return True
        host = parsed.hostname.lower()
        if host in {h.lower() for h in self.blocked_url_hosts}:
            return True
        try:
            addr = ipaddress.ip_address(host)
            for network in self.blocked_url_ips:
                if addr in ipaddress.ip_network(network, strict=False):
                    return True
        except ValueError:
            pass

        return self._resolve_and_check_blocked(host)

    @classmethod
    def from_defaults(cls) -> McpSecurityPolicy:
        """Build a policy from code defaults.

        All policy fields (list-based rules and the
        ``require_key_mcp_permissions`` switch) are UI-managed via the
        ``mcp_security_policy`` database config record; use
        :meth:`from_config` to include them.
        """
        return cls()

    @classmethod
    def _apply_db_config(cls, policy: McpSecurityPolicy, db_value: Any) -> McpSecurityPolicy:
        """Override list-based fields on ``policy`` from a DB config record.

        ``db_value`` may be a ``ServerConfigRecord`` or falsy.
        Returns ``policy`` unchanged if the stored value is malformed.
        """
        if not db_value:
            return policy
        stored_value = db_value if isinstance(db_value, dict) else db_value.value
        if not isinstance(stored_value, dict):
            return policy
        try:
            stored = cls.model_validate(stored_value)
        except ValidationError:
            logger.warning(
                "Failed to validate stored MCP security policy, falling back to defaults"
            )
            return policy
        # Override DB-managed fields from the stored record.
        # NOTE: Keep this tuple in sync with all UI-managed fields in McpSecurityPolicy
        for field_name in (
            "allowed_commands",
            "blocked_commands",
            "allowed_env_keys",
            "blocked_env_keys",
            "blocked_url_hosts",
            "blocked_url_ips",
            "require_key_mcp_permissions",
        ):
            setattr(policy, field_name, getattr(stored, field_name))
        return policy

    @classmethod
    async def from_config(cls) -> McpSecurityPolicy:
        """Build a policy from the database ``mcp_security_policy`` config record.

        All policy fields (allowed/blocked commands, env keys, URL rules, and
        the ``require_key_mcp_permissions`` switch) are UI-managed; falls back
        to code defaults when no DB record exists or the DB is unavailable.
        """
        from llm_proxy.database import ConfigRepository, get_async_session_context

        policy = cls.from_defaults()

        try:
            async with get_async_session_context() as session:
                repo = ConfigRepository(session)
                db_value = await repo.get_server_config("mcp_security_policy")
        except Exception:
            logger.warning(
                "Failed to load MCP security policy from database, falling back to defaults",
                exc_info=True,
            )
            db_value = None

        return cls._apply_db_config(policy, db_value)
