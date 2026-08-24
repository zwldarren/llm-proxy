"""Tests for McpSecurityPolicy."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.mcp.security.policy import McpSecurityPolicy


class TestDefaults:
    """Default policy is secure-by-default: deny all unless explicitly allowed."""

    def test_defaults_reject_all_stdio_commands(self) -> None:
        policy = McpSecurityPolicy()
        # Default allowlist is empty, so all stdio commands are rejected.
        assert policy.is_allowed_command("npx") is False
        assert policy.is_allowed_command("uvx") is False
        assert policy.is_allowed_command("bunx") is False
        # Blocked commands are rejected even if somehow listed.
        assert policy.is_allowed_command("bash") is False
        assert policy.is_allowed_command("node") is False

    def test_defaults_reject_all_env_keys(self) -> None:
        policy = McpSecurityPolicy()
        assert policy.is_allowed_env_key("GITHUB_TOKEN") is False
        assert policy.is_allowed_env_key("OPENAI_API_KEY") is False
        assert policy.is_allowed_env_key("PATH") is False

    def test_defaults_block_dangerous_commands(self) -> None:
        policy = McpSecurityPolicy(allowed_commands=["bash", "python", "node"])
        # Even if allowed, dangerous commands are blocked
        assert policy.is_allowed_command("bash") is False
        assert policy.is_allowed_command("python") is False
        assert policy.is_allowed_command("node") is False

    def test_defaults_block_dangerous_env_keys(self) -> None:
        policy = McpSecurityPolicy(allowed_env_keys=["PATH", "HOME", "LD_PRELOAD"])
        assert policy.is_allowed_env_key("PATH") is False
        assert policy.is_allowed_env_key("HOME") is False
        assert policy.is_allowed_env_key("LD_PRELOAD") is False


class TestAllowedCommands:
    """Allowed commands are permitted when not in the blocked list."""

    def test_allowed_command_works(self) -> None:
        policy = McpSecurityPolicy(allowed_commands=["npx", "uvx"])
        assert policy.is_allowed_command("npx") is True
        assert policy.is_allowed_command("uvx") is True

    def test_allowed_command_case_insensitive(self) -> None:
        policy = McpSecurityPolicy(allowed_commands=["NPX"])
        assert policy.is_allowed_command("npx") is True
        assert policy.is_allowed_command("Npx") is True

    def test_allowed_command_strips_path(self) -> None:
        policy = McpSecurityPolicy(allowed_commands=["npx"])
        assert policy.is_allowed_command("/usr/local/bin/npx") is True
        assert policy.is_allowed_command("./node_modules/.bin/npx") is True

    def test_blocked_command_overrides_allowlist(self) -> None:
        policy = McpSecurityPolicy(
            allowed_commands=["bash", "npx"],
            blocked_commands=["bash"],
        )
        assert policy.is_allowed_command("npx") is True
        assert policy.is_allowed_command("bash") is False

    def test_empty_allowed_commands_rejects_all(self) -> None:
        policy = McpSecurityPolicy(allowed_commands=[])
        assert policy.is_allowed_command("npx") is False
        assert policy.is_allowed_command("uvx") is False

    def test_blocked_command_not_in_defaults(self) -> None:
        """A command not in the default blocked list is allowed if in allowed_commands."""
        policy = McpSecurityPolicy(allowed_commands=["npx"])
        assert policy.is_allowed_command("npx") is True
        assert policy.is_allowed_command("uvx") is False

    def test_exact_invocation_allows_specific_package(self) -> None:
        """Exact entries like 'npx mcp-searxng' only permit that invocation."""
        policy = McpSecurityPolicy(allowed_commands=["npx mcp-searxng"])
        assert policy.is_allowed_command("npx", ["mcp-searxng"]) is True
        assert policy.is_allowed_command("npx", ["-y", "mcp-searxng"]) is True
        assert policy.is_allowed_command("npx", ["other-package"]) is False
        assert policy.is_allowed_command("npx") is False

    def test_broad_and_exact_invocation_can_coexist(self) -> None:
        """Broad entries allow any invocation; exact ones allow only that package."""
        policy = McpSecurityPolicy(allowed_commands=["uvx", "npx mcp-searxng"])
        assert policy.is_allowed_command("uvx", ["anything"]) is True
        assert policy.is_allowed_command("npx", ["mcp-searxng"]) is True
        assert policy.is_allowed_command("npx", ["other"]) is False


class TestAllowedEnvKeys:
    """Allowed env keys are permitted when not in the blocked list."""

    def test_allowed_env_key_works(self) -> None:
        policy = McpSecurityPolicy(allowed_env_keys=["GITHUB_TOKEN", "OPENAI_API_KEY"])
        assert policy.is_allowed_env_key("GITHUB_TOKEN") is True
        assert policy.is_allowed_env_key("OPENAI_API_KEY") is True

    def test_allowed_env_key_case_insensitive(self) -> None:
        policy = McpSecurityPolicy(allowed_env_keys=["github_token"])
        assert policy.is_allowed_env_key("GITHUB_TOKEN") is True
        assert policy.is_allowed_env_key("GitHub_Token") is True

    def test_blocked_env_key_overrides_allowlist(self) -> None:
        policy = McpSecurityPolicy(
            allowed_env_keys=["PATH", "GITHUB_TOKEN"],
            blocked_env_keys=["PATH"],
        )
        assert policy.is_allowed_env_key("GITHUB_TOKEN") is True
        assert policy.is_allowed_env_key("PATH") is False

    def test_empty_allowed_env_keys_rejects_all(self) -> None:
        policy = McpSecurityPolicy()
        assert policy.is_allowed_env_key("GITHUB_TOKEN") is False
        assert policy.is_allowed_env_key("MY_CUSTOM_KEY") is False


class TestBlockedUrls:
    """URL blocking by scheme, host, and IP range."""

    def test_allowed_schemes(self) -> None:
        policy = McpSecurityPolicy()
        assert policy.is_blocked_url("http://example.com/api") is False
        assert policy.is_blocked_url("https://example.com/api") is False

    def test_blocked_scheme(self) -> None:
        policy = McpSecurityPolicy()
        assert policy.is_blocked_url("ftp://example.com") is True
        assert policy.is_blocked_url("file:///etc/passwd") is True
        assert policy.is_blocked_url("data:text/plain,hello") is True

    def test_blocked_private_ip(self) -> None:
        policy = McpSecurityPolicy()
        assert policy.is_blocked_url("http://127.0.0.1:8080") is True
        assert policy.is_blocked_url("http://192.168.1.1") is True
        assert policy.is_blocked_url("http://10.0.0.5") is True
        assert policy.is_blocked_url("http://172.16.0.10") is True

    def test_blocked_metadata_ip(self) -> None:
        policy = McpSecurityPolicy()
        assert policy.is_blocked_url("http://169.254.169.254/latest/meta-data") is True

    def test_blocked_hostname(self) -> None:
        policy = McpSecurityPolicy(blocked_url_hosts=["internal.example.com"])
        assert policy.is_blocked_url("http://internal.example.com/config") is True
        assert policy.is_blocked_url("http://public.example.com") is False

    def test_blocked_localhost_via_dns_resolution(self) -> None:
        """localhost resolves to 127.0.0.1 and is blocked via DNS rebinding guard."""
        policy = McpSecurityPolicy()
        assert policy.is_blocked_url("http://localhost:8080/mcp") is True

    def test_blocked_rebinding_to_private_ip(self) -> None:
        """A public-looking hostname that resolves to a private IP is blocked."""
        policy = McpSecurityPolicy()
        with patch(
            "llm_proxy.mcp.security.policy.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("10.0.0.5", 0))],
        ):
            assert policy.is_blocked_url("http://public-looking.example.com/mcp") is True

    def test_allowed_when_dns_resolution_fails(self) -> None:
        """Unresolvable hostnames are permitted so tests work without DNS."""
        policy = McpSecurityPolicy()
        with patch(
            "llm_proxy.mcp.security.policy.socket.getaddrinfo",
            side_effect=OSError("NXDOMAIN"),
        ):
            assert policy.is_blocked_url("http://fake-public.example.com/mcp") is False

    def test_public_ip_allowed(self) -> None:
        policy = McpSecurityPolicy()
        assert policy.is_blocked_url("http://93.184.216.34") is False  # example.com
        assert policy.is_blocked_url("https://8.8.8.8") is False

    def test_no_hostname_is_blocked(self) -> None:
        policy = McpSecurityPolicy()
        assert policy.is_blocked_url("http:///path") is True


class TestFromDefaults:
    """Factory method from_defaults builds policy from code defaults."""

    def test_from_defaults(self) -> None:
        policy = McpSecurityPolicy.from_defaults()
        assert isinstance(policy, McpSecurityPolicy)
        # No stdio commands are allowed by default.
        assert policy.is_allowed_command("npx") is False
        assert policy.is_allowed_command("uvx") is False
        assert policy.is_allowed_command("bunx") is False
        assert policy.is_allowed_env_key("GITHUB_TOKEN") is False
        # Secure-by-default switches and blocklists.
        assert policy.require_key_mcp_permissions is True
        assert policy.is_allowed_command("bash") is False
        assert policy.is_allowed_command("node") is False


class TestFromConfig:
    """Factory method from_config merges code defaults with DB config."""

    def _mock_session_context(self, session: AsyncMock) -> AsyncMock:
        """Build a patched async context manager for get_async_session_context."""
        async_cm = AsyncMock()
        async_cm.__aenter__ = AsyncMock(return_value=session)
        async_cm.__aexit__ = AsyncMock(return_value=None)
        return patch(
            "llm_proxy.database.get_async_session_context",
            return_value=async_cm,
        )

    @pytest.mark.asyncio
    async def test_from_config_no_db(self) -> None:
        """When no DB config exists, code defaults are used."""
        mock_session = AsyncMock()
        mock_repo = AsyncMock()
        mock_repo.get_server_config.return_value = None

        with (
            self._mock_session_context(mock_session),
            patch("llm_proxy.database.ConfigRepository", return_value=mock_repo),
        ):
            policy = await McpSecurityPolicy.from_config()
            assert isinstance(policy, McpSecurityPolicy)
            # No DB config means secure defaults (no commands allowed).
            assert policy.is_allowed_command("npx") is False
            assert policy.is_allowed_command("uvx") is False
            assert policy.is_allowed_command("bunx") is False

    @pytest.mark.asyncio
    async def test_from_config_with_db_override(self) -> None:
        """DB config fills in list-based fields and the permission switch."""
        mock_session = AsyncMock()
        mock_repo = AsyncMock()
        mock_repo.get_server_config.return_value = type(
            "obj",
            (),
            {"value": {"allowed_commands": ["npx", "uvx"], "require_key_mcp_permissions": False}},
        )()

        with (
            self._mock_session_context(mock_session),
            patch("llm_proxy.database.ConfigRepository", return_value=mock_repo),
        ):
            policy = await McpSecurityPolicy.from_config()
            assert policy.is_allowed_command("npx") is True
            assert policy.is_allowed_command("uvx") is True
            assert policy.require_key_mcp_permissions is False

    @pytest.mark.asyncio
    async def test_from_config_db_error_fallback(self) -> None:
        """DB errors fall back gracefully to the code-default policy."""
        mock_session = AsyncMock()
        mock_repo = AsyncMock()
        mock_repo.get_server_config.side_effect = Exception("DB unavailable")

        with (
            self._mock_session_context(mock_session),
            patch("llm_proxy.database.ConfigRepository", return_value=mock_repo),
        ):
            policy = await McpSecurityPolicy.from_config()
            assert isinstance(policy, McpSecurityPolicy)
            # Falls back to secure defaults
            assert policy.is_allowed_command("npx") is False
            assert policy.is_allowed_command("uvx") is False
            assert policy.is_allowed_command("bunx") is False

    @pytest.mark.asyncio
    async def test_from_config_works_without_manager(self) -> None:
        """Works with code defaults when no DB manager is passed."""
        mock_session = AsyncMock()
        mock_repo = AsyncMock()
        mock_repo.get_server_config.return_value = None

        with (
            self._mock_session_context(mock_session),
            patch("llm_proxy.database.ConfigRepository", return_value=mock_repo),
        ):
            policy = await McpSecurityPolicy.from_config()
            assert isinstance(policy, McpSecurityPolicy)
