"""Tests for the audit-logging middleware helpers.

Covers ``_should_log_audit`` / ``_is_sensitive_read_path`` which decide whether
an admin API request produces an audit log entry, including the read-only GET
coverage for sensitive resources.
"""

from llm_proxy.api.middleware.logging import _is_sensitive_read_path, _should_log_audit


class TestShouldLogAudit:
    def test_non_api_path_never_audited(self):
        assert _should_log_audit("/v1/chat/completions", "POST") is False
        assert _should_log_audit("/health", "GET") is False

    def test_mutating_admin_requests_are_audited(self):
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            assert _should_log_audit("/api/providers", method) is True
            assert _should_log_audit("/api/api-keys/abc", method) is True

    def test_logs_path_is_excluded_to_avoid_feedback_loop(self):
        for method in ("POST", "GET", "DELETE"):
            assert _should_log_audit("/api/logs", method) is False
            assert _should_log_audit("/api/logs/123", method) is False

    def test_sensitive_read_paths_are_audited_on_get(self):
        for path in (
            "/api/providers",
            "/api/providers/openai",
            "/api/models",
            "/api/models/gpt-4",
            "/api/api-keys",
            "/api/api-keys/550e8400-e29b-41d4-a716-446655440000",
            "/api/mcp",
            "/api/users",
            "/api/settings",
            "/api/config",
        ):
            assert _should_log_audit(path, "GET") is True, path

    def test_non_sensitive_read_paths_are_not_audited_on_get(self):
        # Benign / frequently-polled endpoints stay out of the audit log.
        assert _should_log_audit("/api/health", "GET") is False
        assert _should_log_audit("/api/dashboard", "GET") is False
        assert _should_log_audit("/api", "GET") is False

    def test_head_and_options_are_not_audited(self):
        assert _should_log_audit("/api/providers", "HEAD") is False
        assert _should_log_audit("/api/api-keys", "OPTIONS") is False


class TestIsSensitiveReadPath:
    def test_sensitive_prefixes_exact_and_nested(self):
        assert _is_sensitive_read_path("/api/api-keys") is True
        assert _is_sensitive_read_path("/api/api-keys/123") is True
        assert _is_sensitive_read_path("/api/users") is True
        assert _is_sensitive_read_path("/api/settings") is True
        assert _is_sensitive_read_path("/api/config") is True
        assert _is_sensitive_read_path("/api/mcp") is True

    def test_logs_is_not_sensitive(self):
        # /api/logs is excluded from audit entirely (feedback loop), so it is
        # not treated as a sensitive read path either.
        assert _is_sensitive_read_path("/api/logs") is False
        assert _is_sensitive_read_path("/api/logs/123") is False

    def test_unrelated_paths_are_not_sensitive(self):
        assert _is_sensitive_read_path("/api/health") is False
        assert _is_sensitive_read_path("/api/dashboard") is False
        assert _is_sensitive_read_path("/v1/chat/completions") is False
