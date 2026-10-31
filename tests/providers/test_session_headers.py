"""Client upstream-session capture: priority, validation, and forwarding set."""

import pytest

from llm_proxy.providers.session_headers import (
    MAX_TURN_METADATA_LENGTH,
    capture_session_headers,
    clear_session_headers,
    get_client_upstream_session_headers,
    get_client_upstream_session_id,
    normalize_session_id,
)


@pytest.fixture(autouse=True)
def _clean_contextvar():
    clear_session_headers()
    yield
    clear_session_headers()


class TestPriority:
    def test_explicit_opencode_session_wins(self):
        capture_session_headers(
            {
                "x-opencode-session": "ses_client",
                "session_id": "codex-session",
                "x-claude-code-session-id": "claude-session",
            }
        )

        assert get_client_upstream_session_id() == "ses_client"

    @pytest.mark.parametrize(
        ("headers", "expected"),
        [
            ({"x-claude-code-session-id": "claude-1"}, "claude-1"),
            ({"session_id": "codex-1"}, "codex-1"),
            ({"Session-Id": "codex-2"}, "codex-2"),
            ({"thread-id": "thread-1"}, "thread-1"),
            ({"thread_id": "thread-2"}, "thread-2"),
            ({"x-deepseek-harness-session-id": "dsh-1"}, "dsh-1"),
            ({"x-session-affinity": "pi-1"}, "pi-1"),
            ({"n-session": "pi-native-1"}, "pi-native-1"),
            ({"x-session-id": "generic-1"}, "generic-1"),
            ({"x-conversation-id": "conv-1"}, "conv-1"),
            ({"conversation_id": "conv-2"}, "conv-2"),
            ({"conversation-id": "conv-3"}, "conv-3"),
        ],
    )
    def test_each_native_header_is_recognized(self, headers, expected):
        capture_session_headers(headers)

        assert get_client_upstream_session_id() == expected

    @pytest.mark.parametrize(
        "headers",
        [
            {"x-client-request-id": "per-request-nonce"},
            {"acp-connection-id": "one-per-editor-connection"},
            {"x-amp-thread-id": "thread"},
            {"x-omp-session-id": "session"},
        ],
    )
    def test_ambiguous_or_per_request_headers_are_ignored(self, headers):
        """These are not documented as per-conversation affinity keys."""
        capture_session_headers(headers)

        assert get_client_upstream_session_id() is None

    def test_codex_turn_metadata_is_parsed(self):
        capture_session_headers(
            {"x-codex-turn-metadata": '{"session_id": "codex-json", "request": "r1"}'}
        )

        assert get_client_upstream_session_id() == "codex-json"

    def test_invalid_turn_metadata_is_ignored(self):
        capture_session_headers({"x-codex-turn-metadata": "not-json"})

        assert get_client_upstream_session_id() is None

    def test_oversized_turn_metadata_is_ignored(self):
        payload = '{"session_id": "codex-json", "padding": "' + "x" * MAX_TURN_METADATA_LENGTH
        capture_session_headers({"x-codex-turn-metadata": payload})

        assert get_client_upstream_session_id() is None

    def test_invalid_high_priority_value_falls_through(self):
        capture_session_headers(
            {
                "x-opencode-session": "bad\nvalue",
                "x-claude-code-session-id": "claude-good",
            }
        )

        assert get_client_upstream_session_id() == "claude-good"

    def test_case_insensitive_capture(self):
        capture_session_headers({"X-Opencode-Session": "Mixed-Case"})

        assert get_client_upstream_session_id() == "Mixed-Case"


class TestValidation:
    @pytest.mark.parametrize(
        "value",
        [
            None,
            123,
            "",
            "   ",
            "line\nbreak",
            "carriage\rreturn",
            "space inside",
            "comma,separated",
            "x" * 257,
            "é" * 4,
        ],
    )
    def test_rejects_unusable_values(self, value):
        assert normalize_session_id(value) is None

    @pytest.mark.parametrize(
        "value",
        [
            "550e8400-e29b-41d4-a716-446655440000",
            "ses_000000000000AAAAAAAAAAAAAA",
            "opaque.value:with-colons_under_score",
            "base64+/=",
        ],
    )
    def test_accepts_opaque_ids(self, value):
        assert normalize_session_id(value) == value

    def test_trims_surrounding_whitespace(self):
        assert normalize_session_id("  ses_abc  ") == "ses_abc"


class TestForwardingSet:
    def test_forwards_only_the_header_that_carries_the_resolved_id(self):
        capture_session_headers(
            {
                "x-opencode-session": "ses_client",
                "x-claude-code-session-id": "claude-1",
                "session_id": "codex-1",
            }
        )

        # The client sent two conflicting native ids; only the winning one
        # (opencode-session) is resolved, so no native spelling is forwarded.
        assert get_client_upstream_session_headers("ses_client") == {}
        assert get_client_upstream_session_headers("claude-1") == {
            "x-claude-code-session-id": "claude-1"
        }

    def test_forwards_matching_native_aliases(self):
        capture_session_headers(
            {
                "session_id": "codex-1",
                "thread-id": "codex-1",
                "x-client-request-id": "request-1",
                "user-agent": "claude-cli/2.0",
                "authorization": "Bearer secret",
            }
        )

        assert get_client_upstream_session_headers("codex-1") == {
            "session_id": "codex-1",
            "thread-id": "codex-1",
        }

    def test_forwards_codex_turn_metadata_when_it_supplied_the_id(self):
        capture_session_headers(
            {"x-codex-turn-metadata": '{"session_id": "codex-json", "request": "r1"}'}
        )

        assert get_client_upstream_session_headers("codex-json") == {
            "x-codex-turn-metadata": '{"session_id": "codex-json", "request": "r1"}'
        }

    def test_drops_oversized_codex_turn_metadata(self):
        payload = '{"session_id": "codex-json", "padding": "' + "x" * MAX_TURN_METADATA_LENGTH
        capture_session_headers({"x-codex-turn-metadata": payload})

        assert get_client_upstream_session_headers("codex-json") == {}

    def test_drops_non_ascii_codex_turn_metadata(self):
        """A reflected envelope must be header-safe, even when the JSON parses."""
        capture_session_headers(
            {"x-codex-turn-metadata": '{"session_id": "codex-json", "name": "é"}'}
        )

        assert get_client_upstream_session_headers("codex-json") == {}

    def test_does_not_forward_x_opencode_request(self):
        capture_session_headers(
            {"x-opencode-request": "msg_client", "x-opencode-session": "ses_client"}
        )

        assert get_client_upstream_session_headers("ses_client") == {}

    def test_clear_drops_everything(self):
        capture_session_headers({"x-opencode-session": "ses_client"})
        clear_session_headers()

        assert get_client_upstream_session_id() is None
        assert get_client_upstream_session_headers("ses_client") == {}
