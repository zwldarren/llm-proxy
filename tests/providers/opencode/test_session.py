"""Stable upstream-session resolution for OpenCode: header, metadata, content, fallback."""

import pytest

from llm_proxy.models import (
    AnthropicSpecificParams,
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
    OpenAISpecificParams,
    TextBlock,
)
from llm_proxy.providers.opencode.identity import mint_session_id, session_id_from_seed
from llm_proxy.providers.opencode.session import resolve_upstream_session_id
from llm_proxy.providers.session_headers import (
    capture_session_headers,
    clear_session_headers,
)


@pytest.fixture(autouse=True)
def _clean_contextvar():
    clear_session_headers()
    yield
    clear_session_headers()


def make_request(
    *messages: tuple[str, str],
    user: str | None = None,
    params: GenerationParams | None = None,
    **extra,
) -> InternalRequest:
    request = InternalRequest(
        model="glm-5.3",
        conversation=ConversationContext(
            messages=[Message(role=role, content=[TextBlock(text=text)]) for role, text in messages]
        ),
        params=params or GenerationParams(),
        extra=extra,
    )
    request.user = user
    return request


class TestSessionIdShape:
    def test_seed_is_deterministic_and_cli_shaped(self):
        first = session_id_from_seed("conversation-one")
        second = session_id_from_seed("conversation-one")

        assert first == second
        assert first.startswith("ses_")
        assert len(first) == 4 + 12 + 14
        assert all(c.isalnum() for c in first[4:])

    def test_different_seeds_differ(self):
        assert session_id_from_seed("one") != session_id_from_seed("two")

    def test_minted_ids_are_fresh(self):
        assert mint_session_id() != mint_session_id()


class TestResolution:
    def test_client_header_wins(self):
        capture_session_headers({"x-session-id": "client-session"})
        request = make_request(("user", "hello"))

        assert resolve_upstream_session_id(request) == "client-session"

    def test_opencode_session_header_is_authoritative(self):
        capture_session_headers(
            {"x-opencode-session": "ses_client", "x-claude-code-session-id": "claude-1"}
        )
        request = make_request(("user", "hello"))

        assert resolve_upstream_session_id(request) == "ses_client"

    def test_explicit_metadata_session_wins_over_prompt_cache_key(self):
        """An explicit id is more trustworthy than a possibly per-request key."""
        request = make_request(
            ("user", "hello"),
            params=GenerationParams(
                openai=OpenAISpecificParams(
                    prompt_cache_key="per-request-key",
                    metadata={"session_id": "explicit-session"},
                )
            ),
        )

        assert resolve_upstream_session_id(request) == "explicit-session"

    def test_prompt_cache_key_wins_over_content(self):
        request = make_request(
            ("user", "hello"),
            params=GenerationParams(openai=OpenAISpecificParams(prompt_cache_key="cache-key")),
        )

        assert resolve_upstream_session_id(request) == "cache-key"

    def test_anthropic_metadata_session_wins_over_content(self):
        request = make_request(
            ("user", "hello"),
            params=GenerationParams(
                anthropic=AnthropicSpecificParams(
                    metadata={"user_id": "user_1_account_2_session_claude-session"}
                )
            ),
        )

        assert resolve_upstream_session_id(request) == "claude-session"

    def test_openai_metadata_session_is_used(self):
        request = make_request(
            ("user", "hello"),
            params=GenerationParams(
                openai=OpenAISpecificParams(metadata={"session_id": "metadata-session"})
            ),
        )

        assert resolve_upstream_session_id(request) == "metadata-session"

    def test_extra_session_id_is_used(self):
        request = make_request(("user", "hello"), session_id="extra-session")

        assert resolve_upstream_session_id(request) == "extra-session"

    def test_responses_conversation_id_is_used(self):
        request = make_request(("user", "hello"), conversation="conv-from-body")

        assert resolve_upstream_session_id(request) == "conv-from-body"

    def test_responses_conversation_object_id_is_used(self):
        request = make_request(("user", "hello"), conversation={"id": "conv-object-id"})

        assert resolve_upstream_session_id(request) == "conv-object-id"

    def test_content_fingerprint_is_stable_across_turns(self):
        first = make_request(("user", "build the thing"))
        follow_up = make_request(
            ("user", "build the thing"),
            ("assistant", "working on it"),
            ("user", "now test it"),
        )

        assert resolve_upstream_session_id(first) == resolve_upstream_session_id(follow_up)

    def test_content_fingerprint_differs_per_conversation(self):
        first = make_request(("user", "build the thing"))
        second = make_request(("user", "review the thing"))

        assert resolve_upstream_session_id(first) != resolve_upstream_session_id(second)

    def test_user_field_does_not_rotate_the_session(self):
        """Some clients mint ``user`` per request; the conversation id must not follow."""
        first = make_request(("user", "hi"), user="request-1")
        second = make_request(("user", "hi"), user="request-2")

        assert resolve_upstream_session_id(first) == resolve_upstream_session_id(second)

    def test_copilot_preamble_message_is_skipped(self):
        preamble = "<environment_info>same for every chat</environment_info>"
        first = make_request(("user", preamble), ("user", "first real prompt"))
        second = make_request(("user", preamble), ("user", "second real prompt"))

        assert resolve_upstream_session_id(first) != resolve_upstream_session_id(second)

    def test_copilot_preamble_in_the_same_message_does_not_rotate(self):
        """Copilot puts the preamble and the prompt in one message."""
        prompt = "<environment_info>same for every chat</environment_info>\nbuild the thing"
        first = make_request(("user", prompt))
        follow_up = make_request(
            ("user", prompt),
            ("assistant", "working on it"),
            ("user", "now test it"),
        )

        assert resolve_upstream_session_id(first) == resolve_upstream_session_id(follow_up)

    def test_all_boilerplate_still_yields_a_stable_id(self):
        request = make_request(("user", "<context>only preamble</context>"))

        assert resolve_upstream_session_id(request) == resolve_upstream_session_id(
            make_request(("user", "<context>only preamble</context>"))
        )

    def test_per_request_client_id_is_not_used_as_the_session(self):
        """``x-client-request-id`` changes per request; feature on the content instead."""
        capture_session_headers({"x-client-request-id": "per-request-nonce"})
        request = make_request(("user", "build the thing"))

        assert resolve_upstream_session_id(request) != "per-request-nonce"
        assert resolve_upstream_session_id(request) == resolve_upstream_session_id(
            make_request(("user", "build the thing"))
        )

    def test_ambiguous_connection_header_is_not_used_as_the_session(self):
        capture_session_headers({"acp-connection-id": "one-per-editor-connection"})
        request = make_request(("user", "build the thing"))

        assert resolve_upstream_session_id(request) != "one-per-editor-connection"

    def test_no_signal_mints_a_session(self):
        request = make_request()

        first = resolve_upstream_session_id(request)
        second = resolve_upstream_session_id(request)

        assert first.startswith("ses_")
        assert second.startswith("ses_")
        assert first != second
