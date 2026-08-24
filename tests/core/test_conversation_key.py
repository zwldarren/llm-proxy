"""Tests for conversation-key derivation (conversation_key.py).

The derived key is the input to the ``session_sticky`` provider-selection
strategy, so it must be stable across turns of the same conversation for
every protocol request shape the proxy accepts — plain strings, OpenAI
content part lists, Anthropic blocks, and Responses-API items. A None key
degrades session_sticky to random, so any shape that produces None for a
normal chat is a bug.
"""

from llm_proxy.core.conversation_key import conversation_key


def _chat(messages: list[dict]) -> list[dict]:
    """Shape of extract_messages_for_routing output (role/content dicts)."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def test_session_id_wins_over_messages():
    messages = _chat([{"role": "user", "content": "Hello"}])
    assert conversation_key("sess-1", messages) == "sess-1"
    assert conversation_key("sess-1", []) == "sess-1"


def test_plain_string_content_is_stable_across_turns():
    turn1 = _chat([{"role": "user", "content": "What is the capital of France?"}])
    turn2 = _chat(
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "And its population?"},
        ]
    )
    key1 = conversation_key(None, turn1)
    key2 = conversation_key(None, turn2)
    assert key1 is not None
    assert key1 == key2


def test_openai_content_part_list():
    messages = _chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            }
        ]
    )
    key = conversation_key(None, messages)
    assert key is not None
    # Same text with the image part swapped out keeps the same key.
    messages_alt = _chat(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe this image"}],
            }
        ]
    )
    assert conversation_key(None, messages_alt) == key


def test_anthropic_style_blocks():
    messages = _chat(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Summarize this document"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "x", "content": "some output"}],
            },
        ]
    )
    key = conversation_key(None, messages)
    assert key is not None
    # Tool results and assistant blocks do not change the key.
    messages_alt = _chat(
        [
            {"role": "user", "content": [{"type": "text", "text": "Summarize this document"}]},
        ]
    )
    assert conversation_key(None, messages_alt) == key


def test_tool_result_only_first_user_message_is_skipped():
    messages = _chat(
        [
            # First user message carries no text (tool result only) — must
            # fall through to the next textual user message.
            {"role": "user", "content": [{"type": "tool_result", "content": "42"}]},
            {"role": "user", "content": [{"type": "text", "text": "What is 6 times 7?"}]},
        ]
    )
    key = conversation_key(None, messages)
    assert key is not None
    assert key == conversation_key(
        None,
        _chat([{"role": "user", "content": [{"type": "text", "text": "What is 6 times 7?"}]}]),
    )


def test_responses_api_string_input():
    assert conversation_key(None, [{"role": "user", "content": "Hello there"}]) is not None


def test_empty_or_non_text_content_returns_none():
    assert conversation_key(None, []) is None
    assert conversation_key(None, [{"role": "system", "content": "You are helpful"}]) is None
    assert (
        conversation_key(
            None, [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        )
        is None
    )
    assert conversation_key(None, [{"role": "user", "content": "   "}]) is None


def test_different_conversations_get_different_keys():
    a = conversation_key(None, [{"role": "user", "content": "Tell me a joke"}])
    b = conversation_key(None, [{"role": "user", "content": "Tell me a story"}])
    assert a != b
    assert a is not None
    assert b is not None


def test_consecutive_text_parts_join_without_separator():
    split = _chat(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}],
            }
        ]
    )
    joined = _chat([{"role": "user", "content": "Hello world"}])
    assert conversation_key(None, split) == conversation_key(None, joined)
