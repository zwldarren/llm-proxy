"""Unit tests for protocol-agnostic message extraction used by the router.

Covers both the /chat/completions (``messages``) path and the /responses
(``input``) path, including the many OpenAI Responses item types.
"""

from types import SimpleNamespace

from llm_proxy.routing.message_extract import (
    extract_messages_for_routing,
    function_call_output_to_text,
)


def test_chat_messages_path():
    req = SimpleNamespace(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        input=None,
    )
    assert extract_messages_for_routing(req) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_missing_role_defaults_to_user():
    req = SimpleNamespace(messages=[{"content": "only content"}], input=None)
    assert extract_messages_for_routing(req) == [{"role": "user", "content": "only content"}]


def test_responses_str_input():
    req = SimpleNamespace(messages=None, input="hello world")
    assert extract_messages_for_routing(req) == [{"role": "user", "content": "hello world"}]


def test_responses_no_input_returns_empty():
    req = SimpleNamespace(messages=None, input=None)
    assert extract_messages_for_routing(req) == []


def test_responses_message_item():
    req = SimpleNamespace(
        messages=None,
        input=[{"type": "message", "role": "user", "content": "hi"}],
    )
    assert extract_messages_for_routing(req) == [{"role": "user", "content": "hi"}]


def test_responses_reasoning_item():
    req = SimpleNamespace(
        messages=None,
        input=[{"type": "reasoning", "content": [{"type": "text", "text": "thinking"}]}],
    )
    assert extract_messages_for_routing(req) == [{"role": "assistant", "content": "thinking"}]


def test_responses_function_call_item():
    req = SimpleNamespace(
        messages=None,
        input=[{"type": "function_call", "name": "f", "arguments": "{}"}],
    )
    assert extract_messages_for_routing(req) == [{"role": "assistant", "content": "f({})"}]


def test_responses_function_call_output_item():
    req = SimpleNamespace(
        messages=None,
        input=[{"type": "function_call_output", "output": "result"}],
    )
    assert extract_messages_for_routing(req) == [{"role": "tool", "content": "result"}]


def test_responses_function_call_output_list():
    req = SimpleNamespace(
        messages=None,
        input=[
            {
                "type": "function_call_output",
                "output": [
                    {"type": "input_text", "text": "a"},
                    {"type": "input_text", "text": "b"},
                ],
            }
        ],
    )
    assert extract_messages_for_routing(req) == [{"role": "tool", "content": "a\nb"}]


def test_responses_ignored_types_are_skipped():
    req = SimpleNamespace(
        messages=None,
        input=[
            {"type": "web_search_call"},
            {"type": "message", "role": "user", "content": "x"},
        ],
    )
    assert extract_messages_for_routing(req) == [{"role": "user", "content": "x"}]


def test_function_call_output_to_text_str():
    assert function_call_output_to_text("plain") == "plain"


def test_function_call_output_to_text_list_of_strings():
    assert function_call_output_to_text(["a", "b"]) == "a\nb"


def test_function_call_output_to_text_input_text_blocks():
    out = function_call_output_to_text(
        [{"type": "input_text", "text": "hello"}, {"type": "input_image"}],
    )
    assert out == "hello"


def test_function_call_output_to_text_empty():
    assert function_call_output_to_text("") == ""
    assert function_call_output_to_text([]) == ""
