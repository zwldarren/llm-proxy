"""Tests for streamed OpenResponses response persistence (store=true).

Covers the streaming-side storage that mirrors RequestExecutionStage's
non-streaming persistence: without it, streamed store=true turns could never
be continued via previous_response_id or retrieved via GET /v1/responses/{id}.
"""

from unittest.mock import MagicMock

import pytest

from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer


class _FakeResponseStore:
    def __init__(self) -> None:
        self.stored: list[tuple[str, str, dict]] = []

    async def store(self, api_key_name: str, response_id: str, response: dict) -> None:
        self.stored.append((api_key_name, response_id, response))


def _make_transformer(payload: dict | None, store: bool = True):
    transformer = OpenResponsesStreamingTransformer(model="gpt-5.2", request_id="resp_stream_1")
    transformer.state.final_response_payload = payload
    transformer.state.store = store
    return transformer


def _make_request() -> InternalRequest:
    return InternalRequest(
        model="gpt-5.2",
        conversation=ConversationContext(
            system_messages=[],
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
        extra={},
    )


def _completed_payload(store: bool = True) -> dict:
    return {
        "id": "resp_stream_1",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.2",
        "store": store,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "id": "msg_1",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ],
    }


@pytest.mark.asyncio
async def test_persists_completed_stream_response_with_materialized_input():
    """store=true streamed responses are stored with the conversation as input."""
    store = _FakeResponseStore()
    transformer = _make_transformer(_completed_payload(store=True))
    event_context = MagicMock()
    event_context.api_key_name = "test-key"

    await transformer.finalize_persistence(_make_request(), store, event_context)

    assert len(store.stored) == 1
    api_key_name, response_id, body = store.stored[0]
    assert api_key_name == "test-key"
    assert response_id == "resp_stream_1"
    # Stored body matches the completed snapshot the client saw.
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "hello"
    # The materialized conversation is attached so continuations can replay it.
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        }
    ]


@pytest.mark.asyncio
async def test_skips_persistence_when_store_false():
    """Explicit store=false opts out of persistence."""
    store = _FakeResponseStore()
    transformer = _make_transformer(_completed_payload(store=False))
    event_context = MagicMock()
    event_context.api_key_name = "test-key"

    await transformer.finalize_persistence(_make_request(), store, event_context)

    assert store.stored == []


@pytest.mark.asyncio
async def test_falls_back_to_request_side_store_flag():
    """Native upstream snapshots missing ``store`` use the request-side value."""
    store = _FakeResponseStore()
    payload = _completed_payload()
    del payload["store"]
    transformer = _make_transformer(payload, store=True)
    event_context = MagicMock()
    event_context.api_key_name = "test-key"

    await transformer.finalize_persistence(_make_request(), store, event_context)

    assert len(store.stored) == 1


@pytest.mark.asyncio
async def test_skips_persistence_without_payload_or_identity():
    """No snapshot or no API key means nothing to persist."""
    store = _FakeResponseStore()

    # No final payload (e.g. stream failed before completion).
    transformer = _make_transformer(None)
    event_context = MagicMock()
    event_context.api_key_name = "test-key"
    await transformer.finalize_persistence(_make_request(), store, event_context)

    # No api_key_name.
    transformer = _make_transformer(_completed_payload(store=True))
    event_context = MagicMock()
    event_context.api_key_name = None
    await transformer.finalize_persistence(_make_request(), store, event_context)

    assert store.stored == []


class TestStoreDefaultNormalization:
    """set_format_context normalizes omitted ``store`` to true (OpenAI parity)."""

    def setup_method(self):
        from llm_proxy.protocols.openresponses.handler import clear_format_context

        clear_format_context()

    def teardown_method(self):
        from llm_proxy.protocols.openresponses.handler import clear_format_context

        clear_format_context()

    def test_store_defaults_to_true_when_omitted(self):
        from llm_proxy.protocols.openresponses.handler import (
            get_format_context,
            set_format_context,
        )

        set_format_context({"model": "gpt-5.2", "input": "hi"})
        assert get_format_context().store is True

    def test_explicit_store_false_is_preserved(self):
        from llm_proxy.protocols.openresponses.handler import (
            get_format_context,
            set_format_context,
        )

        set_format_context({"model": "gpt-5.2", "input": "hi", "store": False})
        assert get_format_context().store is False

    def test_explicit_store_true_is_preserved(self):
        from llm_proxy.protocols.openresponses.handler import (
            get_format_context,
            set_format_context,
        )

        set_format_context({"model": "gpt-5.2", "input": "hi", "store": True})
        assert get_format_context().store is True


@pytest.mark.asyncio
async def test_persisted_input_excludes_instructions_echo():
    """The instructions-derived system message is not duplicated into input.

    Continuations restore ``instructions`` from the response's own
    ``instructions`` field; serializing the matching system message into the
    stored input would apply it twice.
    """
    from llm_proxy.models import SystemMessage

    store = _FakeResponseStore()
    payload = _completed_payload(store=True)
    payload["instructions"] = "Be helpful."
    transformer = _make_transformer(payload)
    event_context = MagicMock()
    event_context.api_key_name = "test-key"

    request = _make_request()
    request.conversation.system_messages.append(
        SystemMessage.from_text(role="system", text="Be helpful.")
    )
    request.conversation.system_messages.append(
        SystemMessage.from_text(role="system", text="Extra rules.")
    )

    await transformer.finalize_persistence(request, store, event_context)

    assert len(store.stored) == 1
    _, _, body = store.stored[0]
    system_items = [
        i for i in body["input"] if i.get("type") == "message" and i.get("role") == "system"
    ]
    assert [i["content"] for i in system_items] == ["Extra rules."]
