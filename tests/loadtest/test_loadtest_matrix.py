"""Tests for the load-test scenario matrix and the multi-dialect fake upstream.

These do not exercise the proxy (that is the live load run's job); they keep
the harness honest: every protocol/provider pair has a scenario, selection
tokens resolve, payloads carry the fields each protocol requires, and the fake
upstream still answers every wire dialect the matrix relies on.
"""

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loadtest import fake_upstream  # noqa: E402
from loadtest.scenarios import (  # noqa: E402
    PROTOCOLS,
    PROVIDERS,
    SCENARIOS,
    SCENARIOS_BY_KEY,
    build_headers,
    build_payload,
    scenario_tokens,
    select_scenarios,
)


def test_every_protocol_provider_pair_has_scenarios():
    """The matrix covers each (protocol, provider) pair, stream + non-stream."""
    pairs = {(s.protocol, s.provider) for s in SCENARIOS}
    for protocol in PROTOCOLS:
        for provider, spec in PROVIDERS.items():
            if protocol == "embeddings" and not spec.supports_embeddings:
                continue
            assert (protocol, provider) in pairs, f"missing scenarios for {protocol}/{provider}"
    # Non-embeddings protocols get a streaming variant.
    for scenario in SCENARIOS:
        if scenario.protocol != "embeddings":
            assert scenario.key.replace("_nonstream", "_stream") in SCENARIOS_BY_KEY


def test_scenario_keys_and_groups_are_unique_and_resolvable():
    assert len(SCENARIOS_BY_KEY) == len(SCENARIOS)
    tokens = set(scenario_tokens())
    for scenario in SCENARIOS:
        assert scenario.protocol in PROTOCOLS
        assert scenario.provider in PROVIDERS
        assert scenario.groups
        for group in scenario.groups:
            assert group in tokens


def test_select_all_returns_every_scenario():
    assert {s.key for s in select_scenarios("all")} == set(SCENARIOS_BY_KEY)


def test_select_core_is_a_nonempty_subset():
    core = select_scenarios("core")
    assert core
    assert {s.key for s in core} <= set(SCENARIOS_BY_KEY)


def test_select_groups_and_keys():
    assert {s.protocol for s in select_scenarios("messages")} == {"messages"}
    assert {s.provider for s in select_scenarios("gemini")} == {"gemini"}
    assert all(s.stream for s in select_scenarios("stream"))
    assert all(not s.stream for s in select_scenarios("nonstream"))
    one = select_scenarios("chat_openai_compat_stream")
    assert [s.key for s in one] == ["chat_openai_compat_stream"]


def test_unknown_selection_token_fails_loudly():
    with pytest.raises(SystemExit, match="Unknown LOADTEST_SCENARIOS token"):
        select_scenarios("not-a-real-group")


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses", "embeddings"])
def test_payload_has_required_fields(protocol):
    scenario = next(s for s in SCENARIOS if s.protocol == protocol)
    payload = build_payload(scenario)
    assert payload["model"] == scenario.model
    if protocol == "chat":
        assert payload["messages"]
    elif protocol == "messages":
        assert payload["messages"] and "max_tokens" in payload
    elif protocol == "responses" or protocol == "embeddings":
        assert "input" in payload


def test_headers_carry_auth_and_protocol_extras():
    chat = SCENARIOS_BY_KEY["chat_openai_compat_nonstream"]
    assert build_headers(chat, "sk-test")["Authorization"] == "Bearer sk-test"
    messages = SCENARIOS_BY_KEY["messages_anthropic_native_nonstream"]
    assert "anthropic-version" in build_headers(messages, "sk-test")


# ---------------------------------------------------------------------------
# Fake upstream
# ---------------------------------------------------------------------------


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_upstream.app, raise_app_exceptions=True),
        base_url="http://fake-upstream",
    )


async def test_fake_upstream_openai_chat_supports_stream_and_unary():
    async with _client() as client:
        body = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
        unary = await client.post("/v1/chat/completions", json=body)
        assert unary.status_code == 200
        assert unary.json()["choices"][0]["message"]["content"]

        stream = await client.post(
            "/v1/chat/completions",
            json={**body, "stream": True, "stream_options": {"include_usage": True}},
        )
        assert "[DONE]" in stream.text
        assert '"usage"' in stream.text


async def test_fake_upstream_responses_supports_stream_and_unary():
    async with _client() as client:
        body = {"model": "m", "input": "hello"}
        unary = await client.post("/v1/responses", json=body)
        assert unary.json()["output"][0]["content"][0]["text"]
        assert unary.json()["usage"]["input_tokens"] >= 1

        stream = await client.post("/v1/responses", json={**body, "stream": True})
        assert "response.completed" in stream.text


async def test_fake_upstream_anthropic_supports_stream_and_unary():
    async with _client() as client:
        body = {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
        unary = await client.post("/v1/messages", json=body)
        assert unary.json()["content"][0]["text"]
        assert unary.json()["usage"]["output_tokens"] > 0

        stream = await client.post("/v1/messages", json={**body, "stream": True})
        assert "message_stop" in stream.text

        # DeepSeek-native root alias.
        alias = await client.post("/anthropic/v1/messages", json=body)
        assert alias.status_code == 200


async def test_fake_upstream_gemini_and_embeddings():
    async with _client() as client:
        body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        unary = await client.post("/v1beta/models/gemini-2.0:generateContent", json=body)
        assert unary.json()["candidates"][0]["content"]["parts"][0]["text"]

        stream = await client.post(
            "/v1beta/models/gemini-2.0:streamGenerateContent?alt=sse", json=body
        )
        assert "data:" in stream.text

        embed = await client.post(
            "/v1beta/models/text-embedding-004:embedContent",
            json={"content": {"parts": [{"text": "hi"}]}},
        )
        assert embed.json()["embedding"]["values"]


async def test_fake_upstream_ollama_and_openai_embeddings():
    async with _client() as client:
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        unary = await client.post("/api/chat", json=body)
        assert unary.json()["message"]["content"]
        assert unary.json()["done"] is True

        stream = await client.post("/api/chat", json={**body, "stream": True})
        assert '"done":true' in stream.text.replace(" ", "")

        embed = await client.post("/api/embed", json={"model": "m", "input": "hi"})
        assert embed.json()["embeddings"]

        openai_embed = await client.post("/v1/embeddings", json={"model": "m", "input": "hi"})
        assert openai_embed.json()["data"][0]["embedding"]


async def test_fake_upstream_count_tokens():
    async with _client() as client:
        resp = await client.post(
            "/v1/messages/count_tokens",
            json={"messages": [{"role": "user", "content": "hello world"}]},
        )
        assert resp.json()["input_tokens"] >= 1
