"""MiniMax adapter tests — the wire vocabulary, the error channel, the endpoints.

MiniMax's wires do not speak the protocols' vocabulary: ``thinking.type`` is
``disabled|adaptive`` (no ``budget_tokens``), the depth rides
``reasoning_effort`` (chat) or ``output_config.effort`` (Messages), the
service tier is ``standard|priority``, and the Chat Completions endpoint
reports failures in the body (``base_resp``) instead of the status line.

The tests below pin the translation of each of those, the token-count
endpoints, and the ``reasoning_split`` field MiniMax declares on its chat
wire.
"""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.providers.minimax import MiniMaxAdapter
from llm_proxy.providers.minimax.errors import base_resp_error
from llm_proxy.providers.minimax.wire import (
    always_thinks,
    normalize_chat_thinking,
    normalize_effort,
    normalize_messages_thinking,
    normalize_responses_reasoning,
    normalize_service_tier,
)


def _adapter(**kw) -> MiniMaxAdapter:
    return MiniMaxAdapter(api_key="test-key", base_url="https://api.minimax.io/v1", **kw)


def _chat_adapter() -> MiniMaxAdapter:
    """An adapter whose native Messages passthrough is off.

    With the passthrough on, an Anthropic client reaches MiniMax's Messages
    endpoint with its body forwarded verbatim; with it off, the request is
    rebuilt onto the chat wire — the tier the chat-body tests below cover.
    """
    return _adapter(native_passthrough=False)


def _anthropic_request(payload: dict):
    """An Anthropic-protocol request carrying the client's raw body."""
    from llm_proxy.protocols.anthropic import serializer as _anthropic_serializer  # noqa: F401

    ser = get_protocol_serializer("anthropic")
    req = ser.parse_request(payload)
    req.metadata.protocol_name = "anthropic"
    req._raw_protocol_data = payload
    return req


def _openai_request(payload: dict):
    """An OpenAI-protocol request carrying the client's raw body."""
    from llm_proxy.protocols.openai import serializer as _openai_serializer  # noqa: F401

    ser = get_protocol_serializer("openai")
    req = ser.parse_request(payload)
    req.metadata.protocol_name = "openai"
    req._raw_protocol_data = payload
    return req


class TestThinkingHelpers:
    """The pure normalization rules."""

    @pytest.mark.parametrize(
        ("effort", "expected"),
        [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("max", "max"),
            ("MAX", "max"),
            ("minimal", "low"),
            ("none", None),
            ("medium-plus", None),
            ("", None),
            (None, None),
            (7, None),
        ],
    )
    def test_normalize_effort(self, effort, expected):
        assert normalize_effort(effort) == expected

    def test_always_thinks_only_for_m3_1(self):
        assert always_thinks("MiniMax-M3.1-Flash") is True
        assert always_thinks("minimax-m3.1-flash") is True
        assert always_thinks("MiniMax-M3") is False
        assert always_thinks("M2-her") is False
        assert always_thinks(None) is False

    def test_native_body_maps_enabled_to_adaptive(self):
        body = {"model": "MiniMax-M3", "thinking": {"type": "enabled", "budget_tokens": 16000}}
        normalize_messages_thinking(body, "MiniMax-M3")
        assert body["thinking"] == {"type": "adaptive"}

    def test_native_body_drops_budget_tokens(self):
        body = {"model": "MiniMax-M3", "thinking": {"type": "enabled", "budget_tokens": 16000}}
        normalize_messages_thinking(body, "MiniMax-M3")
        assert "budget_tokens" not in body["thinking"]

    def test_native_body_maps_between_tools_to_adaptive(self):
        body = {"model": "MiniMax-M3", "thinking": {"type": "between_tools"}}
        normalize_messages_thinking(body, "MiniMax-M3")
        assert body["thinking"] == {"type": "adaptive"}

    def test_native_body_keeps_disabled_for_m3(self):
        body = {"model": "MiniMax-M3", "thinking": {"type": "disabled"}}
        normalize_messages_thinking(body, "MiniMax-M3")
        assert body["thinking"] == {"type": "disabled"}

    def test_native_body_drops_disabled_for_always_thinking_model(self):
        body = {"model": "MiniMax-M3.1-Flash", "thinking": {"type": "disabled"}}
        normalize_messages_thinking(body, "MiniMax-M3.1-Flash")
        assert "thinking" not in body

    def test_native_body_drops_unknown_thinking_type(self):
        body = {"model": "MiniMax-M3", "thinking": {"type": "wat"}}
        normalize_messages_thinking(body, "MiniMax-M3")
        assert "thinking" not in body

    def test_native_body_maps_effort(self):
        body = {"model": "MiniMax-M3", "output_config": {"effort": "minimal"}}
        normalize_messages_thinking(body, "MiniMax-M3")
        assert body["output_config"] == {"effort": "low"}

    def test_native_body_drops_unsupported_effort_and_disables_thinking(self):
        body = {"model": "MiniMax-M3", "output_config": {"effort": "none"}}
        normalize_messages_thinking(body, "MiniMax-M3")
        assert "output_config" not in body
        assert body["thinking"] == {"type": "disabled"}

    def test_native_body_drops_effort_none_for_always_thinking_model(self):
        body = {"model": "MiniMax-M3.1-Flash", "output_config": {"effort": "none"}}
        normalize_messages_thinking(body, "MiniMax-M3.1-Flash")
        assert "output_config" not in body
        assert "thinking" not in body

    def test_native_body_drops_structured_output_format(self):
        body = {"model": "MiniMax-M3", "output_config": {"format": {"type": "json_schema"}}}
        normalize_messages_thinking(body, "MiniMax-M3")
        assert "output_config" not in body

    def test_chat_body_maps_none_effort_to_disabled_thinking(self):
        body = {"model": "MiniMax-M3", "reasoning_effort": "none"}
        normalize_chat_thinking(body, "MiniMax-M3")
        assert "reasoning_effort" not in body
        assert body["thinking"] == {"type": "disabled"}

    def test_chat_body_drops_none_effort_for_always_thinking_model(self):
        body = {"model": "MiniMax-M3.1-Flash", "reasoning_effort": "none"}
        normalize_chat_thinking(body, "MiniMax-M3.1-Flash")
        assert "reasoning_effort" not in body
        assert "thinking" not in body

    def test_chat_body_keeps_adaptive_when_effort_is_none(self):
        body = {"model": "MiniMax-M3", "reasoning_effort": "none", "thinking": {"type": "adaptive"}}
        normalize_chat_thinking(body, "MiniMax-M3")
        assert "reasoning_effort" not in body
        assert body["thinking"] == {"type": "adaptive"}

    def test_chat_body_maps_enabled_to_adaptive(self):
        body = {"model": "MiniMax-M3", "thinking": {"type": "enabled", "budget_tokens": 1024}}
        normalize_chat_thinking(body, "MiniMax-M3")
        assert body["thinking"] == {"type": "adaptive"}


class TestServiceTier:
    """MiniMax speaks ``standard``/``priority``; the clients speak other words."""

    @pytest.mark.parametrize(
        ("tier", "expected"),
        [
            ("standard", "standard"),
            ("standard_only", "standard"),
            ("default", "standard"),
            ("priority", "priority"),
            ("auto", None),
            ("flex", None),
            ("scale", None),
        ],
    )
    def test_normalize_service_tier(self, tier, expected):
        body = {"service_tier": tier}
        normalize_service_tier(body)
        assert body.get("service_tier") == expected

    def test_native_body_translates_the_anthropic_vocabulary(self):
        adapter = _adapter()
        body = {
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "auto",
        }
        out = adapter.native_body_hook(body)
        assert "service_tier" not in out

    def test_native_body_maps_standard_only_to_standard(self):
        adapter = _adapter()
        body = {
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "standard_only",
        }
        out = adapter.native_body_hook(body)
        assert out["service_tier"] == "standard"

    def test_native_body_keeps_priority(self):
        adapter = _adapter()
        body = {
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "hi"}],
            "service_tier": "priority",
        }
        out = adapter.native_body_hook(body)
        assert out["service_tier"] == "priority"

    def test_chat_body_translates_the_openai_vocabulary(self):
        adapter = _chat_adapter()
        req = _openai_request(
            {
                "model": "MiniMax-M3",
                "messages": [{"role": "user", "content": "hi"}],
                "service_tier": "default",
            }
        )
        body = adapter._build_request_body(req)
        assert body["service_tier"] == "standard"

    def test_chat_body_carries_the_anthropic_client_tier(self):
        """An Anthropic client's ``standard_only`` means MiniMax's ``standard``."""
        adapter = _chat_adapter()
        req = _anthropic_request(
            {
                "model": "MiniMax-M3",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
                "service_tier": "standard_only",
            }
        )
        body = adapter._build_request_body(req)
        assert body["service_tier"] == "standard"


class TestResponsesReasoning:
    """The Responses wire has no ``thinking``; its ``reasoning.effort`` differs."""

    def test_effort_none_is_dropped_for_always_thinking_model(self):
        body = {"model": "MiniMax-M3.1-Flash", "reasoning": {"effort": "none"}}
        normalize_responses_reasoning(body, "MiniMax-M3.1-Flash")
        assert "reasoning" not in body

    def test_effort_none_is_kept_for_m3(self):
        body = {"model": "MiniMax-M3", "reasoning": {"effort": "none"}}
        normalize_responses_reasoning(body, "MiniMax-M3")
        assert body["reasoning"] == {"effort": "none"}

    def test_effort_minimal_is_kept(self):
        body = {"model": "MiniMax-M3", "reasoning": {"effort": "minimal"}}
        normalize_responses_reasoning(body, "MiniMax-M3")
        assert body["reasoning"] == {"effort": "minimal"}

    def test_native_body_drops_effort_none_for_always_thinking_model(self):
        adapter = _adapter()
        body = {
            "model": "MiniMax-M3.1-Flash",
            "input": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "none"},
        }
        out = adapter.native_body_hook(body)
        assert "reasoning" not in out


class TestBaseRespErrors:
    """MiniMax answers 200 and carries the failure in ``base_resp``."""

    def test_success_body_has_no_error(self):
        assert base_resp_error({"base_resp": {"status_code": 0, "status_msg": ""}}) is None

    def test_body_without_base_resp_has_no_error(self):
        assert base_resp_error({"choices": []}) is None

    @pytest.mark.parametrize(
        ("code", "status"),
        [
            (1002, 429),  # rate limit
            (1004, 401),  # not authorized
            (1008, 402),  # insufficient balance
            (1027, 400),  # output sensitive
            (2013, 400),  # invalid params
            (2056, 429),  # usage limit exceeded
        ],
    )
    def test_error_codes_map_to_http_status(self, code, status):
        error = base_resp_error({"base_resp": {"status_code": code, "status_msg": "boom"}})
        assert error is not None
        assert error.status_code == status
        assert str(code) in str(error.message)

    def test_unknown_code_falls_back_to_a_bad_gateway(self):
        error = base_resp_error({"base_resp": {"status_code": 9999, "status_msg": "?"}})
        assert error is not None
        assert error.status_code == 502

    def test_adapter_raises_on_an_error_body(self):
        adapter = _adapter()
        with pytest.raises(ProviderError) as exc:
            adapter._check_upstream_body_error(
                {"base_resp": {"status_code": 1002, "status_msg": "rate limit"}}
            )
        assert exc.value.status_code == 429

    def test_adapter_passes_a_success_body(self):
        adapter = _adapter()
        adapter._check_upstream_body_error({"base_resp": {"status_code": 0}})


class TestNativeMessagesBody:
    """The Anthropic Messages passthrough body (the default tier for MiniMax)."""

    def test_claude_code_thinking_is_translated(self):
        adapter = _adapter()
        body = {
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 16000},
        }
        out = adapter.native_body_hook(body)
        assert out["thinking"] == {"type": "adaptive"}

    def test_output_config_effort_is_mapped(self):
        adapter = _adapter()
        body = {
            "model": "MiniMax-M3.1-Flash",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "minimal"},
        }
        out = adapter.native_body_hook(body)
        assert out["output_config"] == {"effort": "low"}

    def test_responses_body_is_untouched_by_the_thinking_rules(self):
        """A Responses body has no ``messages`` — the Messages rules do not apply."""
        adapter = _adapter()
        body = {
            "model": "MiniMax-M3",
            "input": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 16000},
        }
        out = adapter.native_body_hook(body)
        assert out["thinking"] == {"type": "enabled", "budget_tokens": 16000}

    def test_video_block_is_forwarded_verbatim(self):
        """MiniMax's Messages endpoint reads a ``video`` block — keep it as it is."""
        adapter = _adapter()
        body = {
            "model": "MiniMax-M3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "source": {"type": "url", "url": "https://example.com/clip.mp4"},
                        }
                    ],
                }
            ],
        }
        out = adapter.native_body_hook(body)
        assert out["messages"][0]["content"][0] == {
            "type": "video",
            "source": {"type": "url", "url": "https://example.com/clip.mp4"},
        }


class TestChatBody:
    """The rebuilt / wire-reused chat body."""

    def test_disabled_thinking_is_not_sent_as_reasoning_effort_none(self):
        adapter = _chat_adapter()
        req = _anthropic_request(
            {
                "model": "MiniMax-M3",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "disabled"},
            }
        )
        body = adapter._build_request_body(req)
        assert body.get("reasoning_effort") != "none"
        assert body["thinking"] == {"type": "disabled"}

    def test_disabled_thinking_omitted_for_always_thinking_model(self):
        adapter = _chat_adapter()
        req = _anthropic_request(
            {
                "model": "MiniMax-M3.1-Flash",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "disabled"},
            }
        )
        body = adapter._build_request_body(req)
        assert "reasoning_effort" not in body
        assert "thinking" not in body

    def test_output_config_effort_becomes_reasoning_effort(self):
        adapter = _chat_adapter()
        req = _anthropic_request(
            {
                "model": "MiniMax-M3.1-Flash",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
                "output_config": {"effort": "max"},
            }
        )
        body = adapter._build_request_body(req)
        assert body["reasoning_effort"] == "max"

    def test_openai_client_reasoning_effort_none_is_translated(self):
        adapter = _adapter()
        req = _openai_request(
            {
                "model": "MiniMax-M3",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "none",
            }
        )
        body = adapter._build_request_body(req)
        assert "reasoning_effort" not in body
        assert body["thinking"] == {"type": "disabled"}

    def test_reasoning_split_survives_the_unknown_field_policy(self):
        """``reasoning_split`` is MiniMax's own output switch — it must not be stripped."""
        adapter = _adapter()
        req = _openai_request(
            {
                "model": "MiniMax-M3",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_split": False,
            }
        )
        body = adapter._build_request_body(req)
        assert body["reasoning_split"] is False

    def test_anthropic_video_block_becomes_a_video_url(self):
        """MiniMax's M-series models read video; the block must not degrade to text."""
        adapter = _chat_adapter()
        req = _anthropic_request(
            {
                "model": "MiniMax-M3",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what?"},
                            {
                                "type": "video",
                                "source": {"type": "url", "url": "https://example.com/clip.mp4"},
                            },
                        ],
                    }
                ],
            }
        )
        body = adapter._build_request_body(req)
        parts = body["messages"][0]["content"]
        assert parts[1] == {
            "type": "video_url",
            "video_url": {"url": "https://example.com/clip.mp4"},
        }


class TestTokenEndpoints:
    """MiniMax serves two token counters; both are reachable."""

    def test_count_tokens_targets_the_messages_endpoint(self):
        adapter = _adapter()
        assert adapter._count_tokens_url() == (
            "https://api.minimax.io/anthropic/v1/messages/count_tokens"
        )

    def test_count_tokens_follows_the_anthropic_endpoint_override(self):
        adapter = _adapter(
            endpoint_base_urls={"anthropic_messages": "https://example.com/anthropic/v1/messages"}
        )
        assert (
            adapter._count_tokens_url() == "https://example.com/anthropic/v1/messages/count_tokens"
        )

    def test_input_tokens_targets_the_responses_endpoint(self):
        adapter = _adapter()
        assert adapter._responses_url() == "https://api.minimax.io/v1/responses"
        assert f"{adapter._responses_url().rstrip('/')}/input_tokens" == (
            "https://api.minimax.io/v1/responses/input_tokens"
        )

    def test_the_adapter_exposes_both_counters(self):
        adapter = _adapter()
        assert callable(adapter.count_tokens)
        assert callable(adapter.input_tokens)

    @pytest.mark.asyncio
    async def test_count_tokens_sends_only_the_fields_minimax_documents(self):
        adapter = _adapter()
        sent: dict = {}

        async def _fake_post(url, headers, body):
            sent["url"] = url
            sent["body"] = body
            return {"input_tokens": 12}

        adapter._post_json_with_retry = _fake_post
        result = await adapter.count_tokens(
            {
                "model": "MiniMax-M3",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "enabled", "budget_tokens": 16000},
                "context_management": {"edits": []},
            }
        )
        assert result == {"input_tokens": 12}
        assert sent["url"] == "https://api.minimax.io/anthropic/v1/messages/count_tokens"
        assert "context_management" not in sent["body"]
        assert sent["body"]["thinking"] == {"type": "adaptive"}
