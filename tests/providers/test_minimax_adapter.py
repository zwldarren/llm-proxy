"""MiniMax adapter tests — service tier, error channel, endpoints, verbatim thinking.

Thinking and reasoning parameters pass through to MiniMax verbatim without
proxy-side vocabulary rewriting (ADR-0063). MiniMax still normalizes
``service_tier`` (``standard``/``priority``), translates in-body
``base_resp`` errors into HTTP errors, exempts ``reasoning_split`` on Chat
Completions, and forwards ``count_tokens`` / ``input_tokens``.
"""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.providers.minimax import MiniMaxAdapter
from llm_proxy.providers.minimax.errors import base_resp_error
from llm_proxy.providers.minimax.wire import normalize_service_tier


def _adapter(**kw) -> MiniMaxAdapter:
    return MiniMaxAdapter(api_key="test-key", base_url="https://api.minimax.io/v1", **kw)


def _chat_adapter(**kw) -> MiniMaxAdapter:
    """An adapter whose native Messages passthrough is off."""
    return _adapter(native_passthrough=False, **kw)


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


class TestBaseRespErrors:
    """MiniMax answers 200 and carries the failure in ``base_resp``."""

    def test_success_body_has_no_error(self):
        assert base_resp_error({"base_resp": {"status_code": 0, "status_msg": ""}}) is None

    def test_body_without_base_resp_has_no_error(self):
        assert base_resp_error({"choices": []}) is None

    @pytest.mark.parametrize(
        ("code", "status"),
        [
            (1002, 429),
            (1004, 401),
            (1008, 402),
            (1027, 400),
            (2013, 400),
            (2056, 429),
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


class TestNativeBodiesVerbatim:
    """Native Messages and Responses bodies forward thinking/reasoning verbatim."""

    def test_messages_thinking_and_budget_forwarded_verbatim(self):
        adapter = _adapter()
        body = {
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 16000},
            "output_config": {"effort": "minimal", "format": {"type": "json_schema"}},
        }
        out = adapter.native_body_hook(body)
        assert out["thinking"] == {"type": "enabled", "budget_tokens": 16000}
        assert out["output_config"] == {
            "effort": "minimal",
            "format": {"type": "json_schema"},
        }

    def test_m3_1_disabled_thinking_and_responses_none_forwarded_verbatim(self):
        adapter = _adapter()
        messages_body = {
            "model": "MiniMax-M3.1-Flash",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "disabled"},
        }
        assert adapter.native_body_hook(messages_body)["thinking"] == {"type": "disabled"}

        responses_body = {
            "model": "MiniMax-M3.1-Flash",
            "input": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "none"},
        }
        assert adapter.native_body_hook(responses_body)["reasoning"] == {"effort": "none"}

    def test_video_block_is_forwarded_verbatim(self):
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

    def test_disabled_thinking_maps_to_reasoning_effort_none(self):
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
        assert body["reasoning_effort"] == "none"

    def test_disabled_thinking_forwarded_for_m3_1_flash(self):
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
        assert body["reasoning_effort"] == "none"

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

    def test_openai_wire_reuse_preserves_reasoning_effort_none(self):
        adapter = _adapter()
        req = _openai_request(
            {
                "model": "MiniMax-M3",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "none",
            }
        )
        body = adapter._build_request_body(req)
        assert body["reasoning_effort"] == "none"

    def test_reasoning_split_survives_the_ignore_field_policy(self):
        adapter = _adapter(unknown_fields_policy="ignore")
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
    async def test_count_tokens_keeps_thinking_verbatim_and_trims_undocumented_fields(self):
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
        assert sent["body"]["thinking"] == {"type": "enabled", "budget_tokens": 16000}
