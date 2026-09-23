"""Native chat-completions streaming passthrough (openai protocol).

Covers the OpenAICompatibleBase native stream tier: capability gating
(``native_passthrough`` flag, reasoning-echo veto), body preparation (the native
stream sends the body the converted-path builder produced, including the forced
``include_usage`` — the forcing itself lives on the shared convergence point and
is covered by ``tests/core/test_conversion_tiers.py``), and the verbatim-frame
bookkeeping in ``NativePassthroughHandler.handle_native_openai_chunk``.
"""

import copy
from unittest.mock import MagicMock

import pytest

from llm_proxy.core.conversion import NativePassthroughHandler, plan_conversion
from llm_proxy.models import (
    ConversationContext,
    ConversionTier,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.models.types import StreamOptions
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase


def _adapter(**kwargs) -> OpenAICompatibleBase:
    return OpenAICompatibleBase(
        api_key="test-key",
        base_url="https://upstream.example/v1",
        **kwargs,
    )


def _request(model: str = "glm-5") -> MagicMock:
    req = MagicMock()
    req.model = model
    req.protocol_name = "openai"
    req.native_request_disabled = False
    req._raw_protocol_data = None
    return req


def _request_with_stash(
    raw: dict, model: str = "glm-5", stream_options: StreamOptions | None = None
) -> InternalRequest:
    """A real request (not a MagicMock) so ``_stream_body`` can actually build.

    The native-stream tests below must exercise the real builder: the forced
    ``include_usage`` is applied on the shared convergence point inside it, and a
    stubbed ``_stream_body`` would test a path that no longer exists.
    """
    req = InternalRequest(
        model=model,
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        stream=True,
        stream_options=stream_options,
    )
    req.metadata.protocol_name = "openai"
    req._raw_protocol_data = copy.deepcopy(raw)
    return req


class TestNativeStreamingGate:
    def test_openai_protocol_supported_by_default(self) -> None:
        assert _adapter().supports_native_streaming("openai") is True

    def test_kill_switch_disables(self) -> None:
        assert _adapter(native_passthrough=False).supports_native_streaming("openai") is False

    def test_other_protocols_unaffected(self) -> None:
        # The plain base declares no Anthropic/Responses native endpoints.
        assert _adapter().supports_native_streaming("anthropic") is False

    def test_plan_native_for_regular_model(self) -> None:
        plan = plan_conversion(_adapter(), _request("glm-5"))
        assert plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH

    def test_plan_converted_for_reasoning_echo_model(self) -> None:
        # DeepSeek-style models keep the converted stream so the transformer's
        # accumulation feeds the reasoning cache used by the next turn.
        plan = plan_conversion(_adapter(), _request("deepseek-reasoner"))
        assert plan.stream_mode == ConversionTier.FULL_CONVERSION

    def test_plan_converted_when_disabled(self) -> None:
        req = _request("glm-5")
        req.native_request_disabled = True
        plan = plan_conversion(_adapter(), req)
        assert plan.stream_mode == ConversionTier.FULL_CONVERSION


class TestNativeStreamBody:
    async def test_body_is_the_builder_output_with_forced_include_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The native stream sends exactly what the converted-path builder built:
        a client body that omitted ``stream_options`` arrives with
        ``include_usage`` forced on."""
        adapter = _adapter()
        captured: dict = {}

        def fake_raw_sse(url, body, cancel_token=None):
            captured["url"] = url
            captured["body"] = body

            async def gen():
                yield "data: [DONE]\n\n"
                return

            return gen()

        monkeypatch.setattr(adapter, "_stream_raw_sse", fake_raw_sse)

        req = _request_with_stash(
            {
                "model": "glm-5",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
        )
        stream = await adapter.stream_chat_completion_native(req)
        assert [chunk async for chunk in stream] == ["data: [DONE]\n\n"]

        assert captured["url"] == "https://upstream.example/v1/chat/completions"
        assert captured["body"]["stream_options"] == {"include_usage": True}

    async def test_preserves_client_stream_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A client's other ``stream_options`` keys survive the forced-usage rule
        (it overrides ``include_usage`` only)."""
        adapter = _adapter()
        captured: dict = {}

        def fake_raw_sse(url, body, cancel_token=None):
            captured["body"] = body

            async def gen():
                return
                yield

            return gen()

        monkeypatch.setattr(adapter, "_stream_raw_sse", fake_raw_sse)

        req = _request_with_stash(
            {
                "model": "glm-5",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": False, "include_obfuscation": False},
            },
            stream_options=StreamOptions(include_usage=False, include_obfuscation=False),
        )
        stream = await adapter.stream_chat_completion_native(req)
        assert [chunk async for chunk in stream] == []
        assert captured["body"]["stream_options"] == {
            "include_usage": True,
            "include_obfuscation": False,
        }


class TestHandleNativeOpenAIChunk:
    def _request(self, model: str, echo_model: str | None) -> MagicMock:
        req = MagicMock()
        req.model = model
        req.echo_model = echo_model or model
        return req

    def test_delta_frame_forwarded_verbatim_when_names_match(self) -> None:
        frame = 'data: {"id":"1","model":"glm-5","choices":[{"delta":{"content":"hi"}}]}\n\n'
        out = NativePassthroughHandler.handle_native_openai_chunk(
            frame, self._request("glm-5", "glm-5"), None
        )
        assert out == frame

    def test_model_rewritten_to_client_alias(self) -> None:
        frame = 'data: {"id":"1","model":"upstream-m","choices":[{"delta":{"content":"hi"}}]}\n\n'
        out = NativePassthroughHandler.handle_native_openai_chunk(
            frame, self._request("upstream-m", "alias-m"), None
        )
        assert '"model":"alias-m"' in out
        assert '"model":"upstream-m"' not in out

    def test_model_rewrite_with_spaced_json(self) -> None:
        frame = 'data: {"id": "1", "model": "upstream-m", "choices": []}\n\n'
        out = NativePassthroughHandler.handle_native_openai_chunk(
            frame, self._request("upstream-m", "alias-m"), None
        )
        assert '"model": "alias-m"' in out

    def test_only_first_model_field_rewritten(self) -> None:
        # Content echoing the same shape later in the frame stays untouched:
        # the top-level model field precedes choices in serialization order.
        frame = (
            'data: {"model":"upstream-m","choices":[{"delta":{"content":"'
            'say \\"model\\":\\"upstream-m\\" please"}}]}\n\n'
        )
        out = NativePassthroughHandler.handle_native_openai_chunk(
            frame, self._request("upstream-m", "alias-m"), None
        )
        assert out.count('"model":"alias-m"') == 1
        assert '\\"model\\":\\"upstream-m\\"' in out

    def test_usage_frame_captured_and_forwarded(self) -> None:
        frame = (
            'data: {"id":"1","model":"glm-5","choices":[],"usage":'
            '{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8,'
            '"prompt_tokens_details":{"cached_tokens":2},'
            '"completion_tokens_details":{"reasoning_tokens":1}}}\n\n'
        )
        ctx = MagicMock()
        out = NativePassthroughHandler.handle_native_openai_chunk(
            frame, self._request("glm-5", "glm-5"), ctx
        )
        assert out == frame
        assert ctx.prompt_tokens == 3
        assert ctx.completion_tokens == 5
        assert ctx.total_tokens == 8
        assert ctx.cache_read_input_tokens == 2
        assert ctx.reasoning_tokens == 1

    def test_usage_sniff_does_not_break_regular_frames(self) -> None:
        # A content string containing the word usage still triggers the cheap
        # substring sniff, but the frame carries no usage dict, so nothing
        # is written.
        frame = 'data: {"id":"1","model":"glm-5","choices":[{"delta":{"content":"usage"}}]}\n\n'
        ctx = MagicMock()
        NativePassthroughHandler.handle_native_openai_chunk(
            frame, self._request("glm-5", "glm-5"), ctx
        )
        # prompt_tokens is a MagicMock attribute (truthy) but the handler
        # must not have written it: no usage dict in this frame.
        assert not isinstance(ctx.prompt_tokens, int)

    def test_done_frame_forwarded_verbatim(self) -> None:
        frame = "data: [DONE]\n\n"
        out = NativePassthroughHandler.handle_native_openai_chunk(
            frame, self._request("glm-5", "alias-m"), None
        )
        assert out == frame

    def test_non_str_chunk_passthrough(self) -> None:
        chunk = {"raw": "dict"}
        out = NativePassthroughHandler.handle_native_openai_chunk(
            chunk, self._request("glm-5", "glm-5"), None
        )
        assert out is chunk


class TestNativeUsageFrameBilling:
    """Provider-reported cost and web-search counts on the native stream tier.

    Native frames bypass the streaming transformer, so this capture is the only
    route by which OpenRouter's ``usage.cost`` reaches the billing pipeline.
    """

    def _request(self, model: str, echo_model: str | None) -> MagicMock:
        req = MagicMock()
        req.model = model
        req.echo_model = echo_model or model
        return req

    def test_usage_frame_captures_cost_and_web_search(self) -> None:
        from llm_proxy.observability.event_context import EventContext

        ctx = EventContext(request_id="r", trace_id="t", model="anthropic/claude-sonnet-4.5")
        frame = (
            'data: {"id":"gen-1","choices":[],"usage":{"prompt_tokens":10,'
            '"completion_tokens":20,"total_tokens":30,"cost":0.0123,'
            '"is_byok":false,"server_tool_use":{"web_search_requests":2}}}\n\n'
        )

        out = NativePassthroughHandler.handle_native_openai_chunk(
            frame, self._request("anthropic/claude-sonnet-4.5", "claude-sonnet-4.5"), ctx
        )

        assert out == frame
        assert ctx.prompt_tokens == 10
        assert ctx.provider_reported_cost == 0.0123
        assert ctx.web_search_requests == 2

    def test_zero_cost_is_not_treated_as_reported(self) -> None:
        """A 0.0 cost must not replace local estimation with a fake zero."""
        from llm_proxy.observability.event_context import EventContext

        ctx = EventContext(request_id="r", trace_id="t", model="glm-5")
        frame = 'data: {"id":"1","choices":[],"usage":{"prompt_tokens":5,"cost":0.0}}\n\n'

        NativePassthroughHandler.handle_native_openai_chunk(
            frame, self._request("glm-5", "glm-5"), ctx
        )

        assert ctx.provider_reported_cost is None
