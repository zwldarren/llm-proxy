"""Tests for the Responses input-token estimate route.

The route answers ``POST /v1/responses/input_tokens``: the provider's own
count when the selected upstream serves one (MiniMax does), a local
``o200k_base`` estimate otherwise.
"""

import pytest
from fastapi import Request

from llm_proxy.api.routers.protocol import create_protocol_router
from llm_proxy.protocols.openresponses.handler import openresponses_protocol
from llm_proxy.protocols.openresponses.input_tokens import _estimate_input_tokens
from llm_proxy.protocols.openresponses.schemas import InputTokensRequest


class TestRouteRegistration:
    """The route is mounted next to the Responses endpoint."""

    def test_route_paths(self):
        router = create_protocol_router(openresponses_protocol)
        post_paths = {route.path for route in router.routes if "POST" in (route.methods or set())}
        assert "/v1/responses/input_tokens" in post_paths
        assert "/v1/responses/input_tokens/" in post_paths


class TestLocalEstimate:
    """The fallback estimate counts the prompt text."""

    def test_plain_text_input(self):
        request = InputTokensRequest(model="m", input="hello world")
        assert _estimate_input_tokens(request) > 0

    def test_instructions_are_counted(self):
        without = _estimate_input_tokens(InputTokensRequest(model="m", input="hello"))
        with_instructions = _estimate_input_tokens(
            InputTokensRequest(model="m", input="hello", instructions="a long system prompt")
        )
        assert with_instructions > without

    def test_input_items_are_counted(self):
        request = InputTokensRequest(
            model="m",
            input=[
                {"type": "message", "role": "user", "content": "hello there"},
                {"type": "message", "role": "assistant", "content": "general kenobi"},
            ],
        )
        assert _estimate_input_tokens(request) > 0

    def test_empty_input_is_zero(self):
        assert _estimate_input_tokens(InputTokensRequest(model="m", input="")) == 0


class TestUpstreamForwarding:
    """The upstream is asked first; the estimate is the fallback."""

    @pytest.mark.asyncio
    async def test_upstream_count_is_returned(self, monkeypatch):
        from llm_proxy.api import context as api_context
        from llm_proxy.protocols.openresponses import input_tokens as route

        class _Adapter:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def input_tokens(self, body):
                self.body = body
                return {"input_tokens": 42}

        class _Selection:
            provider_model_name = "MiniMax-M3"

        class _Orchestrator:
            def select_next_provider(self):
                return _Selection()

        class _Services:
            async def adapter_factory(self, _request, _selection):
                return _Adapter()

        class _Context:
            orchestrator = _Orchestrator()
            services = _Services()

        async def _fake_build_request_context(*args, **kwargs):
            return _Context()

        monkeypatch.setattr(api_context, "build_request_context", _fake_build_request_context)

        result = await route.handle_input_tokens(
            InputTokensRequest(model="alias", input="hi"), Request({"type": "http"})
        )
        assert result.input_tokens == 42
        assert result.object == "response.input_tokens"

    @pytest.mark.asyncio
    async def test_provider_without_the_endpoint_falls_back(self, monkeypatch):
        from llm_proxy.api import context as api_context
        from llm_proxy.protocols.openresponses import input_tokens as route

        class _Adapter:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class _Selection:
            provider_model_name = "some-model"

        class _Orchestrator:
            def select_next_provider(self):
                return _Selection()

        class _Services:
            async def adapter_factory(self, _request, _selection):
                return _Adapter()

        class _Context:
            orchestrator = _Orchestrator()
            services = _Services()

        async def _fake_build_request_context(*args, **kwargs):
            return _Context()

        monkeypatch.setattr(api_context, "build_request_context", _fake_build_request_context)

        result = await route.handle_input_tokens(
            InputTokensRequest(model="m", input="hello world"), Request({"type": "http"})
        )
        assert result.input_tokens > 0
