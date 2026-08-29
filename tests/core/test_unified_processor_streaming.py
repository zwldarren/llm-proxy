"""Regression tests for UnifiedProcessor streaming behavior."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.config.types import LoggingConfig
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.processing.base import RequestContext, ServiceDependencies
from llm_proxy.core.processing.strategies import StreamingResponseMarker
from llm_proxy.core.processing.unified import UnifiedProcessor
from llm_proxy.core.provider_selector import ProviderSelectionResult
from llm_proxy.models import (
    ConversationContext,
    FunctionTool,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.observability.event_context import EventContext
from llm_proxy.protocols.anthropic.handler import anthropic_protocol
from llm_proxy.protocols.anthropic.schemas import MessagesRequest
from llm_proxy.protocols.openai.handler import openai_protocol
from llm_proxy.protocols.openai.schemas import ChatCompletionRequest
from llm_proxy.protocols.openresponses.handler import openresponses_protocol
from llm_proxy.web_search import WebSearchInterceptor
from llm_proxy.web_search.provider import SearchResult, WebSearchProvider, WebSearchResponse


class _FakeWebSearchProvider(WebSearchProvider):
    """Stub provider that returns deterministic results for tests."""

    async def search(self, query: str, config=None, **kwargs: Any) -> WebSearchResponse:
        return WebSearchResponse(
            results=[
                SearchResult(url=f"https://example.com?q={query}", title=f"R {query}", snippet="s")
            ],
            search_id="ws-test",
        )

    async def close(self) -> None:
        pass


def _build_unified_request() -> InternalRequest:
    return InternalRequest(
        model="glm-5",
        stream=True,
        conversation=ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[TextBlock(text="hello")],
                )
            ]
        ),
    )


def _build_mock_request() -> Any:
    req = MagicMock()
    req.state = MagicMock()
    req.state.request_id = "req-123"
    req.state.provider = "openai"
    req.is_disconnected = AsyncMock(return_value=False)
    req.url = MagicMock(path="/v1/chat/completions")
    req.method = "POST"
    req.headers = {}
    req.client = MagicMock(host="127.0.0.1")
    return req


def _build_tracing_registry() -> MagicMock:
    """Tracing registry mock with the async hooks the pipeline awaits."""
    registry = MagicMock()
    registry.on_request_start = AsyncMock()
    registry.on_error = AsyncMock()
    registry.on_stream_start = AsyncMock()
    registry.on_stream_chunk = AsyncMock()
    registry.on_stream_end = AsyncMock()
    registry.get_observation_id.return_value = None
    registry.get_trace_id.return_value = None
    return registry


async def _collect_stream_text(response: Any) -> str:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode())
        else:
            chunks.append(chunk)
    return "".join(chunks)


def _build_request_with_web_search() -> InternalRequest:
    return InternalRequest(
        model="glm-5",
        stream=True,
        conversation=ConversationContext(
            messages=[
                Message(
                    role="user", content=[TextBlock(text="search the web for quantum computing")]
                )
            ]
        ),
        tools=[FunctionTool(name="web_search", parameters={"type": "object"})],
        tool_choice=None,
    )


class TestWebSearchStage:
    """Verify WebSearchStage converts web_search tools for every protocol."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("protocol_name", ["anthropic", "openresponses", "openai"])
    async def test_web_search_stage_replaces_builtin_for_all_protocols(
        self, protocol_name: str
    ) -> None:
        from llm_proxy.core.processing.stages import PipelineState, WebSearchStage
        from llm_proxy.models.tools import OpenAIWebSearchTool

        interceptor = WebSearchInterceptor(_FakeWebSearchProvider())
        context = RequestContext(
            orchestrator=MagicMock(),
            services=ServiceDependencies(
                adapter_factory=AsyncMock(), web_search_interceptor=interceptor
            ),
            protocol_name=protocol_name,
        )
        state = PipelineState(
            raw_data={},
            unified_request=InternalRequest(
                model="glm-5",
                stream=True,
                conversation=ConversationContext(
                    messages=[Message(role="user", content=[TextBlock(text="hi")])]
                ),
                tools=[OpenAIWebSearchTool(name="web_search", type="web_search")],
            ),
            req=_build_mock_request(),
            strategy=MagicMock(),
            trace_id="t",
            event_context=MagicMock(),
            selection=ProviderSelectionResult(
                provider_name="ollama",
                provider_config=ProviderConfig(type="ollama"),
                provider_model_name="glm-5",
                priority=0,
            ),
        )

        await WebSearchStage().process(state, context)

        assert state.unified_request.tools is not None
        assert len(state.unified_request.tools) == 1
        assert isinstance(state.unified_request.tools[0], FunctionTool)
        assert state.unified_request.tools[0].name == "web_search"


class TestWebSearchStreaming:
    """Verify web search interception works across Anthropic and OpenResponses."""

    @pytest.mark.asyncio
    async def test_anthropic_streaming_emits_web_search_events(self) -> None:
        """Anthropic protocol should emit web_search_tool_result after provider tool call."""

        async def _provider_stream():
            yield {
                "id": "chatcmpl-ws",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_ws_1",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "quantum computing"}',
                                    },
                                }
                            ]
                        },
                    }
                ],
            }
            yield {
                "id": "chatcmpl-ws",
                "object": "chat.completion.chunk",
                "created": 2,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }

        async def _continuation_stream():
            yield {
                "id": "chatcmpl-ws-cont",
                "object": "chat.completion.chunk",
                "created": 3,
                "model": "glm-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Quantum computing is a rapidly evolving field."},
                    }
                ],
            }
            yield {
                "id": "chatcmpl-ws-cont",
                "object": "chat.completion.chunk",
                "created": 4,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=False)
        adapter.provider_name = "openai-compatible"
        adapter.stream_chat_completion = AsyncMock(
            side_effect=[_provider_stream(), _continuation_stream()]
        )

        orchestrator = MagicMock()
        orchestrator.should_retry.return_value = False
        orchestrator.select_next_provider.return_value = None
        orchestrator.needs_role_transform.return_value = False
        orchestrator.exhausted = False

        interceptor = WebSearchInterceptor(_FakeWebSearchProvider())
        context = RequestContext(
            orchestrator=orchestrator,
            services=ServiceDependencies(
                adapter_factory=AsyncMock(return_value=adapter),
                web_search_interceptor=interceptor,
            ),
            protocol_name="anthropic",
            proxy_web_search_active=True,
        )

        processor = UnifiedProcessor(protocol_endpoint=anthropic_protocol)
        streaming_marker = StreamingResponseMarker(_build_request_with_web_search(), adapter)
        response = await processor._streaming_processor.process(
            streaming_marker=streaming_marker,
            raw_request_data={"model": "glm-5", "stream": True},
            req=_build_mock_request(),
            context=context,
            trace_id="trace-ws-anthropic",
        )

        payload = await _collect_stream_text(response)
        assert '"type":"web_search_tool_result"' in payload
        assert '"web_search_requests":1' in payload
        assert "Quantum computing is a rapidly evolving field." in payload
        assert "event: message_delta" in payload

    @pytest.mark.asyncio
    async def test_openresponses_streaming_emits_web_search_call(self) -> None:
        """OpenResponses protocol should emit web_search_call output item and continue."""

        async def _provider_stream():
            yield {
                "id": "chatcmpl-ws-or",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_ws_or_1",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "quantum computing"}',
                                    },
                                }
                            ]
                        },
                    }
                ],
            }
            yield {
                "id": "chatcmpl-ws-or",
                "object": "chat.completion.chunk",
                "created": 2,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }

        async def _continuation_stream():
            yield {
                "id": "chatcmpl-ws-or-cont",
                "object": "chat.completion.chunk",
                "created": 3,
                "model": "glm-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Quantum computing is a rapidly evolving field."},
                    }
                ],
            }
            yield {
                "id": "chatcmpl-ws-or-cont",
                "object": "chat.completion.chunk",
                "created": 4,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=False)
        adapter.provider_name = "openai-compatible"
        adapter.stream_chat_completion = AsyncMock(
            side_effect=[_provider_stream(), _continuation_stream()]
        )

        orchestrator = MagicMock()
        orchestrator.should_retry.return_value = False
        orchestrator.select_next_provider.return_value = None
        orchestrator.needs_role_transform.return_value = False
        orchestrator.exhausted = False

        interceptor = WebSearchInterceptor(_FakeWebSearchProvider())
        context = RequestContext(
            orchestrator=orchestrator,
            services=ServiceDependencies(
                adapter_factory=AsyncMock(return_value=adapter),
                web_search_interceptor=interceptor,
            ),
            protocol_name="openresponses",
            proxy_web_search_active=True,
        )

        processor = UnifiedProcessor(protocol_endpoint=openresponses_protocol)
        streaming_marker = StreamingResponseMarker(_build_request_with_web_search(), adapter)
        response = await processor._streaming_processor.process(
            streaming_marker=streaming_marker,
            raw_request_data={"model": "glm-5", "stream": True},
            req=_build_mock_request(),
            context=context,
            trace_id="trace-ws-openresponses",
        )

        payload = await _collect_stream_text(response)
        assert '"type":"web_search_call"' in payload
        assert '"query":"quantum computing"' in payload
        assert "Quantum computing is a rapidly evolving field." in payload
        assert "response.completed" in payload

    @pytest.mark.asyncio
    async def test_openai_streaming_builtin_web_search_replaced(self) -> None:
        """Chat Completions protocol with web_search tool is replaced by a function tool.

        Unlike Anthropic/OpenResponses, the OpenAI Chat Completions transformer does
        not emit special web_search events; it streams the function_call back to the
        client, which is responsible for executing the tool. The proxy's job is to make
        sure the upstream provider never receives a native web_search tool.
        """

        async def _provider_stream():
            yield {
                "id": "chatcmpl-ws-cc",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_ws_cc_1",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "quantum computing"}',
                                    },
                                }
                            ]
                        },
                    }
                ],
            }
            yield {
                "id": "chatcmpl-ws-cc",
                "object": "chat.completion.chunk",
                "created": 2,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=False)
        adapter.provider_name = "openai-compatible"
        adapter.stream_chat_completion = AsyncMock(return_value=_provider_stream())

        orchestrator = MagicMock()
        orchestrator.should_retry.return_value = False
        orchestrator.select_next_provider.return_value = None
        orchestrator.needs_role_transform.return_value = False
        orchestrator.exhausted = False

        interceptor = WebSearchInterceptor(_FakeWebSearchProvider())
        context = RequestContext(
            orchestrator=orchestrator,
            services=ServiceDependencies(
                adapter_factory=AsyncMock(return_value=adapter),
                web_search_interceptor=interceptor,
            ),
            protocol_name="openai",
            proxy_web_search_active=True,
        )

        processor = UnifiedProcessor(protocol_endpoint=openai_protocol)
        streaming_marker = StreamingResponseMarker(_build_request_with_web_search(), adapter)
        response = await processor._streaming_processor.process(
            streaming_marker=streaming_marker,
            raw_request_data={"model": "glm-5", "stream": True},
            req=_build_mock_request(),
            context=context,
            trace_id="trace-ws-openai",
        )

        payload = await _collect_stream_text(response)
        assert '"name":"web_search"' in payload
        assert '"finish_reason":"tool_calls"' in payload
        assert "data: [DONE]" in payload

        # Verify the request sent to the provider had a plain function tool, not web_search builtin
        sent_request = adapter.stream_chat_completion.call_args_list[0].args[0]
        assert sent_request.tools is not None
        assert len(sent_request.tools) == 1
        assert isinstance(sent_request.tools[0], FunctionTool)
        assert sent_request.tools[0].name == "web_search"

    @pytest.mark.asyncio
    async def test_client_web_search_function_tool_not_intercepted(self) -> None:
        """A client-defined function tool named 'web_search' must NOT be hijacked.

        When the proxy did NOT take over web search for the request
        (proxy_web_search_active is False), the streaming processor must leave a
        tool call named 'web_search' for the client to handle instead of running
        the proxy's own search engine and injecting web_search_tool_result blocks.
        """

        async def _provider_stream():
            yield {
                "id": "chatcmpl-ws-client",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_ws_client_1",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "quantum", "max_results": 5}',
                                    },
                                }
                            ]
                        },
                    }
                ],
            }
            yield {
                "id": "chatcmpl-ws-client",
                "object": "chat.completion.chunk",
                "created": 2,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=False)
        adapter.provider_name = "openai-compatible"
        adapter.stream_chat_completion = AsyncMock(return_value=_provider_stream())

        orchestrator = MagicMock()
        orchestrator.should_retry.return_value = False
        orchestrator.select_next_provider.return_value = None
        orchestrator.needs_role_transform.return_value = False
        orchestrator.exhausted = False

        interceptor = WebSearchInterceptor(_FakeWebSearchProvider())
        context = RequestContext(
            orchestrator=orchestrator,
            services=ServiceDependencies(
                adapter_factory=AsyncMock(return_value=adapter),
                web_search_interceptor=interceptor,
            ),
            protocol_name="anthropic",
            # Proxy web search is enabled globally, but the proxy did NOT take
            # over this request -- the client owns the 'web_search' tool.
            proxy_web_search_active=False,
        )

        processor = UnifiedProcessor(protocol_endpoint=anthropic_protocol)
        streaming_marker = StreamingResponseMarker(_build_request_with_web_search(), adapter)
        response = await processor._streaming_processor.process(
            streaming_marker=streaming_marker,
            raw_request_data={"model": "glm-5", "stream": True},
            req=_build_mock_request(),
            context=context,
            trace_id="trace-ws-client",
        )

        payload = await _collect_stream_text(response)
        # The client's tool call is emitted to the client...
        assert '"name":"web_search"' in payload
        # ...but the proxy must NOT inject its own web search results or usage.
        assert '"type":"web_search_tool_result"' not in payload
        assert '"web_search_requests"' not in payload


@pytest.mark.asyncio
async def test_streaming_prefetch_accepts_dict_chunks() -> None:
    """Dict chunks should start stream and not trigger empty-stream fallback."""

    async def _mock_stream():
        yield {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "glm-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hello"},
                }
            ],
        }
        yield "[DONE]"

    adapter = MagicMock()
    adapter.supports_native_streaming = MagicMock(return_value=False)
    adapter.provider_name = "openai"
    adapter.stream_chat_completion = AsyncMock(return_value=_mock_stream())

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False
    orchestrator.select_next_provider.return_value = None
    orchestrator.needs_role_transform.return_value = False
    orchestrator.exhausted = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    processor = UnifiedProcessor(protocol_endpoint=openai_protocol)
    unified_request = _build_unified_request()
    streaming_marker = StreamingResponseMarker(unified_request, adapter)
    response = await processor._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": "glm-5", "stream": True},
        req=_build_mock_request(),
        context=context,
        trace_id="trace-1",
    )

    assert response.media_type == "text/event-stream"

    payload = await _collect_stream_text(response)
    assert '"content":"hello"' in payload
    assert "data: [DONE]" in payload

    orchestrator.should_retry.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_prefetch_accepts_future_delta_fields() -> None:
    """Unknown future delta fields should still mark stream as started."""

    async def _mock_stream():
        yield {
            "id": "chatcmpl-2",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "glm-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "future_signal": {
                            "kind": "new_field",
                        }
                    },
                }
            ],
        }
        yield "[DONE]"

    adapter = MagicMock()
    adapter.supports_native_streaming = MagicMock(return_value=False)
    adapter.provider_name = "openai"
    adapter.stream_chat_completion = AsyncMock(return_value=_mock_stream())

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False
    orchestrator.select_next_provider.return_value = None
    orchestrator.needs_role_transform.return_value = False
    orchestrator.exhausted = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    processor = UnifiedProcessor(protocol_endpoint=openai_protocol)
    unified_request = _build_unified_request()
    streaming_marker = StreamingResponseMarker(unified_request, adapter)
    response = await processor._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": "glm-5", "stream": True},
        req=_build_mock_request(),
        context=context,
        trace_id="trace-2",
    )

    payload = await _collect_stream_text(response)
    assert "data: [DONE]" in payload

    orchestrator.should_retry.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_forwards_all_dict_chunks_after_prefetch() -> None:
    """After prefetch, subsequent dict chunks must still be streamed to client."""

    async def _mock_stream():
        yield {
            "id": "chatcmpl-3",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "glm-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hel"},
                }
            ],
        }
        yield {
            "id": "chatcmpl-3",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "glm-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                }
            ],
        }
        yield "[DONE]"

    adapter = MagicMock()
    adapter.supports_native_streaming = MagicMock(return_value=False)
    adapter.provider_name = "openai"
    adapter.stream_chat_completion = AsyncMock(return_value=_mock_stream())

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False
    orchestrator.select_next_provider.return_value = None
    orchestrator.needs_role_transform.return_value = False
    orchestrator.exhausted = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    processor = UnifiedProcessor(protocol_endpoint=openai_protocol)
    unified_request = _build_unified_request()
    streaming_marker = StreamingResponseMarker(unified_request, adapter)
    response = await processor._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": "glm-5", "stream": True},
        req=_build_mock_request(),
        context=context,
        trace_id="trace-3",
    )

    payload = await _collect_stream_text(response)
    assert '"content":"Hel"' in payload
    assert '"content":"lo"' in payload
    assert "data: [DONE]" in payload


@pytest.mark.asyncio
async def test_streaming_prefetch_retries_retryable_finish_reason_before_output() -> None:
    """Retry with next provider when prefetch receives retryable finish_reason."""

    primary_stream_closed = False

    async def _primary_stream():
        nonlocal primary_stream_closed
        try:
            yield {
                "id": "chatcmpl-primary",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {"role": "assistant"}}],
            }
            yield {
                "id": "chatcmpl-primary",
                "object": "chat.completion.chunk",
                "created": 2,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "network_error"}],
            }
        finally:
            primary_stream_closed = True

    async def _fallback_stream():
        yield {
            "id": "chatcmpl-fallback",
            "object": "chat.completion.chunk",
            "created": 3,
            "model": "glm-5",
            "choices": [{"index": 0, "delta": {"content": "hello from fallback"}}],
        }
        yield "[DONE]"

    primary_adapter = MagicMock()
    primary_adapter.supports_native_streaming = MagicMock(return_value=False)
    primary_adapter.provider_name = "primary"
    primary_adapter.close = AsyncMock()
    primary_adapter.stream_chat_completion = AsyncMock(return_value=_primary_stream())

    fallback_adapter = MagicMock()
    fallback_adapter.supports_native_streaming = MagicMock(return_value=False)
    fallback_adapter.provider_name = "fallback"
    fallback_adapter.close = AsyncMock()
    fallback_adapter.stream_chat_completion = AsyncMock(return_value=_fallback_stream())

    selection = MagicMock()
    selection.provider_name = "fallback"
    selection.provider_model_name = "glm-5"
    selection.parameter_overrides = {}

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = True
    orchestrator.select_next_provider.return_value = selection
    orchestrator.needs_role_transform.return_value = False
    orchestrator.exhausted = False

    adapter_factory = AsyncMock(return_value=fallback_adapter)
    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=adapter_factory),
    )

    req = _build_mock_request()
    processor = UnifiedProcessor(protocol_endpoint=openai_protocol)
    unified_request = _build_unified_request()
    streaming_marker = StreamingResponseMarker(unified_request, primary_adapter)
    response = await processor._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": "glm-5", "stream": True},
        req=req,
        context=context,
        trace_id="trace-4",
    )

    payload = await _collect_stream_text(response)
    assert '"content":"hello from fallback"' in payload
    assert "data: [DONE]" in payload
    adapter_factory.assert_awaited_once()
    orchestrator.should_retry.assert_called_once()
    assert req.state.provider == "fallback"
    assert primary_stream_closed is True


@pytest.mark.asyncio
async def test_cancel_token_passed_to_stream_chat_completion() -> None:
    """The cancel_token must be passed to adapter.stream_chat_completion.

    This ensures the provider's inner streaming loop can detect the token
    and force-close the TCP connection, preventing FD racing on reconnect.
    """

    async def _mock_stream():
        yield {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "glm-5",
            "choices": [{"index": 0, "delta": {"content": "hello"}}],
        }
        yield "[DONE]"

    adapter = MagicMock()
    adapter.supports_native_streaming = MagicMock(return_value=False)
    adapter.provider_name = "openai"
    adapter.stream_chat_completion = AsyncMock(return_value=_mock_stream())

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False
    orchestrator.select_next_provider.return_value = None
    orchestrator.needs_role_transform.return_value = False
    orchestrator.exhausted = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    processor = UnifiedProcessor(protocol_endpoint=openai_protocol)
    unified_request = _build_unified_request()
    streaming_marker = StreamingResponseMarker(unified_request, adapter)
    response = await processor._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": "glm-5", "stream": True},
        req=_build_mock_request(),
        context=context,
        trace_id="trace-cancel-token",
    )

    payload = await _collect_stream_text(response)
    assert "data: [DONE]" in payload

    call_args = adapter.stream_chat_completion.call_args
    assert call_args is not None, "stream_chat_completion should have been called"
    cancel_token = call_args.kwargs.get("cancel_token")
    assert cancel_token is not None, "cancel_token kwarg must be passed to stream_chat_completion"
    assert isinstance(cancel_token, asyncio.Event), (
        f"cancel_token must be an asyncio.Event, got {type(cancel_token).__name__}"
    )
    assert cancel_token.is_set() is False, "cancel_token should not be set during normal streaming"


@pytest.mark.asyncio
async def test_cancel_token_set_stops_stream_generator_early() -> None:
    """Setting the cancel_token must cause the stream_generator to stop.

    When the cancel_token is set (e.g. on client disconnect), the
    stream_generator must break its iteration loop early and not yield
    all provider chunks. This forces the provider connection to close.
    """
    captured_cancel_token: asyncio.Event | None = None

    async def _capture_and_stream(*args: Any, **kwargs: Any):
        nonlocal captured_cancel_token
        captured_cancel_token = kwargs.get("cancel_token")
        for chunk in [
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {"content": "first"}}],
            },
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 2,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {"content": "second"}}],
            },
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 3,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {"content": "third"}}],
            },
            "[DONE]",
        ]:
            yield chunk

    adapter = MagicMock()
    adapter.supports_native_streaming = MagicMock(return_value=False)
    adapter.provider_name = "openai"
    adapter.stream_chat_completion = AsyncMock(side_effect=_capture_and_stream)

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False
    orchestrator.select_next_provider.return_value = None
    orchestrator.needs_role_transform.return_value = False
    orchestrator.exhausted = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    processor = UnifiedProcessor(protocol_endpoint=openai_protocol)
    unified_request = _build_unified_request()
    streaming_marker = StreamingResponseMarker(unified_request, adapter)
    response = await processor._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": "glm-5", "stream": True},
        req=_build_mock_request(),
        context=context,
        trace_id="trace-cancel-early",
    )

    assert captured_cancel_token is not None, (
        "cancel_token should have been captured from stream_chat_completion call"
    )

    captured_cancel_token.set()

    payload = await _collect_stream_text(response)
    assert '"content":"first"' in payload, "pre-fetched first chunk should still be streamed"
    assert '"content":"second"' not in payload, (
        "provider chunks after cancel_token.set() must NOT be streamed"
    )
    assert '"content":"third"' not in payload, (
        "provider chunks after cancel_token.set() must NOT be streamed"
    )
    assert captured_cancel_token.is_set(), "cancel_token should have been set by the test"


@pytest.mark.asyncio
async def test_client_disconnect_sets_cancel_token_and_stops_stream() -> None:
    """Client disconnection must set the cancel_token and stop the stream.

    When the client's receive channel reports ``http.disconnect`` in the
    streaming loop, the cancel_token must be set and the stream must stop
    yielding chunks. This is the primary fix for FD racing on disconnect+reconnect.
    """
    disconnect_count = 0
    captured_cancel_token: asyncio.Event | None = None

    async def _receive_disconnect():
        nonlocal disconnect_count
        disconnect_count += 1
        return {"type": "http.disconnect"}

    async def _capture_and_stream(*args: Any, **kwargs: Any):
        nonlocal captured_cancel_token
        captured_cancel_token = kwargs.get("cancel_token")
        for i in range(25):
            yield {
                "id": f"chatcmpl-{i}",
                "object": "chat.completion.chunk",
                "created": i,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {"content": f"chunk{i}"}}],
            }
        yield "[DONE]"

    adapter = MagicMock()
    adapter.supports_native_streaming = MagicMock(return_value=False)
    adapter.provider_name = "openai"
    adapter.stream_chat_completion = AsyncMock(side_effect=_capture_and_stream)

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False
    orchestrator.select_next_provider.return_value = None
    orchestrator.needs_role_transform.return_value = False
    orchestrator.exhausted = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    req = _build_mock_request()
    req._receive = _receive_disconnect

    processor = UnifiedProcessor(protocol_endpoint=openai_protocol)
    unified_request = _build_unified_request()
    streaming_marker = StreamingResponseMarker(unified_request, adapter)
    response = await processor._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": "glm-5", "stream": True},
        req=req,
        context=context,
        trace_id="trace-disconnect",
    )

    assert captured_cancel_token is not None

    payload = await _collect_stream_text(response)
    chunk_count_in_payload = payload.count('"content":"chunk')
    assert '"content":"chunk0"' in payload, "pre-fetched first chunk should be streamed"
    assert disconnect_count >= 1, (
        f"receive() should have been polled for disconnect at least once, got {disconnect_count}"
    )
    assert chunk_count_in_payload <= 11, (
        f"Stream should stop after ~10 chunks when disconnect is detected, "
        f"got {chunk_count_in_payload} chunks"
    )
    assert captured_cancel_token.is_set(), "cancel_token must be set when client disconnects"


@pytest.mark.asyncio
async def test_normal_streaming_completes_without_cancel_token_interference() -> None:
    """Streaming must complete normally when client stays connected.

    Control test: when the client's receive channel never reports a
    disconnect, the full stream (including [DONE]) must be delivered and
    cancel_token must remain unset.
    """
    captured_cancel_token: asyncio.Event | None = None

    async def _capture_and_stream(*args: Any, **kwargs: Any):
        nonlocal captured_cancel_token
        captured_cancel_token = kwargs.get("cancel_token")
        for i in range(3):
            yield {
                "id": f"chatcmpl-{i}",
                "object": "chat.completion.chunk",
                "created": i,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {"content": f"chunk{i}"}}],
            }
        yield "[DONE]"

    adapter = MagicMock()
    adapter.supports_native_streaming = MagicMock(return_value=False)
    adapter.provider_name = "openai"
    adapter.stream_chat_completion = AsyncMock(side_effect=_capture_and_stream)

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False
    orchestrator.select_next_provider.return_value = None
    orchestrator.needs_role_transform.return_value = False
    orchestrator.exhausted = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    processor = UnifiedProcessor(protocol_endpoint=openai_protocol)
    unified_request = _build_unified_request()
    streaming_marker = StreamingResponseMarker(unified_request, adapter)

    async def _receive_connected():
        # A live connection: no message ever arrives (checks time out).
        await asyncio.sleep(30)
        return {"type": "http.request", "more_body": False}

    req = _build_mock_request()
    req._receive = _receive_connected
    response = await processor._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": "glm-5", "stream": True},
        req=req,
        context=context,
        trace_id="trace-normal",
    )

    payload = await _collect_stream_text(response)
    assert "data: [DONE]" in payload, (
        "full stream should complete normally when client stays connected"
    )
    assert captured_cancel_token is not None
    assert captured_cancel_token.is_set() is False, (
        "cancel_token should not be set during normal streaming"
    )


@pytest.mark.asyncio
async def test_native_streaming_passthrough_skips_transformer() -> None:
    """When adapter supports native streaming, raw SSE frames pass through unchanged."""

    async def _mock_native_stream(*args: Any, **kwargs: Any):
        yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
        yield 'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'

    adapter = MagicMock()
    adapter.supports_native_streaming = MagicMock(return_value=True)
    adapter.provider_name = "anthropic"
    adapter.stream_chat_completion_native = AsyncMock(return_value=_mock_native_stream())

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False
    orchestrator.select_next_provider.return_value = None
    orchestrator.needs_role_transform.return_value = False
    orchestrator.exhausted = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    processor = UnifiedProcessor(protocol_endpoint=anthropic_protocol)
    unified_request = _build_unified_request()
    unified_request.model = "claude-3-sonnet"
    streaming_marker = StreamingResponseMarker(unified_request, adapter)
    response = await processor._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": "claude-3-sonnet", "stream": True},
        req=_build_mock_request(),
        context=context,
        trace_id="trace-native",
    )

    assert response.media_type == "text/event-stream"

    payload = await _collect_stream_text(response)
    assert "event: message_start" in payload
    assert 'data: {"type":"message_start"}' in payload
    assert "event: content_block_delta" in payload
    # No OpenAI-style data: [DONE] because native path skips the transformer
    assert "data: [DONE]" not in payload


@pytest.mark.asyncio
async def test_native_streaming_traces_chunks() -> None:
    """Native streaming path must still call tracing on_stream_chunk for every chunk."""

    async def _mock_native_stream(*args: Any, **kwargs: Any):
        yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
        yield 'event: content_block_delta\ndata: {"delta":"hello"}\n\n'

    adapter = MagicMock()
    adapter.supports_native_streaming = MagicMock(return_value=True)
    adapter.provider_name = "anthropic"
    adapter.stream_chat_completion_native = AsyncMock(return_value=_mock_native_stream())

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False
    orchestrator.select_next_provider.return_value = None
    orchestrator.needs_role_transform.return_value = False
    orchestrator.exhausted = False

    mock_registry = MagicMock()
    mock_registry.on_stream_chunk = AsyncMock()
    mock_registry.on_stream_start = AsyncMock()
    mock_registry.on_stream_end = AsyncMock()

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(
            adapter_factory=AsyncMock(return_value=adapter),
            tracing_registry=mock_registry,
        ),
    )

    processor = UnifiedProcessor(protocol_endpoint=anthropic_protocol)
    unified_request = _build_unified_request()
    unified_request.model = "claude-3-sonnet"
    streaming_marker = StreamingResponseMarker(unified_request, adapter)
    response = await processor._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": "claude-3-sonnet", "stream": True},
        req=_build_mock_request(),
        context=context,
        trace_id="trace-native-tracing",
    )

    # Consume the stream so on_stream_chunk is invoked
    await _collect_stream_text(response)

    chunk_calls = mock_registry.on_stream_chunk.call_args_list
    assert len(chunk_calls) >= 2, "tracing should record native SSE chunks"
    # First call should be the message_start frame
    assert "message_start" in str(chunk_calls[0])


@pytest.mark.asyncio
async def test_native_streaming_injects_model_name() -> None:
    """Native streaming must overwrite provider's model with the user-facing alias."""

    async def _mock_native_stream(*args: Any, **kwargs: Any):
        yield (
            "event: message_start\n"
            'data: {"type":"message_start","message":'
            '{"id":"msg_123","model":"claude-3-sonnet-20240229"}}\n\n'
        )
        yield 'event: content_block_delta\ndata: {"delta":"hello"}\n\n'

    adapter = MagicMock()
    adapter.supports_native_streaming = MagicMock(return_value=True)
    adapter.provider_name = "anthropic"
    adapter.stream_chat_completion_native = AsyncMock(return_value=_mock_native_stream())

    orchestrator = MagicMock()
    orchestrator.should_retry.return_value = False
    orchestrator.select_next_provider.return_value = None
    orchestrator.needs_role_transform.return_value = False
    orchestrator.exhausted = False

    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=adapter)),
    )

    processor = UnifiedProcessor(protocol_endpoint=anthropic_protocol)
    unified_request = _build_unified_request()
    unified_request.model = "claude-3-sonnet"
    streaming_marker = StreamingResponseMarker(unified_request, adapter)
    response = await processor._streaming_processor.process(
        streaming_marker=streaming_marker,
        raw_request_data={"model": "claude-3-sonnet", "stream": True},
        req=_build_mock_request(),
        context=context,
        trace_id="trace-native-model",
    )

    payload = await _collect_stream_text(response)
    assert "claude-3-sonnet-20240229" not in payload, (
        "provider's internal model name should be masked"
    )
    assert 'model":"claude-3-sonnet"' in payload, (
        "user-facing model alias must be injected into message_start"
    )


class TestModelEchoConsistency:
    """The client-requested alias must be echoed consistently across the
    streaming, native passthrough, and web-search continuation paths."""

    @staticmethod
    async def _run_pipeline(protocol_endpoint, selection, adapter, request_model, monkeypatch):
        """Run the real pipeline for a streaming request; returns the SSE text."""
        orchestrator = MagicMock()
        orchestrator.should_retry.return_value = False
        orchestrator.select_next_provider.return_value = selection
        orchestrator.needs_role_transform.return_value = False
        orchestrator.exhausted = False

        context = RequestContext(
            orchestrator=orchestrator,
            services=ServiceDependencies(
                adapter_factory=AsyncMock(return_value=adapter),
                tracing_registry=_build_tracing_registry(),
            ),
        )

        monkeypatch.setattr(
            "llm_proxy.core.processing.unified.load_logging_config",
            lambda: LoggingConfig(enable_database_logging=True),
        )

        processor = UnifiedProcessor(protocol_endpoint=protocol_endpoint)
        response = await processor.process(request_model, _build_mock_request(), context)
        return await _collect_stream_text(response)

    @pytest.mark.asyncio
    async def test_stream_chunks_echo_user_facing_alias(self, monkeypatch) -> None:
        """Streaming chunks must echo the client-requested alias, not the
        resolved provider model name — through the real pipeline with a
        model mapping (fast -> gpt-4o-mini)."""

        async def _provider_stream():
            yield {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "hi"},
                        "finish_reason": None,
                    }
                ],
            }
            yield {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 2,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield "[DONE]"

        selection = ProviderSelectionResult(
            provider_name="openai",
            provider_config=ProviderConfig(type="openai", api_key="test-key"),
            provider_model_name="gpt-4o-mini",
            priority=1,
        )

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=False)
        adapter.provider_name = "openai"
        adapter.stream_chat_completion = AsyncMock(return_value=_provider_stream())

        request_model = ChatCompletionRequest(
            model="fast",
            messages=[{"role": "user", "content": "hello", "id": "user-1"}],
            stream=True,
        )
        payload = await self._run_pipeline(
            openai_protocol, selection, adapter, request_model, monkeypatch
        )

        assert '"model":"fast"' in payload
        assert '"model":"gpt-4o-mini"' not in payload

    @pytest.mark.asyncio
    async def test_native_streaming_injects_user_facing_alias(self, monkeypatch) -> None:
        """Native anthropic streaming must inject the client-requested alias
        (not the resolved provider model name) into message_start — through
        the real pipeline with a model mapping (fast -> claude-3-5-sonnet)."""

        async def _mock_native_stream(*args: Any, **kwargs: Any):
            yield (
                "event: message_start\n"
                'data: {"type":"message_start","message":'
                '{"id":"msg_123","model":"claude-3-5-sonnet-20241022"}}\n\n'
            )
            yield 'event: content_block_delta\ndata: {"delta":"hello"}\n\n'

        selection = ProviderSelectionResult(
            provider_name="anthropic",
            provider_config=ProviderConfig(type="anthropic", api_key="test-key"),
            provider_model_name="claude-3-5-sonnet-20241022",
            priority=1,
        )

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=True)
        adapter.provider_name = "anthropic"
        adapter.stream_chat_completion_native = AsyncMock(return_value=_mock_native_stream())

        request_model = MessagesRequest(
            model="fast",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        payload = await self._run_pipeline(
            anthropic_protocol, selection, adapter, request_model, monkeypatch
        )

        assert "claude-3-5-sonnet-20241022" not in payload, (
            "provider's internal model name should be masked"
        )
        assert 'model":"fast"' in payload, (
            "client-requested alias must be injected into message_start"
        )

    @pytest.mark.asyncio
    async def test_web_search_continuation_echoes_user_facing_alias(self) -> None:
        """Web search continuation chunks must echo the same alias as the
        main stream."""

        async def _provider_stream():
            yield {
                "id": "chatcmpl-ws",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_ws_1",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "quantum computing"}',
                                    },
                                }
                            ]
                        },
                    }
                ],
            }
            yield {
                "id": "chatcmpl-ws",
                "object": "chat.completion.chunk",
                "created": 2,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }

        async def _continuation_stream():
            yield {
                "id": "chatcmpl-ws-cont",
                "object": "chat.completion.chunk",
                "created": 3,
                "model": "glm-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Quantum computing is a rapidly evolving field."},
                    }
                ],
            }
            yield {
                "id": "chatcmpl-ws-cont",
                "object": "chat.completion.chunk",
                "created": 4,
                "model": "glm-5",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=False)
        adapter.provider_name = "openai-compatible"
        adapter.stream_chat_completion = AsyncMock(
            side_effect=[_provider_stream(), _continuation_stream()]
        )

        orchestrator = MagicMock()
        orchestrator.should_retry.return_value = False
        orchestrator.select_next_provider.return_value = None
        orchestrator.needs_role_transform.return_value = False
        orchestrator.exhausted = False

        interceptor = WebSearchInterceptor(_FakeWebSearchProvider())
        context = RequestContext(
            orchestrator=orchestrator,
            services=ServiceDependencies(
                adapter_factory=AsyncMock(return_value=adapter),
                web_search_interceptor=interceptor,
            ),
            protocol_name="anthropic",
            proxy_web_search_active=True,
        )

        processor = UnifiedProcessor(protocol_endpoint=anthropic_protocol)
        unified_request = _build_request_with_web_search()
        unified_request.model = "glm-5"
        unified_request.user_facing_model = "fast"
        streaming_marker = StreamingResponseMarker(unified_request, adapter)
        response = await processor._streaming_processor.process(
            streaming_marker=streaming_marker,
            raw_request_data={"model": "fast", "stream": True},
            req=_build_mock_request(),
            context=context,
            trace_id="trace-ws-alias",
            event_context=EventContext(
                request_id="req-123", trace_id="trace-ws-alias", model="fast"
            ),
        )

        payload = await _collect_stream_text(response)
        assert "Quantum computing is a rapidly evolving field." in payload
        assert '"model":"fast"' in payload
        assert '"model":"glm-5"' not in payload

    def test_parameter_override_service_preserves_user_facing_model(self) -> None:
        """apply() must carry the client-requested alias across the re-parse
        (fallback path)."""
        from llm_proxy.core.processing.stages.parameter_override import (
            ParameterOverrideService,
        )
        from llm_proxy.protocols.registry import get_protocol_serializer

        serializer = get_protocol_serializer("openai")
        service = ParameterOverrideService(serializer)
        request = _build_unified_request()
        request.model = "gpt-4o-mini"
        request.user_facing_model = "fast"
        raw = {
            "model": "fast",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        _, new_request = service.apply(
            raw_data=raw,
            unified_request=request,
            parameter_overrides={"max_tokens": 100},
            provider_model_name="gpt-4o-mini",
            request_id="req-1",
        )
        assert new_request.user_facing_model == "fast"
        assert new_request.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_setup_fallback_provider_preserves_user_facing_model(self) -> None:
        """setup_fallback_provider must keep the client-requested alias on
        the fallback request — both the in-place branch (no overrides) and
        the re-parsed branch (with overrides)."""
        from llm_proxy.core.processing.fallback import setup_fallback_provider
        from llm_proxy.core.processing.stages.parameter_override import (
            ParameterOverrideService,
        )
        from llm_proxy.protocols.registry import get_protocol_serializer

        serializer = get_protocol_serializer("openai")
        service = ParameterOverrideService(serializer)
        context = RequestContext(
            orchestrator=MagicMock(),
            services=ServiceDependencies(adapter_factory=AsyncMock(return_value=MagicMock())),
        )
        req = _build_mock_request()
        raw = {
            "model": "fast",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }

        for overrides in (None, {"max_tokens": 100}):
            selection = ProviderSelectionResult(
                provider_name="openai",
                provider_config=ProviderConfig(type="openai", api_key="test-key"),
                provider_model_name="gpt-4o-mini",
                priority=1,
                parameter_overrides=overrides,
            )
            unified_request = _build_unified_request()
            unified_request.model = "glm-5"
            unified_request.user_facing_model = "fast"
            _, new_request = await setup_fallback_provider(
                selection,
                req,
                unified_request,
                raw,
                context,
                service,
            )
            assert new_request.user_facing_model == "fast"
            assert new_request.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_setup_fallback_provider_applies_overrides_to_pristine_body() -> None:
    """Regression: a failed provider's parameter overrides must not leak into
    the fallback attempt — overrides are applied to the pristine client body."""
    from llm_proxy.core.processing.fallback import setup_fallback_provider
    from llm_proxy.core.processing.stages.parameter_override import (
        ParameterOverrideService,
    )
    from llm_proxy.protocols.registry import get_protocol_serializer

    serializer = get_protocol_serializer("openai")
    service = ParameterOverrideService(serializer)
    context = RequestContext(
        orchestrator=MagicMock(),
        services=ServiceDependencies(adapter_factory=AsyncMock(return_value=MagicMock())),
    )
    req = _build_mock_request()
    pristine_raw = {
        "model": "fast",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "stream": True,
    }

    # The failed provider's attempt: its overrides raised temperature and
    # injected a provider-specific key.
    _failed_raw, failed_request = service.apply(
        raw_data=pristine_raw,
        unified_request=_build_unified_request(),
        parameter_overrides={"temperature": 1.0, "x_provider1_flag": True},
        provider_model_name="provider-1-model",
        request_id="req-1",
    )
    assert failed_request.params is not None
    assert failed_request.params.temperature == 1.0

    selection = ProviderSelectionResult(
        provider_name="provider-2",
        provider_config=ProviderConfig(type="openai", api_key="test-key"),
        provider_model_name="gpt-4o-mini",
        priority=2,
        parameter_overrides={"max_tokens": 64},
    )
    result = await setup_fallback_provider(
        selection, req, failed_request, pristine_raw, context, service
    )
    assert result is not None
    _, new_request = result

    # The fallback provider's own overrides apply...
    assert new_request.params is not None
    assert new_request.params.max_tokens == 64
    # ...the pristine client value survives (not the failed provider's 1.0)...
    assert new_request.params.temperature == 0.2
    # ...and the failed provider's injected key neither persists nor stays
    # exempt from field-policy stripping.
    assert "x_provider1_flag" not in (new_request.extra or {})
    assert new_request._override_injected_keys == {"max_tokens"}


@pytest.mark.asyncio
async def test_setup_fallback_provider_skips_provider_rejecting_stage_rerun() -> None:
    """When the per-provider stage re-run rejects the request (e.g. an
    unresolvable proxy-local previous_response_id on a non-native upstream),
    the selector advances to the next provider instead of aborting."""
    from llm_proxy.core.processing.fallback import setup_fallback_provider
    from llm_proxy.core.processing.stages.parameter_override import (
        ParameterOverrideService,
    )
    from llm_proxy.protocols.registry import get_protocol_serializer

    serializer = get_protocol_serializer("openai")
    service = ParameterOverrideService(serializer)

    bad_selection = ProviderSelectionResult(
        provider_name="bad",
        provider_config=ProviderConfig(type="openai", api_key="test-key"),
        provider_model_name="gpt-4o-mini",
        priority=1,
        parameter_overrides=None,
    )
    good_selection = ProviderSelectionResult(
        provider_name="good",
        provider_config=ProviderConfig(type="openai", api_key="test-key"),
        provider_model_name="gpt-4o",
        priority=2,
        parameter_overrides=None,
    )
    orchestrator = MagicMock()
    orchestrator.select_next_provider.side_effect = [good_selection, None]

    bad_adapter = MagicMock()
    bad_adapter.provider_name = "bad"
    bad_adapter.close = AsyncMock()
    # Non-native upstream: cannot resolve a proxy-local previous_response_id.
    bad_adapter._target_endpoint = lambda: "chat_completions"
    good_adapter = MagicMock()
    good_adapter.provider_name = "good"
    good_adapter.close = AsyncMock()
    # Native Responses upstream: forwards the id server-side, no rejection.
    good_adapter._target_endpoint = lambda: "responses"
    context = RequestContext(
        orchestrator=orchestrator,
        services=ServiceDependencies(
            adapter_factory=AsyncMock(side_effect=[bad_adapter, good_adapter])
        ),
    )
    req = _build_mock_request()
    # previous_response_id with no response store and a non-native upstream
    # makes PreviousResponseResolutionStage reject the request.
    raw = {
        "model": "fast",
        "messages": [{"role": "user", "content": "hi"}],
        "previous_response_id": "resp_proxy_local",
        "stream": True,
    }

    result = await setup_fallback_provider(
        bad_selection, req, _build_unified_request(), raw, context, service
    )

    assert result is not None
    adapter, new_request = result
    assert adapter is good_adapter
    assert new_request.model == "gpt-4o"


def test_merge_transformer_usage_sums_independent_calls() -> None:
    """Web-search continuation turns are independent billed upstream calls,
    so usage must be summed, not maxed."""
    from llm_proxy.core.processing.web_search_streaming import merge_continuation_usage

    original = MagicMock()
    original._pending_stop_reason = None
    original._pending_usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 50,
    }
    continuation = MagicMock()
    continuation._pending_stop_reason = "end_turn"
    continuation._pending_usage = {
        "input_tokens": 150,
        "output_tokens": 30,
        "cache_read_input_tokens": 10,
    }

    merge_continuation_usage(original, continuation)

    assert continuation._pending_usage == {
        "input_tokens": 250,
        "output_tokens": 50,
        "cache_read_input_tokens": 60,
    }
