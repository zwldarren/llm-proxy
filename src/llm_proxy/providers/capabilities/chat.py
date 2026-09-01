"""Chat completion capability mixin."""

from collections.abc import AsyncIterator
from typing import Any

import orjson

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import InternalRequest
from llm_proxy.providers.capabilities.host import ChatSelf
from llm_proxy.streaming.sse_parse import parse_sse_data_line


class ChatCapabilityMixin:
    """Mixin for provider adapters that support chat completions.

    Provides the template-method streaming pattern with hooks:
    _stream_url, _stream_headers, _stream_body, _stream_filter_line,
    _stream_transform_chunk, _stream_finalize.

    This template method is designed for **OpenAI-compatible** providers
    whose streaming SSE is already in canonical OpenAI
    ``chat.completion.chunk`` format with ``data:``-prefixed lines.

    Providers with non-OpenAI streaming formats (Anthropic, Gemini, Ollama,
    OpenAI Responses) override ``stream_chat_completion()`` entirely and
    use ``ProviderSerializer.get_chunk_converter()`` to convert chunks.
    """

    # Endpoint constants — overridden by subclasses
    CHAT_ENDPOINT: str = ""

    def _build_chat_raw(self: ChatSelf, request: Any, context: Any) -> dict[str, Any]:
        """Build raw chat body without extra merge or field policy."""
        return self._get_serializer().build_provider_request(request, context)

    def _stream_url(self: ChatSelf, request: InternalRequest) -> str:
        return self._resolve_endpoint_url(
            "chat_completion", self.CHAT_ENDPOINT, model=request.model
        )

    def _stream_headers(self: ChatSelf) -> dict[str, str]:
        return self._build_headers()

    def _stream_body(self: ChatSelf, request: InternalRequest) -> dict[str, Any]:
        raise NotImplementedError(
            "Subclasses must implement _stream_body or stream_chat_completion"
        )

    def _stream_filter_line(self: ChatSelf, line_str: str) -> str | None:
        return parse_sse_data_line(line_str)

    def _stream_transform_chunk(
        self: ChatSelf, chunk: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any] | str | None:
        return chunk

    def _stream_finalize(self: ChatSelf, context: dict[str, Any]) -> dict[str, Any] | str | None:
        return None

    async def stream_chat_completion(
        self: ChatSelf,
        request: InternalRequest,
        cancel_token=None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | dict[str, Any]]:
        url = self._stream_url(request)
        headers = self._stream_headers()
        body = self._stream_body(request)
        stream_timeout = self._get_stream_timeout()
        if context is None:
            context = {}

        async def _stream_generator():
            client = await self._get_client()
            try:
                async with self._streaming_post(
                    client,
                    url,
                    headers=headers,
                    json=body,
                    timeout=stream_timeout,
                ) as response:
                    await self._raise_for_stream_status(response)

                    async for line_str in self._iter_stream_lines(response, cancel_token):
                        parsed = self._stream_filter_line(line_str)
                        if parsed is None:
                            continue
                        if parsed == "[DONE]":
                            final = self._stream_finalize(context)
                            if final is not None:
                                yield final
                            yield "[DONE]"
                            continue
                        try:
                            chunk = orjson.loads(parsed)
                            transformed = self._stream_transform_chunk(chunk, context)
                            if transformed is not None:
                                yield transformed
                        except orjson.JSONDecodeError:
                            continue

            except ProviderError:
                raise
            except Exception as e:
                error = await self._handle_http_error(e)
                raise error from e

        return self._with_retry_generator(_stream_generator, cancel_token=cancel_token)


__all__ = ["ChatCapabilityMixin"]
