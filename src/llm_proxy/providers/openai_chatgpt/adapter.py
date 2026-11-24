"""OpenAI ChatGPT OAuth provider adapter (Responses API with Sign in with ChatGPT)."""

import inspect
from collections.abc import AsyncIterator
from typing import Any

import orjson

from llm_proxy.core.adapter import register_adapter
from llm_proxy.core.conversion import plan_conversion
from llm_proxy.core.errors import get_error_type_for_status
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.reasoning_cache import (
    try_cache_reasoning_from_response,
    try_cache_reasoning_from_responses_output,
)
from llm_proxy.core.utils import optional_int
from llm_proxy.http.client import AsyncSession
from llm_proxy.models import ConversionTier, InternalRequest, InternalResponse
from llm_proxy.models.provider import ProviderModelInfo
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.base import (
    STREAM_READ_TIMEOUT,
    BaseHttpProvider,
    extract_rate_limit_headers,
)
from llm_proxy.providers.capabilities import ChatCapabilityMixin
from llm_proxy.providers.openai.adapter import _strip_input_item_id
from llm_proxy.providers.openai.client_headers import get_client_headers
from llm_proxy.providers.openai.sse_fallback import parse_json_or_sse
from llm_proxy.providers.openai_chatgpt.token_manager import get_oauth_token_manager
from llm_proxy.serialization.providers import get_provider_serializer

logger = get_logger(__name__)
_serializer = get_provider_serializer("openai")


@register_adapter("openai-chatgpt")
class OpenAIChatGPTAdapter(ChatCapabilityMixin, BaseHttpProvider):
    """OpenAI provider authenticated via ChatGPT OAuth (``chatgpt.tokens.use.direct``).

    Routes chat and OpenResponses requests through ``POST /v1/responses`` using
    the user's ChatGPT subscription allowance rather than a platform API key.
    Per OpenAI's Sign in with ChatGPT direct-usage requirements, every upstream
    call sends ``stream=True`` and ``store=False``.
    """

    _DEFAULT_PROVIDER_NAME = "openai-chatgpt"

    #: Branding for the admin provider catalog (GET /api/config/providers/provider-types).
    DISPLAY_NAME_EN = "OpenAI (ChatGPT OAuth)"
    DISPLAY_NAME_ZH = "OpenAI (ChatGPT OAuth)"
    LOBE_ICON_ID = "openai"
    LOBE_ICON_VARIANT = "mono"

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    native_protocols = frozenset({"openresponses"})
    RESPONSES_ENDPOINT = "/responses"

    def _get_serializer(self):
        return _serializer

    def _target_endpoint(self) -> str:
        """OpenAI ChatGPT OAuth targets the Responses API endpoint."""
        return "responses"

    def native_body_hook(self, body: dict[str, Any]) -> dict[str, Any]:
        """Apply Responses API repairs and direct plan-usage flags (stream=True, store=False)."""
        text = body.get("text")
        if isinstance(text, dict) and "format" not in text and "type" in text:
            body["text"] = {"format": text}
        input_items = body.get("input")
        if isinstance(input_items, list):
            body["input"] = [_strip_input_item_id(item) for item in input_items]
        body["stream"] = True
        body["store"] = False
        return body

    def _prepare_responses_body(
        self, request_body: dict[str, Any], *, model: str | None = None
    ) -> dict[str, Any]:
        """Copy non-null fields, apply routed model, and enforce stream=True, store=False."""
        body = {k: v for k, v in request_body.items() if v is not None}
        if model:
            body["model"] = model
        return self.native_body_hook(body)

    def _build_chat_raw(self, request, context):
        return _serializer.build_provider_request(request, context)

    def _responses_url(self, model: str | None = None) -> str:
        return self._resolve_endpoint_url("chat_completion", self.RESPONSES_ENDPOINT, model=model)

    async def _get_access_token(
        self, *, force_refresh: bool = False, failed_token: str | None = None
    ) -> str:
        token_provider = self._extra_config.get("token_provider")
        if callable(token_provider):
            sig = inspect.signature(token_provider)
            kwargs: dict[str, Any] = {}
            if "force_refresh" in sig.parameters:
                kwargs["force_refresh"] = force_refresh
            if "failed_token" in sig.parameters:
                kwargs["failed_token"] = failed_token
            token = token_provider(**kwargs)
            if inspect.isawaitable(token):
                token = await token
            return str(token)

        client = await self._get_client()
        return await get_oauth_token_manager().get_access_token(
            self.provider_name,
            http_client=client,
            force_refresh=force_refresh,
            failed_token=failed_token,
        )

    def _build_oauth_headers(self, access_token: str) -> dict[str, str]:
        """Build upstream headers authenticated with the ChatGPT OAuth access token."""
        headers = self._build_headers(auth_header="", auth_prefix="")
        headers["Authorization"] = f"Bearer {access_token}"
        client_headers = get_client_headers()
        if client_headers:
            existing = {k.lower() for k in headers}
            for key, value in client_headers.items():
                if key.lower() not in existing:
                    headers[key] = value
        return headers

    def _parse_error_response(self, status_code: int, error_body: dict[str, Any]) -> ProviderError:
        """Handle pre-stream ``{"detail": "..."}`` diagnostic bodies as well as standard errors."""
        if isinstance(error_body, dict) and "error" not in error_body and "detail" in error_body:
            detail = error_body["detail"]
            message = (
                detail.get("message") if isinstance(detail, dict) else str(detail)
            ) or f"HTTP {status_code}"
            return ProviderError(
                message=str(message),
                error_type=get_error_type_for_status(status_code),
                status_code=status_code,
                provider_name=self.provider_name,
                original_error=error_body,
            )
        return super()._parse_error_response(status_code, error_body)

    async def _execute_with_oauth_retry(self, operation):
        """Run ``operation(token)`` and refresh + retry once on HTTP 401."""
        token = await self._get_access_token()
        try:
            return await operation(token)
        except ProviderError as exc:
            if exc.status_code != 401:
                raise
            refreshed = await self._get_access_token(force_refresh=True, failed_token=token)
            return await operation(refreshed)

    async def chat_completion(self, request: InternalRequest, **_kwargs: Any) -> InternalResponse:
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for request_type='chat'")

        url = self._responses_url(model=request.model)
        json_body = self._prepare_responses_body(outbound.json_body, model=request.model)

        response = await self._execute_with_oauth_retry(
            lambda token: self._post_json_response_with_retry(
                url,
                self._build_oauth_headers(token),
                json_body,
            )
        )
        body = parse_json_or_sse(response)
        if plan_conversion(self, request).response_mode == ConversionTier.NATIVE_PASSTHROUGH:
            result = self._build_passthrough_response(body, request)
            if isinstance(body, dict):
                try_cache_reasoning_from_responses_output(
                    body.get("output"),
                    body.get("id", "") or "",
                    logger_prefix="OpenAIChatGPTNative",
                )
        else:
            result = self._parse_response(
                _serializer,
                body,
                model=request.model,
                request_id=request.request_id,
                request=request,
            )
            try_cache_reasoning_from_response(result)
        result.provider_info["_rate_limit_headers"] = extract_rate_limit_headers(
            getattr(response, "headers", None)
        )
        return result

    async def stream_chat_completion(
        self,
        request: InternalRequest,
        cancel_token=None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Stream a chat completion, converting Responses SSE events to OpenAI chunks."""
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for request_type='chat'")
        json_body: dict[str, Any] = self._prepare_responses_body(
            outbound.json_body, model=request.model
        )

        url = self._responses_url(model=json_body.get("model"))
        stream_timeout = (self._connect_timeout, STREAM_READ_TIMEOUT)

        async def _stream_generator():
            token = await self._get_access_token()
            for attempt in range(2):
                converter = _serializer.get_chunk_converter(
                    model=request.model, request_id=request.request_id or ""
                )
                yielded_any = False
                client = await self._get_client()
                headers = self._build_oauth_headers(token)
                try:
                    async with self._streaming_post(
                        client,
                        url,
                        headers=headers,
                        json=json_body,
                        timeout=stream_timeout,
                    ) as response:
                        await self._raise_for_stream_status(response)
                        self._stash_stream_response_headers(response)

                        current_event_type = None
                        async for line_str in self._iter_stream_lines(response, cancel_token):
                            if line_str.startswith("event:"):
                                current_event_type = line_str[6:].strip()
                                continue
                            if line_str.startswith("data:"):
                                data_str = line_str[5:].strip()
                                if data_str == "[DONE]":
                                    for final_chunk in converter.finalize_chunks():
                                        yielded_any = True
                                        yield final_chunk
                                    yielded_any = True
                                    yield "[DONE]"
                                    return
                                try:
                                    data = orjson.loads(data_str)
                                    event_type = current_event_type or data.get("type")
                                    current_event_type = None
                                    data["event_type"] = event_type
                                    chunk = converter.convert_chunk(data)
                                    if chunk is not None:
                                        yielded_any = True
                                        yield chunk
                                except orjson.JSONDecodeError:
                                    logger.warning(
                                        f"Failed to decode SSE data payload: {data_str[:200]}"
                                    )
                                    continue

                        for final_chunk in converter.finalize_chunks():
                            yielded_any = True
                            yield final_chunk
                        yield "[DONE]"
                        return

                except ProviderError as exc:
                    if exc.status_code == 401 and attempt == 0 and not yielded_any:
                        token = await self._get_access_token(force_refresh=True, failed_token=token)
                        continue
                    raise
                except Exception as e:
                    error = await self._handle_http_error(e)
                    raise error from e

        return self._with_retry_generator(_stream_generator, cancel_token=cancel_token)

    async def stream_chat_completion_native(
        self,
        request: InternalRequest,
        cancel_token=None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding native Responses API SSE blocks."""
        outbound = self._build_outbound_body(request, request_type="chat")
        if outbound.json_body is None:
            raise ValueError("_build_outbound_body returned no json_body for request_type='chat'")
        json_body: dict[str, Any] = self._prepare_responses_body(
            outbound.json_body, model=request.model
        )

        url = self._responses_url(model=json_body.get("model"))
        stream_timeout = (self._connect_timeout, STREAM_READ_TIMEOUT)

        async def _stream_generator():
            token = await self._get_access_token()
            for attempt in range(2):
                yielded_any = False
                client = await self._get_client()
                headers = self._build_oauth_headers(token)
                try:
                    async with self._streaming_post(
                        client,
                        url,
                        headers=headers,
                        json=json_body,
                        timeout=stream_timeout,
                    ) as response:
                        await self._raise_for_stream_status(response)
                        self._stash_stream_response_headers(response)

                        buf: list[str] = []
                        async for raw_line in response.iter_lines():
                            if cancel_token and cancel_token.is_set():
                                logger.debug(f"{self.provider_name} stream cancelled")
                                break
                            line_str = (
                                raw_line.decode("utf-8")
                                if isinstance(raw_line, bytes)
                                else raw_line
                            )
                            for part in line_str.split("\n"):
                                if part == "":
                                    if buf:
                                        yielded_any = True
                                        yield "\n".join(buf) + "\n\n"
                                        buf = []
                                else:
                                    buf.append(part)
                        if buf and not (cancel_token and cancel_token.is_set()):
                            yield "\n".join(buf) + "\n\n"
                        return

                except ProviderError as exc:
                    if exc.status_code == 401 and attempt == 0 and not yielded_any:
                        token = await self._get_access_token(force_refresh=True, failed_token=token)
                        continue
                    raise
                except Exception as e:
                    error = await self._handle_http_error(e)
                    raise error from e

        return self._with_retry_generator(_stream_generator, cancel_token=cancel_token)

    async def list_models(self, client: AsyncSession | None = None) -> list[ProviderModelInfo]:
        """Fetch account-available models from ``GET /v1/models`` using OAuth Bearer auth."""
        http_client = client or await self._get_client()
        url = self._models_url()
        params = self._models_params()

        token = await self._get_access_token()
        response = await http_client.request(
            "GET",
            url,
            headers=self._build_oauth_headers(token),
            params=params,
        )
        if response.status_code == 401:
            token = await self._get_access_token(force_refresh=True, failed_token=token)
            response = await http_client.request(
                "GET",
                url,
                headers=self._build_oauth_headers(token),
                params=params,
            )

        response.raise_for_status()
        data = response.json()

        raw_models = data.get("models")
        if isinstance(raw_models, list):
            dict_items = [item for item in raw_models if isinstance(item, dict)]
            has_visibility = any("visibility" in item for item in dict_items)
            if has_visibility:
                dict_items = [item for item in dict_items if item.get("visibility") == "list"]
            parsed = [
                ProviderModelInfo(
                    id=str(item.get("slug") or item.get("id") or ""),
                    name=str(
                        item.get("display_name")
                        or item.get("name")
                        or item.get("slug")
                        or item.get("id")
                        or ""
                    ),
                    description=item.get("description"),
                    owned_by=item.get("owned_by"),
                    context_length=optional_int(
                        item.get("context_window") or item.get("context_length")
                    ),
                )
                for item in dict_items
                if item.get("slug") or item.get("id")
            ]
            return sorted(parsed, key=lambda m: m.id)

        standard_models = data.get(self._models_data_key, [])
        return sorted(
            [self._parse_model(m) for m in standard_models if isinstance(m, dict)],
            key=lambda m: m.id,
        )


__all__ = ["OpenAIChatGPTAdapter"]
