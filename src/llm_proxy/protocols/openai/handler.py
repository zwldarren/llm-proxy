"""OpenAI protocol endpoint configuration."""

from typing import Any

from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openai.schemas import ChatCompletionRequest
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer  # noqa: F401
from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer


def _clear_client_headers() -> None:
    """Drop the captured client headers after request formatting completes."""
    from llm_proxy.providers.openai.client_headers import clear_client_headers

    clear_client_headers()


async def _capture_client_headers(request: Any, fastapi_request: Any) -> None:
    """Protocol middleware: capture client headers for upstream passthrough.

    Runs before UnifiedProcessor. The OpenRouter adapter merges the captured
    per-request control headers (``X-OpenRouter-Metadata``,
    ``X-OpenRouter-Cache*``) when building upstream requests, so a client's
    opt-ins are not lost at the proxy hop.
    """
    from llm_proxy.providers.openai.client_headers import capture_client_headers

    headers = getattr(fastapi_request, "headers", None)
    if headers is not None:
        capture_client_headers(headers)


openai_protocol = ProtocolEndpoint(
    name="openai",
    # Path aliases: clients whose base_url is missing /v1 or double-writes it
    # ("{base}/v1" + "/v1/chat/completions") still reach the endpoint. Mirrors
    # the openresponses protocol's alias set.
    paths=["/v1/chat/completions", "/chat/completions", "/v1/v1/chat/completions"],
    request_model=ChatCompletionRequest,
    streaming_transformer=OpenAIStreamingTransformer,
    middleware=[_capture_client_headers],
    on_format_done=_clear_client_headers,
    tags=["chat"],
    description="OpenAI Chat Completions API",
)


__all__ = ["openai_protocol"]
