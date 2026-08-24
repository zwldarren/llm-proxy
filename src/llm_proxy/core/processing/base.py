"""Base types and interfaces for request processing."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_proxy.core.identity import RequestIdentity
from llm_proxy.core.request_type import RequestType
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from fastapi import Request

    from llm_proxy.core.adapter import BaseAdapter
    from llm_proxy.core.provider_selector import ProviderSelectionResult, ProviderSelector
    from llm_proxy.models import InternalRequest
    from llm_proxy.observability.event_context import EventContext
    from llm_proxy.observability.tracing.handlers import TracingRegistry
    from llm_proxy.web_search.interceptor import WebSearchInterceptor
    from llm_proxy.web_search.provider import WebSearchToolConfig

logger = get_logger(__name__)

AdapterFactoryFunc = Callable[
    ["Request", "ProviderSelectionResult"],
    Awaitable["BaseAdapter"],
]
RequestMiddlewareFunc = Callable[[Any, "BaseAdapter"], Awaitable[Any]]
ResponseMiddlewareFunc = Callable[[Any, "BaseAdapter", Any], Awaitable[Any]]


def mirror_conversion_tier(
    request: InternalRequest,
    event_context: EventContext | None,
    provider_name: str,
    *,
    stream: bool = False,
) -> None:
    """Mirror the serving attempt's conversion tiers into logs/audit.

    The request tier is stamped when the adapter builds the outbound body
    (``core.conversion`` seam preparations or the serializer's full
    conversion); the response tier is stamped by the response chokepoints
    (``_build_passthrough_response`` / ``_parse_response``). The request
    object tracks the live attempt across retries and fallbacks.
    """
    if event_context is None:
        return
    # Defensive getattr: tests/handlers may pass partially-mocked requests.
    tier = getattr(request, "conversion_tier", None)
    response_tier = getattr(request, "response_tier", None)
    if tier is None and response_tier is None:
        return
    if tier is not None:
        event_context.metadata["conversion_tier"] = tier
    if response_tier is not None:
        event_context.metadata["response_tier"] = response_tier
    kind = "streaming request" if stream else "request"
    logger.debug(
        f"Conversion tiers for {kind}: request={tier} response={response_tier} "
        f"[provider={provider_name}]"
    )


@dataclass
class ServiceDependencies:
    adapter_factory: AdapterFactoryFunc
    config_manager: Any = None
    tracing_registry: TracingRegistry | None = None
    web_search_interceptor: WebSearchInterceptor | None = None


@dataclass
class RequestContext:
    """Request-scoped services and hooks — the "who/what can be used" object.

    Owns things that exist for the whole request and are **stable**: service
    wiring (adapter factory, config manager, tracing registry, web search
    interceptor), middleware hooks (process_request/process_response),
    identity/attribution fields (user, session, client_ip), and opt-in
    features (response store, smart-routing hooks). Pipeline stages read it
    to get capabilities; they do NOT write mutable working state here — that
    belongs on :class:`PipelineState`. Observability capture belongs on
    :class:`EventContext` (``context.event_context``).
    """

    orchestrator: ProviderSelector
    services: ServiceDependencies
    process_request: RequestMiddlewareFunc | None = None
    process_response: ResponseMiddlewareFunc | None = None
    request_type: RequestType = RequestType.CHAT
    trace_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    response_store: Any = None
    web_search_tool_config: WebSearchToolConfig | None = None
    proxy_web_search_active: bool = False
    event_context: EventContext | None = None
    identity: RequestIdentity | None = None
    client_ip: str | None = None
    protocol_name: str | None = None
    should_log_input_output: bool = True
    routing_decision: Any = None
    requested_model: str | None = None
    verbose_routing_logs: bool = False
    # Optional post-request hook (event_context, success) -> None, wired by the
    # API layer when smart routing is active (e.g. routing.model_experience
    # EWMA observation). Keeps core from depending on the routing package.
    on_request_completed: Callable[[Any, bool], Awaitable[None]] | None = None

    @property
    def adapter_factory(self) -> AdapterFactoryFunc:
        return self.services.adapter_factory

    @property
    def config_manager(self) -> Any:
        return self.services.config_manager

    @config_manager.setter
    def config_manager(self, value: Any) -> None:
        self.services.config_manager = value

    @property
    def tracing_registry(self) -> TracingRegistry | None:
        return self.services.tracing_registry

    @tracing_registry.setter
    def tracing_registry(self, value: TracingRegistry | None) -> None:
        self.services.tracing_registry = value

    @property
    def web_search_interceptor(self) -> WebSearchInterceptor | None:
        return self.services.web_search_interceptor

    @web_search_interceptor.setter
    def web_search_interceptor(self, value: WebSearchInterceptor | None) -> None:
        self.services.web_search_interceptor = value


__all__ = [
    "AdapterFactoryFunc",
    "RequestContext",
    "RequestMiddlewareFunc",
    "ResponseMiddlewareFunc",
    "ServiceDependencies",
]
