"""OpenAI Realtime API support package.

The Realtime API is a bidirectional WebSocket protocol (audio + text events)
that cannot be expressed through the request/response pipeline used by the
REST protocols. The proxy therefore acts as a transparent WebSocket relay:
clients connect to ``WS /v1/realtime?model=<proxy model>``, the proxy
authenticates them with its own API keys, resolves the model to a provider,
opens a WebSocket to the provider's native Realtime endpoint, and pumps
messages in both directions while observing ``response.done`` events for
usage-based billing and logging.
"""

from llm_proxy.realtime.relay import RealtimeRelay, RealtimeRelayConfig
from llm_proxy.realtime.upstream import (
    build_realtime_url,
    build_upstream_headers,
    connect_upstream,
)
from llm_proxy.realtime.usage import RealtimeUsageObserver

__all__ = [
    "RealtimeRelay",
    "RealtimeRelayConfig",
    "RealtimeUsageObserver",
    "build_realtime_url",
    "build_upstream_headers",
    "connect_upstream",
]
