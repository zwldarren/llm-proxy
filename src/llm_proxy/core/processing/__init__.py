"""Request processing modules.

This package provides modular request processing capabilities
using the UnifiedProcessor for all v2 protocols.
"""

from llm_proxy.core.processing.base import (
    AdapterFactoryFunc,
    RequestContext,
    RequestMiddlewareFunc,
    ResponseMiddlewareFunc,
    ServiceDependencies,
)
from llm_proxy.core.processing.fallback import record_fallback_attempt, setup_fallback_provider
from llm_proxy.core.processing.stages import (
    ParameterOverrideService,
    ParameterOverrideStage,
    PipelineStage,
    PipelineState,
    ProviderSelectionStage,
    RequestExecutionStage,
    RetryExecutor,
    WebSearchStage,
    normalize_developer_roles,
)
from llm_proxy.core.processing.strategies import (
    ProcessingStrategy,
    StreamingResponseMarker,
    get_strategy,
)
from llm_proxy.core.processing.strategies.chunk_parser import OpenAIStreamChunkParser
from llm_proxy.core.processing.unified import (
    UnifiedProcessor,
    create_unified_processor,
)
from llm_proxy.core.processing.web_search_streaming import WebSearchStreamProcessor

__all__ = [
    "AdapterFactoryFunc",
    "OpenAIStreamChunkParser",
    "ParameterOverrideService",
    "ParameterOverrideStage",
    "PipelineStage",
    "PipelineState",
    "ProcessingStrategy",
    "ProviderSelectionStage",
    "RequestContext",
    "RequestExecutionStage",
    "RequestMiddlewareFunc",
    "ResponseMiddlewareFunc",
    "RetryExecutor",
    "ServiceDependencies",
    "StreamingResponseMarker",
    "UnifiedProcessor",
    "WebSearchStage",
    "WebSearchStreamProcessor",
    "create_unified_processor",
    "get_strategy",
    "normalize_developer_roles",
    "record_fallback_attempt",
    "setup_fallback_provider",
]
