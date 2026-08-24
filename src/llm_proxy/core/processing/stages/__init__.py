"""Pipeline stages for request processing.

Each stage is an independent class that can be tested in isolation.
The UnifiedProcessor orchestrates stages in sequence via PipelineState.
"""

from llm_proxy.core.processing.stages.base import PipelineStage, PipelineState
from llm_proxy.core.processing.stages.parameter_override import (
    ParameterOverrideService,
    ParameterOverrideStage,
)
from llm_proxy.core.processing.stages.previous_response import (
    PreviousResponseResolutionStage,
)
from llm_proxy.core.processing.stages.provider_selection import ProviderSelectionStage
from llm_proxy.core.processing.stages.request_execution import (
    RequestExecutionStage,
    RetryExecutor,
)
from llm_proxy.core.processing.stages.role_normalization import normalize_developer_roles
from llm_proxy.core.processing.stages.web_search import WebSearchStage

__all__ = [
    "ParameterOverrideService",
    "ParameterOverrideStage",
    "PipelineStage",
    "PipelineState",
    "PreviousResponseResolutionStage",
    "ProviderSelectionStage",
    "RequestExecutionStage",
    "RetryExecutor",
    "WebSearchStage",
    "normalize_developer_roles",
]
