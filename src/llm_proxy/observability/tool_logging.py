"""Logging utilities for tool operations (MCP, Web Search).

This module provides logging support for tool invocations like
MCP server calls and web search queries.
"""

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_proxy.observability.service import RequestLogCreate, RequestLogService
from llm_proxy.security.passwords import SENSITIVE_KEYS, mask_sensitive

if TYPE_CHECKING:
    from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.types import (
    LogType,
    McpOperationType,
    McpResourceType,
    WebSearchStatus,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class McpLogEntry:
    """Input data for creating an MCP operation log."""

    server_name: str
    operation: McpOperationType
    resource_type: McpResourceType
    resource_name: str | None = None  # tool name, resource URI, or prompt name
    arguments: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    status_code: int = 200
    response_time_ms: int | None = None


def _mask_query(query: str) -> str:
    """Partially mask a web search query for privacy while keeping correlation.

    Returns the first 32 characters of the query followed by a short hash of the
    full query. This reduces the risk of logging PII while still allowing
    operators to see the general topic and detect repeated queries.
    """
    if not isinstance(query, str):
        query = str(query)
    prefix = query[:32]
    suffix_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}... [hash:{suffix_hash}]"


@dataclass(frozen=True)
class WebSearchLogEntry:
    """Input data for creating a web search log."""

    query: str
    status: WebSearchStatus
    result_count: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None
    status_code: int = 200
    response_time_ms: int | None = None
    provider: str = "searxng"
    max_uses: int | None = None
    current_use: int | None = None


class ToolLogService:
    """Service for logging MCP and Web Search operations.

    This service wraps RequestLogService to provide a simpler interface
    for tool-specific logging. It generates unique request IDs and
    formats log entries appropriately.
    """

    def __init__(self, log_service: RequestLogService):
        self._log_service = log_service

    def _generate_request_id(self) -> str:
        """Generate a unique request ID for tool operations."""
        return f"tool_{uuid.uuid4().hex[:16]}"

    def log_mcp_background(
        self,
        entry: McpLogEntry,
        user_id: int | None = None,
        user_identity: str | None = None,
        api_key_name: str | None = None,
        auth_method: str | None = None,
    ) -> str:
        """Log an MCP operation in the background.

        Args:
            entry: The MCP log entry data
            user_id: The numeric user ID for attribution
            user_identity: The user-facing identifier (username)
            api_key_name: The name of the API key used
            auth_method: The authentication method used

        Returns:
            The request ID for the log entry
        """
        request_id = self._generate_request_id()
        timestamp = time.time()

        # Mask sensitive fields in tool arguments before persistence.
        masked_arguments = mask_sensitive(entry.arguments, SENSITIVE_KEYS)

        # Build metadata with MCP-specific fields
        metadata: dict[str, Any] = {
            "mcp_server": entry.server_name,
            "mcp_operation": entry.operation.value,
            "mcp_resource_type": entry.resource_type.value,
            "mcp_resource_name": entry.resource_name,
            "mcp_arguments": masked_arguments,
            "mcp_result_summary": entry.result_summary,
        }

        # Determine outcome based on status_code
        outcome = "success" if 200 <= entry.status_code < 300 else "failure"

        log = RequestLogCreate(
            request_id=request_id,
            timestamp=timestamp,
            endpoint=f"/mcp/{entry.server_name}/{entry.operation.value}",
            method="POST",  # MCP operations are all POST-like
            log_type=LogType.MCP,
            status_code=entry.status_code,
            response_time_ms=entry.response_time_ms,
            error_message=entry.error_message,
            log_metadata=metadata,
            model=entry.server_name,  # Use model field for server name
            outcome=outcome,
            user_id=user_id,
            user_identity=user_identity,
            api_key_name=api_key_name,
            auth_method=auth_method,
        )

        self._log_service.create_log_background(log)
        return request_id

    def log_web_search_background(
        self,
        entry: WebSearchLogEntry,
        user_id: int | None = None,
        user_identity: str | None = None,
        api_key_name: str | None = None,
        auth_method: str | None = None,
    ) -> str:
        """Log a web search operation in the background.

        Args:
            entry: The web search log entry data
            user_id: The numeric user ID for attribution
            user_identity: The user-facing identifier (username)
            api_key_name: The name of the API key used
            auth_method: The authentication method used

        Returns:
            The request ID for the log entry
        """
        request_id = self._generate_request_id()
        timestamp = time.time()

        # Build metadata with web search fields
        metadata: dict[str, Any] = {
            "web_search_query": _mask_query(entry.query),
            "web_search_status": entry.status.value,
            "web_search_result_count": entry.result_count,
            # Limit to 10 results for storage efficiency
            "web_search_results": entry.results[:10] if entry.results else [],
            "web_search_provider": entry.provider,
            "web_search_max_uses": entry.max_uses,
            "web_search_current_use": entry.current_use,
        }

        # Determine outcome based on status_code
        outcome = "success" if 200 <= entry.status_code < 300 else "failure"

        log = RequestLogCreate(
            request_id=request_id,
            timestamp=timestamp,
            endpoint="/tools/web_search",
            method="POST",
            log_type=LogType.WEB_SEARCH,
            status_code=entry.status_code,
            response_time_ms=entry.response_time_ms,
            error_message=entry.error_message,
            log_metadata=metadata,
            provider=entry.provider,
            outcome=outcome,
            user_id=user_id,
            user_identity=user_identity,
            api_key_name=api_key_name,
            auth_method=auth_method,
        )

        self._log_service.create_log_background(log)
        return request_id


# Global service instance (initialized lazily)
_tool_log_service: ToolLogService | None = None
_tool_log_service_lock = threading.RLock()


def get_tool_log_service(
    config_or_service: LoggingConfig | RequestLogService | None = None,
) -> ToolLogService:
    """Get or create the tool log service instance.

    Args:
        config_or_service: Either a LoggingConfig, RequestLogService, or None.
                          If None, returns the existing instance or auto-creates one.

    Returns:
        ToolLogService instance.
    """
    global _tool_log_service

    if config_or_service is None:
        with _tool_log_service_lock:
            if _tool_log_service is None:
                from llm_proxy.config.manager import load_logging_config

                _tool_log_service = ToolLogService(RequestLogService(load_logging_config()))
        return _tool_log_service

    with _tool_log_service_lock:
        if isinstance(config_or_service, RequestLogService):
            _tool_log_service = ToolLogService(config_or_service)
        elif isinstance(config_or_service, LoggingConfig):
            _tool_log_service = ToolLogService(RequestLogService(config_or_service))
    return _tool_log_service
