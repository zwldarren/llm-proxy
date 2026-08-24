"""Web search tool interceptor for request/response processing.

This module intercepts web_search tool calls in LLM requests/responses,
executes the search via a configured provider, and injects the results
in Anthropic-compatible format.

When the interceptor is active (web_search enabled in proxy configuration):
- web_search tools are converted to a plain function tool for every provider
- The proxy intercepts tool calls and executes searches itself
- Results are injected back into the response

This intentionally replaces any native web search implementation provided
by the upstream model or provider."""

import base64
import hashlib
import secrets
import time
from typing import Any, cast

import orjson

from llm_proxy.core.context import get_request_user_context
from llm_proxy.models import (
    ContentBlock,
    InternalResponse,
    ServerToolUseBlock,
    ToolUseBlock,
)
from llm_proxy.models.content_blocks.anthropic_builtin import WebSearchToolResultBlock
from llm_proxy.models.tools import (
    FunctionTool,
    OpenAIWebSearchTool,
    WebSearchTool,
    is_web_search_tool_name,
)
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.tool_logging import WebSearchLogEntry, get_tool_log_service
from llm_proxy.observability.types import WebSearchStatus

from .provider import WebSearchExecutionResult, WebSearchProvider, WebSearchToolConfig

logger = get_logger(__name__)


def _is_web_search_tool_type(tool_type: str) -> bool:
    """Check if a tool type string corresponds to web search.

    Handles both Anthropic-style (web_search_20250305) and
    OpenAI-style (web_search, web_search_preview) type values.
    """
    return isinstance(tool_type, str) and tool_type.startswith("web_search")


class WebSearchInterceptor:
    """Handles web search tool interception and execution.

    This class is responsible for:
    1. Detecting web_search tool definitions in requests
    2. Detecting server_tool_use blocks for web_search in responses
    3. Executing web searches via the configured provider
    4. Formatting results as Anthropic-compatible web_search_tool_result blocks
    5. Tracking and reporting usage for billing
    """

    # JSON schema for web_search function tool
    WEB_SEARCH_FUNCTION_SCHEMA = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string",
            },
        },
        "required": ["query"],
    }

    def __init__(self, provider: WebSearchProvider):
        """Initialize the web search interceptor.

        Args:
            provider: The web search provider to use for executing searches
        """
        self._provider = provider

    def _get_search_state(self, search_state: dict[str, int] | None) -> dict[str, int]:
        """Return a mutable per-request search counter.

        ``search_state`` is supplied by callers that need to enforce ``max_uses``
        across multiple searches within the same request. If not provided, a fresh
        counter is returned so that standalone calls are still validated.
        """
        return search_state if search_state is not None else {"count": 0}

    def has_web_search_tool(self, tools: list[Any] | None) -> bool:
        """Check whether the tools list contains a web search tool.

        Supports both Anthropic and OpenAI WebSearchTool dataclasses,
        plus dict passthrough formats.

        Args:
            tools: List of tool definitions from the request

        Returns:
            True if a web_search tool is present, False otherwise
        """
        if not tools:
            return False

        for tool in tools:
            if isinstance(tool, (WebSearchTool, OpenAIWebSearchTool)):
                return True

            if isinstance(tool, dict):
                tool_type = tool.get("type", "")
                if _is_web_search_tool_type(tool_type):
                    return True

        return False

    def convert_web_search_to_function(
        self, tool_config: WebSearchToolConfig | None = None
    ) -> FunctionTool:
        """Convert web_search server tool to a function tool for non-Anthropic providers.

        Args:
            tool_config: Optional configuration from the original tool definition

        Returns:
            FunctionTool that models can call
        """
        return FunctionTool(
            name="web_search",
            parameters=self.WEB_SEARCH_FUNCTION_SCHEMA,
            description=(
                "Search the web for current information. "
                "Use this tool when you need up-to-date information beyond your knowledge cutoff."
            ),
        )

    def extract_web_search_tool_config(self, tools: list[Any] | None) -> WebSearchToolConfig | None:
        """Extract configuration from web_search tool definition.

        Args:
            tools: List of tool definitions from the request

        Returns:
            WebSearchToolConfig if found, None otherwise
        """
        if not tools:
            return None

        for tool in tools:
            if isinstance(tool, (WebSearchTool, OpenAIWebSearchTool)):
                allowed_domains = tool.allowed_domains
                blocked_domains = tool.blocked_domains
                user_location = None
                if hasattr(tool, "user_location"):
                    ul = tool.user_location
                    if ul is not None:
                        # Anthropic uses UserLocation dataclass; OpenAI uses dict.
                        user_location = (
                            {
                                "city": ul.city,
                                "country": ul.country,
                                "region": ul.region,
                                "timezone": ul.timezone,
                            }
                            if hasattr(ul, "city")
                            else ul
                        )
                return WebSearchToolConfig(
                    max_uses=tool.max_uses,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                    user_location=user_location,
                    search_context_size=getattr(tool, "search_context_size", None),
                    external_web_access=getattr(tool, "external_web_access", None),
                    return_token_budget=getattr(tool, "return_token_budget", None),
                    search_content_types=getattr(tool, "search_content_types", None),
                    image_settings=getattr(tool, "image_settings", None),
                )

            if isinstance(tool, dict):
                tool_type = tool.get("type", "")
                if _is_web_search_tool_type(tool_type):
                    # Extract allowed/blocked domains from both top-level and filters wrapper
                    allowed_domains = tool.get("allowed_domains")
                    blocked_domains = tool.get("blocked_domains")
                    filters = tool.get("filters")
                    if isinstance(filters, dict):
                        if allowed_domains is None:
                            allowed_domains = filters.get("allowed_domains")
                        if blocked_domains is None:
                            blocked_domains = filters.get("blocked_domains")
                    return WebSearchToolConfig(
                        max_uses=tool.get("max_uses"),
                        allowed_domains=allowed_domains,
                        blocked_domains=blocked_domains,
                        user_location=tool.get("user_location"),
                        search_context_size=tool.get("search_context_size"),
                        external_web_access=tool.get("external_web_access"),
                        return_token_budget=tool.get("return_token_budget"),
                        search_content_types=tool.get("search_content_types"),
                        image_settings=tool.get("image_settings"),
                    )

        return None

    def filter_web_search_tools(self, tools: list[Any] | None) -> list[Any] | None:
        """Remove web_search tools from the tools list.

        This is used before sending the request to non-Anthropic providers
        that don't recognize web_search tool type.

        Args:
            tools: List of tool definitions

        Returns:
            Filtered list without web_search tools, or None if input was None
        """
        if not tools:
            return None

        filtered = []
        for tool in tools:
            if isinstance(tool, (WebSearchTool, OpenAIWebSearchTool)):
                continue
            if isinstance(tool, dict):
                tool_type = tool.get("type", "")
                if _is_web_search_tool_type(tool_type):
                    continue
            filtered.append(tool)

        return filtered if filtered else None

    def is_web_search_server_tool_use(self, block: ContentBlock) -> bool:
        """Check if a content block is a web_search server_tool_use or tool_use.

        For Anthropic, web_search calls appear as ServerToolUseBlock.
        For non-Anthropic providers, they appear as ToolUseBlock with name="web_search".

        Note: The comparison normalizes both case and underscores to handle
        models that output tool names in different formats (e.g., "WebSearch",
        "web_search", "WEB_SEARCH" are all treated as web_search).

        Args:
            block: Content block to check

        Returns:
            True if this is a web_search tool call
        """
        if isinstance(block, ServerToolUseBlock):
            return is_web_search_tool_name(block.name)
        if isinstance(block, ToolUseBlock):
            return is_web_search_tool_name(block.name)
        return False

    async def execute_search(
        self,
        tool_use: ServerToolUseBlock,
        tool_config: WebSearchToolConfig | None = None,
        request_id: str | None = None,
        search_state: dict[str, int] | None = None,
    ) -> WebSearchExecutionResult:
        """Execute a web search and return the execution result.

        Args:
            tool_use: The server_tool_use block containing the search query
            tool_config: Optional configuration from the original tool definition
            request_id: Request ID for tracking search count
            search_state: Optional per-request counter dict used to enforce
                ``max_uses`` across multiple searches in the same request.
                Callers should pass the same dict for all searches within a
                single request.

        Returns:
            WebSearchExecutionResult containing the result block and usage info
        """
        query = tool_use.input.get("query", "")
        start_time = time.time()

        if not query:
            self._log_search(
                query="",
                status=WebSearchStatus.ERROR,
                start_time=start_time,
                result_count=0,
                error_message="Missing 'query' parameter",
                status_code=400,
            )
            return WebSearchExecutionResult(
                tool_use_block=tool_use,
                result_block=self._create_error_result(
                    tool_use.id, "invalid_input", "Missing 'query' parameter"
                ),
                web_search_count=0,
            )

        # Check max_uses using a per-request counter supplied by the caller.
        state = self._get_search_state(search_state)
        if tool_config and tool_config.max_uses:
            count = state["count"]
            if count >= tool_config.max_uses:
                self._log_search(
                    query=query,
                    status=WebSearchStatus.MAX_USES_EXCEEDED,
                    start_time=start_time,
                    result_count=0,
                    error_message=f"Maximum searches ({tool_config.max_uses}) exceeded",
                    status_code=429,
                    max_uses=tool_config.max_uses,
                    current_use=count + 1,
                )
                return WebSearchExecutionResult(
                    tool_use_block=tool_use,
                    result_block=self._create_error_result(
                        tool_use.id,
                        "max_uses_exceeded",
                        f"Maximum searches ({tool_config.max_uses}) exceeded",
                    ),
                    web_search_count=0,
                )
            state["count"] = count + 1

        try:
            logger.info(f"Executing web search: query='{query}'")
            response = await self._provider.search(query, tool_config)

            # Format results as Anthropic-compatible content
            content_blocks = []
            results_for_log = []
            for idx, result in enumerate(response.results):
                result_item = {
                    "type": "web_search_result",
                    "url": result.url,
                    "title": result.title,
                    "encoded_content": self._encode_content(result.snippet),
                    # Generate a unique encoded_index for each result
                    # This is used for multi-turn citation references
                    "encoded_index": self._generate_encoded_index(result.url, idx),
                }
                if result.page_age:
                    result_item["page_age"] = result.page_age
                content_blocks.append(result_item)
                # Collect results for logging (without snippets for privacy)
                results_for_log.append({"url": result.url, "title": result.title})

            # Log successful search
            self._log_search(
                query=query,
                status=WebSearchStatus.SUCCESS,
                start_time=start_time,
                result_count=len(response.results),
                results=results_for_log,
                status_code=200,
                max_uses=tool_config.max_uses if tool_config else None,
                current_use=state["count"] if tool_config and tool_config.max_uses else None,
            )

            # Successful search counts toward usage
            return WebSearchExecutionResult(
                tool_use_block=tool_use,
                result_block=WebSearchToolResultBlock(
                    tool_use_id=tool_use.id,
                    content=content_blocks,
                    is_error=False,
                ),
                web_search_count=1,
            )

        except Exception as e:
            error_code = getattr(e, "error_code", "unavailable")
            error_message = str(e) if str(e) else "Search failed"
            logger.error(f"Web search failed: query='{query}', error={e}")

            # Determine status based on error type
            status = WebSearchStatus.ERROR
            status_code = 500
            if "rate" in error_message.lower() or "limit" in error_message.lower():
                status = WebSearchStatus.RATE_LIMITED
                status_code = 429

            self._log_search(
                query=query,
                status=status,
                start_time=start_time,
                result_count=0,
                error_message=error_message,
                status_code=status_code,
                max_uses=tool_config.max_uses if tool_config else None,
                current_use=state["count"] if tool_config and tool_config.max_uses else None,
            )

            return WebSearchExecutionResult(
                tool_use_block=tool_use,
                result_block=self._create_error_result(tool_use.id, error_code, error_message),
                web_search_count=0,
            )

    def _log_search(
        self,
        query: str,
        status: WebSearchStatus,
        start_time: float,
        result_count: int,
        status_code: int,
        results: list[dict[str, str]] | None = None,
        error_message: str | None = None,
        max_uses: int | None = None,
        current_use: int | None = None,
    ) -> None:
        """Log a web search operation if logging is configured."""
        try:
            # Get user context for attribution
            user_ctx = get_request_user_context()
            user_id = user_ctx.user_id if user_ctx else None
            user_identity = user_ctx.user_identity if user_ctx else None
            api_key_name = user_ctx.api_key_name if user_ctx else None
            auth_method = user_ctx.auth_method if user_ctx else None

            log_service = get_tool_log_service()

            entry = WebSearchLogEntry(
                query=query,
                status=status,
                result_count=result_count,
                results=results or [],
                error_message=error_message,
                status_code=status_code,
                response_time_ms=int((time.time() - start_time) * 1000),
                provider=getattr(self._provider, "name", "searxng"),
                max_uses=max_uses,
                current_use=current_use,
            )
            log_service.log_web_search_background(
                entry,
                user_id=user_id,
                user_identity=user_identity,
                api_key_name=api_key_name,
                auth_method=auth_method,
            )
        except Exception:
            logger.debug("Failed to log web search operation", exc_info=True)

    async def inject_results_into_response(
        self,
        response: InternalResponse,
        tool_config: WebSearchToolConfig | None = None,
        request_id: str | None = None,
        search_state: dict[str, int] | None = None,
    ) -> tuple[InternalResponse, list[tuple[ServerToolUseBlock, WebSearchExecutionResult]]]:
        """Inject web search results into a InternalResponse.

        This method finds all server_tool_use blocks for web_search in the response,
        executes the searches, and injects web_search_tool_result blocks.
        It also updates usage tracking for billing.

        Args:
            response: The response to modify
            tool_config: Optional configuration from the original tool definition
            request_id: Request ID for tracking
            search_state: Optional per-request counter dict used to enforce
                ``max_uses`` across multiple searches in the same request.

        Returns:
            Tuple of (modified response, list of (ServerToolUseBlock,
            WebSearchExecutionResult) pairs). The pairs let callers build a
            continuation request so the model can answer from the results
            (the non-streaming counterpart of the streaming continuation
            loop).
        """
        # Find all web_search server_tool_use blocks
        web_search_blocks = []
        other_blocks = []

        for block in response.output:
            if self.is_web_search_server_tool_use(block):
                web_search_blocks.append(block)
            else:
                other_blocks.append(block)

        if not web_search_blocks:
            return response, []

        # Execute searches and collect results
        output_blocks: list[ContentBlock] = []
        search_results: list[tuple[ServerToolUseBlock, WebSearchExecutionResult]] = []
        total_search_count = 0
        state = self._get_search_state(search_state)

        for tool_use in web_search_blocks:
            if isinstance(tool_use, ToolUseBlock):
                tool_use = ServerToolUseBlock(
                    id=tool_use.id,
                    name=tool_use.name,
                    input=tool_use.input,
                )
            assert isinstance(tool_use, ServerToolUseBlock)
            execution_result = await self.execute_search(
                tool_use, tool_config, request_id, search_state=state
            )
            search_results.append((tool_use, execution_result))
            output_blocks.append(execution_result.tool_use_block)
            output_blocks.append(execution_result.result_block)
            total_search_count += execution_result.web_search_count

        # Rebuild output with proper ordering:
        # Anthropic expects: text blocks -> tool_use -> tool_result -> remaining text
        # For simplicity, we interleave tool_use and tool_result
        response.output = other_blocks + output_blocks

        # Update usage tracking for server tool use
        if total_search_count > 0:
            server_tool_use = response.provider_info.get("server_tool_use", {})
            current_count = server_tool_use.get("web_search_requests", 0)
            server_tool_use["web_search_requests"] = current_count + total_search_count
            response.provider_info["server_tool_use"] = server_tool_use

        return response, search_results

    def _create_error_result(
        self,
        tool_use_id: str,
        error_code: str,
        error_message: str,
    ) -> WebSearchToolResultBlock:
        """Create an error result block.

        Args:
            tool_use_id: The tool_use_id to reference
            error_code: Anthropic-compatible error code
            error_message: Human-readable error message

        Returns:
            WebSearchToolResultBlock with error content
        """
        error_content = orjson.dumps(
            {
                "type": "web_search_tool_result_error",
                "error_code": error_code,
            }
        ).decode()
        return WebSearchToolResultBlock(
            tool_use_id=tool_use_id,
            content=error_content,
            is_error=True,
        )

    def decode_search_results(self, result_block: WebSearchToolResultBlock) -> str:
        if isinstance(result_block.content, str):
            return result_block.content
        results = []
        items = result_block.content
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    item_dict = cast("dict[str, Any]", item)
                    raw = item_dict.get("encoded_content", "")
                    content = base64.b64decode(raw).decode("utf-8")
                    results.append(
                        {
                            "url": item_dict.get("url", ""),
                            "title": item_dict.get("title", ""),
                            "snippet": content,
                        }
                    )
        return orjson.dumps({"results": results}).decode()

    def _encode_content(self, content: str) -> str:
        """Encode content as base64 for Anthropic compatibility.

        Note: This is encoding, not encryption. Anthropic uses a more
        sophisticated encryption scheme for multi-turn conversation support.
        For now, we use base64 encoding as a placeholder.

        Args:
            content: The content to encode

        Returns:
            Base64-encoded content string
        """
        return base64.b64encode(content.encode("utf-8")).decode("utf-8")

    def _generate_encoded_index(self, url: str, index: int) -> str:
        """Generate an encoded index for citation references.

        The encoded_index is used in multi-turn conversations to reference
        specific search results in citations. This implementation generates
        a deterministic but unique identifier based on the URL and index.

        Note: This is encoding, not encryption. Anthropic uses a more
        sophisticated encryption scheme for multi-turn citation support.

        Args:
            url: The URL of the search result
            index: The index of the result in the results list

        Returns:
            Encoded index string (base64 encoded)
        """
        # Generate a unique but deterministic identifier
        # Combine URL, index, and a random nonce for uniqueness
        nonce = secrets.token_hex(8)
        data = f"{url}:{index}:{nonce}"
        hashed = hashlib.sha256(data.encode()).digest()
        return base64.b64encode(hashed).decode("utf-8")

    async def close(self) -> None:
        """Clean up resources."""
        await self._provider.close()
