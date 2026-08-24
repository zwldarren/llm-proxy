"""Helper functions for configuration management."""

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_config_manager
from llm_proxy.database import ConfigRepository
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


def get_config_repository(session: AsyncSession) -> ConfigRepository:
    """Get configuration repository dependency."""
    return ConfigRepository(session)


async def rebuild_web_search_interceptor(request: Request) -> None:
    """Rebuild the web search interceptor from the current configuration.

    Closes the existing interceptor (if any) and creates a new one using
    the freshly loaded config. This ensures runtime state (e.g. max_results,
    URL, auth credentials) stays in sync with persisted configuration.
    """
    app = request.app

    existing = getattr(app.state, "web_search_interceptor", None)
    if existing is not None:
        try:
            await existing.close()
        except Exception as e:
            logger.warning(f"Failed to close old web search interceptor: {e}")

    config_manager = get_config_manager(request)
    config = await config_manager.get_config()

    web_search_interceptor = None
    if config.server_params.web_search and config.server_params.web_search.enabled:
        from llm_proxy.web_search import create_web_search_provider
        from llm_proxy.web_search.interceptor import WebSearchInterceptor

        try:
            search_provider = create_web_search_provider(config.server_params.web_search)
            if search_provider:
                web_search_interceptor = WebSearchInterceptor(
                    provider=search_provider,
                )
                logger.debug("Web search interceptor rebuilt after config reload")
        except Exception as e:
            logger.error(f"Failed to rebuild web search interceptor: {e}")

    app.state.web_search_interceptor = web_search_interceptor


async def commit_and_reload(session: AsyncSession, request: Request) -> None:
    """Persist changes before refreshing the in-memory config cache."""
    await session.commit()
    await get_config_manager(request).reload()


def _extract_metadata_fields(
    metadata: dict | None,
    fields_to_extract: list[str],
) -> tuple[dict, dict]:
    """Extract specific fields from metadata dict.

    Args:
        metadata: Source metadata dict (copied if provided)
        fields_to_extract: List of field names to extract

    Returns:
        Tuple of (remaining_metadata, extracted_fields)
    """
    metadata = metadata.copy() if metadata else {}

    extracted = {}
    for field in fields_to_extract:
        if field in metadata:
            extracted[field] = metadata.pop(field)
    return metadata, extracted
