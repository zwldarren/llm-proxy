"""Helper functions for configuration management."""

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_config_manager
from llm_proxy.database import ConfigRepository


def get_config_repository(session: AsyncSession) -> ConfigRepository:
    """Get configuration repository dependency."""
    return ConfigRepository(session)


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
