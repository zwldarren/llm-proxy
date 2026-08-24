"""Base repository module with shared utilities."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    """Base repository class with common session handling."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with async session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    def _prepare_metadata_data(
        self,
        data: dict[str, Any],
        parameter_overrides_key: str = "parameter_overrides",
        metadata_key: str = "metadata",
        keys_to_remove: list[str] | None = None,
    ) -> dict[str, Any]:
        """Prepare data by merging parameter overrides into metadata.

        This is a shared helper to reduce duplication between provider and model
        repository data preparation methods.

        Args:
            data: The data dictionary to process (will be copied)
            parameter_overrides_key: Key for parameter overrides in data
            metadata_key: Key for metadata in data
            keys_to_remove: Additional keys to remove from metadata

        Returns:
            Processed data dictionary with parameter_overrides merged into metadata
        """
        result = data.copy()
        parameter_overrides = result.pop(parameter_overrides_key, {})

        # Get existing metadata or create empty dict
        metadata = result.get(metadata_key, {})
        if not isinstance(metadata, dict):
            metadata = {}

        # Merge parameter_overrides into metadata
        if parameter_overrides:
            metadata[parameter_overrides_key] = parameter_overrides

        # Remove specified keys from metadata
        if keys_to_remove:
            for key in keys_to_remove:
                metadata.pop(key, None)

        result[metadata_key] = metadata
        return result
