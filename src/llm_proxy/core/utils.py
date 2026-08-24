"""Core utility functions for LLM Proxy."""

import asyncio
import re
import uuid
from collections.abc import Callable
from typing import Any

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


async def quiet_aclose(agen: Any) -> None:
    """Close an async iterator, never raising.

    During disconnect teardown the same async generator can be closed from
    two places at once: an explicit ``finally`` (often inside an
    ``asyncio.shield`` task) and asyncio's async-gen GC finalizer, which
    schedules its own ``agen.aclose()`` task. The second ``aclose()`` on a
    generator whose close is already in flight raises ``RuntimeError:
    aclose(): asynchronous generator is already running``; when that happens
    inside an orphaned shield/finalizer task it surfaces as "Task exception
    was never retrieved". The race is benign — whichever close wins finishes
    the cleanup — so it (and any other cleanup failure) is swallowed here.
    """
    aclose = getattr(agen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except asyncio.CancelledError:
        pass
    except RuntimeError as exc:
        if "already running" not in str(exc):
            logger.debug(f"Ignoring RuntimeError while closing async generator: {exc}")
    except Exception as exc:
        logger.debug(f"Ignoring error while closing async generator: {exc}")


def install_asyncgen_close_race_filter() -> Callable[[], None]:
    """Demote the benign async-gen close race to DEBUG on the running loop.

    Even with :func:`quiet_aclose` guarding every explicit close, asyncio's
    async-gen GC finalizer schedules its own raw ``agen.aclose()`` task when
    a generator is collected mid-teardown. Under disconnect/cancellation
    timing that finalizer task can lose the race against an explicit close
    that is already in flight, and its ``RuntimeError: aclose(): asynchronous
    generator is already running`` then surfaces as an ERROR-level "Task
    exception was never retrieved" log line. The race is benign — whichever
    close wins finishes the cleanup — so exactly that context is demoted to
    DEBUG here; every other loop exception is delegated unchanged.

    Returns a restore function that reinstalls the previous handler.
    """
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()

    def _handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "asynchronous generator is already running" in str(
            exc
        ):
            logger.debug(f"Ignoring benign async generator close race: {exc}")
            return
        if previous is not None:
            previous(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)

    def _restore() -> None:
        loop.set_exception_handler(previous)

    return _restore


def generate_response_id() -> str:
    """Generate a unique response ID.

    Format: resp_<24-char hex uuid>
    Used across all serializers and adapters for consistent response ID generation.

    Returns:
        A unique response ID string.
    """
    return f"resp_{uuid.uuid4().hex[:24]}"


def safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int safely, returning default on failure.

    Args:
        value: The value to convert
        default: Default to return if conversion fails (default: 0)

    Returns:
        Converted int, or default on failure
    """
    if value is None:
        return default
    try:
        return int(value)
    except ValueError, TypeError:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely, returning default on failure.

    Args:
        value: The value to convert
        default: Default to return if conversion fails (default: 0.0)

    Returns:
        Converted float, or default on failure
    """
    if value is None:
        return default
    try:
        return float(value)
    except ValueError, TypeError:
        return default


# Data URI regex pattern for parsing base64 data URLs
# Matches: data:<media_type>;base64,<data>
_DATA_URI_PATTERN = re.compile(r"data:([^;]+);base64,(.+)")


def parse_data_uri(url: str) -> tuple[str | None, str] | None:
    """Parse a data URI and extract media type and base64 data.

    Args:
        url: The data URI string (e.g., "data:image/png;base64,iVBORw0KGgo...")

    Returns:
        A tuple of (media_type, data) if parsing succeeds, None otherwise.
        media_type may be None if not present in the URI.
    """
    if not url.startswith("data:"):
        return None

    # Try to extract media type from the prefix
    media_type = None
    if ";" in url:
        media_type = url.split(";")[0].split(":")[1] if ":" in url else None

    match = _DATA_URI_PATTERN.match(url)
    if match:
        # Use extracted media type from regex if available, fallback to prefix extraction
        extracted_media_type = match.group(1) if match.group(1) else media_type
        return extracted_media_type, match.group(2)

    return None


def create_image_source_from_url(url: str, _detail: str | None = None) -> Any:
    """Create an ImageSource from a URL string.

    Handles both data URIs and regular HTTP URLs.

    Args:
        url: The image URL (data URI or HTTP URL)
        detail: Optional detail level for the image

    Returns:
        ImageSource if parsing succeeds, None otherwise.
    """
    from llm_proxy.models.types import ImageSource

    if url.startswith("data:"):
        parsed = parse_data_uri(url)
        if parsed:
            media_type, data = parsed
            return ImageSource(type="base64", data=data, media_type=media_type)
        return None
    else:
        return ImageSource(type="url", data=url, media_type=None)
