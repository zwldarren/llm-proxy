"""Retry with exponential backoff for provider operations."""

import asyncio
import os
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, TypeVar

import httpx2

from llm_proxy.core.errors.classification import is_same_provider_retryable
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.utils import quiet_aclose
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.providers.components.error_translator import ErrorTranslator

T = TypeVar("T")

# Cap Retry-After delays to keep retry windows aligned with the existing
# exponential backoff ceiling (30s + jitter). Values above this are clamped.
MAX_RETRY_AFTER_SECONDS = 60.0

# Transport-level exceptions that indicate a transient connection problem and
# are safe to retry mid-stream. Classification is by exception type, never by
# message text, so it is robust to provider/httpx2 wording changes. httpx2
# mirrors httpx's hierarchy:
# - NetworkError covers ConnectError, ReadError, WriteError, CloseError
# - TimeoutException covers ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout
# - RemoteProtocolError covers a peer closing the connection mid-stream or
#   sending a malformed response
# asyncio.TimeoutError (== builtin TimeoutError, an OSError subclass) covers
# client-side timeouts raised outside httpx2; OSError covers raw socket errors
# (ConnectionResetError, ConnectionRefusedError, BrokenPipeError, ...).
#
# Deliberate exclusions (trade-off vs. the old message matching):
# - asyncio.CancelledError derives from BaseException, so `except Exception`
#   never catches it; it must never be classified as retryable.
# - httpx2.StreamError / LocalProtocolError are client-side misuse errors
#   ("the developer made an error"), not transport failures; retrying them
#   would only burn attempts. The old message matching could retry a
#   StreamError whose text mentioned "connection" -- that was a
#   misclassification.
# - httpx2.ProxyError is a configuration-level failure, not a transient
#   transport error; it propagates immediately.
# - Provider-specific details embedded in error *messages* (e.g. a provider
#   saying "connection reset" inside a JSON error body) are no longer retried
#   unless the exception type itself is transport-level. Such messages are
#   surfaced to the caller as-is instead.
RETRYABLE_STREAM_EXCEPTIONS = (
    httpx2.NetworkError,
    httpx2.TimeoutException,
    httpx2.RemoteProtocolError,
    asyncio.TimeoutError,
    OSError,
)

logger = get_logger(__name__)


def _is_retryable_stream_error(error: Exception) -> bool:
    """Return True when *error* is a transient transport-level failure.

    Used by ``execute_generator`` to decide whether a stream that failed
    before yielding any data can be retried. Classification is purely by
    exception type (see ``RETRYABLE_STREAM_EXCEPTIONS``); message text is
    never inspected.
    """
    return isinstance(error, RETRYABLE_STREAM_EXCEPTIONS)


def _in_test_mode() -> bool:
    """Check if we are running under pytest.

    Returns True when PYTEST_RUNNING env var is set (e.g. via pyproject.toml
    addopts or CI workflow). Tests can also set this explicitly.
    """
    return os.environ.get("PYTEST_RUNNING") == "1"


class RetryPolicy:
    """Retry operations with exponential backoff.

    Handles both single-shot operations (_with_retry) and stream generators
    (_with_retry_generator), using ErrorTranslator to map transport exceptions
    into ProviderError before deciding whether to retry.

    When ``_testing_disable_backoff=True`` or the environment variable
    ``PYTEST_RUNNING=1`` is set, ``_backoff()`` becomes a no-op so that
    retry tests complete quickly without real wall-clock delays.
    """

    def __init__(
        self,
        max_retries: int = 3,
        provider_name: str = "",
        error_translator: ErrorTranslator | None = None,
        _testing_disable_backoff: bool | None = None,
    ):
        self._max_retries = max_retries
        self._provider_name = provider_name
        self._error_translator: ErrorTranslator | None = error_translator
        # Disable backoff when running under pytest or when explicitly requested.
        if _testing_disable_backoff is None:
            _testing_disable_backoff = _in_test_mode()
        self._testing_disable_backoff = _testing_disable_backoff
        self._recorder: Callable[[dict[str, Any]], None] | None = None

    def set_recorder(self, recorder: Callable[[dict[str, Any]], None] | None) -> None:
        """Attach (or clear) a retry-attempt recorder.

        The recorder receives one dict per failed retryable attempt and is
        typically wired to append to ``EventContext.retry_attempts``.
        """
        self._recorder = recorder

    def _get_error_translator(self) -> ErrorTranslator:
        if self._error_translator is None:
            from llm_proxy.providers.components.error_translator import ErrorTranslator

            self._error_translator = ErrorTranslator(provider_name=self._provider_name)
        return self._error_translator

    def _should_retry(self, error: ProviderError, retryable_errors: set[str] | None) -> bool:
        """Decide whether to retry the same provider after *error*.

        An explicit ``retryable_errors`` override (used by some tests) takes
        precedence; otherwise the central ``is_same_provider_retryable``
        classifier decides (transient + 5xx/408/429).
        """
        if retryable_errors is not None:
            return error.error_type in retryable_errors
        return is_same_provider_retryable(error=error, status_code=error.status_code)

    def _record_attempt(self, attempt: int, error: ProviderError, will_retry: bool) -> None:
        """Emit an INFO log and record a same-provider retry attempt.

        ``attempt`` is 0-indexed; the recorded ``attempt`` field is 1-indexed
        for human readability. ``will_retry`` is False on the final (exhausted)
        attempt so consumers can distinguish a retry from a terminal failure.
        """
        if will_retry:
            logger.info(
                f"{self._provider_name} retrying after attempt "
                f"{attempt + 1}/{self._max_retries} "
                f"({error.error_type}, status={error.status_code}): {error.message}"
            )
        else:
            logger.info(
                f"{self._provider_name} retries exhausted "
                f"({self._max_retries}/{self._max_retries}) "
                f"after {error.error_type} (status={error.status_code}): {error.message}"
            )
        entry = {
            "provider": self._provider_name,
            "attempt": attempt + 1,
            "total": self._max_retries,
            "error_type": error.error_type,
            "status_code": error.status_code,
            "error_message": error.message,
            "retried": will_retry,
        }
        if self._recorder is not None:
            try:
                self._recorder(entry)
            except Exception:
                logger.debug("Retry recorder callback failed", exc_info=True)

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        retryable_errors: set[str] | None = None,
    ) -> T:
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                return await operation()
            except ProviderError as e:
                last_error = e
                if not self._should_retry(e, retryable_errors):
                    raise
                will_retry = attempt < self._max_retries - 1
                self._record_attempt(attempt, e, will_retry)
                if will_retry:
                    await self._backoff(attempt, e)
            except Exception as e:
                mapped_error = await self._get_error_translator().translate_error(e)
                last_error = mapped_error
                if isinstance(mapped_error, ProviderError) and not self._should_retry(
                    mapped_error, retryable_errors
                ):
                    raise mapped_error from e
                will_retry = attempt < self._max_retries - 1
                if isinstance(mapped_error, ProviderError):
                    self._record_attempt(attempt, mapped_error, will_retry)
                if will_retry:
                    await self._backoff(attempt, mapped_error)
                else:
                    raise mapped_error from e

        if last_error:
            raise last_error
        raise RuntimeError("Retry logic failed")

    async def execute_generator(
        self,
        generator_factory: Callable[[], AsyncIterator[T]],
        retryable_errors: set[str] | None = None,
        cancel_token=None,
    ) -> AsyncIterator[T]:
        """Execute a stream generator with retry logic for connection errors only.

        Once data has been yielded, subsequent errors are raised immediately
        to prevent duplicate data on retry.
        """
        last_error: Exception | None = None
        yielded_data = False

        for attempt in range(self._max_retries):
            gen = None
            try:
                gen = generator_factory()
                async for line in gen:
                    yield line
                    yielded_data = True
                    if cancel_token and cancel_token.is_set():
                        return
                return

            except ProviderError as e:
                if yielded_data:
                    raise
                last_error = e
                if not self._should_retry(e, retryable_errors):
                    raise
                will_retry = attempt < self._max_retries - 1
                self._record_attempt(attempt, e, will_retry)
                if will_retry:
                    await self._backoff(attempt, e)
                else:
                    raise

            except Exception as e:
                if yielded_data:
                    raise
                last_error = e
                if not _is_retryable_stream_error(e):
                    raise
                if isinstance(e, (asyncio.TimeoutError, httpx2.TimeoutException)):
                    label = "timeout"
                else:
                    label = "connection"
                logger.warning(
                    f"{self._provider_name} {label} error "
                    f"(attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    await self._backoff(attempt, e)
                else:
                    raise
            finally:
                if gen is not None:
                    await quiet_aclose(gen)

        if last_error:
            raise last_error
        raise RuntimeError("Retry logic failed to execute generator")

    async def _backoff(self, attempt: int, error: Exception | None = None) -> None:
        if self._testing_disable_backoff:
            return
        retry_after = self._extract_retry_after(error)
        if retry_after is not None:
            delay = retry_after
        else:
            delay = min(0.5 * (2**attempt), 30.0)
            delay = delay * (0.5 + random.random())
        logger.debug(f"Retrying after {delay:.2f}s (attempt {attempt + 1}/{self._max_retries})")
        await asyncio.sleep(delay)

    @staticmethod
    def _extract_retry_after(error: Exception | None) -> float | None:
        """Return the upstream Retry-After delay in seconds, if present and valid.

        Looks at the ProviderError's ``original_error`` payload (populated by
        ErrorHandler for HTTP status errors) for a ``retry_after`` value. The
        value may be an integer seconds string or an HTTP date string.
        """
        if not isinstance(error, ProviderError):
            return None
        original_error = getattr(error, "original_error", None)
        if not isinstance(original_error, dict):
            return None
        value = original_error.get("retry_after")
        if value is None:
            return None
        return RetryPolicy._parse_retry_after(value)

    @staticmethod
    def _parse_retry_after(value: Any) -> float | None:
        """Parse a Retry-After value into seconds, clamped to a maximum.

        Supports integer seconds ("5") and HTTP date strings
        ("Wed, 21 Oct 2025 07:28:00 GMT"). Values above
        ``MAX_RETRY_AFTER_SECONDS`` are clamped.
        """
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            return min(int(text), MAX_RETRY_AFTER_SECONDS)
        try:
            retry_date = parsedate_to_datetime(text)
            delay = (retry_date - datetime.now(UTC)).total_seconds()
            if delay <= 0:
                return None
            return min(delay, MAX_RETRY_AFTER_SECONDS)
        except Exception:
            return None


__all__ = ["RetryPolicy"]
