"""Web search tool logs must record the query verbatim.

The stored query used to be truncated to 32 characters with a SHA-256 suffix
(``site:github.com pi-web-access ni... [hash:6e692f98]``), which made the admin
log useless for diagnosing a failed search — an operator could not see what was
actually searched. The query is still a *content* field: with the body-logging
master switch off, ``scrub_log_metadata`` replaces it with ``_bodies_disabled``.

Exercised through the ``record_web_search`` intake verb (ADR-0020); assertions
are on the row handed to the background writer.
"""

from unittest.mock import MagicMock, patch

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.observability import log_intake
from llm_proxy.observability.log_intake import WebSearchLogEntry, record_web_search
from llm_proxy.observability.redaction import BODIES_DISABLED_MARKER, scrub_log_metadata
from llm_proxy.observability.types import WebSearchStatus

QUERY = "site:github.com pi-web-access nicobailon and a very long tail for good measure"


def _record(entry: WebSearchLogEntry, **attribution) -> MagicMock:
    log_intake.configure(config=LoggingConfig())
    service = MagicMock()
    with patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service):
        record_web_search(entry, **attribution)
    return service


def test_web_search_query_is_stored_verbatim():
    """Regression: the query was masked down to a 32-char prefix plus a hash."""
    service = _record(
        WebSearchLogEntry(
            query=QUERY,
            status=WebSearchStatus.ERROR,
            error_message="SearXNG web search failed: 401",
            status_code=500,
            provider="searxng",
        ),
        user_identity="admin",
    )

    log = service.create_log_background.call_args[0][0]
    assert log.log_metadata["web_search_query"] == QUERY
    assert "[hash:" not in log.log_metadata["web_search_query"]


def test_failure_keeps_provider_reason_and_outcome():
    """The row stays diagnosable: which provider failed, why, and the outcome."""
    service = _record(
        WebSearchLogEntry(
            query="q",
            status=WebSearchStatus.ERROR,
            error_message="Ollama web search failed: 401",
            status_code=500,
            provider="ollama",
        )
    )

    log = service.create_log_background.call_args[0][0]
    assert log.provider == "ollama"
    assert log.log_metadata["web_search_provider"] == "ollama"
    assert log.error_message == "Ollama web search failed: 401"
    assert log.outcome == "failure"


def test_query_is_still_scrubbed_when_body_logging_is_off():
    """The verbatim query must obey the body-logging privacy switch."""
    service = _record(WebSearchLogEntry(query=QUERY, status=WebSearchStatus.SUCCESS))

    metadata = service.create_log_background.call_args[0][0].log_metadata
    assert scrub_log_metadata(metadata)["web_search_query"] == BODIES_DISABLED_MARKER
