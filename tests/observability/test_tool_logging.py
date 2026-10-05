"""Web search tool logs must record the query verbatim.

The stored query used to be truncated to 32 characters with a SHA-256 suffix
(``site:github.com pi-web-access ni... [hash:6e692f98]``), which made the admin
log useless for diagnosing a failed search — an operator could not see what was
actually searched. The query is still a *content* field: with the body-logging
master switch off, ``scrub_log_metadata`` replaces it with ``_bodies_disabled``.
"""

from unittest.mock import MagicMock

from llm_proxy.observability.redaction import BODIES_DISABLED_MARKER, scrub_log_metadata
from llm_proxy.observability.tool_logging import ToolLogService, WebSearchLogEntry
from llm_proxy.observability.types import WebSearchStatus

QUERY = "site:github.com pi-web-access nicobailon and a very long tail for good measure"


def _service() -> tuple[ToolLogService, MagicMock]:
    log_service = MagicMock()
    return ToolLogService(log_service), log_service


def test_web_search_query_is_stored_verbatim():
    """Regression: the query was masked down to a 32-char prefix plus a hash."""
    service, log_service = _service()

    service.log_web_search_background(
        WebSearchLogEntry(
            query=QUERY,
            status=WebSearchStatus.ERROR,
            error_message="SearXNG web search failed: 401",
            status_code=500,
            provider="searxng",
        ),
        user_identity="admin",
    )

    log = log_service.create_log_background.call_args[0][0]
    assert log.log_metadata["web_search_query"] == QUERY
    assert "[hash:" not in log.log_metadata["web_search_query"]


def test_failure_keeps_provider_reason_and_outcome():
    """The row stays diagnosable: which provider failed, why, and the outcome."""
    service, log_service = _service()

    service.log_web_search_background(
        WebSearchLogEntry(
            query="q",
            status=WebSearchStatus.ERROR,
            error_message="Ollama web search failed: 401",
            status_code=500,
            provider="ollama",
        )
    )

    log = log_service.create_log_background.call_args[0][0]
    assert log.provider == "ollama"
    assert log.log_metadata["web_search_provider"] == "ollama"
    assert log.error_message == "Ollama web search failed: 401"
    assert log.outcome == "failure"


def test_query_is_still_scrubbed_when_body_logging_is_off():
    """The verbatim query must obey the body-logging privacy switch."""
    service, log_service = _service()

    service.log_web_search_background(
        WebSearchLogEntry(query=QUERY, status=WebSearchStatus.SUCCESS)
    )

    metadata = log_service.create_log_background.call_args[0][0].log_metadata
    assert scrub_log_metadata(metadata)["web_search_query"] == BODIES_DISABLED_MARKER
