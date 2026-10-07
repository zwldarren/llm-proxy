"""Logging for internal model calls made on a client request's behalf.

MCP and web-search calls already have their own log type and their own row
(:mod:`llm_proxy.observability.tool_logging`); a routing-judge call is the same
kind of thing (ADR-0018): an auxiliary model call the client did not make
itself, billed on the client's behalf, and worth its own row in the log store
and its own usage record in the spend ledger. This module is that row's single
writer, so the judge seam never has to know a log store exists.

Bodies are never recorded. The judge's input is a bounded excerpt of a request
the parent row already holds, and its output is a verdict that is recorded in
full in ``log_metadata["judge"]``; a second copy of user content buys nothing.
Rows carry attribution, tokens, cost and that metadata only.

Rows and usage records written here are attributed to the request that triggered
the call — same API key, user and session — because the spend is that request's.
Budget enforcement therefore has to read ``judge`` usage records as well as
``endpoint`` ones; the query layer does exactly that.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.observability.service import (
    RequestLogCreate,
    RequestLogService,
    UsageRecordCreate,
    UsageService,
)
from llm_proxy.observability.types import LogType

#: Endpoint recorded on an internal call's row. The judge is not a client
#: endpoint; the ``/internal/`` prefix keeps it clearly out of the protocol paths.
JUDGE_CALL_ENDPOINT = "/internal/routing-judge"


@dataclass(frozen=True)
class InternalCallLogEntry:
    """One internal model call, ready to become a log row and a usage record.

    ``request_id`` is the caller's, not this service's: the child id derives from
    the parent request so a judge row is idempotent (the same parent cannot
    silently produce two rows) and greppable next to the request it served.
    """

    request_id: str
    timestamp: float
    model: str
    endpoint: str = JUDGE_CALL_ENDPOINT
    method: str = "POST"
    provider: str | None = None
    provider_model_name: str | None = None
    status_code: int | None = None
    response_time_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    error_message: str | None = None
    log_metadata: dict[str, Any] = field(default_factory=dict)


class InternalCallLogService:
    """Writes internal calls to the request-log and usage stores.

    Mirrors :class:`~llm_proxy.observability.tool_logging.ToolLogService`: the
    caller supplies the attribution, the service formats the records. Never
    raises — accounting for an internal call must not break the client request
    the call was made for.
    """

    def __init__(
        self,
        log_service: RequestLogService,
        usage_service: UsageService | None = None,
    ) -> None:
        self._log_service = log_service
        self._usage_service = usage_service or UsageService()

    def log_call_background(
        self,
        entry: InternalCallLogEntry,
        *,
        user_id: int | None = None,
        user_identity: str | None = None,
        api_key_name: str | None = None,
        auth_method: str | None = None,
        session_id: str | None = None,
        client_ip: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        """Enqueue one internal call's log row and usage record. MUST NOT block.

        ``model`` is the judge model the operator configured; the endpoint's own
        ``model`` column keeps that name even though the request that triggered it
        is logged under its virtual-model alias. The concrete upstream model — when
        the provider reports one — rides in metadata under ``provider_model_name``,
        the same place an endpoint row keeps it.
        """
        metadata = dict(entry.log_metadata)
        if entry.provider_model_name:
            metadata.setdefault("provider_model_name", entry.provider_model_name)

        log = RequestLogCreate(
            request_id=entry.request_id,
            timestamp=entry.timestamp,
            endpoint=entry.endpoint,
            method=entry.method,
            log_type=LogType.JUDGE,
            status_code=entry.status_code,
            response_time_ms=entry.response_time_ms,
            model=entry.model,
            provider=entry.provider,
            prompt_tokens=entry.prompt_tokens,
            completion_tokens=entry.completion_tokens,
            total_tokens=entry.total_tokens,
            cost_usd=entry.cost_usd,
            error_message=entry.error_message,
            log_metadata=metadata,
            user_id=user_id,
            user_identity=user_identity,
            api_key_name=api_key_name,
            auth_method=auth_method,
            session_id=session_id,
            client_ip=client_ip,
            user_agent=user_agent,
            outcome="success" if entry.status_code == 200 else "failure",
        )

        self._log_service.create_log_background(log)
        self._usage_service.create_usage_background(UsageRecordCreate.from_request_log(log))


# Global service instance (initialized lazily), mirroring tool_logging.
_internal_call_log_service: InternalCallLogService | None = None
_internal_call_log_service_lock = threading.RLock()


def get_internal_call_log_service(
    config_or_service: LoggingConfig | RequestLogService | None = None,
) -> InternalCallLogService:
    """Get or create the internal-call log service.

    ``None`` returns the existing instance, or auto-creates one from the
    on-disk logging config (tests and embedded callers). Startup passes the
    shared :class:`RequestLogService` so these rows land in the same batch
    writers as every other log.
    """
    global _internal_call_log_service

    if config_or_service is None:
        with _internal_call_log_service_lock:
            if _internal_call_log_service is None:
                from llm_proxy.config.manager import load_logging_config

                _internal_call_log_service = InternalCallLogService(
                    RequestLogService(load_logging_config())
                )
        return _internal_call_log_service

    with _internal_call_log_service_lock:
        if isinstance(config_or_service, RequestLogService):
            _internal_call_log_service = InternalCallLogService(config_or_service)
        elif isinstance(config_or_service, LoggingConfig):
            _internal_call_log_service = InternalCallLogService(
                RequestLogService(config_or_service)
            )
    return _internal_call_log_service


__all__ = [
    "JUDGE_CALL_ENDPOINT",
    "InternalCallLogEntry",
    "InternalCallLogService",
    "get_internal_call_log_service",
]
