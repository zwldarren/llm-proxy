"""Routing training-data export (admin API).

The offline counterpart of smart routing's measurement posture: the feedback
labels and the judge's recorded verdicts — the corpora the third
content-reading signal will be trained on (ADR-0029) — exported as JSONL so a
training run reads them without touching the log store. Not a product surface
for clients; admin-only like the settings it measures.

The response streams. Rows are fetched and serialized one page at a time, so an
export up to the cap never holds a whole corpus — whose stored request bodies
reach a megabyte each — in memory, and a caller can consume the first lines
while the rest is still being read.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_async_session_dep, require_admin_role
from llm_proxy.database.repositories.log_repository import LogRepository
from llm_proxy.routing.dataset import (
    feedback_export_row,
    jsonl_line,
    judge_shadow_export_row,
)

router = APIRouter(
    prefix="/api/routing", tags=["routing"], dependencies=[Depends(require_admin_role)]
)

#: Per-corpus cap on exported rows; a training sweep pages through with
#: ``since``. ``format=all`` takes up to this many rows from *each* corpus.
_MAX_EXPORT_ROWS = 50_000

#: Rows per database round-trip: bounded enough that one page's raw request
#: bodies stay cheap, large enough that the round-trips are not the bottleneck.
_PAGE_ROWS = 1000

#: Fetch one page of a corpus as ``(page_size, offset) -> rows``.
_PageFetcher = Callable[[int, int], Awaitable[list[dict[str, Any]]]]
#: Turn one fetched item into an export row, or ``None`` when it is unusable.
_RowBuilder = Callable[[dict[str, Any]], dict[str, Any] | None]


async def _pages(fetch: _PageFetcher, build: _RowBuilder, *, limit: int) -> AsyncIterator[str]:
    """Yield JSONL lines one page at a time, fetching at most ``limit`` rows.

    ``limit`` bounds the rows *fetched*, not the lines yielded: a row the
    builder drops (returns ``None``) is not replaced, so a call can return
    fewer usable lines than ``limit``, and a ``since``-advance sweep recovers
    the rest. Paging stops early on a short page, so a caller asking for more
    rows than exist pays one extra query rather than walking offsets to the
    cap.
    """
    fetched = 0
    while fetched < limit:
        page_size = min(_PAGE_ROWS, limit - fetched)
        page = await fetch(page_size, fetched)
        if not page:
            return
        fetched += len(page)
        for item in page:
            row = build(item)
            if row is not None:
                yield jsonl_line(row)
        if len(page) < page_size:
            return


def _feedback_pages(repo: LogRepository, *, since: float | None, limit: int) -> AsyncIterator[str]:
    """The feedback corpus, one page of rows at a time."""

    async def fetch(page_limit: int, offset: int) -> list[dict[str, Any]]:
        return await repo.get_feedback_dataset_rows(since=since, limit=page_limit, offset=offset)

    def build(pair: dict[str, Any]) -> dict[str, Any] | None:
        return feedback_export_row(feedback=pair["feedback"], log=pair["log"])

    return _pages(fetch, build, limit=limit)


def _judge_shadow_pages(
    repo: LogRepository, *, since: float | None, limit: int
) -> AsyncIterator[str]:
    """The judge-shadow corpus, one page of rows at a time."""

    async def fetch(page_limit: int, offset: int) -> list[dict[str, Any]]:
        return await repo.get_judge_dataset_rows(since=since, limit=page_limit, offset=offset)

    def build(item: dict[str, Any]) -> dict[str, Any] | None:
        return judge_shadow_export_row(
            judge_log=item["judge_log"],
            parent_log=item["parent_log"],
            signal=item["signal"],
        )

    return _pages(fetch, build, limit=limit)


async def _export_lines(
    repo: LogRepository, *, corpus: str, since: float | None, limit: int
) -> AsyncIterator[str]:
    """The requested corpora as JSONL lines, in that order."""
    if corpus in {"feedback", "all"}:
        async for line in _feedback_pages(repo, since=since, limit=limit):
            yield line
    if corpus in {"judge_shadow", "all"}:
        async for line in _judge_shadow_pages(repo, since=since, limit=limit):
            yield line


@router.get(
    "/export/dataset",
    response_class=StreamingResponse,
    responses={200: {"content": {"application/x-ndjson": {}}}},
)
async def export_dataset(
    format: str = Query(  # noqa: A002
        "all",
        pattern="^(feedback|judge_shadow|all)$",
        description="Which corpus to export: feedback labels, judge shadow verdicts, or both",
    ),
    since: float | None = Query(
        None, description="Only rows created at or after this Unix timestamp"
    ),
    limit: int = Query(5000, ge=1, le=_MAX_EXPORT_ROWS),
    session: AsyncSession = get_async_session_dep,
) -> StreamingResponse:
    """Export routing training data as JSONL (one JSON object per line).

    - ``feedback`` — one row per explicit user rating (``ok``/``weak``/``strong``)
      on a smart-routed request, with the routed prompt and the tier decision.
    - ``judge_shadow`` — one row per recorded judge verdict, with the ensemble's
      decision it would have replaced, whether it owned the tier, and the
      user's feedback signal when one was given.
    - ``all`` — both, in that order.

    Streamed, so the document is never assembled in memory. Both corpora page
    in ascending time order, so a sweep advances by re-issuing with
    ``since`` set to the newest row it saw.

    ``limit`` bounds the rows fetched per corpus, not the usable lines: a row
    with no usable prompt is dropped rather than replaced, so the ``since``
    sweep is what eventually exports the remainder.
    """
    repo = LogRepository(session)
    return StreamingResponse(
        _export_lines(repo, corpus=format, since=since, limit=limit),
        media_type="application/x-ndjson",
    )
