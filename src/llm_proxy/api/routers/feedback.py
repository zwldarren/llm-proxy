"""Explicit user feedback on smart-routing decisions.

Closes the feedback loop: a user rates a routed request (ok/weak/strong),
the signal is applied to the model's experience (Thompson sampling picks it
up on the next routing decision), and the row doubles as a calibration eval
sample for future Platt temperature re-fitting.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import (
    get_async_session_dep,
    get_current_user,
    require_authenticated,
)
from llm_proxy.api.schemas.me import MeFeedback
from llm_proxy.core.exceptions import ConflictError, NotFoundError
from llm_proxy.database.repositories.log_repository import LogRepository
from llm_proxy.database.tables import FeedbackRecord, UserRecord
from llm_proxy.observability.logger import get_logger
from llm_proxy.routing.model_experience import ModelExperienceStore

router = APIRouter(
    prefix="/api/me", tags=["Feedback"], dependencies=[Depends(require_authenticated)]
)
logger = get_logger(__name__)

# Cap on batch lookups so the query string stays bounded.
_MAX_BATCH_IDS = 200


@router.get("/feedback", response_model=dict[str, str])
async def list_feedback(
    request_ids: str = Query(..., description="Comma-separated request IDs"),
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
) -> dict[str, str]:
    """Return recorded feedback signals for a batch of request IDs.

    Powers the logs UI, which marks rows that already received feedback.
    Only IDs with recorded feedback appear in the response; non-admin users
    only see their own submissions.
    """
    ids = [rid.strip() for rid in request_ids.split(",") if rid.strip()][:_MAX_BATCH_IDS]
    if not ids:
        return {}
    stmt = select(FeedbackRecord.request_id, FeedbackRecord.signal).where(
        FeedbackRecord.request_id.in_(ids)
    )
    if user.role != "admin":
        stmt = stmt.where(FeedbackRecord.user_id == user.id)
    result = await session.execute(stmt)
    return {row.request_id: row.signal for row in result.all()}


@router.post("/feedback", status_code=204)
async def submit_feedback(
    body: MeFeedback,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
) -> None:
    """Record explicit feedback for a routed request.

    - 404: unknown request_id (or owned by another user, for non-admins)
    - 409: feedback already recorded for this request (one per request)
    - 422: the request was not smart-routed (no resolved model to credit)
    """
    # Non-admin users may only rate their own requests; admins are unscoped,
    # matching the logs endpoints.
    user_filter = None if user.role == "admin" else user.id
    repo = LogRepository(session)
    log = await repo.get_log_by_request_id_for_api(body.request_id, user_id=user_filter)
    if log is None:
        raise NotFoundError(message="Log not found")

    routing = (log.get("log_metadata") or {}).get("routing") or {}
    resolved_model = routing.get("resolved_model")
    if not resolved_model:
        raise HTTPException(status_code=422, detail="Request was not smart-routed")

    # Idempotency: the primary key on feedback_records enforces one feedback
    # per request; check first for a clean 409 instead of an integrity error.
    existing = await session.execute(
        select(FeedbackRecord.request_id).where(FeedbackRecord.request_id == body.request_id)
    )
    if existing.scalar_one_or_none() is not None:
        raise ConflictError(message="Feedback already recorded for this request")

    store = ModelExperienceStore(session=session)
    await store.record_feedback_async(resolved_model, body.signal)
    session.add(
        FeedbackRecord(
            request_id=body.request_id,
            signal=body.signal,
            model=resolved_model,
            user_id=user.id,
        )
    )
    await session.commit()
