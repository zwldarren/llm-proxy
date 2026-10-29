"""Email invitations: admin management and invitee acceptance."""

from typing import Literal, cast

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.auth_session import issue_login_response
from llm_proxy.api.dependencies import (
    get_async_session_dep,
    get_config_from_state,
    require_admin_role,
    require_recent_auth_for_mutations,
)
from llm_proxy.api.middleware.rate_limiting import get_rate_limiter
from llm_proxy.api.schemas.auth import LoginResponse
from llm_proxy.api.schemas.invitations import (
    InvitationAccept,
    InvitationCreate,
    InvitationCreatedResponse,
    InvitationListResponse,
    InvitationPublic,
    InvitationRead,
)
from llm_proxy.core.budget import effective_budget_window, validate_budget_envelope
from llm_proxy.core.exceptions import ConflictError, NotFoundError, ValidationError
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import (
    InvitationRecord,
    InvitationRepository,
    UserRecord,
    UserRepository,
)
from llm_proxy.notifications import send_email
from llm_proxy.observability.log_intake import record_auth_event, record_member_action
from llm_proxy.observability.types import ActionCategory, Outcome
from llm_proxy.security.console_url import public_origin
from llm_proxy.security.passwords import hash_password
from llm_proxy.security.tokens import generate_invite_token, hash_token

router = APIRouter(
    prefix="/api/team/invitations",
    tags=["Team"],
    dependencies=[Depends(require_admin_role), Depends(require_recent_auth_for_mutations)],
)
public_router = APIRouter(prefix="/api/auth/invitations", tags=["authentication"])
limiter = get_rate_limiter()


def _invitation_read(invitation: InvitationRecord) -> InvitationRead:
    return InvitationRead(
        id=invitation.id,
        email=invitation.email,
        role=cast(Literal["admin", "viewer"], invitation.role),
        status=cast(
            Literal["pending", "accepted", "revoked", "expired"],
            InvitationRepository.status(invitation),
        ),
        allowed_models=invitation.allowed_models,
        budget_usd=invitation.budget_usd,
        budget_period=invitation.budget_period,
        budget_reset_day=invitation.budget_reset_day,
        created_at=invitation.created_at,
        expires_at=invitation.expires_at,
        accepted_at=invitation.accepted_at,
        revoked_at=invitation.revoked_at,
        last_sent_at=invitation.last_sent_at,
    )


async def _deliver_invite(
    request: Request, invitation: InvitationRecord, token: str, *, resend: bool = False
) -> tuple[str, bool]:
    console_auth = (await get_config_from_state(request)).server_params.console_auth
    base_url = public_origin(console_auth) or str(request.base_url).rstrip("/")
    invite_url = f"{base_url}/invite/{token}"
    subject = "You have been invited to LLM Proxy"
    if resend:
        subject = "Your LLM Proxy invitation was renewed"
    body = (
        f"You have been invited to the LLM Proxy console as {invitation.role}.\n\n"
        f"Open this single-use link within 7 days:\n{invite_url}\n\n"
        "If you were not expecting this invitation, ignore it."
    )
    sent = await send_email(console_auth.smtp, invitation.email, subject, body)
    return invite_url, sent


@router.post("", response_model=InvitationCreatedResponse, status_code=201)
async def create_invitation(
    data: InvitationCreate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    validate_budget_envelope(data.budget_usd, data.budget_period, data.budget_reset_day)
    budget_usd, budget_period, budget_reset_day = effective_budget_window(
        data.budget_usd, data.budget_period, data.budget_reset_day
    )
    users = UserRepository(session)
    if await users.get_by_email(data.email) is not None:
        raise ConflictError(message="An account already uses that email address")

    invitations = InvitationRepository(session)
    if await invitations.get_pending_by_email(data.email) is not None:
        raise ConflictError(message="A pending invitation already exists for that email")

    token = generate_invite_token()
    invitation = await invitations.create(
        email=data.email,
        token_hash=hash_token(token),
        role=data.role,
        created_by_user_id=admin.id,
        allowed_models=data.allowed_models,
        budget_usd=budget_usd,
        budget_period=budget_period,
        budget_reset_day=budget_reset_day,
    )
    await session.commit()
    invite_url, sent = await _deliver_invite(request, invitation, token)
    record_member_action(
        request,
        actor=admin.username,
        action_category=ActionCategory.CREATE,
        target_user=invitation.email,
        extra={"role": invitation.role, "invitation_id": invitation.id},
        status_code=201,
    )
    return InvitationCreatedResponse(
        invitation=_invitation_read(invitation),
        token=token,
        invite_url=invite_url,
        email_sent=sent,
    )


@router.get("", response_model=InvitationListResponse)
async def list_invitations(
    session: AsyncSession = get_async_session_dep,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    repo = InvitationRepository(session)
    items = await repo.list_all(limit=limit, offset=offset)
    return InvitationListResponse(
        items=[_invitation_read(item) for item in items],
        total=await repo.count_all(),
    )


@router.delete("/{invitation_id}", response_model=InvitationRead)
async def revoke_invitation(
    invitation_id: str,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    repo = InvitationRepository(session)
    existing = await repo.get_by_id(invitation_id)
    if existing is None:
        raise NotFoundError(message="Invitation not found")
    if InvitationRepository.status(existing) == "accepted":
        raise ValidationError(message="An accepted invitation cannot be revoked")
    invitation = await repo.revoke(invitation_id)
    if invitation is None:
        # Revoked concurrently between the read above and the conditional
        # update: report the current state instead of rewriting the audit trail.
        raise ValidationError(message="This invitation is already revoked")
    await session.commit()
    record_member_action(
        request,
        actor=admin.username,
        action_category=ActionCategory.DELETE,
        target_user=invitation.email,
        extra={"invitation_id": invitation.id},
    )
    return _invitation_read(invitation)


@router.post("/{invitation_id}/resend", response_model=InvitationCreatedResponse)
async def resend_invitation(
    invitation_id: str,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    repo = InvitationRepository(session)
    existing = await repo.get_by_id(invitation_id)
    if existing is None:
        raise NotFoundError(message="Invitation not found")
    if InvitationRepository.status(existing) != "pending":
        raise ValidationError(message="Only a pending invitation can be resent")
    token = generate_invite_token()
    invitation = await repo.refresh_token(invitation_id, hash_token(token))
    if invitation is None:
        raise NotFoundError(message="Invitation not found")
    await session.commit()
    invite_url, sent = await _deliver_invite(request, invitation, token, resend=True)
    record_member_action(
        request,
        actor=admin.username,
        action_category=ActionCategory.UPDATE,
        target_user=invitation.email,
        extra={"invitation_id": invitation.id, "resend": True},
    )
    return InvitationCreatedResponse(
        invitation=_invitation_read(invitation),
        token=token,
        invite_url=invite_url,
        email_sent=sent,
    )


async def _pending_invitation(repo: InvitationRepository, token: str) -> InvitationRecord:
    invitation = await repo.get_by_token_hash(hash_token(token))
    if invitation is None:
        raise NotFoundError(message="Invitation not found or already used")
    if InvitationRepository.status(invitation) != "pending":
        raise ValidationError(
            message="This invitation is no longer valid", code="invitation_invalid"
        )
    return invitation


@public_router.get("/{token}", response_model=InvitationPublic)
@limiter.limit("auth.invite_status")
async def get_invitation(
    token: str,
    request: Request,
    session: AsyncSession = get_async_session_dep,
):
    # ``request`` is part of the signature because the rate limiter resolves the
    # client key from it; without it the decorator above silently does nothing.
    invitation = await _pending_invitation(InvitationRepository(session), token)
    return InvitationPublic(
        email=invitation.email,
        role=cast(Literal["admin", "viewer"], invitation.role),
        expires_at=invitation.expires_at,
    )


@public_router.post("/{token}/accept", response_model=LoginResponse)
@limiter.limit("auth.invite_accept")
async def accept_invitation(
    token: str,
    data: InvitationAccept,
    request: Request,
    response: Response,
    session: AsyncSession = get_async_session_dep,
):
    invitations = InvitationRepository(session)
    invitation = await _pending_invitation(invitations, token)

    users = UserRepository(session)
    if await users.get_by_email(invitation.email) is not None:
        raise ConflictError(message="An account already uses that email address")
    try:
        user = await users.create_user(
            data.username,
            hash_password(data.password),
            role=invitation.role,
            must_change_password=False,
            email=invitation.email,
            email_verified_by="invite",
        )
    except ValueError as exc:
        raise ConflictError(message=str(exc)) from exc

    if invitation.allowed_models is not None:
        await users.set_allowed_models(user.id, invitation.allowed_models)
    if invitation.budget_usd is not None:
        await users.set_budget(
            user.id,
            budget_usd=invitation.budget_usd,
            budget_period=invitation.budget_period,
            budget_reset_day=invitation.budget_reset_day,
        )
    # Claim the invitation before minting the session: the conditional UPDATE
    # makes a concurrent second redemption of the same token fail here instead
    # of being caught incidentally by the unique email constraint.
    if await invitations.mark_accepted(invitation.id, user.id) is None:
        await session.rollback()
        raise ConflictError(message="This invitation has already been used")
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="invite",
        metadata_extra={"invitation_id": invitation.id},
    )
    return await issue_login_response(request, response, session, user, auth_method="invite")
