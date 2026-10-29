"""Team member management endpoints (admin-only)."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import (
    get_async_session_dep,
    get_auth_config,
    publish_config_generation,
    require_admin_role,
    require_recent_auth,
    require_recent_auth_for_mutations,
)
from llm_proxy.api.middleware.api_key_cache import invalidate_api_key_cache
from llm_proxy.api.routers.logs import invalidate_user_role_cache
from llm_proxy.api.schemas.common import validate_email
from llm_proxy.api.schemas.team import (
    TeamMemberBudgetUpdate,
    TeamMemberCreate,
    TeamMemberEmailUpdate,
    TeamMemberListResponse,
    TeamMemberModelsUpdate,
    TeamMemberPasswordReset,
    TeamMemberRateLimitUpdate,
    TeamMemberRead,
    TeamMemberRoleUpdate,
    TeamMemberUsernameUpdate,
    TeamMemberUsernameUpdateResponse,
)
from llm_proxy.api.session_cookies import set_console_cookie
from llm_proxy.core.budget import BudgetEnvelope, effective_budget_window, validate_budget_envelope
from llm_proxy.core.exceptions import ConflictError, NotFoundError, ValidationError
from llm_proxy.core.request_facts import get_request_identity
from llm_proxy.database import (
    ApiKeyRepository,
    OAuthIdentityRepository,
    PasskeyRepository,
    TotpRepository,
    UserRepository,
    UserSessionRepository,
)
from llm_proxy.database.repositories.usage_repository import UsageRepository
from llm_proxy.database.tables import UserRecord
from llm_proxy.observability.log_intake import record_member_action
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.types import ActionCategory, Outcome
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.passwords import hash_password, is_bcrypt_hash

router = APIRouter(
    prefix="/api/team",
    tags=["Team"],
    dependencies=[Depends(require_admin_role), Depends(require_recent_auth_for_mutations)],
)
logger = get_logger(__name__)


async def _get_member_or_404(repo: UserRepository, user_id: int) -> UserRecord:
    """Resolve a team member by id or raise a 404."""
    user = await repo.get_by_id(user_id)
    if not user:
        raise NotFoundError(message=f"User {user_id} not found")
    return user


async def _guard_last_active_admin(repo: UserRepository, target: UserRecord, action: str) -> None:
    """Refuse an operation that would remove the last active admin.

    Applies to delete / demote / deactivate: any of them may target an admin,
    but never the final active one, or the deployment would be left without a
    usable admin account.
    """
    if (
        target.role == "admin"
        and target.is_active
        and await repo.count_active_admins(for_update=True) <= 1
    ):
        raise ValidationError(message=f"Cannot {action} the last active admin")


@dataclass
class MemberOperation:
    """A completed member-management operation, for post-commit bookkeeping.

    Bundles the actor/action/target triple with the optional log, cache
    invalidation, and audit-status details so the shared commit hook takes
    one parameter instead of a travelling bundle of keyword arguments.
    """

    actor: str
    action: ActionCategory
    target_user: str
    outcome: Outcome = Outcome.SUCCESS
    extra: dict[str, Any] | None = None
    log_message: str | None = None
    log_fields: dict[str, Any] | None = None
    invalidate_api_keys: bool = False
    invalidate_role_usernames: tuple[str, ...] = ()
    status_code: int = 200


async def _commit_member_operation(request: Request, op: MemberOperation) -> None:
    """Post-commit bookkeeping shared by member-management endpoints.

    Invalidates the API-key / role caches so the change takes effect without
    waiting for cache TTLs, logs the operation, and writes the audit entry.
    Must run after ``session.commit()``.
    """
    if op.invalidate_api_keys:
        invalidate_api_key_cache()
    if op.invalidate_role_usernames:
        invalidate_user_role_cache(*op.invalidate_role_usernames)
    if op.invalidate_api_keys or op.invalidate_role_usernames:
        # Both caches are process-local and built from the users/api_keys tables
        # (not from ProxyConfig). Wake peer workers so they drop their snapshot
        # when they adopt the new generation, instead of honouring a demoted
        # role or a deactivated key until the 60-second TTL.
        await publish_config_generation(request)
    if op.log_message:
        logger.info(op.log_message, **(op.log_fields or {}))
    record_member_action(
        request,
        actor=op.actor,
        action_category=op.action,
        target_user=op.target_user,
        outcome=op.outcome,
        extra=op.extra,
        status_code=op.status_code,
    )


async def _with_budget_spend(
    session: AsyncSession, users: list[UserRecord]
) -> list[TeamMemberRead]:
    """Attach current-window spend to budgeted members (one grouped query).

    Members without a budget keep ``budget_spend_usd`` / ``budget_period_start``
    as null.
    """
    members = [TeamMemberRead.model_validate(u) for u in users]
    window_starts: dict[int, float] = {}
    for user in users:
        if user.budget_usd is not None:
            window_starts[user.id] = BudgetEnvelope.from_orm_fields(user).effective_start_ts()
    if not window_starts:
        return members
    spend_by_user = await UsageRepository(session).get_spend_since_by_user(window_starts)
    by_id = {member.id: member for member in members}
    for user_id, start_ts in window_starts.items():
        member = by_id[user_id]
        member.budget_period_start = datetime.fromtimestamp(start_ts, tz=UTC)
        member.budget_spend_usd = spend_by_user.get(user_id, 0.0)
    return members


async def _member_with_spend(session: AsyncSession, user: UserRecord) -> TeamMemberRead:
    """Attach the member's current-window spend to a single member."""
    return (await _with_budget_spend(session, [user]))[0]


@router.get("/members", response_model=TeamMemberListResponse)
async def list_members(
    session: AsyncSession = get_async_session_dep,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of members to return"),
    offset: int = Query(0, ge=0, description="Number of members to skip"),
    search: str | None = Query(None, max_length=100, description="Username substring filter"),
    sort_by: Literal["username", "role", "status", "created"] = Query("username"),
    sort_dir: Literal["asc", "desc"] = Query("asc"),
):
    repo = UserRepository(session)
    users = await repo.list_users(
        limit=limit,
        offset=offset,
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )
    return TeamMemberListResponse(
        items=await _with_budget_spend(session, users),
        total=await repo.count_users(search=search),
    )


@router.post("/members", response_model=TeamMemberRead, status_code=201)
async def create_member(
    data: TeamMemberCreate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    repo = UserRepository(session)
    try:
        # The initial password is chosen by the admin, so it must be treated
        # as temporary: the member is forced to set their own password on
        # first login before they can use the rest of the API.
        user = await repo.create_user(
            data.username, hash_password(data.password), role=data.role, must_change_password=True
        )
    except ValueError as e:
        raise ConflictError(message=str(e)) from e
    if data.allowed_models is not None:
        await repo.set_allowed_models(user.id, data.allowed_models)
    await session.commit()
    await session.refresh(user)
    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.CREATE,
            target_user=user.username,
            extra={"role": user.role},
            log_message="Admin created a team member",
            log_fields={
                "admin_username": admin.username,
                "target_user_id": user.id,
                "target_username": user.username,
                "role": user.role,
            },
            status_code=201,
        ),
    )
    return TeamMemberRead.model_validate(user)


@router.delete("/members/{user_id}", response_model=dict)
async def delete_member(
    user_id: int,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    repo = UserRepository(session)
    user = await _get_member_or_404(repo, user_id)
    if user.id == admin.id:
        raise ValidationError(message="You cannot delete your own account")
    await _guard_last_active_admin(repo, user, "delete")

    # Delete their API keys using a scoped query rather than loading all keys
    # into memory and filtering in Python.
    api_key_repo = ApiKeyRepository(session)
    await api_key_repo.delete_api_keys_by_user(user_id)

    # A hard delete is refused while the user still has an active console
    # session (UserRepository.delete_user guards it). Revoke sessions first:
    # "delete member" means delete, not "delete after the admin discovers the
    # two-step deactivate-then-delete workaround".
    session_repo = UserSessionRepository(session)
    await session_repo.deactivate_user_sessions(user_id)

    # Delete the user (deactivated sessions are cascadingly deleted)
    await repo.delete_user(user_id)

    await session.commit()
    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.DELETE,
            target_user=user.username,
            log_message="Admin deleted a team member",
            log_fields={
                "admin_username": admin.username,
                "target_user_id": user_id,
                "target_username": user.username,
            },
            invalidate_api_keys=True,
            # Drop the logs router's username-keyed role cache entry so a
            # recycled username cannot inherit the deleted user's cached role.
            invalidate_role_usernames=(user.username,),
        ),
    )
    return {"message": f"User {user_id} deleted"}


@router.put("/members/{user_id}/role", response_model=TeamMemberRead)
async def update_member_role(
    user_id: int,
    data: TeamMemberRoleUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    """Change a member's role (promote viewer -> admin, demote admin -> viewer).

    The role rides in the JWT ``type`` claim, so on an actual change all of
    the member's existing tokens and session keys are revoked and they must
    log in again at the new privilege level. Self-service role changes and
    demoting the last active admin are refused.
    """
    repo = UserRepository(session)
    target = await _get_member_or_404(repo, user_id)
    if target.id == admin.id:
        raise ValidationError(message="You cannot change your own role")
    if target.role == data.role:
        # No-op: return current state without revoking sessions.
        return await _member_with_spend(session, target)
    if data.role == "viewer":
        await _guard_last_active_admin(repo, target, "demote")

    # A demoted viewer must obey the public-host tracing rule. Re-running DNS
    # validation during a role change is slow and can fail on transient
    # resolution; drop the personal config instead. The user can re-add a
    # public collector after the demotion.
    had_tracing_config = target.tracing_config is not None
    if data.role == "viewer" and had_tracing_config:
        await repo.set_tracing_config(user_id, None)

    updated = await repo.set_role(user_id, data.role)
    assert updated is not None  # existence checked above
    await repo.increment_token_version(user_id)
    session_repo = UserSessionRepository(session)
    await session_repo.deactivate_user_sessions(user_id)
    await session.commit()

    if data.role == "viewer" and had_tracing_config:
        from llm_proxy.observability.user_tracing import get_user_tracing_manager

        await get_user_tracing_manager().publish_invalidation(user_id)

    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.UPDATE,
            target_user=updated.username,
            extra={
                "operation": "role",
                "old_role": target.role,
                "new_role": data.role,
            },
            log_message="Admin changed a member's role",
            log_fields={
                "admin_username": admin.username,
                "target_user_id": user_id,
                "target_username": updated.username,
                "old_role": target.role,
                "new_role": data.role,
            },
            invalidate_api_keys=True,
            invalidate_role_usernames=(updated.username,),
        ),
    )
    return await _member_with_spend(session, updated)


@router.post("/members/{user_id}/deactivate", response_model=TeamMemberRead)
async def deactivate_member(
    user_id: int,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    """Deactivate a member without deleting them.

    A deactivated member cannot log in and all of their API keys stop
    authenticating (the key cache snapshots the owner's status). Their keys,
    usage history, and logs are preserved, and the account can be reactivated
    later. Idempotent: deactivating an already-inactive member is a no-op.
    """
    repo = UserRepository(session)
    target = await _get_member_or_404(repo, user_id)
    if target.id == admin.id:
        raise ValidationError(message="You cannot deactivate your own account")
    await _guard_last_active_admin(repo, target, "deactivate")

    if target.is_active:
        await repo.deactivate_user(user_id)
        # Revoke outstanding JWTs and session keys so the account goes quiet
        # immediately rather than at token expiry.
        await repo.increment_token_version(user_id)
        session_repo = UserSessionRepository(session)
        await session_repo.deactivate_user_sessions(user_id)
        await session.commit()
        # The key cache snapshots user.is_active; drop it so the member's
        # API keys are rejected without waiting for the cache TTL.
        await _commit_member_operation(
            request,
            MemberOperation(
                actor=admin.username,
                action=ActionCategory.UPDATE,
                target_user=target.username,
                extra={"operation": "deactivate"},
                log_message="Admin deactivated a team member",
                log_fields={
                    "admin_username": admin.username,
                    "target_user_id": user_id,
                    "target_username": target.username,
                },
                invalidate_api_keys=True,
                invalidate_role_usernames=(target.username,),
            ),
        )
        await session.refresh(target)
    return await _member_with_spend(session, target)


@router.post("/members/{user_id}/reactivate", response_model=TeamMemberRead)
async def reactivate_member(
    user_id: int,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    """Reactivate a previously deactivated member. Idempotent.

    The member's API keys resume working (subject to their own is_active
    flags); their JWTs stay revoked (deactivation bumped token_version), so
    they must log in again.
    """
    repo = UserRepository(session)
    target = await _get_member_or_404(repo, user_id)
    if not target.is_active:
        await repo.set_active(user_id, True)
        await session.commit()
        await _commit_member_operation(
            request,
            MemberOperation(
                actor=admin.username,
                action=ActionCategory.UPDATE,
                target_user=target.username,
                extra={"operation": "reactivate"},
                log_message="Admin reactivated a team member",
                log_fields={
                    "admin_username": admin.username,
                    "target_user_id": user_id,
                    "target_username": target.username,
                },
                invalidate_api_keys=True,
                invalidate_role_usernames=(target.username,),
            ),
        )
        await session.refresh(target)
    return await _member_with_spend(session, target)


@router.put("/members/{user_id}/models", response_model=TeamMemberRead)
async def update_member_models(
    user_id: int,
    data: TeamMemberModelsUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    """Set a member's model allowlist (null = unrestricted, [] = deny all).

    The constraint applies immediately: every request made with the member's
    API keys is intersected with this list at authentication time, so keys
    created before a tightening are restricted as well.
    """
    repo = UserRepository(session)
    user = await _get_member_or_404(repo, user_id)
    await repo.set_allowed_models(user_id, data.allowed_models)
    await session.commit()
    await session.refresh(user)
    # The key cache snapshots user constraints; drop it so the new allowlist
    # takes effect without waiting for the cache TTL.
    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.UPDATE,
            target_user=user.username,
            extra={"operation": "models", "allowed_models": data.allowed_models},
            invalidate_api_keys=True,
        ),
    )
    return await _member_with_spend(session, user)


@router.put("/members/{user_id}/budget", response_model=TeamMemberRead)
async def update_member_budget(
    user_id: int,
    data: TeamMemberBudgetUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    """Set or clear a member's account-level budget (admin-only).

    The cap aggregates spend across all of the member's API keys; when the
    current-window spend reaches it, all of their keys are rejected (429).
    Key-level budgets stay purely self-service — this envelope is the
    admin-controlled guardrail. Explicit ``null`` for ``budget_usd`` clears
    the budget and its window configuration.
    """
    repo = UserRepository(session)
    target = await _get_member_or_404(repo, user_id)

    update_fields = data.model_dump(exclude_unset=True)
    clearing_cap = "budget_usd" in update_fields and update_fields["budget_usd"] is None
    # Clearing the cap implicitly clears the window config as well; setting
    # window fields in the same request is contradictory.
    if clearing_cap and ("budget_period" in update_fields or "budget_reset_day" in update_fields):
        raise ValidationError(
            message="budget_period and budget_reset_day cannot be set while clearing the budget"
        )
    new_usd = update_fields.get("budget_usd", target.budget_usd)
    new_period = update_fields.get("budget_period", target.budget_period)
    new_reset_day = update_fields.get("budget_reset_day", target.budget_reset_day)
    if not clearing_cap:
        # A window without any cap is meaningless; a reset day needs a monthly
        # window. Checked against the effective values (request wins over stored).
        validate_budget_envelope(new_usd, new_period, new_reset_day)
    # Clearing the cap clears the window configuration too; the repository
    # applies the same rule, so the stale stored fields are dropped here to
    # keep validation and storage consistent (see :func:`effective_budget_window`).
    new_usd, new_period, new_reset_day = effective_budget_window(new_usd, new_period, new_reset_day)

    updated = await repo.set_budget(
        user_id,
        budget_usd=new_usd,
        budget_period=new_period,
        budget_reset_day=new_reset_day,
    )
    assert updated is not None  # existence checked above
    await session.commit()
    # The key cache snapshots user budget config; drop it so the new envelope
    # takes effect without waiting for the cache TTL.
    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.UPDATE,
            target_user=updated.username,
            extra={
                "operation": "budget",
                "budget_usd": updated.budget_usd,
                "budget_period": updated.budget_period,
                "budget_reset_day": updated.budget_reset_day,
            },
            log_message="Admin set a member's account budget",
            log_fields={
                "admin_username": admin.username,
                "target_user_id": user_id,
                "target_username": updated.username,
                "budget_usd": updated.budget_usd,
                "budget_period": updated.budget_period,
            },
            invalidate_api_keys=True,
        ),
    )
    return await _member_with_spend(session, updated)


@router.post("/members/{user_id}/budget/reset", response_model=TeamMemberRead)
async def reset_member_budget(
    user_id: int,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    """Manually reset a member's current budget window (admin-only).

    Stamps ``budget_reset_at=now``; current-period spend then counts only
    usage at or after this point. Re-enables the member's keys when they were
    blocked for exceeding the account budget (until the new window's spend
    reaches the cap again).
    """
    repo = UserRepository(session)
    await _get_member_or_404(repo, user_id)
    updated = await repo.reset_budget(user_id)
    assert updated is not None  # existence checked above
    await session.commit()
    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.UPDATE,
            target_user=updated.username,
            extra={"operation": "budget_reset"},
            log_message="Admin reset a member's budget window",
            log_fields={
                "admin_username": admin.username,
                "target_user_id": user_id,
                "target_username": updated.username,
            },
            invalidate_api_keys=True,
        ),
    )
    return await _member_with_spend(session, updated)


@router.put("/members/{user_id}/rate-limit", response_model=TeamMemberRead)
async def update_member_rate_limit(
    user_id: int,
    data: TeamMemberRateLimitUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    """Set or clear a member's account-level rate limit (admin-only).

    The cap applies across every key the member owns, including the UI
    session key. Per-key limits are owner-scoped and admin-only, so they
    cannot target a member's key; this is the member-level quota control the
    Team view can actually apply.
    """
    repo = UserRepository(session)
    await _get_member_or_404(repo, user_id)
    updated = await repo.set_rate_limit(user_id, data.rate_limit_rpm)
    assert updated is not None  # existence checked above
    await session.commit()
    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.UPDATE,
            target_user=updated.username,
            extra={"operation": "rate_limit", "rate_limit_rpm": updated.rate_limit_rpm},
            log_message="Admin set a member's account rate limit",
            log_fields={
                "admin_username": admin.username,
                "target_user_id": user_id,
                "target_username": updated.username,
                "rate_limit_rpm": updated.rate_limit_rpm,
            },
            invalidate_api_keys=True,
        ),
    )
    return await _member_with_spend(session, updated)


@router.put("/members/{user_id}/username", response_model=TeamMemberUsernameUpdateResponse)
async def update_member_username(
    user_id: int,
    data: TeamMemberUsernameUpdate,
    request: Request,
    response: Response,
    session: AsyncSession = get_async_session_dep,
    # Same callable as the router-level dependency, so FastAPI reuses the
    # already-resolved admin record instead of querying twice.
    admin: UserRecord = Depends(require_admin_role),
):
    """Rename a team member.

    When an admin renames their own account, a fresh JWT is returned because
    the current token's `sub` claim references the old username and would stop
    resolving. Members' session API keys are keyed by user id and keep working.

    The member's token_version is bumped during a successful rename so any
    previously issued JWT stops working and cannot be revalidated later.
    """
    repo = UserRepository(session)
    target = await _get_member_or_404(repo, user_id)
    old_username = target.username
    try:
        updated = await repo.update_username(user_id, data.username)
    except ValueError as e:
        raise ConflictError(message=str(e)) from e
    assert updated is not None  # existence checked above
    await session.commit()

    is_self_rename = updated.id == admin.id
    access_token: str | None = None
    if is_self_rename:
        auth_config = await get_auth_config(request)
        access_token = JWTManager(auth_config).create_token(
            updated.username,
            role=updated.role,
            token_version=updated.token_version,
            session_id=get_request_identity(request).console_session_id,
        )
        set_console_cookie(response, request, access_token=access_token)

    # The logs router caches role/user_id keyed by username; drop both names
    # so a recycled username cannot inherit the previous owner's cached entry.
    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.UPDATE,
            target_user=updated.username,
            extra={"operation": "username", "old_username": old_username},
            log_message=(
                "Admin renamed themselves" if is_self_rename else "Admin renamed a team member"
            ),
            log_fields=(
                {"old_username": old_username, "new_username": updated.username}
                if is_self_rename
                else {
                    "admin_username": admin.username,
                    "target_user_id": user_id,
                    "old_username": old_username,
                    "new_username": updated.username,
                }
            ),
            invalidate_role_usernames=(old_username, updated.username),
        ),
    )
    return TeamMemberUsernameUpdateResponse(
        **(await _member_with_spend(session, updated)).model_dump(),
        access_token=access_token,
    )


@router.put("/members/{user_id}/email", response_model=TeamMemberRead)
async def set_member_email(
    user_id: int,
    data: TeamMemberEmailUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_recent_auth),
):
    """Set an admin-attested email when SMTP verification is unavailable.

    The invite flow and OAuth set a verified email automatically. This route
    exists for existing password accounts and for operators who deliberately
    attest the address themselves.
    """
    repo = UserRepository(session)
    await _get_member_or_404(repo, user_id)
    try:
        updated = await repo.set_email(user_id, validate_email(data.email), verified_by="admin")
    except ValueError as exc:
        raise ConflictError(message=str(exc)) from exc
    assert updated is not None  # existence checked above
    await session.commit()
    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.UPDATE,
            target_user=updated.username,
            extra={"email": updated.email, "verified_by": "admin"},
            log_message="Admin set a member email",
            log_fields={"admin_username": admin.username, "target_user_id": updated.id},
        ),
    )
    return TeamMemberRead.model_validate(updated)


@router.delete("/members/{user_id}/2fa", response_model=dict)
async def reset_member_2fa(
    user_id: int,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_recent_auth),
):
    """Revoke a member's TOTP secret and recovery codes (admin recovery)."""
    repo = UserRepository(session)
    user = await _get_member_or_404(repo, user_id)
    await TotpRepository(session).delete(user_id)
    await session.commit()
    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.DELETE,
            target_user=user.username,
            extra={"operation": "2fa_reset"},
            log_message="Admin reset a member's two-factor authentication",
            log_fields={"admin_username": admin.username, "target_user_id": user.id},
        ),
    )
    return {"success": True}


@router.delete("/members/{user_id}/passkeys", response_model=dict)
async def revoke_member_passkeys(
    user_id: int,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_recent_auth),
):
    """Revoke every passkey for a member, refusing to remove their last factor."""
    repo = UserRepository(session)
    user = await _get_member_or_404(repo, user_id)
    if not is_bcrypt_hash(user.password_hash):
        identities = await OAuthIdentityRepository(session).list_for_user(user_id)
        if not identities:
            raise ValidationError(
                message=(
                    "This member has no password or linked sign-in provider; "
                    "resetting passkeys would lock them out"
                ),
                code="last_factor",
            )
    removed = await PasskeyRepository(session).delete_all_for_user(user_id)
    await session.commit()
    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.DELETE,
            target_user=user.username,
            extra={"operation": "passkeys_reset", "removed": removed},
            log_message="Admin revoked a member's passkeys",
            log_fields={"admin_username": admin.username, "target_user_id": user.id},
        ),
    )
    return {"success": True, "removed": removed}


@router.put("/members/{user_id}/password", response_model=dict)
async def reset_member_password(
    user_id: int,
    data: TeamMemberPasswordReset,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    admin: UserRecord = Depends(require_admin_role),
):
    repo = UserRepository(session)
    target = await _get_member_or_404(repo, user_id)
    await repo.update_password(user_id, hash_password(data.password))
    # The admin chose the new password, so the member must replace it with
    # their own on next login.
    await repo.set_must_change_password(user_id, True)
    # Revoke the member's existing JWTs and session API keys: an admin password
    # reset is typically a response to a suspected compromise, so stale
    # credentials must stop working immediately.
    await repo.increment_token_version(user_id)
    session_repo = UserSessionRepository(session)
    await session_repo.deactivate_user_sessions(user_id)
    await session.commit()
    await _commit_member_operation(
        request,
        MemberOperation(
            actor=admin.username,
            action=ActionCategory.UPDATE,
            target_user=target.username,
            extra={"operation": "password_reset"},
            # The forced-change flag is read from the key-cache snapshot;
            # drop it so regular API keys are rejected immediately.
            invalidate_api_keys=True,
        ),
    )
    return {"message": f"Password reset for user {user_id}"}
