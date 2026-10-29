"""WebAuthn registration, login and credential management."""

import json
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.auth_session import (
    begin_mfa_challenge,
    issue_login_response,
    user_needs_second_factor,
)
from llm_proxy.api.dependencies import (
    get_async_session_dep,
    get_config_from_state,
    get_current_user,
    get_jwt_manager,
    require_recent_auth,
)
from llm_proxy.api.middleware.rate_limiting import get_rate_limiter
from llm_proxy.api.schemas.auth import LoginResponse, StepUpResponse
from llm_proxy.api.schemas.passkeys import (
    PasskeyLoginBeginRequest,
    PasskeyLoginBeginResponse,
    PasskeyLoginFinishRequest,
    PasskeyRead,
    PasskeyRegisterBeginResponse,
    PasskeyRegisterFinishRequest,
    PasskeyRenameRequest,
)
from llm_proxy.core.exceptions import AuthenticationFailedError, NotFoundError, ValidationError
from llm_proxy.core.request_facts import get_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import (
    OAuthIdentityRepository,
    PasskeyRepository,
    UserRecord,
    UserRepository,
    UserSessionRepository,
)
from llm_proxy.observability.log_intake import record_auth_event
from llm_proxy.observability.types import Outcome
from llm_proxy.security.base64url import b64url_decode, b64url_encode
from llm_proxy.security.console_url import public_origin, rp_id
from llm_proxy.security.passwords import is_bcrypt_hash

router = APIRouter(prefix="/api/me/passkeys", tags=["Profile"])
public_router = APIRouter(prefix="/api/auth/passkeys", tags=["authentication"])
limiter = get_rate_limiter()

PASSKEY_REGISTER_PURPOSE = "passkey_register"
PASSKEY_LOGIN_PURPOSE = "passkey_login"
PASSKEY_STEP_UP_PURPOSE = "passkey_step_up"
PASSKEY_CHALLENGE_TTL_SECONDS = 300


def _webauthn():
    """Import py_webauthn only when a WebAuthn endpoint is called."""
    import webauthn
    from webauthn.helpers import options_to_json
    from webauthn.helpers.structs import (
        AuthenticatorSelectionCriteria,
        PublicKeyCredentialDescriptor,
        ResidentKeyRequirement,
        UserVerificationRequirement,
    )

    return (
        webauthn,
        options_to_json,
        AuthenticatorSelectionCriteria,
        PublicKeyCredentialDescriptor,
        ResidentKeyRequirement,
        UserVerificationRequirement,
    )


def _webauthn_module():
    """The py_webauthn module alone, for the two verify-response calls."""
    return _webauthn()[0]


def _options_dict(options: Any, options_to_json: Any) -> dict[str, Any]:
    return json.loads(options_to_json(options))


async def _passkey_context(request: Request) -> tuple[str, str]:
    console_auth = (await get_config_from_state(request)).server_params.console_auth
    origin = public_origin(console_auth)
    relying_party_id = rp_id(console_auth)
    if origin is None or relying_party_id is None:
        raise ValidationError(
            message=(
                "Passkeys require an HTTPS public_base_url (or http://localhost for development)"
            ),
            code="passkey_unavailable",
        )
    return origin, relying_party_id


def _challenge_claims(token: dict[str, Any]) -> bytes:
    challenge = token.get("challenge")
    if not isinstance(challenge, str):
        raise AuthenticationFailedError(message="Invalid passkey challenge")
    return b64url_decode(challenge)


def _passkey_read(record) -> PasskeyRead:
    return PasskeyRead(
        id=record.id,
        name=record.name,
        transports=record.transports,
        created_at=record.created_at,
        last_used_at=record.last_used_at,
    )


#: Longest transport hint the WebAuthn spec defines; anything longer is noise.
_MAX_TRANSPORT_LEN = 32


def _credential_transports(credential: dict[str, Any]) -> list[str] | None:
    """Coerce the client-supplied ``transports`` into the stored shape.

    ``credential`` is an opaque dict, so pydantic never validates this field.
    Persisting something that is not a list of strings makes every later
    ``GET /api/me/passkeys`` fail response-model validation, which would take
    the whole listing (and this account's passkey management) down with it.
    """
    raw = credential.get("transports")
    if not isinstance(raw, list):
        return None
    transports = [
        item for item in raw if isinstance(item, str) and 0 < len(item) <= _MAX_TRANSPORT_LEN
    ]
    return transports or None


@router.get("", response_model=list[PasskeyRead])
async def list_passkeys(
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
):
    records = await PasskeyRepository(session).list_for_user(user.id)
    return [_passkey_read(record) for record in records]


@router.post("/register/begin", response_model=PasskeyRegisterBeginResponse)
@limiter.limit("auth.passkey")
async def begin_registration(
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    _, relying_party_id = await _passkey_context(request)
    (
        webauthn,
        options_to_json,
        AuthenticatorSelectionCriteria,
        PublicKeyCredentialDescriptor,
        ResidentKeyRequirement,
        UserVerificationRequirement,
    ) = _webauthn()
    existing = await PasskeyRepository(session).list_for_user(user.id)
    exclude = [
        PublicKeyCredentialDescriptor(
            id=PasskeyRepository.decode_credential_id(record),
            transports=None,
        )
        for record in existing
    ]
    options = webauthn.generate_registration_options(
        rp_id=relying_party_id,
        rp_name="LLM Proxy",
        user_id=str(user.id).encode("utf-8"),
        user_name=user.username,
        user_display_name=user.username,
        exclude_credentials=exclude or None,
        authenticator_selection=AuthenticatorSelectionCriteria(
            user_verification=UserVerificationRequirement.REQUIRED,
            resident_key=ResidentKeyRequirement.REQUIRED,
        ),
    )
    auth_manager = await get_jwt_manager(request)
    token = auth_manager.create_scoped_token(
        PASSKEY_REGISTER_PURPOSE,
        {"uid": user.id, "challenge": b64url_encode(options.challenge)},
        ttl_seconds=PASSKEY_CHALLENGE_TTL_SECONDS,
    )
    return PasskeyRegisterBeginResponse(
        options=_options_dict(options, options_to_json),
        challenge_token=token,
    )


@router.post("/register/finish", response_model=PasskeyRead)
@limiter.limit("auth.passkey")
async def finish_registration(
    data: PasskeyRegisterFinishRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    origin, relying_party_id = await _passkey_context(request)
    auth_manager = await get_jwt_manager(request)
    try:
        claims = auth_manager.verify_scoped_token(data.challenge_token, PASSKEY_REGISTER_PURPOSE)
    except Exception as exc:
        raise AuthenticationFailedError(
            message="The passkey challenge expired. Please try again.",
            code="passkey_challenge_expired",
        ) from exc
    if int(claims.get("uid", -1)) != user.id:
        raise AuthenticationFailedError(message="Invalid passkey challenge")

    webauthn = _webauthn_module()
    try:
        verified = webauthn.verify_registration_response(
            credential=data.credential,
            expected_challenge=_challenge_claims(claims),
            expected_rp_id=relying_party_id,
            expected_origin=origin,
            require_user_verification=True,
        )
    except Exception as exc:
        raise ValidationError(
            message="Passkey registration could not be verified",
            code="passkey_verification_failed",
        ) from exc

    aaguid = getattr(verified, "aaguid", None)
    record = await PasskeyRepository(session).create(
        user_id=user.id,
        credential_id=verified.credential_id,
        public_key=verified.credential_public_key,
        sign_count=verified.sign_count,
        transports=_credential_transports(data.credential),
        aaguid=aaguid.hex() if isinstance(aaguid, bytes) else (str(aaguid) if aaguid else None),
        name=data.name,
    )
    await session.commit()
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="passkey",
        metadata_extra={"operation": "passkey_register", "passkey_id": record.id},
    )
    return _passkey_read(record)


@router.patch("/{credential_row_id}", response_model=PasskeyRead)
@limiter.limit("auth.passkey")
async def rename_passkey(
    credential_row_id: int,
    data: PasskeyRenameRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    repo = PasskeyRepository(session)
    if not await repo.rename(user.id, credential_row_id, data.name):
        raise NotFoundError(message="Passkey not found")
    await session.commit()
    record = await repo.get_by_id(credential_row_id)
    assert record is not None  # rename() just confirmed it exists
    return _passkey_read(record)


@router.delete("/{credential_row_id}", response_model=dict)
async def delete_passkey(
    credential_row_id: int,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    repo = PasskeyRepository(session)
    record = await repo.get_by_id(credential_row_id)
    if record is None or record.user_id != user.id:
        raise NotFoundError(message="Passkey not found")
    # Lock the account row so the "is this the last factor?" check and the
    # delete below cannot interleave with a concurrent removal of the other
    # passkey; without it both requests observe two credentials and both win.
    locked = await UserRepository(session).get_for_update(user.id)
    if locked is None or not locked.is_active:
        raise AuthenticationFailedError(message="Account is disabled")
    if await repo.count_for_user(user.id) <= 1 and not is_bcrypt_hash(locked.password_hash):
        # OAuth is another usable primary factor, so a passkey-only account
        # with a linked provider may still remove its last passkey.
        identities = await OAuthIdentityRepository(session).list_for_user(user.id)
        if not identities:
            raise ValidationError(
                message=("Set a password or link a sign-in provider before removing it"),
                code="last_factor",
            )
    await repo.delete(user.id, credential_row_id)
    await session.commit()
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="passkey",
        metadata_extra={"operation": "passkey_delete", "passkey_id": credential_row_id},
    )
    return {"success": True}


@public_router.post("/login/begin", response_model=PasskeyLoginBeginResponse)
@limiter.limit("auth.passkey")
async def begin_login(
    data: PasskeyLoginBeginRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
):
    _, relying_party_id = await _passkey_context(request)
    webauthn, options_to_json, _, PublicKeyCredentialDescriptor, _, UserVerificationRequirement = (
        _webauthn()
    )
    user = None
    records = []
    if data.username:
        user = await UserRepository(session).get_by_login(data.username)
        if user is not None and user.is_active:
            records = await PasskeyRepository(session).list_for_user(user.id)
    allow_credentials = [
        PublicKeyCredentialDescriptor(
            id=PasskeyRepository.decode_credential_id(record),
            transports=None,
        )
        for record in records
    ]
    options = webauthn.generate_authentication_options(
        rp_id=relying_party_id,
        allow_credentials=allow_credentials or None,
        user_verification=UserVerificationRequirement.REQUIRED,
    )
    auth_manager = await get_jwt_manager(request)
    token = auth_manager.create_scoped_token(
        PASSKEY_LOGIN_PURPOSE,
        {
            "uid": user.id if user is not None else None,
            "challenge": b64url_encode(options.challenge),
        },
        ttl_seconds=PASSKEY_CHALLENGE_TTL_SECONDS,
    )
    return PasskeyLoginBeginResponse(
        options=_options_dict(options, options_to_json),
        challenge_token=token,
    )


async def _verify_login_credential(
    request: Request,
    session: AsyncSession,
    *,
    challenge_token: str,
    credential: dict[str, Any],
    purpose: str,
) -> tuple[UserRecord, Any, dict[str, Any]]:
    """Verify a WebAuthn assertion and return the user, the credential and the claims.

    Every rejection is audited: these endpoints are public, so a success-only
    trail leaves credential probing against the passkey flow invisible.
    """
    try:
        return await _verify_login_credential_impl(
            request,
            session,
            challenge_token=challenge_token,
            credential=credential,
            purpose=purpose,
        )
    except AuthenticationFailedError as exc:
        client_ip = get_client_ip(request)
        record_auth_event(
            request,
            # No account is proven on a failed public login, so the client IP
            # stands in as the actor (the same convention as record_failed_auth).
            username=client_ip,
            client_ip=client_ip,
            outcome=Outcome.FAILURE,
            auth_method="passkey",
            metadata_extra={
                "operation": purpose,
                "reason": exc.code,
                "credential_id": str(credential.get("id") or credential.get("rawId") or "")[:128],
            },
        )
        raise


async def _verify_login_credential_impl(
    request: Request,
    session: AsyncSession,
    *,
    challenge_token: str,
    credential: dict[str, Any],
    purpose: str,
) -> tuple[UserRecord, Any, dict[str, Any]]:
    origin, relying_party_id = await _passkey_context(request)
    auth_manager = await get_jwt_manager(request)
    try:
        claims = auth_manager.verify_scoped_token(challenge_token, purpose)
    except Exception as exc:
        raise AuthenticationFailedError(
            message="The passkey challenge expired. Please try again.",
            code="passkey_challenge_expired",
        ) from exc

    raw_id = credential.get("id") or credential.get("rawId")
    if not isinstance(raw_id, str):
        raise AuthenticationFailedError(message="Invalid passkey response")
    normalized_id = raw_id.rstrip("=")
    repo = PasskeyRepository(session)
    record = await repo.get_by_credential_id(normalized_id)
    if record is None:
        raise AuthenticationFailedError(
            message="Passkey not recognized. Please sign in another way.",
            code="passkey_not_recognized",
        )
    user = await UserRepository(session).get_by_id(record.user_id)
    if user is None or not user.is_active:
        raise AuthenticationFailedError(message="Account is disabled")

    webauthn = _webauthn_module()
    try:
        verified = webauthn.verify_authentication_response(
            credential=credential,
            expected_challenge=_challenge_claims(claims),
            expected_rp_id=relying_party_id,
            expected_origin=origin,
            credential_public_key=PasskeyRepository.decode_public_key(record),
            credential_current_sign_count=record.sign_count,
            require_user_verification=True,
        )
    except Exception as exc:
        raise AuthenticationFailedError(
            message="Passkey could not be verified",
            code="passkey_verification_failed",
        ) from exc

    await repo.record_use(record, getattr(verified, "new_sign_count", record.sign_count))
    return user, verified, claims


@public_router.post("/login/finish", response_model=LoginResponse)
@limiter.limit("auth.passkey")
async def finish_login(
    data: PasskeyLoginFinishRequest,
    request: Request,
    response: Response,
    session: AsyncSession = get_async_session_dep,
):
    user, _, _ = await _verify_login_credential(
        request,
        session,
        challenge_token=data.challenge_token,
        credential=data.credential,
        purpose=PASSKEY_LOGIN_PURPOSE,
    )
    if await user_needs_second_factor(session, user):
        return await begin_mfa_challenge(request, user, auth_method="passkey")
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="passkey",
        metadata_extra={"operation": "passkey_login"},
    )
    return await issue_login_response(request, response, session, user, auth_method="passkey")


@router.post("/step-up/begin", response_model=PasskeyLoginBeginResponse)
@limiter.limit("auth.passkey")
async def begin_step_up(
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
):
    _, relying_party_id = await _passkey_context(request)
    webauthn, options_to_json, _, PublicKeyCredentialDescriptor, _, UserVerificationRequirement = (
        _webauthn()
    )
    records = await PasskeyRepository(session).list_for_user(user.id)
    if not records:
        raise ValidationError(
            message="No passkey is registered for this account", code="no_passkey"
        )
    options = webauthn.generate_authentication_options(
        rp_id=relying_party_id,
        allow_credentials=[
            PublicKeyCredentialDescriptor(
                id=PasskeyRepository.decode_credential_id(record),
                transports=None,
            )
            for record in records
        ],
        user_verification=UserVerificationRequirement.REQUIRED,
    )
    auth_manager = await get_jwt_manager(request)
    identity = get_request_identity(request)
    token = auth_manager.create_scoped_token(
        PASSKEY_STEP_UP_PURPOSE,
        {
            "uid": user.id,
            "sid": identity.console_session_id,
            "challenge": b64url_encode(options.challenge),
        },
        ttl_seconds=PASSKEY_CHALLENGE_TTL_SECONDS,
    )
    return PasskeyLoginBeginResponse(
        options=_options_dict(options, options_to_json),
        challenge_token=token,
    )


@router.post("/step-up/finish", response_model=StepUpResponse)
@limiter.limit("auth.passkey")
async def finish_step_up(
    data: PasskeyLoginFinishRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
):
    verified_user, _, claims = await _verify_login_credential(
        request,
        session,
        challenge_token=data.challenge_token,
        credential=data.credential,
        purpose=PASSKEY_STEP_UP_PURPOSE,
    )
    if verified_user.id != user.id:
        raise AuthenticationFailedError(message="Invalid passkey challenge")
    identity = get_request_identity(request)
    if not identity.console_session_id:
        raise AuthenticationFailedError(message="Session not found")
    # The challenge token was minted for one console session; refusing it in
    # any other one keeps a captured challenge+assertion pair from being
    # replayed against a different login of the same account.
    if claims.get("sid") != identity.console_session_id:
        raise AuthenticationFailedError(message="Invalid passkey challenge")
    repo = UserSessionRepository(session)
    if await repo.touch_verified(identity.console_session_id) is None:
        raise AuthenticationFailedError(message="Session has been revoked")
    await session.commit()
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="passkey",
        metadata_extra={"operation": "passkey_step_up"},
    )
    return StepUpResponse(verified=True)
