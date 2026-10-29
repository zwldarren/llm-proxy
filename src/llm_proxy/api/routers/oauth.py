"""Generic OpenID Connect login and account linking."""

import hashlib
import re
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx2
import jwt
from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.auth_session import begin_mfa_challenge, issue_login_response
from llm_proxy.api.dependencies import (
    get_async_session_dep,
    get_config_from_state,
    get_current_user,
    get_jwt_manager,
    require_recent_auth,
)
from llm_proxy.api.middleware.rate_limiting import get_rate_limiter
from llm_proxy.api.schemas.auth import LoginResponse
from llm_proxy.api.schemas.oauth import (
    OAuthCompleteRequest,
    OAuthIdentityListResponse,
    OAuthIdentityRead,
    OAuthLinkCompleteRequest,
    OAuthProviderListResponse,
    OAuthProviderRead,
)
from llm_proxy.config.types.console_auth import OidcProviderSettings
from llm_proxy.core.exceptions import (
    AuthenticationFailedError,
    ConflictError,
    NotFoundError,
    ValidationError,
)
from llm_proxy.core.request_facts import get_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import (
    InvitationRepository,
    OAuthIdentityRepository,
    PasskeyRepository,
    TotpRepository,
    UserRecord,
    UserRepository,
)
from llm_proxy.observability.log_intake import record_auth_event
from llm_proxy.observability.types import Outcome
from llm_proxy.security.base64url import b64url_encode
from llm_proxy.security.console_url import is_secure_issuer, public_origin
from llm_proxy.security.passwords import hash_password, is_bcrypt_hash
from llm_proxy.security.tokens import hash_token

public_router = APIRouter(prefix="/api/auth/oauth", tags=["authentication"])
router = APIRouter(prefix="/api/me/oauth", tags=["Profile"])
limiter = get_rate_limiter()

OAUTH_STATE_PURPOSE = "oauth_state"
OAUTH_COMPLETE_PURPOSE = "oauth_complete"
#: Bridges the (unauthenticated) provider callback back to an authenticated
#: browser session, so that binding an identity needs more than the state.
OAUTH_LINK_PURPOSE = "oauth_link"
OAUTH_STATE_TTL_SECONDS = 600
OAUTH_COMPLETE_TTL_SECONDS = 600

#: The PKCE verifier never leaves the browser. It is handed to the callback in
#: an HttpOnly cookie instead of the ``state`` parameter: the state travels
#: through the identity provider, browser history and IdP-side logs, so a
#: verifier inside it protects nothing — whoever can intercept the
#: authorization code can also read the verifier meant to protect it.
OAUTH_PKCE_COOKIE = "llm_proxy_oidc_pkce"
OAUTH_PKCE_COOKIE_PATH = "/api/auth/oauth"

#: ID-token signature algorithms accepted from configured providers.
_ID_TOKEN_ALGORITHMS = frozenset({"RS256", "ES256", "PS256", "EdDSA"})


async def _console_auth(request: Request):
    return (await get_config_from_state(request)).server_params.console_auth


def _provider(console_auth, provider_id: str) -> OidcProviderSettings:
    for provider in console_auth.oidc_providers:
        if provider.id == provider_id and provider.enabled:
            return provider
    raise NotFoundError(message=f"OIDC provider '{provider_id}' is not configured")


async def _discover(provider: OidcProviderSettings) -> dict[str, Any]:
    """Fetch and sanity-check the provider's discovery document.

    The scheme is re-checked here as well as at save time: a settings row can
    predate that validation, and an ``http://`` issuer would send the client
    secret and the authorization code over plaintext.
    """
    if not is_secure_issuer(provider.issuer):
        raise ValidationError(
            message=(
                f"OIDC provider '{provider.id}' issuer must be an HTTPS URL "
                "(or http://localhost for development)"
            ),
            code="oidc_insecure_issuer",
        )
    url = f"{provider.issuer}/.well-known/openid-configuration"
    try:
        async with httpx2.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            document = response.json()
    except Exception as exc:
        raise ValidationError(
            message=f"Could not load OIDC discovery document for '{provider.id}'",
            code="oidc_discovery_failed",
        ) from exc
    if not isinstance(document, dict):
        raise ValidationError(message="OIDC discovery document is not a JSON object")
    discovered_issuer = document.get("issuer")
    if isinstance(discovered_issuer, str) and discovered_issuer.rstrip("/") != provider.issuer:
        raise ValidationError(
            message=(
                f"OIDC discovery issuer mismatch for '{provider.id}': "
                f"expected {provider.issuer}, got {discovered_issuer}"
            ),
            code="oidc_issuer_mismatch",
        )
    for field in ("authorization_endpoint", "token_endpoint", "userinfo_endpoint"):
        if not isinstance(document.get(field), str):
            raise ValidationError(message=f"OIDC provider is missing {field}")
    return document


def _redirect_error(origin: str, message: str) -> RedirectResponse:
    return RedirectResponse(
        f"{origin}/login?oauth_error={b64url_encode(message.encode('utf-8'))}",
        status_code=303,
    )


def _split_pkce_cookie(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    state_id, separator, verifier = value.partition(":")
    if not separator or not state_id or not verifier:
        return None
    return state_id, verifier


def _set_pkce_cookie(response: Response, state_id: str, verifier: str, *, secure: bool) -> None:
    response.set_cookie(
        OAUTH_PKCE_COOKIE,
        f"{state_id}:{verifier}",
        max_age=OAUTH_STATE_TTL_SECONDS,
        path=OAUTH_PKCE_COOKIE_PATH,
        httponly=True,
        secure=secure,
        samesite="lax",
    )


def _clear_pkce_cookie(response: Response) -> None:
    response.delete_cookie(OAUTH_PKCE_COOKIE, path=OAUTH_PKCE_COOKIE_PATH)


@dataclass(slots=True)
class _Authorization:
    """An authorization redirect plus the PKCE state to keep in the browser."""

    url: str
    state_id: str
    verifier: str
    secure_cookie: bool


def _username_base(email: str) -> str:
    local = email.split("@", 1)[0].lower()
    base = re.sub(r"[^a-z0-9_-]", "-", local).strip("-")
    return (base or "user")[:48]


async def _create_oauth_user(session: AsyncSession, *, email: str) -> UserRecord:
    users = UserRepository(session)
    base = _username_base(email)
    last_error: Exception | None = None
    for suffix in range(0, 100):
        username = base if suffix == 0 else f"{base}-{suffix}"
        try:
            return await users.create_user(
                username,
                "",
                role="viewer",
                email=email,
                email_verified_by="oidc",
            )
        except ValueError as exc:
            last_error = exc
    raise ConflictError(
        message="Could not allocate a username for the OAuth account"
    ) from last_error


@public_router.get("/providers", response_model=OAuthProviderListResponse)
async def list_providers(request: Request):
    console_auth = await _console_auth(request)
    if public_origin(console_auth) is None:
        # OIDC needs an exact redirect URI and a secure origin. Without the
        # public URL, advertising buttons that can only fail is worse than
        # showing none.
        return OAuthProviderListResponse(providers=[])
    return OAuthProviderListResponse(
        providers=[
            OAuthProviderRead(id=provider.id, name=provider.name)
            for provider in console_auth.oidc_providers
            if provider.enabled
        ]
    )


async def _begin_authorization(
    request: Request,
    session: AsyncSession,
    *,
    provider_id: str,
    invitation_token: str | None,
    link_user_id: int | None,
    link_session_id: str | None = None,
) -> _Authorization:
    console_auth = await _console_auth(request)
    provider = _provider(console_auth, provider_id)
    origin = public_origin(console_auth)
    if origin is None:
        raise ValidationError(
            message="OIDC login requires an HTTPS public_base_url (or http://localhost)",
            code="oauth_unavailable",
        )
    discovery = await _discover(provider)
    redirect_uri = f"{origin}/api/auth/oauth/{provider.id}/callback"

    state_id = secrets.token_urlsafe(24)
    claims: dict[str, Any] = {
        "provider": provider.id,
        "nonce": secrets.token_urlsafe(24),
        "redirect_uri": redirect_uri,
        "state_id": state_id,
    }
    if invitation_token:
        invitation = await InvitationRepository(session).get_by_token_hash(
            hash_token(invitation_token)
        )
        if invitation is None or InvitationRepository.status(invitation) != "pending":
            raise ValidationError(
                message="This invitation is no longer valid", code="invitation_invalid"
            )
        claims["invitation_id"] = invitation.id
        claims["invitation_email"] = invitation.email
        # Re-sending an invitation rotates its token. Pin the one that started
        # this flow so the callback can refuse a link the admin has replaced.
        claims["invitation_token_hash"] = invitation.token_hash
    if link_user_id is not None:
        claims["link_user_id"] = link_user_id
        if link_session_id:
            claims["link_session_id"] = link_session_id

    verifier = secrets.token_urlsafe(64)
    challenge = b64url_encode(hashlib.sha256(verifier.encode("ascii")).digest())
    state = (await get_jwt_manager(request)).create_scoped_token(
        OAUTH_STATE_PURPOSE, claims, ttl_seconds=OAUTH_STATE_TTL_SECONDS
    )
    params = {
        "response_type": "code",
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri,
        "scope": provider.scopes,
        "state": state,
        "nonce": claims["nonce"],
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return _Authorization(
        url=f"{discovery['authorization_endpoint']}?{urlencode(params)}",
        state_id=state_id,
        verifier=verifier,
        secure_cookie=origin.startswith("https"),
    )


@public_router.get("/{provider_id}/begin")
@limiter.limit("auth.oauth")
async def begin_login(
    provider_id: str,
    request: Request,
    invite: str | None = Query(default=None),
    session: AsyncSession = get_async_session_dep,
):
    try:
        authorization = await _begin_authorization(
            request,
            session,
            provider_id=provider_id,
            invitation_token=invite,
            link_user_id=None,
        )
    except (NotFoundError, ValidationError) as exc:
        console_auth = await _console_auth(request)
        origin = public_origin(console_auth)
        if origin is None:
            raise
        return _redirect_error(origin, exc.message)
    response = RedirectResponse(authorization.url, status_code=303)
    _set_pkce_cookie(
        response,
        authorization.state_id,
        authorization.verifier,
        secure=authorization.secure_cookie,
    )
    return response


@router.get("/{provider_id}/begin")
@limiter.limit("auth.oauth")
async def begin_link(
    provider_id: str,
    request: Request,
    response: Response,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    authorization = await _begin_authorization(
        request,
        session,
        provider_id=provider_id,
        invitation_token=None,
        link_user_id=user.id,
        link_session_id=get_request_identity(request).console_session_id,
    )
    _set_pkce_cookie(
        response,
        authorization.state_id,
        authorization.verifier,
        secure=authorization.secure_cookie,
    )
    return {"authorization_url": authorization.url}


async def _exchange_code(
    provider: OidcProviderSettings,
    discovery: dict[str, Any],
    *,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": provider.client_id,
        "code_verifier": code_verifier,
    }
    if provider.client_secret:
        data["client_secret"] = provider.client_secret
    try:
        async with httpx2.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.post(discovery["token_endpoint"], data=data)
            response.raise_for_status()
            tokens = response.json()
    except Exception as exc:
        raise AuthenticationFailedError(
            message="OIDC token exchange failed", code="oidc_token_exchange_failed"
        ) from exc
    if not isinstance(tokens, dict) or not isinstance(tokens.get("access_token"), str):
        raise AuthenticationFailedError(message="OIDC provider returned no access token")
    return tokens


async def _fetch_userinfo(
    discovery: dict[str, Any],
    access_token: str,
) -> dict[str, Any]:
    try:
        async with httpx2.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(
                discovery["userinfo_endpoint"],
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()
            claims = response.json()
    except Exception as exc:
        raise AuthenticationFailedError(
            message="OIDC userinfo request failed", code="oidc_userinfo_failed"
        ) from exc
    if not isinstance(claims, dict):
        raise AuthenticationFailedError(message="OIDC userinfo is not a JSON object")
    return claims


async def _fetch_json(url: str) -> dict[str, Any]:
    async with httpx2.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        document = response.json()
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return document


def _signing_key(jwks: dict[str, Any], kid: str | None) -> Any:
    keys = jwks.get("keys")
    if not isinstance(keys, list):
        raise ValueError("JWKS document has no keys")
    for entry in keys:
        if isinstance(entry, dict) and (kid is None or entry.get("kid") == kid):
            return jwt.PyJWK(entry).key
    raise ValueError(f"No JWK matches kid={kid!r}")


async def _verify_id_token(
    provider: OidcProviderSettings,
    discovery: dict[str, Any],
    tokens: dict[str, Any],
    *,
    nonce: str,
) -> dict[str, Any] | None:
    """Validate the OIDC ID token when the provider issues one.

    The token exchange is a direct, PKCE-bound call, but the ID token is the
    provider's signed statement about the user. Verify its signature against
    the published JWKS and its issuer/audience/nonce before trusting any claim.
    Providers that issue no ID token fall back to the UserInfo endpoint.
    """
    raw = tokens.get("id_token")
    if raw is None:
        return None
    message = "The identity provider returned an invalid ID token"
    if not isinstance(raw, str):
        raise AuthenticationFailedError(message=message, code="oidc_id_token_invalid")
    jwks_uri = discovery.get("jwks_uri")
    if not isinstance(jwks_uri, str):
        raise AuthenticationFailedError(message=message, code="oidc_id_token_invalid")
    try:
        header = jwt.get_unverified_header(raw)
        algorithm = header.get("alg")
        if algorithm not in _ID_TOKEN_ALGORITHMS:
            raise ValueError(f"Unsupported ID token algorithm: {algorithm}")
        jwks = await _fetch_json(jwks_uri)
        claims = jwt.decode(
            raw,
            key=_signing_key(jwks, header.get("kid")),
            algorithms=[algorithm],
            audience=provider.client_id,
            issuer=provider.issuer,
            options={"require": ["exp", "iat", "aud", "iss", "sub"]},
        )
    except Exception as exc:
        raise AuthenticationFailedError(message=message, code="oidc_id_token_invalid") from exc
    presented_nonce = claims.get("nonce")
    if (
        not nonce
        or not isinstance(presented_nonce, str)
        or not secrets.compare_digest(presented_nonce, nonce)
    ):
        raise AuthenticationFailedError(message=message, code="oidc_id_token_invalid")
    return claims


def _invitation_token_still_current(claims: dict[str, Any], invitation) -> bool:
    """Whether the invitation link that started this flow is still the current one.

    Re-sending an invitation rotates ``token_hash``; comparing it against the
    hash pinned in the state claims is what makes the old link stop working
    for an in-flight OAuth flow too.
    """
    pinned = claims.get("invitation_token_hash")
    if not isinstance(pinned, str):
        return False
    return secrets.compare_digest(pinned, str(invitation.token_hash))


def _verified_email(claims: dict[str, Any]) -> str:
    email = claims.get("email")
    verified = claims.get("email_verified")
    if not isinstance(email, str) or not email:
        raise AuthenticationFailedError(
            message="The identity provider did not return an email address",
            code="oidc_email_missing",
        )
    if verified is not True:
        raise AuthenticationFailedError(
            message="The identity provider did not verify the email address",
            code="oidc_email_unverified",
        )
    return email.strip().lower()


@public_router.get("/{provider_id}/callback")
@limiter.limit("auth.oauth")
async def callback(
    provider_id: str,
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    session: AsyncSession = get_async_session_dep,
):
    console_auth = await _console_auth(request)
    origin = public_origin(console_auth)
    if origin is None:
        raise ValidationError(
            message="OIDC requires a configured public_base_url", code="oauth_unavailable"
        )

    def fail(message: str, reason: str, response: Response | None = None) -> Response:
        """Audit a rejected callback and send the browser back to the login page.

        The flow is unauthenticated, so the trail is the only signal that an
        account-linking or login attempt was refused (and why).
        """
        record_auth_event(
            request,
            # Nothing about the account is proven yet, so the client IP stands
            # in as the actor (the same convention as record_failed_auth).
            username=get_client_ip(request),
            client_ip=get_client_ip(request),
            outcome=Outcome.FAILURE,
            auth_method="oidc",
            metadata_extra={
                "operation": "oidc_callback",
                "provider": provider_id,
                "reason": reason,
            },
        )
        redirect = _redirect_error(origin, message)
        _clear_pkce_cookie(redirect)
        return redirect

    if not code or not state:
        return fail("The identity provider did not return a code", "missing_code_or_state")

    provider = _provider(console_auth, provider_id)
    auth_manager = await get_jwt_manager(request)
    try:
        claims = auth_manager.verify_scoped_token(state, OAUTH_STATE_PURPOSE)
    except Exception:
        return fail(
            "The OIDC login state expired. Please try again.",
            "state_invalid_or_expired",
        )
    if claims.get("provider") != provider.id:
        return fail("OIDC provider mismatch", "provider_mismatch")

    # The verifier comes from the HttpOnly cookie set when this flow started,
    # never from the state that travelled through the provider.
    pkce = _split_pkce_cookie(request.cookies.get(OAUTH_PKCE_COOKIE))
    if pkce is None or pkce[0] != claims.get("state_id"):
        return fail(
            "The sign-in could not be completed in this browser. Please try again.",
            "pkce_verifier_missing",
        )
    _, verifier = pkce

    discovery = await _discover(provider)
    try:
        tokens = await _exchange_code(
            provider,
            discovery,
            code=code,
            redirect_uri=str(claims["redirect_uri"]),
            code_verifier=verifier,
        )
    except AuthenticationFailedError as exc:
        return fail("The identity provider rejected the sign-in.", exc.code)
    try:
        userinfo = await _fetch_userinfo(discovery, tokens["access_token"])
        expected_nonce = claims.get("nonce")
        id_claims = await _verify_id_token(
            provider,
            discovery,
            tokens,
            nonce=expected_nonce if isinstance(expected_nonce, str) else "",
        )
        if id_claims is not None:
            userinfo = {
                **userinfo,
                **{
                    key: id_claims[key]
                    for key in ("sub", "email", "email_verified")
                    if key in id_claims
                },
            }
        email = _verified_email(userinfo)
    except AuthenticationFailedError as exc:
        return fail(exc.message, exc.code)
    subject = userinfo.get("sub")
    if not isinstance(subject, str) or not subject:
        return fail("The identity provider returned no subject identifier", "subject_missing")

    identities = OAuthIdentityRepository(session)
    users = UserRepository(session)
    identity = await identities.get(provider.id, subject)
    link_user_id = claims.get("link_user_id")
    invitation_id = claims.get("invitation_id")
    created_account = False
    if link_user_id is not None:
        # Linking is deliberately NOT completed here: this endpoint is
        # unauthenticated, so the signed state would be the only proof of who
        # asked to link. Hand the verified identity to the browser instead and
        # let an authenticated, step-up-protected call finish the binding.
        if identity is not None:
            if identity.user_id != int(link_user_id):
                return fail(
                    "This provider is already linked to another account",
                    "identity_linked_elsewhere",
                )
            redirect = RedirectResponse(f"{origin}/account/security?linked=1", status_code=303)
            _clear_pkce_cookie(redirect)
            return redirect
        user = await users.get_by_id(int(link_user_id))
        if user is None or not user.is_active:
            return fail("Your account is no longer available", "link_user_inactive")
        if user.email and user.email != email:
            return fail(
                "The provider email does not match your account email",
                "link_email_mismatch",
            )
        link_token = auth_manager.create_scoped_token(
            OAUTH_LINK_PURPOSE,
            {
                "uid": user.id,
                "provider": provider.id,
                "subject": subject,
                "email": email,
                "sid": claims.get("link_session_id"),
            },
            ttl_seconds=OAUTH_COMPLETE_TTL_SECONDS,
        )
        redirect = RedirectResponse(
            f"{origin}/oauth/complete?{urlencode({'link_token': link_token})}",
            status_code=303,
        )
        _clear_pkce_cookie(redirect)
        return redirect

    if identity is not None:
        user = await users.get_by_id(identity.user_id)
        if user is None or not user.is_active:
            return fail("This account is disabled", "account_disabled")
        if isinstance(invitation_id, str):
            invitation = await InvitationRepository(session).get_by_id(invitation_id)
            if invitation is not None and invitation.email != email:
                return fail(
                    "The provider email does not match the invitation",
                    "invitation_email_mismatch",
                )
            if invitation is not None and not _invitation_token_still_current(claims, invitation):
                # Re-sending an invitation rotates its token. A flow started
                # from the replaced link must not be able to provision or
                # claim the account afterwards.
                return fail(
                    "This invitation link has been replaced. Use the latest one.",
                    "invitation_token_rotated",
                )
            if invitation is not None and InvitationRepository.status(invitation) == "pending":
                await InvitationRepository(session).mark_accepted(invitation.id, user.id)
        await identities.record_login(identity)
    else:
        if not isinstance(invitation_id, str):
            return _redirect_error(
                origin,
                "No account is linked to this provider. Ask an admin for an invitation.",
            )
        invitations = InvitationRepository(session)
        invitation = await invitations.get_by_id(invitation_id)
        if invitation is None or InvitationRepository.status(invitation) != "pending":
            return _redirect_error(origin, "This invitation is no longer valid")
        if not _invitation_token_still_current(claims, invitation):
            return fail(
                "This invitation link has been replaced. Use the latest one.",
                "invitation_token_rotated",
            )
        if invitation.email != email:
            return _redirect_error(origin, "The provider email does not match the invitation")
        existing_user = await users.get_by_email(email)
        if existing_user is not None:
            return _redirect_error(
                origin, "An account already uses that email. Sign in and link the provider."
            )
        user = await _create_oauth_user(session, email=email)
        created_account = True
        try:
            await identities.create(
                user_id=user.id,
                provider=provider.id,
                subject=subject,
                email=email,
                email_verified=True,
            )
        except ValueError as exc:
            return fail(str(exc), "identity_linked_elsewhere")
        if invitation.allowed_models is not None:
            await users.set_allowed_models(user.id, invitation.allowed_models)
        if invitation.budget_usd is not None:
            await users.set_budget(
                user.id,
                budget_usd=invitation.budget_usd,
                budget_period=invitation.budget_period,
                budget_reset_day=invitation.budget_reset_day,
            )
        await invitations.mark_accepted(invitation.id, user.id)

    await session.commit()
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="oidc",
        metadata_extra={"operation": "oidc_login", "provider": provider.id},
    )
    completion = auth_manager.create_scoped_token(
        OAUTH_COMPLETE_PURPOSE,
        {"uid": user.id, "amr": "oidc", "new_account": created_account},
        ttl_seconds=OAUTH_COMPLETE_TTL_SECONDS,
    )
    params = {"token": completion}
    if created_account:
        # The account was provisioned from an invitation just now. Let the
        # invitee pick their username (and, optionally, a fallback password)
        # before the console session is minted. These query values only drive
        # the UI; the completion endpoint re-validates everything it accepts.
        params["onboarding"] = "1"
        params["suggested_username"] = user.username
    redirect = RedirectResponse(
        f"{origin}/oauth/complete?{urlencode(params)}",
        status_code=303,
    )
    _clear_pkce_cookie(redirect)
    return redirect


@public_router.post("/complete", response_model=LoginResponse)
@limiter.limit("auth.oauth")
async def complete_login(
    data: OAuthCompleteRequest,
    request: Request,
    response: Response,
    session: AsyncSession = get_async_session_dep,
):
    auth_manager = await get_jwt_manager(request)
    try:
        claims = auth_manager.verify_scoped_token(data.completion_token, OAUTH_COMPLETE_PURPOSE)
    except Exception as exc:
        raise AuthenticationFailedError(
            message="The OIDC login expired. Please try again.", code="oauth_complete_expired"
        ) from exc
    user = await UserRepository(session).get_by_id(int(claims.get("uid", -1)))
    if user is None or not user.is_active:
        raise AuthenticationFailedError(message="This account is disabled")
    if claims.get("new_account"):
        if not isinstance(data.username, str) or not data.username.strip():
            raise ValidationError(
                message="Choose a username before finishing account creation",
                code="username_required",
            )
    elif data.username or data.password:
        # A completion token only proves the OAuth sign-in, not a fresh
        # step-up. Letting it rename the account or install a password would
        # turn a 3-minute bridge token into a persistent credential change.
        # Existing accounts must use the authenticated account-security flows.
        raise ValidationError(
            message="Account details can only be set while completing a new invitation",
            code="oauth_onboarding_only",
        )
    if data.username and data.username.strip().lower() != user.username:
        try:
            user = await UserRepository(session).update_username(user.id, data.username)
        except ValueError as exc:
            raise ConflictError(message=str(exc)) from exc
        if user is None:
            raise NotFoundError(message="User not found")
    if data.password:
        await UserRepository(session).update_password(user.id, hash_password(data.password))
        await session.commit()

    # The completion token is only a bridge from the browser redirect to this
    # JSON call. A 2FA-enabled account still needs the normal second factor.
    totp_repo = TotpRepository(session)
    record = await totp_repo.get(user.id)
    if record is not None and record.confirmed_at is not None:
        return await begin_mfa_challenge(request, user, auth_method="oidc")
    return await issue_login_response(request, response, session, user, auth_method="oidc")


@router.post("/link/complete", response_model=dict)
@limiter.limit("auth.oauth")
async def complete_link(
    data: OAuthLinkCompleteRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    """Finish an account link that an identity provider verified.

    The provider callback is unauthenticated, so it only proves which identity
    the IdP vouched for; this call proves that the person completing the link
    still holds the console session that started it.
    """
    auth_manager = await get_jwt_manager(request)
    try:
        claims = auth_manager.verify_scoped_token(data.link_token, OAUTH_LINK_PURPOSE)
    except Exception as exc:
        raise AuthenticationFailedError(
            message="The provider link expired. Please try again.", code="oauth_link_expired"
        ) from exc
    if int(claims.get("uid", -1)) != user.id:
        raise AuthenticationFailedError(
            message="The provider link belongs to another account", code="oauth_link_mismatch"
        )
    identity = get_request_identity(request)
    if claims.get("sid") and claims.get("sid") != identity.console_session_id:
        # The link was started from a console session that is no longer the
        # active one; a leaked link token must not outlive its own session.
        raise AuthenticationFailedError(
            message="Please sign in again and retry linking", code="oauth_link_session_changed"
        )

    provider_id = str(claims.get("provider") or "")
    subject = str(claims.get("subject") or "")
    email = str(claims.get("email") or "").strip().lower()
    if not provider_id or not subject or not email:
        raise AuthenticationFailedError(message="The provider link is incomplete")

    identities = OAuthIdentityRepository(session)
    existing = await identities.get(provider_id, subject)
    if existing is not None:
        if existing.user_id == user.id:
            return {"success": True}
        raise ConflictError(message="This provider account is already linked to someone else")
    await UserRepository(session).get_for_update(user.id)
    if user.email and user.email != email:
        raise ConflictError(message="The provider email does not match your account email")
    if user.email is None:
        await UserRepository(session).set_email(user.id, email, verified_by="oidc")
    try:
        await identities.create(
            user_id=user.id,
            provider=provider_id,
            subject=subject,
            email=email,
            email_verified=True,
        )
    except ValueError as exc:
        raise ConflictError(message=str(exc)) from exc
    await session.commit()
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="oidc",
        metadata_extra={"operation": "oidc_link", "provider": provider_id},
    )
    return {"success": True}


@router.get("/identities", response_model=OAuthIdentityListResponse)
async def list_identities(
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
):
    records = await OAuthIdentityRepository(session).list_for_user(user.id)
    return OAuthIdentityListResponse(
        identities=[
            OAuthIdentityRead(
                id=record.id,
                provider=record.provider,
                email=record.email,
                created_at=record.created_at,
                last_login_at=record.last_login_at,
            )
            for record in records
        ]
    )


@router.delete("/identities/{identity_id}", response_model=dict)
async def unlink_identity(
    identity_id: int,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    identities = OAuthIdentityRepository(session)
    # Lock the account before counting what is left: two concurrent unlinks
    # must not both observe a spare factor and both remove one.
    await UserRepository(session).get_for_update(user.id)
    if not is_bcrypt_hash(user.password_hash):
        remaining = await identities.list_for_user(user.id)
        passkeys = await PasskeyRepository(session).count_for_user(user.id)
        if len(remaining) <= 1 and passkeys == 0:
            raise ValidationError(
                message="Set a password or add a passkey before unlinking your last sign-in method",
                code="last_factor",
            )
    if not await identities.delete(user.id, identity_id):
        raise NotFoundError(message="Linked identity not found")
    await session.commit()
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="oidc",
        metadata_extra={"operation": "oidc_unlink", "identity_id": identity_id},
    )
    return {"success": True}
