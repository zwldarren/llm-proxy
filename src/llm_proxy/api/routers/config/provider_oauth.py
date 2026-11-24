"""Admin OAuth management and callback endpoints for OAuth-backed providers."""

import html
import json
import secrets
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_async_session_dep, require_admin_role
from llm_proxy.api.routers.config.helpers import commit_and_reload, get_config_repository
from llm_proxy.api.schemas.providers import (
    ProviderOAuthCompleteRequest,
    ProviderOAuthStartRequest,
    ProviderOAuthStartResponse,
    ProviderOAuthStatusRead,
)
from llm_proxy.core.exceptions import NotFoundError, ValidationError
from llm_proxy.database.repositories.provider_oauth import ProviderOAuthCredentialRepository
from llm_proxy.http.client import ProviderHTTPClientManager
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.openai_chatgpt.oauth import (
    DEFAULT_LOOPBACK_CALLBACK_URI,
    DYNAMIC_AGENT_CLIENT_ID,
    PENDING_OAUTH_TTL_SECONDS,
    PendingOAuthTransaction,
    build_authorization_url,
    exchange_authorization_code,
    extract_id_token_claims,
    generate_pkce_pair,
    get_or_create_host_id,
    get_pending_oauth_store,
    parse_callback_url,
    validate_oauth_redirect_uri,
)
from llm_proxy.providers.openai_chatgpt.token_manager import get_oauth_token_manager
from llm_proxy.services import runtime_services

logger = get_logger(__name__)

public_router = APIRouter(prefix="/providers/oauth", tags=["configuration"])
router = APIRouter(
    prefix="/providers/{name}/oauth",
    tags=["configuration"],
    dependencies=[Depends(require_admin_role)],
)

_OAUTH_PROVIDER_TYPE = "openai-chatgpt"


async def _optional_http_client(request: Request) -> Any:
    manager = runtime_services(request).http_client_or_none()
    if manager is None:
        return None
    if isinstance(manager, ProviderHTTPClientManager):
        return await manager.get_client("__default__")
    return getattr(manager, "client", None)


def _optional_redis_client(request: Request) -> Any:
    return runtime_services(request).redis_client()


def _default_redirect_uri(request: Request) -> str:
    host = (request.url.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return str(request.url_for("provider_oauth_callback"))
    return DEFAULT_LOOPBACK_CALLBACK_URI


async def _require_oauth_provider(session: AsyncSession, name: str):
    repo = get_config_repository(session)
    provider = await repo.get_provider(name, decrypt=False)
    if not provider:
        raise NotFoundError(message=f"Provider '{name}' not found")
    if provider.type.strip().lower() != _OAUTH_PROVIDER_TYPE:
        raise ValidationError(
            message=(
                f"Provider '{name}' has type '{provider.type}'; "
                f"OAuth sign-in is only supported for '{_OAUTH_PROVIDER_TYPE}'."
            ),
            code="oauth_unsupported_provider_type",
        )
    return provider


async def _finalize_oauth_transaction(
    *,
    session: AsyncSession,
    request: Request,
    tx: PendingOAuthTransaction,
    code: str,
    callback_client_id: str | None,
) -> ProviderOAuthStatusRead:
    repo = get_config_repository(session)
    provider = await repo.get_provider(tx.provider_name, decrypt=False)
    if not provider:
        raise NotFoundError(message=f"Provider '{tx.provider_name}' not found")

    if tx.client_id == DYNAMIC_AGENT_CLIENT_ID and not (
        callback_client_id and callback_client_id.strip()
    ):
        raise ValidationError(
            message="OpenAI callback did not return an issued client_id for dynamic registration.",
            code="oauth_missing_client_id",
        )

    effective_client_id = (callback_client_id or tx.client_id).strip()
    if not effective_client_id:
        raise ValidationError(
            message="OAuth callback did not provide a client_id.",
            code="oauth_missing_client_id",
        )

    http_client = await _optional_http_client(request)
    payload = await exchange_authorization_code(
        code=code,
        client_id=effective_client_id,
        code_verifier=tx.code_verifier,
        redirect_uri=tx.redirect_uri,
        client_secret=tx.client_secret,
        http_client=http_client,
    )
    claims = extract_id_token_claims(payload.id_token, expected_nonce=tx.nonce)
    expires_at = (
        datetime.fromtimestamp(time.time() + payload.expires_in, tz=UTC)
        if payload.expires_in is not None
        else None
    )

    cred_repo = ProviderOAuthCredentialRepository(session)
    record = await cred_repo.upsert_for_provider(
        provider.id,
        client_id=effective_client_id,
        client_secret=tx.client_secret,
        ext_agent_host_id=tx.ext_agent_host_id,
        access_token=payload.access_token,
        refresh_token=payload.refresh_token,
        token_type=payload.token_type,
        scope=payload.scope,
        expires_at=expires_at,
        account_sub=claims["sub"],
        account_email=claims["email"],
        account_name=claims["name"],
    )
    await commit_and_reload(session, request)
    get_oauth_token_manager().invalidate(tx.provider_name)
    return ProviderOAuthStatusRead.from_credential(record)


def _render_callback_html(*, ok: bool, title: str, message: str, provider_name: str | None) -> str:
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    payload_json = json.dumps(
        {
            "type": "llm-proxy-provider-oauth",
            "status": "success" if ok else "error",
            "provider": provider_name,
            "message": message,
        }
    ).replace("<", "\\u003c")
    status_badge = "Connected" if ok else "Authorization failed"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: #0f1318;
      color: #e6ebf2;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      padding: 24px;
    }}
    .card {{
      max-width: 440px;
      width: 100%;
      background: #171c23;
      border: 1px solid #2b3340;
      border-radius: 10px;
      padding: 24px;
    }}
    .badge {{
      display: inline-block;
      font-size: 12px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 999px;
      margin-bottom: 12px;
      background: {"#163526" if ok else "#3b1c22"};
      color: {"#7be3a8" if ok else "#f598a6"};
    }}
    h1 {{ font-size: 18px; margin: 0 0 8px; }}
    p {{ font-size: 14px; line-height: 1.5; color: #b5c0cf; margin: 0 0 16px; }}
    button {{
      background: #27303f;
      color: #e6ebf2;
      border: 1px solid #3a4559;
      border-radius: 6px;
      padding: 8px 14px;
      font-size: 13px;
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <div class="card">
    <span class="badge">{status_badge}</span>
    <h1>{safe_title}</h1>
    <p>{safe_message}</p>
    <button type="button" onclick="window.close()">Close window</button>
  </div>
  <script>
    (function () {{
      var payload = {payload_json};
      if (window.opener && typeof window.opener.postMessage === "function") {{
        try {{
          window.opener.postMessage(payload, window.location.origin);
        }} catch (e) {{}}
      }}
      if (payload.status === "success") {{
        setTimeout(function () {{ window.close(); }}, 800);
      }}
    }})();
  </script>
</body>
</html>"""


@public_router.get("/callback", name="provider_oauth_callback", response_class=HTMLResponse)
async def oauth_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    client_id: str | None = Query(default=None),
    error: str | None = Query(default=None),
    error_description: str | None = Query(default=None),
    session: AsyncSession = get_async_session_dep,
) -> HTMLResponse:
    """Browser callback endpoint for Sign in with ChatGPT."""
    if not state:
        return HTMLResponse(
            _render_callback_html(
                ok=False,
                title="Missing OAuth state",
                message="The OAuth callback is missing the state parameter.",
                provider_name=None,
            ),
            status_code=400,
        )

    store = get_pending_oauth_store()
    tx = await store.pop(state, redis_client=_optional_redis_client(request))
    if tx is None:
        return HTMLResponse(
            _render_callback_html(
                ok=False,
                title="Invalid or expired OAuth session",
                message=(
                    "The sign-in state expired or was already used. "
                    "Start sign-in again from LLM Proxy."
                ),
                provider_name=None,
            ),
            status_code=400,
        )

    if error:
        detail = error_description or error
        return HTMLResponse(
            _render_callback_html(
                ok=False,
                title="ChatGPT sign-in cancelled or denied",
                message=str(detail),
                provider_name=tx.provider_name,
            ),
            status_code=400,
        )

    if not code:
        return HTMLResponse(
            _render_callback_html(
                ok=False,
                title="Missing authorization code",
                message="OpenAI did not return an authorization code.",
                provider_name=tx.provider_name,
            ),
            status_code=400,
        )

    try:
        status = await _finalize_oauth_transaction(
            session=session,
            request=request,
            tx=tx,
            code=code,
            callback_client_id=client_id,
        )
    except (ValidationError, NotFoundError) as exc:
        # Expected, user-actionable failures: safe to render verbatim.
        return HTMLResponse(
            _render_callback_html(
                ok=False,
                title="ChatGPT OAuth connection failed",
                message=str(exc),
                provider_name=tx.provider_name,
            ),
            status_code=400,
        )
    except Exception:
        # Transport, database and programming errors carry internals, and this
        # page is reachable unauthenticated by whoever holds the state URL: log
        # the detail, show a generic message.
        logger.error(f"Provider OAuth callback failed for '{tx.provider_name}'", exc_info=True)
        return HTMLResponse(
            _render_callback_html(
                ok=False,
                title="ChatGPT OAuth connection failed",
                message=(
                    "The token exchange with OpenAI did not complete. "
                    "Check the server logs and try signing in again."
                ),
                provider_name=tx.provider_name,
            ),
            status_code=502,
        )

    account_label = status.account_email or status.account_name or tx.provider_name
    return HTMLResponse(
        _render_callback_html(
            ok=True,
            title="ChatGPT account connected",
            message=(
                f"Connected {account_label} to provider '{tx.provider_name}'. "
                "You can close this window."
            ),
            provider_name=tx.provider_name,
        )
    )


@router.get("/status", response_model=ProviderOAuthStatusRead)
async def get_provider_oauth_status(
    name: str = Path(...),
    session: AsyncSession = get_async_session_dep,
) -> ProviderOAuthStatusRead:
    """Return non-secret OAuth status for a provider."""
    provider = await _require_oauth_provider(session, name)
    cred_repo = ProviderOAuthCredentialRepository(session)
    cred = await cred_repo.get_by_provider_id(provider.id, decrypt=False)
    return ProviderOAuthStatusRead.from_credential(cred)


@router.post("/start", response_model=ProviderOAuthStartResponse)
async def start_provider_oauth(
    request: Request,
    body: ProviderOAuthStartRequest | None = None,
    name: str = Path(...),
    session: AsyncSession = get_async_session_dep,
) -> ProviderOAuthStartResponse:
    """Start Sign in with ChatGPT OAuth (PKCE + dynamic client registration)."""
    provider = await _require_oauth_provider(session, name)
    payload = body or ProviderOAuthStartRequest()

    redirect_uri = validate_oauth_redirect_uri(
        payload.redirect_uri or _default_redirect_uri(request)
    )

    cred_repo = ProviderOAuthCredentialRepository(session)
    existing = await cred_repo.get_by_provider_id(provider.id, decrypt=True)

    explicit_client_id = payload.client_id.strip() if payload.client_id else None
    if explicit_client_id:
        client_id = explicit_client_id
    elif not payload.force_new_registration and existing is not None and existing.client_id:
        client_id = existing.client_id
    else:
        client_id = DYNAMIC_AGENT_CLIENT_ID

    if payload.client_secret:
        client_secret = payload.client_secret.strip()
    elif (
        existing is not None
        and client_id == existing.client_id
        and not payload.force_new_registration
    ):
        # A stored secret belongs to the stored client registration; pairing it
        # with a different client_id makes the exchange fail with invalid_client.
        client_secret = existing.client_secret
    else:
        client_secret = None

    ext_agent_host_id: str | None = None
    if client_id == DYNAMIC_AGENT_CLIENT_ID or (
        existing is not None and existing.ext_agent_host_id
    ):
        ext_agent_host_id = (
            existing.ext_agent_host_id
            if existing is not None and existing.ext_agent_host_id
            else await get_or_create_host_id(session)
        )
        await session.commit()

    code_verifier, code_challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(24)

    tx = PendingOAuthTransaction(
        state=state,
        provider_name=name,
        client_id=client_id,
        client_secret=client_secret,
        ext_agent_host_id=ext_agent_host_id,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
        nonce=nonce,
        created_at=time.time(),
    )
    await get_pending_oauth_store().put(tx, redis_client=_optional_redis_client(request))

    authorization_url = build_authorization_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        nonce=nonce,
        code_challenge=code_challenge,
        ext_agent_host_id=ext_agent_host_id,
    )
    return ProviderOAuthStartResponse(
        authorization_url=authorization_url,
        state=state,
        redirect_uri=redirect_uri,
        expires_in=PENDING_OAUTH_TTL_SECONDS,
    )


@router.post("/complete", response_model=ProviderOAuthStatusRead)
async def complete_provider_oauth(
    body: ProviderOAuthCompleteRequest,
    request: Request,
    name: str = Path(...),
    session: AsyncSession = get_async_session_dep,
) -> ProviderOAuthStatusRead:
    """Complete OAuth from a pasted callback URL or explicit ``code`` + ``state``."""
    await _require_oauth_provider(session, name)

    code = body.code
    state = body.state
    client_id = body.client_id

    if body.callback_url:
        parsed = parse_callback_url(body.callback_url)
        if parsed["error"]:
            raise ValidationError(
                message=parsed["error_description"] or parsed["error"] or "OAuth error",
                code="oauth_authorization_denied",
            )
        code = code or parsed["code"]
        state = state or parsed["state"]
        client_id = client_id or parsed["client_id"]

    if not state or not code:
        raise ValidationError(
            message="Both 'code' and 'state' (or a callback_url containing them) are required.",
            code="oauth_missing_parameters",
        )

    tx = await get_pending_oauth_store().pop(state, redis_client=_optional_redis_client(request))
    if tx is None:
        raise ValidationError(
            message="Invalid or expired OAuth state. Please start sign-in again.",
            code="oauth_invalid_state",
        )
    if tx.provider_name != name:
        raise ValidationError(
            message=f"OAuth state belongs to provider '{tx.provider_name}', not '{name}'.",
            code="oauth_provider_mismatch",
        )

    return await _finalize_oauth_transaction(
        session=session,
        request=request,
        tx=tx,
        code=code,
        callback_client_id=client_id,
    )


@router.post("/refresh", response_model=ProviderOAuthStatusRead)
async def refresh_provider_oauth(
    request: Request,
    name: str = Path(...),
    session: AsyncSession = get_async_session_dep,
) -> ProviderOAuthStatusRead:
    """Force a refresh-token exchange for ``name`` and return updated status."""
    provider = await _require_oauth_provider(session, name)
    cred_repo = ProviderOAuthCredentialRepository(session)
    existing = await cred_repo.get_by_provider_id(provider.id, decrypt=False)
    if existing is None:
        raise ValidationError(
            message=f"Provider '{name}' is not signed in with ChatGPT.",
            code="oauth_not_connected",
        )

    http_client = await _optional_http_client(request)
    await get_oauth_token_manager().get_access_token(
        name,
        http_client=http_client,
        force_refresh=True,
        redis_client=_optional_redis_client(request),
    )

    # The token manager rotates tokens in a session of its own and commits
    # there; drop anything this session already holds so the re-read is real.
    session.expire_all()
    refreshed = await cred_repo.get_by_provider_id(provider.id, decrypt=False)
    return ProviderOAuthStatusRead.from_credential(refreshed)


@router.delete("")
async def disconnect_provider_oauth(
    request: Request,
    name: str = Path(...),
    session: AsyncSession = get_async_session_dep,
) -> dict[str, str]:
    """Delete stored OAuth credentials for ``name``."""
    provider = await _require_oauth_provider(session, name)
    cred_repo = ProviderOAuthCredentialRepository(session)
    await cred_repo.delete_by_provider_id(provider.id)
    await commit_and_reload(session, request)
    get_oauth_token_manager().invalidate(name)
    return {"message": f"Disconnected OAuth credentials for provider '{name}'."}
