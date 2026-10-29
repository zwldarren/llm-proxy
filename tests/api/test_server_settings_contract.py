"""Contract test for the server settings registry (ADR-0026).

Walks ``SERVER_SETTINGS`` key by key and asserts the wire contract each
descriptor declares: default GET responses, PUT persistence per write mode,
merge-vs-replace semantics, DELETE, and pre-persist validation. This file
replaces the per-key endpoint test files and the schema-drift guard: with one
canonical model per key, the default-response assertions below ARE the drift
guard — the wire defaults cannot diverge from the runtime model because both
come from the same class.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import (
    require_admin_role,
    require_authenticated,
)
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.config import server as server_router
from llm_proxy.api.routers.config.server import router
from llm_proxy.config.server_settings import (
    SERVER_SETTINGS,
    ReadMode,
    ServerSetting,
    WriteMode,
)
from llm_proxy.config.types.rate_limits import DEFAULT_RATE_LIMITS
from llm_proxy.config.types.smart_routing import SmartRoutingConfig

BASE = "/api/config/server"


class FakeServerConfigRepo:
    """In-memory server_config rows behind the repository's async interface."""

    def __init__(self, rows: dict | None = None):
        self.rows = dict(rows or {})

    async def get_server_config(self, key: str):
        if key not in self.rows:
            return None
        return SimpleNamespace(value=self.rows[key], description="")

    async def set_server_config(self, key: str, value, description: str | None = None):
        self.rows[key] = value
        return SimpleNamespace(value=value, description=description)

    async def delete_server_config(self, key: str) -> bool:
        return self.rows.pop(key, None) is not None


@pytest.fixture
def repo() -> FakeServerConfigRepo:
    return FakeServerConfigRepo()


@pytest.fixture
def client(repo: FakeServerConfigRepo, monkeypatch) -> TestClient:
    app = FastAPI()
    app.dependency_overrides[require_authenticated] = lambda: None
    app.dependency_overrides[require_admin_role] = lambda: None
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/config")
    monkeypatch.setattr(server_router, "get_config_repository", lambda session: repo)
    monkeypatch.setattr(server_router, "commit_and_reload", AsyncMock())
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _full_payload(setting: ServerSetting) -> dict:
    """A complete PUT body: the canonical model dump, in the wire dialect if any."""
    source = setting.wire_model if setting.wire_model is not None else setting.model
    return source().model_dump()


def _expected_default(setting: ServerSetting) -> dict:
    match setting.read:
        case ReadMode.MODEL_DUMP:
            return setting.model().model_dump()
        case ReadMode.WIRE_MERGED:
            return setting.wire_model().model_dump()  # type: ignore[union-attr]
        case ReadMode.RAW_OR_DEFAULT:
            return dict(setting.default_doc or {})
        case ReadMode.LIMITS_MERGED:
            return {"limits": dict(DEFAULT_RATE_LIMITS)}
        case ReadMode.LIST_WRAP:
            return {"origins": []}
    raise AssertionError(f"unhandled read mode {setting.read}")  # pragma: no cover


def _expected_stored(setting: ServerSetting, payload: dict):
    """What the stored row must be after PUT-ing a complete body onto an empty key."""
    match setting.write:
        case WriteMode.REPLACE | WriteMode.MERGE_FIELDS | WriteMode.MERGE_WIRE_PARTIAL:
            return payload
        case WriteMode.REPLACE_EXCLUDE_NONE:
            return {k: v for k, v in payload.items() if v is not None}
        case WriteMode.REPLACE_INNER:
            return payload["limits"]
        case WriteMode.REPLACE_LIST:
            return [o.strip() for o in payload["origins"] if o.strip()]
    raise AssertionError(f"unhandled write mode {setting.write}")  # pragma: no cover


@pytest.mark.parametrize("setting", SERVER_SETTINGS, ids=lambda s: s.key)
def test_get_returns_defaults_without_row(client, setting: ServerSetting):
    res = client.get(f"{BASE}{setting.path}")
    assert res.status_code == 200
    assert res.json() == _expected_default(setting)


@pytest.mark.parametrize("setting", SERVER_SETTINGS, ids=lambda s: s.key)
def test_put_persists_per_write_mode(client, repo, setting: ServerSetting):
    payload = _full_payload(setting)
    res = client.put(f"{BASE}{setting.path}", json=payload)
    assert res.status_code == 200
    assert repo.rows[setting.key] == _expected_stored(setting, payload)
    # The response and a follow-up GET both render the stored row.
    assert client.get(f"{BASE}{setting.path}").json() == res.json()


# --- Merge semantics: only the keys that declare a merge write mode ---


def test_logging_put_merges_over_existing_row(client, repo):
    repo.rows["logging"] = {"log_input_output": True, "log_retention_days": 7}
    res = client.put(f"{BASE}/logging", json={"sampling_rate": 0.5})
    assert res.status_code == 200
    data = res.json()
    # Sent field wins, omitted stored fields survive, defaults fill the rest.
    assert data["sampling_rate"] == 0.5
    assert repo.rows["logging"]["log_input_output"] is True
    assert repo.rows["logging"]["log_retention_days"] == 7
    assert data["mask_sensitive_data"] is True


def test_logging_put_explicit_null_clears_stored_value(client, repo):
    repo.rows["logging"] = {"audit_sampling_rate": 0.5}
    res = client.put(f"{BASE}/logging", json={"audit_sampling_rate": None})
    assert res.status_code == 200
    assert repo.rows["logging"]["audit_sampling_rate"] is None


def test_logging_get_merges_legacy_row_without_validation(client, repo):
    """Legacy/junk rows round-trip verbatim: defaults fill gaps, no re-validation."""
    repo.rows["logging"] = {"log_input_output": True, "log_retention_days": "abc"}
    res = client.get(f"{BASE}/logging")
    assert res.status_code == 200
    data = res.json()
    assert data["log_input_output"] is True
    assert data["log_retention_days"] == "abc"  # echoed raw, not validated
    assert data["sampling_rate"] == 1.0  # gap filled from wire defaults


def test_smart_routing_put_merges_explicit_fields_only(client, repo):
    # Stored rows are full model dumps; a partial judge would be expanded on read.
    judge = SmartRoutingConfig.from_row(
        {"judge": {"enabled": True, "model": "jev", "deadline_s": 0.5}}
    ).model_dump()["judge"]
    repo.rows["smart_routing"] = {
        "enabled": True,
        "mode_weights": {"fast": 0.5, "auto": 0.5, "best": 1.0},
        "judge": judge,
    }
    res = client.put(f"{BASE}/smart-routing", json={"mode_weights": {"fast": 0.1}})
    assert res.status_code == 200
    data = res.json()
    assert data["mode_weights"] == {"fast": 0.1}
    assert data["enabled"] is True
    assert data["judge"] == judge


def test_provider_selection_put_ignores_fields_outside_update_narrowing(client, repo):
    """Today's update schema carries only ``strategy``; that narrowing is declared."""
    repo.rows["provider_selection"] = {"strategy": "cost_optimized"}
    res = client.put(f"{BASE}/provider-selection", json={"strategy": "balanced"})
    assert res.status_code == 200
    assert repo.rows["provider_selection"] == {"strategy": "balanced"}
    # An empty body changes nothing.
    res = client.put(f"{BASE}/provider-selection", json={})
    assert res.status_code == 200
    assert repo.rows["provider_selection"] == {"strategy": "balanced"}


# --- Replace semantics: omitted fields reset to canonical defaults ---


@pytest.mark.parametrize(
    "path,key,model,field",
    [
        ("/request-policy", "request_policy", "RequestPolicyConfig", "unknown_fields_policy"),
        ("/resilience", "resilience", "ResilienceConfig", "max_retries"),
        ("/security", "security", "SecurityParams", "login_lockout_enabled"),
        ("/keepalive", "keepalive", "KeepaliveParams", "enabled"),
        ("/mcp-security", "mcp_security_policy", "McpSecurityPolicyConfig", "allowed_commands"),
    ],
)
def test_replace_put_resets_omitted_fields_to_defaults(client, repo, path, key, model, field):
    from llm_proxy.config import types as config_types

    canonical = getattr(config_types, model)
    repo.rows[key] = {field: True if field != "unknown_fields_policy" else "error"}
    res = client.put(f"{BASE}{path}", json={})
    assert res.status_code == 200
    # Omitted field is back to the canonical default, not the stored value.
    assert repo.rows[key] == canonical().model_dump()
    assert res.json() == canonical().model_dump()


# --- Storage shapes ---


@pytest.mark.parametrize("value", ["****edited", "prefix****suffix", "***", "*****"])
def test_edited_secret_mask_values_are_rejected(value):
    from llm_proxy.config.server_settings import _reject_edited_secret_mask

    with pytest.raises(ValueError, match="looks like the secret mask"):
        _reject_edited_secret_mask(value, label="SMTP password")


def test_exact_or_empty_secret_mask_is_accepted():
    from llm_proxy.config.server_settings import _reject_edited_secret_mask

    for value in ("****", "", None):
        _reject_edited_secret_mask(value, label="SMTP password")


def test_web_search_secrets_are_encrypted_masked_and_preserved(client, repo, monkeypatch):
    """Search credentials never round-trip: stored ciphertext, masked response."""
    from llm_proxy.security import encryption

    monkeypatch.setattr(encryption, "encrypt_api_key", lambda value: f"enc:{value}")
    payload = {
        "enabled": True,
        "provider": "searxng",
        "searxng": {
            "url": "https://search.example.com",
            "api_key": "search-secret",
            "basic_auth_username": "operator",
            "basic_auth_password": "http-secret",
        },
    }
    res = client.put(f"{BASE}/web-search", json=payload)
    assert res.status_code == 200

    stored = repo.rows["web_search_config"]["searxng"]
    assert stored["api_key"] == "enc:search-secret"
    assert stored["basic_auth_password"] == "enc:http-secret"
    assert res.json()["searxng"]["api_key"] == "****"
    assert res.json()["searxng"]["basic_auth_password"] == "****"

    # The UI echoes the mask back for untouched fields: keep the ciphertext
    # instead of encrypting the mask or clearing the credential.
    payload["searxng"]["api_key"] = "****"
    payload["searxng"]["basic_auth_password"] = "****"
    res = client.put(f"{BASE}/web-search", json=payload)
    assert res.status_code == 200
    assert repo.rows["web_search_config"]["searxng"]["api_key"] == "enc:search-secret"
    assert repo.rows["web_search_config"]["searxng"]["basic_auth_password"] == "enc:http-secret"


def test_web_search_rejects_an_edited_secret_mask(client, repo):
    """A mask that was edited must not be stored as the new credential.

    The read shape returns ``****`` for a stored secret. If the operator edits
    that placeholder (instead of clearing it or replacing it completely), the
    write must fail loudly rather than overwrite the real credential with the
    edited mask.
    """
    stored = {
        "enabled": True,
        "provider": "searxng",
        "searxng": {
            "url": "https://search.example.com",
            "api_key": "enc:stored-api-key",
            "basic_auth_password": "enc:stored-password",
        },
    }
    repo.rows["web_search_config"] = stored
    payload = {
        **stored,
        "searxng": {**stored["searxng"], "api_key": "****edited"},
    }

    res = client.put(f"{BASE}/web-search", json=payload)

    assert res.status_code == 422
    assert "SearXNG API key" in res.text
    assert repo.rows["web_search_config"] == stored


def test_web_search_rejects_an_edited_basic_auth_password_mask(client, repo):
    stored = {
        "enabled": True,
        "provider": "searxng",
        "searxng": {
            "url": "https://search.example.com",
            "basic_auth_username": "operator",
            "basic_auth_password": "enc:stored-password",
        },
    }
    repo.rows["web_search_config"] = stored
    payload = {
        **stored,
        "searxng": {**stored["searxng"], "basic_auth_password": "****edited"},
    }

    res = client.put(f"{BASE}/web-search", json=payload)

    assert res.status_code == 422
    assert "SearXNG basic auth password" in res.text
    assert repo.rows["web_search_config"] == stored


def test_web_search_rejects_an_edited_ollama_mask(client, repo):
    stored = {
        "enabled": True,
        "provider": "ollama",
        "ollama": {"api_key": "enc:stored-api-key", "base_url": "https://ollama.com"},
    }
    repo.rows["web_search_config"] = stored
    payload = {
        **stored,
        "ollama": {**stored["ollama"], "api_key": "****edited"},
    }

    res = client.put(f"{BASE}/web-search", json=payload)

    assert res.status_code == 422
    assert "Ollama API key" in res.text
    assert repo.rows["web_search_config"] == stored


def test_console_auth_preserves_exact_masks_and_rejects_edited_ones(client, repo, monkeypatch):
    """Echoed masks keep secrets; an edited mask is rejected, not written."""
    from llm_proxy.security import encryption

    monkeypatch.setattr(encryption, "encrypt_api_key", lambda value: f"enc:{value}")
    stored = {
        "public_base_url": "https://proxy.example.com",
        "require_2fa_for_admins": False,
        "smtp": {
            "enabled": True,
            "host": "smtp.example.com",
            "port": 587,
            "username": "operator",
            "password": "enc:stored-smtp-password",
            "from_address": "proxy@example.com",
            "starttls": True,
            "use_ssl": False,
        },
        "oidc_providers": [
            {
                "id": "idp",
                "name": "Example IdP",
                "issuer": "https://idp.example.com",
                "client_id": "client-id",
                "client_secret": "enc:stored-client-secret",
                "scopes": "openid email profile",
                "enabled": True,
            }
        ],
    }
    repo.rows["console_auth"] = stored

    unchanged = {
        **stored,
        "smtp": {**stored["smtp"], "password": "****"},
        "oidc_providers": [{**stored["oidc_providers"][0], "client_secret": "****"}],
    }
    res = client.put(f"{BASE}/console-auth", json=unchanged)
    assert res.status_code == 200
    assert repo.rows["console_auth"] == stored

    edited_smtp = {
        **unchanged,
        "smtp": {**unchanged["smtp"], "password": "****edited"},
    }
    res = client.put(f"{BASE}/console-auth", json=edited_smtp)
    assert res.status_code == 422
    assert "SMTP password" in res.text
    assert repo.rows["console_auth"] == stored

    edited_oidc = {
        **unchanged,
        "oidc_providers": [{**unchanged["oidc_providers"][0], "client_secret": "****edited"}],
    }
    res = client.put(f"{BASE}/console-auth", json=edited_oidc)
    assert res.status_code == 422
    assert "client secret" in res.text
    assert repo.rows["console_auth"] == stored

    rotated = {
        **unchanged,
        "smtp": {**unchanged["smtp"], "password": "new-smtp-password"},
        "oidc_providers": [
            {**unchanged["oidc_providers"][0], "client_secret": "new-client-secret"}
        ],
    }
    res = client.put(f"{BASE}/console-auth", json=rotated)
    assert res.status_code == 200
    saved = repo.rows["console_auth"]
    assert saved["smtp"]["password"] == "enc:new-smtp-password"
    assert saved["oidc_providers"][0]["client_secret"] == "enc:new-client-secret"


def test_rate_limits_persist_overrides_only(client, repo):
    res = client.put(f"{BASE}/rate-limits", json={"limits": {"auth.login": "10/minute"}})
    assert res.status_code == 200
    assert repo.rows["rate_limits"] == {"auth.login": "10/minute"}
    assert res.json() == {"limits": {**DEFAULT_RATE_LIMITS, "auth.login": "10/minute"}}


def test_cors_persists_bare_list_stripped(client, repo):
    res = client.put(
        f"{BASE}/cors", json={"origins": [" https://a.example.com ", "", "http://b.local"]}
    )
    assert res.status_code == 200
    assert repo.rows["cors_origins"] == ["https://a.example.com", "http://b.local"]
    assert res.json() == {"origins": ["https://a.example.com", "http://b.local"]}


def test_web_search_put_stores_only_sent_sub_configs(client, repo):
    payload = {"enabled": True, "provider": "searxng", "searxng": None, "ollama": None}
    res = client.put(f"{BASE}/web-search", json=payload)
    assert res.status_code == 200
    assert repo.rows["web_search_config"] == {"enabled": True, "provider": "searxng"}
    assert res.json() == {"enabled": True, "provider": "searxng"}


def test_web_search_put_dumps_provided_sub_config_in_full(client, repo):
    """A provided sub-config keeps its null-valued keys (top-level-only exclusion)."""
    payload = {
        "enabled": True,
        "provider": "searxng",
        "searxng": {"url": "https://search.example.com"},
        "ollama": None,
    }
    res = client.put(f"{BASE}/web-search", json=payload)
    assert res.status_code == 200
    stored = repo.rows["web_search_config"]
    assert stored["searxng"] == {
        "url": "https://search.example.com",
        "api_key": None,
        "basic_auth_username": None,
        "basic_auth_password": None,
        "engines": None,
        "timeout": 30.0,
        "max_results": 10,
    }
    # RAW_OR_DEFAULT echoes the stored row verbatim, nulls included.
    assert client.get(f"{BASE}/web-search").json() == stored


def test_web_search_put_refuses_an_internal_endpoint(client):
    """The proxy calls the search endpoint itself, so a literal metadata IP is a 422."""
    payload = {
        "enabled": True,
        "provider": "searxng",
        "searxng": {"url": "http://169.254.169.254/latest/meta-data"},
        "ollama": None,
    }
    res = client.put(f"{BASE}/web-search", json=payload)
    assert res.status_code == 422


def test_load_server_settings_repairs_out_of_range_web_search_values(repo):
    """Rows written before the bound tightening load with clamped values, not None."""
    from llm_proxy.config.server_settings import load_server_settings

    snapshot = load_server_settings(
        {
            "web_search_config": {
                "enabled": True,
                "provider": "searxng",
                "searxng": {"url": "http://localhost:8080", "max_results": 15, "timeout": 400},
            }
        }
    )
    assert snapshot.web_search is not None
    assert snapshot.web_search.searxng is not None
    assert snapshot.web_search.searxng.max_results == 10
    assert snapshot.web_search.searxng.timeout == 300.0


def test_web_search_delete_removes_row(client, repo):
    repo.rows["web_search_config"] = {"enabled": True, "provider": "searxng"}
    res = client.delete(f"{BASE}/web-search")
    assert res.status_code == 200
    assert res.json() == {"message": "Web search configuration deleted"}
    assert "web_search_config" not in repo.rows
    # Back to the default document.
    assert client.get(f"{BASE}/web-search").json() == {
        "enabled": False,
        "provider": "searxng",
        "searxng": None,
        "ollama": None,
    }


# --- Pre-persist validation ---


def test_rate_limits_rejects_unknown_bucket(client, repo):
    res = client.put(f"{BASE}/rate-limits", json={"limits": {"nope": "5/minute"}})
    assert res.status_code == 422
    assert "Unknown rate limit buckets: nope" in res.json()["error"]["message"]
    assert repo.rows == {}


def test_rate_limits_rejects_invalid_spec(client, repo):
    res = client.put(f"{BASE}/rate-limits", json={"limits": {"auth.login": "five/minute"}})
    assert res.status_code == 422
    assert "Invalid rate limit value for bucket 'auth.login'" in res.json()["error"]["message"]
    assert repo.rows == {}


@pytest.mark.parametrize("spec", ["0/minute", "-1/minute", "5/0second", "5/-1minute"])
def test_rate_limits_rejects_non_positive_operands(client, repo, spec):
    """Both operands have to be positive: the middleware stores nothing that would
    make the bucket deny every request (limit <= 0) or allow every one (window <= 0)."""
    res = client.put(f"{BASE}/rate-limits", json={"limits": {"auth.login": spec}})
    assert res.status_code == 422
    assert "Invalid rate limit value for bucket 'auth.login'" in res.json()["error"]["message"]
    assert repo.rows == {}


def test_a_stored_null_loads_as_unset_instead_of_wedging_the_reload(repo):
    """The PUT body is all-optional, so a stored null is reachable (and legacy rows
    carry them). The runtime read falls back to the wire default rather than raising,
    which would fail every later config reload until the row was repaired by hand."""
    from llm_proxy.config.server_settings import load_server_settings

    snapshot = load_server_settings(
        {
            "logging": {
                "log_input_output": None,
                "log_retention_days": None,
                "max_logged_body_bytes": None,
                "sampling_rate": None,
                "mask_sensitive_data": None,
                "sensitive_keys": None,
            }
        }
    )

    assert snapshot.logging.retention_days == 30
    assert snapshot.logging.max_logged_body_bytes == 1024 * 1024
    assert snapshot.logging.sampling_rate == 1.0
    assert snapshot.logging.mask_sensitive_data is True
    assert snapshot.logging.log_input_output is False


def test_request_policy_rejects_invalid_literal(client, repo):
    res = client.put(f"{BASE}/request-policy", json={"unknown_fields_policy": "shrug"})
    assert res.status_code == 422
    assert repo.rows == {}


def test_put_rejects_out_of_range_value(client, repo):
    res = client.put(f"{BASE}/logging", json={"sampling_rate": 2.0})
    assert res.status_code == 422
    assert repo.rows == {}
