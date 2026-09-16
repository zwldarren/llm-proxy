"""Seed a fresh llm-proxy container with the full load-test topology.

Creates (idempotently) the admin account, every provider/model in
``loadtest/scenarios.py`` (OpenAI Chat Completions, OpenAI Responses, Anthropic
Messages, DeepSeek multi-dialect, Gemini and Ollama), and one API key with
access to all of them. The key is written to ``loadtest/.api_key-<host>`` and
auto-loaded by ``locustfile.py``.

    uv run loadtest/seed.py

Then smoke-tests every selected scenario through the proxy so a broken wire
dialect fails here, not in the middle of a load run.

Env overrides: PROXY_BASE_URL, ADMIN_USERNAME, ADMIN_PASSWORD,
FAKE_UPSTREAM_URL (must resolve from INSIDE the proxy container),
SEED_SMOKE (0 to skip), SEED_SMOKE_SCENARIOS (default ``all``).
"""

import hashlib
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loadtest.scenarios import (  # noqa: E402
    PROVIDERS,
    ProviderSpec,
    build_headers,
    build_payload,
    select_scenarios,
)

BASE = os.getenv("PROXY_BASE_URL", "http://localhost:8180").rstrip("/")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Loadtest-Admin-2026!")
KEY_NAME = "loadtest"
UPSTREAM_API_KEY = "sk-fake-upstream"


def key_file_for(base: str) -> str:
    """Per-instance key cache: one file per base URL so container and local
    seeds never clobber each other's credentials."""
    digest = hashlib.sha1(base.encode()).hexdigest()[:8]
    return os.path.join(os.path.dirname(__file__), f".api_key-{digest}")


KEY_FILE = key_file_for(BASE)


def wait_ready(client: httpx.Client, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = client.get(f"{BASE}/api/health")
            if resp.status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(2)
    sys.exit(f"proxy at {BASE} did not become ready within {timeout}s")


def admin_token(client: httpx.Client) -> str:
    resp = client.get(f"{BASE}/api/auth/setup-status")
    resp.raise_for_status()
    if resp.json().get("needs_setup"):
        resp = client.post(
            f"{BASE}/api/auth/setup",
            json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
        )
    else:
        resp = client.post(
            f"{BASE}/api/auth/login",
            json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
        )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _provider_payload(spec: ProviderSpec) -> dict:
    return {
        "type": spec.type,
        "base_url": spec.base_url,
        "api_key": UPSTREAM_API_KEY,
        "timeout": 300.0,
        "enabled": True,
        "provider_metadata": spec.metadata,
        "endpoint_base_urls": spec.endpoint_base_urls,
    }


def ensure_provider(client: httpx.Client, headers: dict, spec: ProviderSpec) -> None:
    providers = client.get(f"{BASE}/api/config/providers", headers=headers)
    providers.raise_for_status()
    existing = next((p for p in providers.json() if p["name"] == spec.name), None)
    payload = _provider_payload(spec)
    if existing is None:
        resp = client.post(
            f"{BASE}/api/config/providers",
            headers=headers,
            json={"name": spec.name, **payload},
        )
        resp.raise_for_status()
        print(f"created provider {spec.name} ({spec.type}) -> {spec.base_url}")
        return
    # Re-apply on every seed: provider metadata / base URL drift is exactly the
    # kind of thing a stale load-test stack gets wrong.
    resp = client.put(f"{BASE}/api/config/providers/{spec.name}", headers=headers, json=payload)
    resp.raise_for_status()
    print(f"updated provider {spec.name} ({spec.type}) -> {spec.base_url}")


def ensure_model(client: httpx.Client, headers: dict, spec: ProviderSpec) -> None:
    desired = {
        "providers": [
            {
                "provider_name": spec.name,
                "provider_model_name": spec.provider_model,
                "priority": 1,
            }
        ],
        "supports_embedding": spec.supports_embeddings,
    }
    models = client.get(f"{BASE}/api/config/models", headers=headers)
    models.raise_for_status()
    existing = next((m for m in models.json() if m["name"] == spec.model), None)
    if existing is None:
        resp = client.post(
            f"{BASE}/api/config/models",
            headers=headers,
            json={"name": spec.model, **desired},
        )
        resp.raise_for_status()
        print(f"created model {spec.model} -> {spec.name}")
        return
    resp = client.put(
        f"{BASE}/api/config/models/{spec.model}",
        headers=headers,
        json=desired,
    )
    resp.raise_for_status()
    print(f"updated model {spec.model} -> {spec.name}")


def key_works(client: httpx.Client, key: str) -> bool:
    resp = client.post(
        f"{BASE}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": PROVIDERS["openai_compat"].model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        },
        timeout=30.0,
    )
    return resp.status_code == 200


def ensure_api_key(client: httpx.Client, headers: dict) -> str:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as fh:
            key = fh.read().strip()
        if key and key_works(client, key):
            print(f"reusing existing key from {KEY_FILE}")
            return key

    resp = client.post(f"{BASE}/api/api-keys", headers=headers, json={"name": KEY_NAME})
    if resp.status_code == 409:
        # Key exists but its plaintext is unrecoverable — mint a fresh one.
        resp = client.post(
            f"{BASE}/api/api-keys",
            headers=headers,
            json={"name": f"{KEY_NAME}-{int(time.time())}"},
        )
    resp.raise_for_status()
    key = resp.json()["key"]
    with open(KEY_FILE, "w") as fh:
        fh.write(key)
    os.chmod(KEY_FILE, 0o600)
    print(f"created API key, saved to {KEY_FILE}")
    return key


_EXPECTED_BODY_FRAGMENTS = {
    "chat": '"choices"',
    "messages": '"content"',
    "responses": '"output"',
    "embeddings": '"embedding"',
}


def smoke_test(client: httpx.Client, key: str, scenarios) -> list[str]:
    """Drive each scenario end-to-end and record failures."""
    failures: list[str] = []
    for scenario in scenarios:
        payload = build_payload(scenario)
        headers = build_headers(scenario, key)
        try:
            with client.stream(
                "POST",
                f"{BASE}{scenario.path}",
                json=payload,
                headers=headers,
                timeout=60.0,
            ) as resp:
                body = "".join(resp.iter_text())
                status = resp.status_code
        except httpx.HTTPError as exc:
            failures.append(f"{scenario.key}: transport error {exc}")
            continue

        # Cheap shape check: a 200 with a malformed body is still a broken
        # conversion, and would only surface as a client parse error under load.
        expected_fragment = _EXPECTED_BODY_FRAGMENTS.get(scenario.protocol, "")
        marker = scenario.terminal_marker
        if status != 200:
            failures.append(f"{scenario.key}: status={status} body={body[:200]}")
        elif scenario.stream and marker and marker not in body:
            failures.append(f"{scenario.key}: missing terminal marker {marker!r}")
        elif not scenario.stream and expected_fragment and expected_fragment not in body:
            failures.append(f"{scenario.key}: body missing {expected_fragment!r}: {body[:200]}")
    return failures


def main() -> None:
    smoke_spec = os.getenv("SEED_SMOKE_SCENARIOS", "all")
    with httpx.Client(timeout=15.0) as client:
        wait_ready(client)
        token = admin_token(client)
        headers = {"Authorization": f"Bearer {token}"}

        for spec in PROVIDERS.values():
            ensure_provider(client, headers, spec)
        for spec in PROVIDERS.values():
            ensure_model(client, headers, spec)

        key = ensure_api_key(client, headers)

        if os.getenv("SEED_SMOKE", "1") == "0":
            print("seed complete (smoke skipped)")
            return

        scenarios = select_scenarios(smoke_spec)
        print(f"smoke-testing {len(scenarios)} scenario(s) ...")
        failures = smoke_test(client, key, scenarios)
        for scenario in scenarios:
            status = "FAIL" if any(f.startswith(scenario.key + ":") for f in failures) else "ok"
            print(f"  smoke {scenario.key:44s} {status}")
        if failures:
            print("\nsmoke failures:")
            for failure in failures:
                print(f"  - {failure}")
            sys.exit(1)

    print("seed complete")


if __name__ == "__main__":
    main()
