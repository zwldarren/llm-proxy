"""Seed a fresh llm-proxy container for load testing.

Creates (idempotently): admin account -> fake-openai provider -> fake-model ->
loadtest API key (written to loadtest/.api_key, auto-loaded by locustfile.py).

    uv run loadtest/seed.py

Env overrides: PROXY_BASE_URL, ADMIN_USERNAME, ADMIN_PASSWORD, UPSTREAM_URL.
UPSTREAM_URL must resolve from INSIDE the proxy container.
"""

import hashlib
import os
import sys
import time

import httpx

BASE = os.getenv("PROXY_BASE_URL", "http://localhost:8180").rstrip("/")
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://fake-upstream:8900/v1")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Loadtest-Admin-2026!")
PROVIDER_NAME = "fake-openai"
MODEL_NAME = "fake-model"
KEY_NAME = "loadtest"


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


def ensure_provider_and_model(client: httpx.Client, headers: dict) -> None:
    providers = client.get(f"{BASE}/api/config/providers", headers=headers)
    providers.raise_for_status()
    existing = next((p for p in providers.json() if p["name"] == PROVIDER_NAME), None)
    if existing is None:
        resp = client.post(
            f"{BASE}/api/config/providers",
            headers=headers,
            json={
                "name": PROVIDER_NAME,
                # openai-compatible targets {base_url}/chat/completions; the
                # "openai" type natively targets the Responses API instead.
                "type": "openai-compatible",
                "base_url": UPSTREAM_URL,
                "api_key": "sk-fake-upstream",
                "timeout": 300.0,
                "enabled": True,
            },
        )
        resp.raise_for_status()
        print(f"created provider {PROVIDER_NAME} -> {UPSTREAM_URL}")
    elif existing["type"] != "openai-compatible" or existing["base_url"] != UPSTREAM_URL:
        resp = client.put(
            f"{BASE}/api/config/providers/{PROVIDER_NAME}",
            headers=headers,
            json={"type": "openai-compatible", "base_url": UPSTREAM_URL, "enabled": True},
        )
        resp.raise_for_status()
        print(f"updated provider {PROVIDER_NAME} -> {UPSTREAM_URL}")

    models = client.get(f"{BASE}/api/config/models", headers=headers)
    models.raise_for_status()
    if not any(m["name"] == MODEL_NAME for m in models.json()):
        resp = client.post(
            f"{BASE}/api/config/models",
            headers=headers,
            json={
                "name": MODEL_NAME,
                "providers": [
                    {
                        "provider_name": PROVIDER_NAME,
                        "provider_model_name": MODEL_NAME,
                        "priority": 1,
                    }
                ],
            },
        )
        resp.raise_for_status()
        print(f"created model {MODEL_NAME}")


def key_works(client: httpx.Client, key: str) -> bool:
    resp = client.post(
        f"{BASE}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": MODEL_NAME,
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


def main() -> None:
    with httpx.Client(timeout=15.0) as client:
        wait_ready(client)
        token = admin_token(client)
        headers = {"Authorization": f"Bearer {token}"}
        ensure_provider_and_model(client, headers)
        key = ensure_api_key(client, headers)

        # End-to-end smoke: non-streaming and streaming through the proxy.
        for stream in (False, True):
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hello"}],
            }
            if stream:
                payload["stream"] = True
            resp = client.post(
                f"{BASE}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=30.0,
            )
            overhead = resp.headers.get("x-llm-proxy-overhead-duration-ms", "?")
            print(f"smoke stream={stream}: status={resp.status_code} overhead={overhead}ms")
            if resp.status_code != 200:
                sys.exit(f"smoke test failed: {resp.text[:500]}")

    print("seed complete")


if __name__ == "__main__":
    main()
