"""Locust load test for llm-proxy — mirrors the LiteLLM benchmark shape.

Prereqs: seed the proxy once (creates admin, provider, model, API key):

    uv run loadtest/seed.py

Against the proxy (default port 8080):

    uv run locust -f loadtest/locustfile.py --host http://localhost:8180 \
        --headless -u 200 -r 50 -t 60s

Baseline against the fake upstream directly (same routes, auth ignored):

    uv run locust -f loadtest/locustfile.py --host http://localhost:8900 \
        --headless -u 200 -r 50 -t 60s

Think time matters: with wait_time between 0.5–1s, N users offer ~N/0.86 RPS
and hold roughly N * response_time in flight. Report RPS alongside latency —
see docs.litellm.ai/docs/benchmarks#locust-settings.
"""

import contextlib
import hashlib
import os
import uuid
from pathlib import Path

from locust import HttpUser, between, events, task

TEST_MODEL = os.getenv("TEST_MODEL", "fake-model")


def _load_api_key(host: str | None) -> str:
    key = os.getenv("API_KEY")
    if key:
        return key
    if host:
        # Mirror seed.py's per-instance key cache naming.
        digest = hashlib.sha1(host.rstrip("/").encode()).hexdigest()[:8]
        candidate = Path(__file__).parent / f".api_key-{digest}"
        if candidate.exists():
            return candidate.read_text().strip()
    legacy = Path(__file__).parent / ".api_key"
    if legacy.exists():
        return legacy.read_text().strip()
    return "sk-loadtest"


# Custom metric: proxy-reported overhead (wall time minus upstream wait),
# parsed from the x-llm-proxy-overhead-duration-ms response header.
@events.request.add_listener
def on_request(response=None, request_type=None, **kwargs):
    # The custom metric is fired from inside this listener with no response
    # object; skip it so re-entry does not recurse or raise.
    if request_type == "Custom" or response is None or not hasattr(response, "headers"):
        return
    overhead = response.headers.get("x-llm-proxy-overhead-duration-ms")
    if overhead:
        with contextlib.suppress(ValueError, TypeError):
            events.request.fire(
                request_type="Custom",
                name="Proxy Overhead (ms)",
                response_time=float(overhead),
                response_length=0,
            )


def _payload(stream: bool) -> dict:
    payload = {
        "model": TEST_MODEL,
        # UUID prefix defeats any response caching; long body exercises the
        # serialization hot path.
        "messages": [{"role": "user", "content": f"{uuid.uuid4()} This is a load test " * 30}],
    }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


class ProxyUser(HttpUser):
    wait_time = between(0.5, 1)

    def on_start(self):
        self.client.headers.update({"Authorization": f"Bearer {_load_api_key(self.host)}"})

    @task(4)
    def chat_completion(self):
        with self.client.post(
            "/v1/chat/completions", json=_payload(stream=False), catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code} body={resp.text[:200]}")

    @task(1)
    def chat_completion_stream(self):
        with self.client.post(
            "/v1/chat/completions",
            json=_payload(stream=True),
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code} body={resp.text[:200]}")
                return
            # Consume the full SSE body so the measurement covers the stream.
            for _ in resp.iter_lines():
                pass
