"""Locust load test for llm-proxy — the conversion-tier matrix.

The proxy's cost per request depends on the *conversion tier* the pipeline
picks for a (client protocol, provider dialect) pair: native passthrough,
wire reuse, or full conversion. This harness drives the matrix in
``loadtest/scenarios.py`` so each tier, protocol and provider can be measured
separately or as a weighted mix.

Prereqs: seed the proxy once (creates every provider/model in the matrix and
an API key with access to all of them):

    uv run loadtest/seed.py

Run the curated core mix against the proxy (port 8180):

    uv run locust -f loadtest/locustfile.py --host http://localhost:8180 \\
        --headless -u 100 -r 20 -t 60s

Pick scenarios by key or group (protocol / provider / tier / stream):

    LOADTEST_SCENARIOS=messages,responses uv run locust -f loadtest/locustfile.py ...
    LOADTEST_SCENARIOS=native uv run locust ...
    LOADTEST_SCENARIOS=chat_openai_compat_stream uv run locust ...
    LOADTEST_SCENARIOS=all uv run locust ...

No-proxy baseline (the fake upstream serves the same wire dialects directly):

    uv run locust -f loadtest/locustfile.py --host http://localhost:8900 ...

Think time matters: with wait_time between 0.5–1s, N users offer ~N/0.86 RPS
and hold roughly N * response_time in flight. Report RPS alongside latency.
"""

import os
import random
import sys
import threading
import time
from pathlib import Path

from locust import HttpUser, between, events, task

# Import the scenario matrix from the repo root regardless of how locust was
# invoked (`uv run locust -f loadtest/locustfile.py` does not put the repo root
# on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loadtest.scenarios import (  # noqa: E402
    build_headers,
    build_payload,
    select_scenarios,
)

SCENARIOS = select_scenarios()
_WEIGHTS = [s.weight for s in SCENARIOS]

# Per-request latency dump. When LOADTEST_LATENCY_DUMP is set, every POST
# request is appended as "<elapsed_s>,<stat_name>,<response_time_ms>" so the
# analysis can compute exact percentiles over any time window — locust's own
# summary CSV only carries whole-run cumulative percentiles, which the
# connection storm in the first seconds of a run otherwise dominates at p99.
_LATENCY_PATH = os.getenv("LOADTEST_LATENCY_DUMP")
_latency_fh = None
_latency_start = 0.0
_latency_lock = threading.Lock()


def _load_api_key(host: str | None) -> str:
    import hashlib
    import os

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


@events.test_start.add_listener
def on_test_start(environment, **_kwargs):
    global _latency_fh, _latency_start
    if _LATENCY_PATH:
        # Held open for the whole test run and closed in the test_stop listener.
        _latency_fh = open(_LATENCY_PATH, "w", buffering=1 << 16)  # noqa: SIM115
        _latency_start = time.perf_counter()
    print(f"llm-proxy load test: {len(SCENARIOS)} scenario(s) selected")
    for scenario in SCENARIOS:
        print(f"  - {scenario.key:44s} tier={scenario.expected_tier}")


@events.test_stop.add_listener
def on_test_stop(environment, **_kwargs):
    if _latency_fh is not None:
        _latency_fh.close()


@events.request.add_listener
def on_request(request_type, name, response_time, exception, **_kwargs):
    """Record every gateway request (not the Custom overhead/ttft events)."""
    if _latency_fh is None or request_type != "POST":
        return
    elapsed = time.perf_counter() - _latency_start
    with _latency_lock:
        _latency_fh.write(f"{elapsed:.3f},{name},{response_time:.3f}\n")


def _report_overhead(response, scenario) -> None:
    """Emit the proxy-reported overhead (wall time minus upstream wait) as a
    per-scenario custom metric. llm-proxy emits x-llm-proxy-overhead-duration-ms;
    LiteLLM emits x-litellm-overhead-duration-ms (checked so the same harness
    can A/B both gateways)."""
    for header in ("x-llm-proxy-overhead-duration-ms", "x-litellm-overhead-duration-ms"):
        value = response.headers.get(header)
        if not value:
            continue
        try:
            overhead = float(value)
        except TypeError, ValueError:
            return
        events.request.fire(
            request_type="Custom",
            name=f"overhead:{scenario.key}",
            response_time=overhead,
            response_length=0,
        )
        return


class ProxyUser(HttpUser):
    wait_time = between(0.5, 1)

    def on_start(self):
        self.api_key = _load_api_key(self.host)

    @task
    def run_scenario(self):
        scenario = random.choices(SCENARIOS, weights=_WEIGHTS, k=1)[0]
        payload = build_payload(scenario)
        headers = build_headers(scenario, self.api_key)
        if scenario.stream:
            self._run_streaming(scenario, payload, headers)
        else:
            self._run_unary(scenario, payload, headers)

    def _run_unary(self, scenario, payload, headers):
        with self.client.post(
            scenario.path,
            json=payload,
            headers=headers,
            name=scenario.stat_name,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code} body={response.text[:200]}")
                return
            _report_overhead(response, scenario)
            if scenario.protocol == "embeddings" and '"embedding"' not in response.text:
                response.failure(f"embeddings body missing 'embedding': {response.text[:200]}")

    def _run_streaming(self, scenario, payload, headers):
        start = time.perf_counter()
        with self.client.post(
            scenario.path,
            json=payload,
            headers=headers,
            name=scenario.stat_name,
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code} body={response.text[:200]}")
                return
            _report_overhead(response, scenario)

            marker = scenario.terminal_marker
            seen_marker = False
            first_line = True
            for line in response.iter_lines():
                # locust's requests-based client yields bytes unless asked to
                # decode; normalise so marker matching works for both.
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if first_line:
                    # TTFB is locust's own response_time for streamed responses;
                    # this is the first *body* line, a stricter TTFT signal.
                    events.request.fire(
                        request_type="Custom",
                        name=f"ttft:{scenario.key}",
                        response_time=(time.perf_counter() - start) * 1000,
                        response_length=0,
                    )
                    first_line = False
                if marker and marker in line:
                    seen_marker = True
            events.request.fire(
                request_type="Custom",
                name=f"stream_total:{scenario.key}",
                response_time=(time.perf_counter() - start) * 1000,
                response_length=0,
            )
            if marker and not seen_marker:
                response.failure(f"stream ended without terminal marker {marker!r}")
