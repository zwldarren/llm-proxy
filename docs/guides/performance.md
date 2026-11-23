# Performance & Benchmarks

The proxy adds a measurable amount of work to every request: protocol parsing,
provider routing, usage accounting, request logging. This page quantifies that
work and puts it next to a comparable gateway, LiteLLM.

Measured 2026-11-24. Both gateways ran on the same host, one at a time, with
the same CPU quota, worker count, datastores and upstream.

## Results

| Concurrent users | Gateway | Requests/s | p50 | p95 | p99 |
| --- | --- | --- | --- | --- | --- |
| 400 | Upstream direct | 532 | 1.7 ms | 3.3 ms | 4.7 ms |
| 400 | LLM Proxy | 527 | 5.8 ms | 11.5 ms | 17.0 ms |
| 400 | LiteLLM 1.108.15 LTS | 303 | 360 ms | 748 ms | 2.9 s |
| 800 | Upstream direct | 1060 | 2.1 ms | 6.7 ms | 20 ms |
| 800 | LLM Proxy | 1029 | 18.2 ms | 55.8 ms | 96 ms |
| 800 | LiteLLM 1.108.15 LTS | 313 | 1100 ms | 2.9 s | 3.9 s |
| 1200 | Upstream direct | 1565 | 6.0 ms | 51 ms | 91 ms |
| 1200 | LLM Proxy | 1251 | 159 ms | 314 ms | 403 ms |
| 1200 | LiteLLM 1.108.15 LTS | 295 | 1475 ms | 4.1 s | 6.5 s |

"Upstream direct" is the same fake upstream with no gateway in front of it: the
ceiling this host can drive. Latency percentiles drop the first 5 s of each run,
so the connection ramp does not set p99; requests per second cover the full run.

CPU and memory tell the same story:

| Concurrent users | Gateway | Mean CPU (cores) | CPU ms per request | Peak memory |
| --- | --- | --- | --- | --- |
| 400 | LLM Proxy | 2.0 of 8 | 3.8 | 1.2 GiB |
| 400 | LiteLLM 1.108.15 LTS | 7.4 of 8 | 24.6 | 5.0 GiB |
| 800 | LLM Proxy | 3.7 of 8 | 3.6 | 1.2 GiB |
| 800 | LiteLLM 1.108.15 LTS | 7.9 of 8 | 25.6 | 5.5 GiB |
| 1200 | LLM Proxy | 4.1 of 8 | 3.3 | 1.2 GiB |
| 1200 | LiteLLM 1.108.15 LTS | 7.9 of 8 | 27.2 | 5.6 GiB |

Both gateways returned zero failed requests across the 18 measured runs.

## Setup

| | LLM Proxy | LiteLLM |
| --- | --- | --- |
| Version | 0.6.1, commit `562e410a4` | 1.108.15 (LTS line 1.108) |
| Image | built from source | `ghcr.io/berriai/litellm:v1.108.15-lts` |
| Python | 3.15 | 3.13 |
| Workers | 8 | 8 |
| CPU quota | 8 cores | 8 cores |
| Datastores | PostgreSQL 18 + Redis 8 | same containers, own database |
| Log level | `WARNING` | `LITELLM_LOG=ERROR` |

Both sides use their documented production settings: batched request-log, usage
and spend writes, Redis for cross-worker state, retries disabled, response
caching disabled. LiteLLM's spend tracking and LLM Proxy's logging and budget
accounting stay on, because they are part of a normal deployment.

The client is Locust 2.46.7, running on the same 12-core host. It sends
`/v1/chat/completions` requests, non-streaming and streaming at equal weight,
with ~430-token prompts. Each run lasts 60 s after a 400 users/s ramp; each
level is measured twice, and the second round walks the levels in reverse. The
upstream is `loadtest/fake_upstream.py`, which answers with zero model latency so
the measurement isolates gateway cost.

## What the numbers mean

LiteLLM reaches capacity at roughly 300 requests/s on 8 cores, so throughput
stays flat as offered concurrency triples and the extra load turns into queue
time. Its cost is CPU inside its workers: 8 cores divided by about 25 ms per
request is about 320 requests/s, which is where all three levels land.

LLM Proxy matches the direct upstream to about 1000 requests/s offered. At 400
and 800 users it delivers 99% and 97% of the no-gateway throughput with 4 ms and
16 ms of added p50. At 1200 users it serves 80% of direct throughput. It does so
while using about half of its CPU quota, so the queue at that level is not a CPU
limit; this run does not isolate what it is.

The per-request CPU difference is larger than the throughput difference, because
LLM Proxy is not CPU-saturated below 1200 users: about 3.5 ms versus about 25 ms,
a factor of 6.5 to 8. Memory is 1.2 GiB versus 5.0–5.6 GiB.

For stability, the September 2026 comparison measured LiteLLM's then-current
`main-stable` image: it returned 3 and 21 failed requests at 400 and 800 users,
all 500s from `RuntimeError: dictionary changed size during iteration`, and
collapsed at 1200. The 1.108.15 LTS line did not fail any request in 18 runs. Its ceiling is
unchanged: the cost of exceeding it is now queue time instead of errors.

## What this does not show

- The upstream has zero latency. A real provider adds tens to thousands of
  milliseconds, which dominates end-to-end time and makes the relative
  difference between gateways smaller. The numbers describe gateway cost, not
  the latency your users will see.
- It is one host, shared with unrelated containers, with the load generator on
  the same machine. Absolute throughput is understated for both gateways, and
  the 1200-user point is the noisiest.
- The generator is closed-loop with 0.5–1 s of think time. A slow gateway
  receives less offered load, so the table shows achieved throughput, not
  capacity under an open-loop arrival process.
- Different Python versions (3.15 vs 3.13) account for part of the per-request
  cost gap, but not for a factor of seven.
- Features differ. LiteLLM covers more providers and callbacks; the comparison
  measures the request path both were configured to serve, not feature parity.

## Reproduce

The harness is in the repository, with the full matrix and scenario selection
documented in [loadtest/README.md](https://github.com/zwldarren/llm-proxy/blob/main/loadtest/README.md).
The multi-gateway driver is
[loadtest/gateway-ab.sh](https://github.com/zwldarren/llm-proxy/blob/main/loadtest/gateway-ab.sh).

```bash
git clone https://github.com/zwldarren/llm-proxy.git
cd llm-proxy
COMPOSE="docker compose -f docker-compose.yaml -f docker-compose.loadtest.yaml \
  -f loadtest/docker-compose.bench.yaml"

docker build -t llm-proxy:loadtest .
$COMPOSE up -d db redis fake-upstream
PROXY_BASE_URL=http://localhost:8180 uv run loadtest/seed.py

OUT=loadtest/results/$(date -u +%F)-litellm-lts LEVELS="400 800 1200" \
  REPS=2 MIX=core uv run loadtest/gateway-ab.sh

uv run loadtest/analyze_ab.py "$OUT"
uv run loadtest/analyze_latency.py "$OUT" --skip 5
```

LiteLLM needs its own database in the shared Postgres container. Create it once:

```bash
docker compose -f docker-compose.yaml -f docker-compose.loadtest.yaml \
  -f loadtest/docker-compose.bench.yaml exec db \
  psql -U llmproxy -c 'CREATE DATABASE litellm'
```

`gateway-ab.sh` stops the idle gateway before each run, recreates both
containers so every worker loads the seeded topology, and samples container CPU,
memory and host load. The LiteLLM image is pinned in
[loadtest/docker-compose.bench.yaml](https://github.com/zwldarren/llm-proxy/blob/main/loadtest/docker-compose.bench.yaml)
so a rerun measures the same build.
