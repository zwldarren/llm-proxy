# Load testing llm-proxy

The proxy is not one performance profile. A request's cost depends on the
**conversion tier** the pipeline picks for the (client protocol, provider
dialect) pair — see `llm_proxy.core.conversion` and the "Conversion plan" entry
in [`CONTEXT.md`](../CONTEXT.md):

| Tier | What happens |
| --- | --- |
| **Native passthrough** | The stashed client body and the upstream SSE are forwarded verbatim (client protocol == provider dialect). |
| **Wire reuse** | The client body is reused with `model`/`stream` rewritten; the response body may also ride verbatim. |
| **Full conversion** | Canonical parse → `InternalRequest` → provider serializer, and the reverse for the response/stream. |

A single "/v1/chat/completions against one openai-compatible upstream" test only
measures the wire-reuse tier. This harness drives the full matrix.

## Topology

`scenarios.py` defines six upstream providers, each a different wire dialect,
seeded 1:1 onto a proxy model:

| Model | Provider type | Dialect | chat (`/v1/chat/completions`) | messages (`/v1/messages`) | responses (`/v1/responses`) | embeddings |
| --- | --- | --- | --- | --- | --- | --- |
| `fake-model` | `openai-compatible` | Chat Completions | wire reuse + native stream | conversion | conversion | native |
| `fake-openai` | `openai` | Responses | conversion | conversion | **native** | native |
| `fake-anthropic` | `anthropic` | Messages | conversion | **native** | conversion | — |
| `fake-deepseek` | `deepseek` | Chat + Messages + Responses | wire reuse + converted stream | **native** | **native** | native |
| `fake-gemini` | `gemini` | Gemini | conversion | conversion | conversion | conversion |
| `fake-ollama` | `ollama` | Ollama | conversion | conversion | conversion | conversion |

Each dialect is served by the deterministic fake upstream
(`fake_upstream.py`) with **zero model latency**, streaming and non-streaming,
so measurements isolate gateway overhead. The fake upstream also listens on the
host (`:8900`) and can be used as a no-proxy baseline.

## Running

One-shot (build stack → seed → restart workers → locust):

```bash
uv run loadtest/run.sh --headless -u 100 -r 20 -t 60s
```

Manually, against an already-running stack:

```bash
docker compose -f docker-compose.yaml -f docker-compose.loadtest.yaml up -d --build
uv run loadtest/seed.py
docker compose -f docker-compose.yaml -f docker-compose.loadtest.yaml restart llm-proxy
uv run locust -f loadtest/locustfile.py --host http://localhost:8180 --headless -u 100 -r 20 -t 60s
```

> **Why the restart?** Configuration is held per worker in memory. `seed.py`
> changes reach only the worker that served the admin request; Redis does not
> broadcast config reloads. With `LOADTEST_WORKERS>1` the other workers keep
> the pre-seed topology and return `model_not_found` for the new models. Restart
> after seeding, or run with `LOADTEST_WORKERS=1` (`run.sh` restarts for you).

No-proxy baseline (the fake upstream serves the same wire dialects directly):

```bash
uv run locust -f loadtest/locustfile.py --host http://localhost:8900 --headless -u 100 -r 20 -t 60s
```

## Selecting scenarios

`LOADTEST_SCENARIOS` takes comma-separated scenario keys or groups. Default is
`core`; `all` selects the whole 40-point matrix.

```bash
LOADTEST_SCENARIOS=messages,responses uv run locust ...   # by protocol
LOADTEST_SCENARIOS=native uv run locust ...               # native tier only
LOADTEST_SCENARIOS=conversion uv run locust ...           # converted paths only
LOADTEST_SCENARIOS=stream nonstream...                    # stream / non-stream
LOADTEST_SCENARIOS=gemini,ollama uv run locust ...        # by provider dialect
LOADTEST_SCENARIOS=chat_openai_compat_stream uv run locust ...  # one point
LOADTEST_SCENARIOS=all uv run loadtest/seed.py            # smoke every point
```

Groups: `chat` `messages` `responses` `embeddings` · `openai_compat`
`openai_native` `anthropic_native` `deepseek_multi` `gemini` `ollama` · `native`
`wire_reuse` `conversion` · `stream` `nonstream` · `core` `all`.

## Interpreting results

Locust reports one request name per matrix point (`messages:anthropic_native
[stream]`), so latencies stay attributable. Each scenario also emits custom
metrics:

- `overhead:<key>` — proxy-reported overhead (wall time minus upstream wait),
  parsed from `x-llm-proxy-overhead-duration-ms`.
- `ttft:<key>` — first content line for streaming scenarios.
- `stream_total:<key>` — full stream duration. For streamed requests locust's
  own `response_time` is the time to response headers.

Compare tiers by running two groups back to back at the same user count, e.g.
`LOADTEST_SCENARIOS=messages_anthropic_native_stream` vs
`LOADTEST_SCENARIOS=messages_openai_compat_stream`.

## Fake upstream knobs

| Env | Default | Effect |
| --- | --- | --- |
| `FAKE_RESPONSE_DELAY_MS` | `0` | Fixed delay before any response (simulate provider latency). |
| `FAKE_STREAM_CHUNKS` | `8` | Content chunks per streaming response. |
| `FAKE_STREAM_CHUNK_DELAY_MS` | `0` | Delay between stream chunks. |
| `FAKE_EMBED_DIMS` | `8` | Embedding vector length. |
| `FAKE_UPSTREAM_WORKERS` | `4` | Fake-upstream uvicorn workers. |

Set them on the `fake-upstream` service, e.g.
`FAKE_STREAM_CHUNKS=100 FAKE_STREAM_CHUNK_DELAY_MS=1`.

## Adding a scenario

Add a `ProviderSpec` or `(protocol, provider)` entry to `_EXPECTED_TIERS` in
`scenarios.py`; the scenario list, seed topology and smoke test all derive from
it. New wire dialects need matching handlers in `fake_upstream.py`.
