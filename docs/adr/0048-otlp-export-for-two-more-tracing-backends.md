# ADR-0048: Two more tracing backends behind one OTLP exporter

## Status

Accepted (2026-10). `otel` and `langsmith` join `langfuse` as built-in tracing
providers; all three are per-user and share one lifecycle adapter.

## Context

ADR-0044 settled the shape of a tracing backend: a *live exporter* that reads the
immutable `RequestObservation` and talks to the SDK, driven by a thin lifecycle
adapter, because a span must be open before the response headers are written (the
trace id they echo) and TTFT must be recorded while the stream still runs. Only
Langfuse implemented it, and an earlier generic OTLP handler had been removed when
tracing was consolidated to Langfuse (`7f0aa52`, "remove OTLP provider and
consolidate to Langfuse-only tracing").

Two gaps remained for operators:

- **No vendor-neutral destination.** A deployment that runs Tempo, Jaeger, Grafana
  or Phoenix had no way to send traces there, although the proxy already ships the
  OpenTelemetry SDK and spans.
- **No LangSmith.** LangSmith exposes an OTLP/HTTP ingest gateway, so it is reachable
  with the SDK the proxy already has — but its run model is not the GenAI semconv: it
  ingests **traces only, over HTTP only**, reads the pre-semconv `gen_ai.system`
  spelling and flattened `gen_ai.prompt.*`/`gen_ai.completion.*` attributes, routes by
  `x-api-key`/`Langsmith-Project` headers, and drops attributes it does not map.

Tracing is strictly per user (`observability/user_tracing.py`): a user's backend sees
only that user's requests and its endpoint is user-supplied, so the URL is an SSRF
surface on the same footing as a provider `base_url`.

## Decision

**One OTLP exporter, two provider handlers, one lifecycle adapter.**

1. **`observability/sinks/otlp.py` owns every OTel SDK call.** It builds a
   per-handler `TracerProvider` (never the process-wide one: two users exporting to
   different collectors must not cross-export), supports OTLP **HTTP/protobuf and
   gRPC**, and shapes one span per request. `observability/sinks/send_content.py` is
   the one place that resolves a backend's `send_content` against the request's
   `ContentPolicy`; Langfuse and both OTLP exporters share it.

2. **`langsmith` is not a second SDK.** It is the OpenTelemetry handler with a fixed
   HTTP transport and LangSmith's credential headers
   (`LangsmithTracingHandler(OtlpTracingHandler)`), plus a second attribute
   vocabulary behind the exporter's `backend` flag. This avoids a new dependency and
   keeps one payload model; the rejected alternative was the official `langsmith`
   SDK, which would add a second exporter, a second content projection and ~7
   transitive packages for the same data.

3. **Attributes follow the GenAI semantic conventions.** The span is `CLIENT`, named
   `{gen_ai.operation.name} {model}`, and carries `gen_ai.request.model`,
   `gen_ai.response.model`, `gen_ai.request.*` sampling parameters, and
   `gen_ai.usage.input_tokens`/`output_tokens`/`total_tokens` plus the cache and
   reasoning details. Usage counters stay **inclusive**, as the conventions define
   them, unlike Langfuse's exclusive `usage_details` buckets. Projected content is
   `input.value`/`output.value` (OpenInference spelling); the proxy's own facts are
   `llm_proxy.*`. The LangSmith backend adds the spellings its ingest reads:
   `gen_ai.system`, flattened `gen_ai.prompt.*`/`gen_ai.completion.*`,
   `langsmith.trace.session_id`, `langsmith.span.tags`, `langsmith.metadata.*`.

4. **Content policy is unchanged.** `send_content` defaults to `full` for every
   backend (matching the Langfuse default restored in ADR-0044's amendment),
   `inherit` follows `log_input_output`/`mask_sensitive_data`, and `none` withholds
   every payload while usage, cost and metadata still export. A backend never widens
   the request's mode.

5. **The shared lifecycle adapter is `ExporterBackedHandler`.** All three providers
   implement the same six hooks by delegating to the exporter, so the streaming
   ordering that ADR-0044 fixed (usage folded before export; TTFT once per stream;
   `release` flushes without tearing the backend down and schedules the close
   after a grace period, so a released registry cannot strand a provider's
   batching thread and export client) exists once. Per-request state
   is keyed by request id on the exporter, not held in a `ContextVar`, because the
   streaming hooks run in a different asyncio task than the pipeline.

6. **A sampled-out span publishes no trace id.** Sampling is a per-handler
   `TraceIdRatioBased`; the response header must not point at a trace nobody will
   ever receive, so the handler answers `None` for a non-recording span. A
   caller-supplied canonical 32-hex `x-trace-id` (or `x-langfuse-trace-id`) is adopted
   as the span's parent context, so the proxy's span continues the caller's trace.
   **LangSmith is the exception:** its ingest only materializes a trace whose root
   span arrives, so `langsmith` starts every request as its own root run rather
   than parent it to a caller run that LangSmith never receives. The correlation id
   travels as `llm_proxy.correlation_id` / `langsmith.metadata.correlation_id`
   instead; `otel` keeps adopting the caller's trace id.

7. **Endpoints are validated like any other outbound URL.** `endpoint` joins
   `base_url` in `_validate_tracing_provider_urls`: format-only for an admin (who can
   already configure providers), `validate_server_url(..., fail_closed=True)` for
   everyone else, with `HTTP_ALLOWED_HOSTS` still the escape hatch. Endpoints must
   include a scheme for both transports (`https://collector:4317`), which is also what
   lets the gRPC exporter infer TLS.

## Consequences

- No new Python dependency; `otel` works against any OTLP collector and `langsmith`
  against the gateway, both on the SDK already pinned in `pyproject.toml`.
- LangSmith's OTLP ingest is HTTP-only and trace-only. An operator who wants gRPC to
  LangSmith must run a collector; the settings sheet and docs say so. Unmapped
  attributes are dropped by LangSmith, which is why the legacy flattened message
  attributes are emitted deliberately rather than relying on the semconv ones.
- Multi-part content (the semconv `parts` shape) reaches LangSmith as a literal JSON
  string in a message's `content`; the proxy does not reshape it.
- The OTLP exporters own their own HTTP clients, so they are covered by the
  save-time URL check but not by the connect-time resolver (the limitation ADR-0047
  records for the MCP, judge/flywheel and Langfuse clients).
- Content attributes are bounded by the content registry's previews and a last-resort
  16 KiB cap per attribute, so one pathological payload cannot push a span past the
  collector's message limit.
- A future OTLP-shaped backend needs only a handler subclass (transport, headers,
  metadata); a non-OTLP backend implements the exporter interface and reuses
  `ExporterBackedHandler`, which is the seam ADR-0044 was written to provide.
