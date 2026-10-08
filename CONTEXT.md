# LLM Proxy

Multi-protocol LLM API proxy: clients speak a Protocol, upstreams are Providers, and a unified pipeline converts between the two.

## Language

### Request flow

**Protocol**:
A client-facing API format the proxy accepts (OpenAI Chat, Anthropic Messages, OpenResponses).
_Avoid_: format, frontend API

**Provider**:
An upstream LLM API the proxy forwards requests to (OpenAI, Anthropic, Gemini, Ollama, …).
_Avoid_: backend, upstream service

**Service** (process-lifetime):
A subsystem constructed once per worker by the lifespan and shared by every request —
the config manager, the pooled HTTP client, Redis, the circuit-breaker and provider-stats
stores, the MCP manager, the per-protocol processors, the web-search runtime. Reach it
through `llm_proxy.services` (`runtime_services(request).config_manager()`); never by
probing `app.state` for an attribute name. See ADR-0021.
_Avoid_: app state, global, singleton, dependency (that is a FastAPI `Depends`)

**Request facts**:
What the proxy learns about one in-flight request, on the record: its
`request_id`, `identity`, `allowed_models`, the routing facts (`model`,
`provider`, `session_id`), the parsed request body, and the two latches
(`client_disconnected`, `audit_log_written`) plus the audit capture buffers.
One `request.state` slot, owned by `llm_proxy.core.request_facts`: read and
write it through `facts_for(request)` / `facts_from_scope(scope)` — never by
getting or setting a `request.state` attribute name. `RequestIdentity` is a
field of the record, reached through `get_request_identity`. See ADR-0022.
_Avoid_: request state, request context (that is `RequestContext`, the processing
carrier), session context

### Evaluation endpoints

**Decision model**:
A model that answers typed questions with calibrated probabilities instead of
prose. The proxy reaches two families of them: System One (TypeSafe's Jev, also
resold by OpenRouter and served locally by Ollama) and OpenAI's Decisions
(`gpt-6-luna`). One capability flag, `supports_systemone`, marks a model as a
decision model, which gates both evaluation endpoints and being selected as the
routing judge. See ADR-0019.
_Avoid_: classifier (that is routing signal B), judge (that is the router's
consumer of one), decision endpoint (that is the route, not the model)

**Envelope**:
One of the two wire formats over the same three evaluation primitives. System One
(`state` + `questions` map, kinds `noul`/`choice`/`score`) and Decisions (`input`
+ ordered `questions`, kinds `predicate`/`choice`/`score`). A request names one
by its endpoint — `/v1/systemone` or `/v1/decisions`. See ADR-0019.
_Avoid_: protocol, dialect, format (it is a payload shape, not a client-facing
API)

**Envelope bridge**:
The provider-side conversion that lets an upstream speaking one envelope answer
the other's endpoint: `llm_proxy.models.decisions_bridge`, applied by the
`DecisionsOverSystemOneMixin` / `SystemOneOverDecisionsMixin` capability mixins.
As lossless as the two envelopes allow; each dropped fact is marked in the
module docstring and in `docs/api/decisions.md`. Each evaluation endpoint resolves
under its own route, so `/v1/decisions` can reach an upstream route that
`/v1/systemone` does not — one upstream route can serve both, or two can, without
the code differing.
_Avoid_: adapter translation, protocol conversion (that is the chat path)

### Smart routing

**Virtual model**:
A model name (`auto`, `fast`, `best`) the proxy resolves to a concrete configured model by routing instead of forwarding. Each carries an intent — `fast` is cost-pressed, `best` is quality-pressed — and virtual models are chat-only.
_Avoid_: router model, alias, pseudo-model

**Routing complexity**:
The router's continuous estimate of how hard a request is to answer, mapped by `_derive_tier` onto the routing tier it is served from. It describes the request; a model's capability is its served quality lane.
_Avoid_: difficulty score, prompt score

**Routing tier**:
The band a request's routing complexity falls into — `SIMPLE`, `MEDIUM` or `COMPLEX`. A different axis from a model's served quality lane (`economy`/`balanced`/`premium`), which describes the model, and from a billing tier (ADR-0014), which is a price band.
_Avoid_: tier (unqualified), model tier, quality tier

**Routing signal**:
One independent in-process vote on the routing tier: metadata (A), structural (B), or embedding (C). The ensemble combines the signals that did not abstain; a judge verdict is not a signal. See ADR-0018.
_Avoid_: classifier (that is signal B), heuristic, score

**Routing judge**:
A System One decision model the router consults when the signals are not confident. It answers typed questions about the request and returns calibrated probabilities, never text. Inert until an operator points it at a model marked as a System One model. Every consulted call is also logged as a row of its own (log type `judge`), attributed to the request it served. See ADR-0018.
_Avoid_: classifier, arbiter, LLM-as-a-judge, decision model

**Judge verdict**:
The judge's typed answers for one request: a tier choice with its distribution, and an escalation probability. An *ambiguous* answer is a verdict of no verdict.
_Avoid_: judge output, prediction

**Judge gate**:
The configured predicate that decides whether an eligible turn is worth asking the judge about: ensemble confidence below a threshold, complexity inside a band, or no predicate at all — which asks about every eligible turn and is what the seed corpus measured best. Eligibility itself is narrower: a first human turn, in a mode where the judge is enabled. The gate is the knob that sizes the judge's added latency and cost.
_Avoid_: trigger, threshold

**Judge abstention**:
The judge declining to decide, whether by answering *ambiguous* or by not answering at all (deadline, failure, open circuit). The ensemble decides the tier in every abstention case.
_Avoid_: fallback model, failure

**Judge shadow**:
Running the judge and recording its verdict without letting that verdict own the tier — the state a judge ships in first, so agreement with the ensemble can be measured on live traffic before anything depends on it. A shadow window also samples a small share of turns the gate did *not* fire, because agreement measured only inside the gate cannot show whether the gate itself is leaking ambiguous turns.
_Avoid_: dry run, test mode, canary

**Judge context**:
The bounded excerpt of the request the judge receives as its `state`: the current ask, a capped number of prior user turns, and structured capability flags. Its content is data, never instruction.
_Avoid_: prompt, summary

**Judge deadline**:
The wall-clock budget a single judge call gets before the router gives up and treats it as an abstention.
_Avoid_: timeout (unqualified)

### OpenResponses protocol module

**OpenResponses protocol module**:
The deep module at `llm_proxy.protocols.openresponses` that owns everything about the OpenResponses protocol: endpoint registration, request/response conversion, streaming, and store=true persistence. Its public interface is four verbs — parse, format, replay, materialize.
_Avoid_: openresponses handler, openresponses serializer (those are internals)

**Replay**:
The conversion of stored Responses items back into unified conversation messages (used for previous_response_id continuations and item_reference resolution). Public entry: `replay_stored_response`.
_Avoid_: materialize (that's the opposite direction), prepend

**Materialize**:
The conversion of the unified conversation into Responses input items (used for store=true persistence). Public entry: `conversation_to_input_items`.
_Avoid_: replay (that's the opposite direction), serialize input

**Responses toolkit**:
The shared module at `llm_proxy.serialization.responses_toolkit` holding Responses-shaped helpers (tool-name namespaces, reasoning extraction, item conversion) used by the OpenResponses protocol module and by provider-family serializers.
_Avoid_: shared utils, openresponses helpers

### Conversion tiers

**Conversion plan**:
The single verdict for how a chat request reaches the upstream, computed by `plan_conversion(adapter, request, context=None)` in `llm_proxy.core.conversion` from adapter capability (`native_protocols`, `allows_native_request`), serializer capability (`compatible_protocols`, carried by `BuildContext`), request flags (`native_request_disabled`, `previous_response_materialized`), and stash presence. Three independent fields — `request_tier`, `stream_mode`, `response_mode` — because the sides legitimately disagree (rebuilt request + native stream). Tiers: `NATIVE_PASSTHROUGH`, `WIRE_REUSE`, `FULL_CONVERSION`. `response_mode` is three-valued like `request_tier`: a non-native request on a wire-compatible provider still gets a verbatim (wire-reuse) response — raw body plus the two load-bearing transforms (reasoning-field rename, model aliasing), with usage parsed for billing and the reasoning cache written from the wire shape. Kill switch: provider metadata `response_passthrough: false`. See ADR-0011, ADR-0012.
_Avoid_: passthrough check, fast-path gate, per-adapter native branch

**Native passthrough**:
Forwarding a request body and/or response stream to the upstream verbatim, because client protocol and provider API are wire-identical. One of the three tiers in the Conversion plan; body preparation is `prepare_native_body` in `llm_proxy.core.conversion` (fresh copy, top-level `None` strip, routed-model substitution, stream flag) plus per-family repairs behind `BaseAdapter.native_body_hook`. Adapters never prepare native bodies themselves.
_Avoid_: native request, passthrough mode, raw forwarding, passthrough body builder

**Wire-compatible rebuild shortcut**:
The WIRE_REUSE tier: the client's stashed raw body is reused instead of a full rebuild, but `model`/`stream` are rewritten and top-level `None` fields stripped. Decided by the Conversion plan (serializer declares `compatible_protocols` as data) and prepared by `prepare_wire_reuse_body` in `llm_proxy.core.conversion`, which returns a deep-copied body fully detached from the stash. Not native passthrough — the field policy and post-build repairs (reasoning echo) still run.
_Avoid_: passthrough, fast path passthrough, serializer fast path

**Tier-independent forced field**:
A field the proxy writes onto the upstream body regardless of which tier produced it (currently `stream_options.include_usage`, ADR-0008/ADR-0017). Tier-independent by construction, so it is enforced on a path every tier converges on — `OpenAICompatibleBase._force_include_usage`, reached from `_build_request_body` — never inside one tier's builder: a forced field added to a single tier is silently dropped on the raw-reuse tiers, which is the billing-estimation bug ADR-0017 records. The seam deliberately does not apply the policy (`prepare_wire_reuse_body` is raw-body preparation and stays protocol-agnostic, since `prepare_native_body` also serves Anthropic/Responses bodies where the field does not exist). Guarded twice: `TestTierIndependentStreamUsage` (per field, across the tier × stream matrix) and `TestTierFieldParity` (the raw-reuse and rebuild tiers must expose the same top-level field set, with the legitimate exceptions declared and checked for exactness).
_Avoid_: forced param, default field, global field policy

**Reasoning-field preference**:
The per-`(base_url, model)` learned cache of which assistant reasoning field the upstream expects (`reasoning` vs `reasoning_content`), held by `OpenAIRequestBuilder` with a TTL/LRU bound and a model-less fallback read. Every response path teaches the model before the client-facing rename — parsed non-stream (`OpenAIResponseParser`), verbatim wire-reuse, and streaming chunks (`_stream_transform_chunk`) — keyed by routed model plus upstream-reported model (aliasing), via the single shared write `record_reasoning_field_preference`. Request-side normalization and the reasoning-echo placeholder resolve the field per body model; never-seen models default to `reasoning_content`. See ADR-0013.
_Avoid_: reasoning convention, per-base_url reasoning cache, detect reasoning field

**Stream prefetch**:
The leading items the proxy reads from a provider stream before the first byte
reaches the client, parsed only far enough to see the first user-visible content
and any in-band fallback signal (a context-length or retryable finish reason)
that arrived before it. One loop owns the stop rule and the finish-reason policy
(`prefetch_stream` in `core/processing/stream_prefetch.py`); a per-tier decoder
owns what an item means — `ConvertedStreamDecoder` re-encodes chunks through the
protocol transformer, `NATIVE_BLOCKS` replays raw SSE blocks verbatim. An item
that carries a signal is neither replayed nor counted as content. Tiers the proxy
does not read ahead (`PrefetchResult.without_prefetch`) report
`stream_started=True`. See ADR-0024.
_Avoid_: peek, sniff, read-ahead

### Conversion layer

**Provider serializer**:
The registered per-provider-key conversion class (`build_provider_request` / `parse_provider_response` / `get_chunk_converter`), living in `serialization/<family>/serializer.py` for the four dialect families (openai, anthropic, gemini, ollama). Adapters obtain it through the registry, never by direct import. Small provider-specific serializers (chutes, nanogpt) stay provider-local.
_Avoid_: family serializer, provider converter

**API variant** (Gemini):
The per-provider upstream dialect switch `metadata.api_variant` — `generate_content` (default, legacy) or `interactions` (Google's GA Interactions API, serializer at `serialization/gemini_interactions/`). The Gemini adapter picks the serializer and endpoint shape from it; embeddings/models are untouched. See ADR-0010.
_Avoid_: dialect, flavor, api_version

**Canonical usage record**:
`Usage` / `StreamingUsage` express each billable fact in exactly one field — cache read → `cache_read_input_tokens`, cache write → `cache_creation_input_tokens`, thinking → `reasoning_tokens`. Provider serializers normalize dialect aliases at parse time; the alias fields do not exist on the canonical record. Billing reads canonical fields only (`extract_tokens_from_usage` tolerates the OpenAI-dialect nested expression as a fallback, never alongside the flat field).
_Avoid_: cached_content_tokens, thoughts_tokens (deleted provider-flavored aliases)

**Web-search continuation**:
The loop that injects proxy-executed search results and re-calls the provider when a streamed turn ends waiting on `web_search` results. Owned by `WebSearchStreamProcessor` (`core/processing/web_search_streaming.py`): result processing, continuation request building, the loop itself (`generate_continuation`), state hand-off (`ContinuationState`), and usage merge (`merge_continuation_usage`, delegating to the transformer's public `merge_terminal_state` verb).
_Avoid_: continuation logic in streaming_processor (that was the pre-ADR-0007 arrangement); getattr probes into transformer `_pending_*` / `_current_block_index` (the pending terminal state is `PendingTerminalState` in `streaming/transformer.py`, reached via public verbs)

**Fallback re-parse**:
Each provider fallback attempt re-parses from the pristine client body (`PipelineState.original_raw_data` / `fallback_raw_data`), re-applies its own parameter overrides, and re-runs the per-provider request stages (`rerun_per_provider_stages` in `core/processing/stages/composition.py`, consuming the single composition owner `create_per_provider_stages` shared with `UnifiedProcessor._stages`) so a failed provider's overrides and stage decisions never leak into the next attempt.
_Avoid_: fallback reusing the failed provider's mutated request; a second hand-maintained per-provider stage list (composition owner: `stages/composition.py`)

### Realtime relay

**Realtime relay**:
The transparent bidirectional WebSocket relay for the OpenAI Realtime API (`WS /v1/realtime?model=…`). The Realtime protocol is a long-lived two-way event stream (audio + text) that cannot be expressed through the request/response pipeline, so the proxy authenticates the client with its own API keys, resolves the model to a provider, opens a WebSocket to the provider's native Realtime endpoint, and pumps messages verbatim in both directions. Owned by `llm_proxy.realtime` (relay, upstream connection, usage observer) plus the endpoint in `api/routers/realtime.py`.
_Avoid_: realtime proxy, realtime passthrough, realtime handler

**Realtime turn**:
One model response within a Realtime session, delimited by the upstream `response.done` event. Each completed turn is written as one background request log entry (endpoint `/v1/realtime`, method `WS`) with the turn's usage and cost, so the dashboard shows per-call billing for realtime sessions.
_Avoid_: realtime request, realtime message

**Realtime usage dialect**:
The usage shape carried by `response.done` — top-level `input_tokens`/`output_tokens` plus nested `input_token_details`/`output_token_details` (singular "token"). `extract_tokens_from_usage` treats it as a fallback dialect alongside the chat (`prompt_tokens_details`) and Responses (`input_tokens_details`) shapes; the same token fact is never counted twice.
_Avoid_: realtime usage format, audio usage

**Realtime close code**:
The WebSocket close code a Realtime connection ends with, from the endpoint's own table (`api/routers/realtime.py`). Two codes follow the official OpenAI Realtime scheme (4000-4009 client errors, 4100-4108 server errors) where a semantic match exists (4004 invalid model, 4007 rate limited); the rest are proxy conventions in the RFC 6455 private-use range, matching the OpenResponses WebSocket transport (4401 auth failure, 4403 forbidden, 1011 upstream/provider failure) — the official 4005 invalid-authentication and 4100-4108 server-error codes are intentionally not used so both proxy WS transports share one close-code language. The reason always precedes the close as a Realtime `error` event.
_Avoid_: reusing HTTP status codes as close codes

### Request logging

**Logged response body**:
The response-side content stored on a request-log row. For a streaming request it is the protocol-native non-streaming body: reassembled from the streaming transformer's accumulated content blocks on the converted tiers, or from the native passthrough frames when the client's protocol and the provider speak the same wire format (`native_frame_accumulation` — accumulated Chat Completions chunks, rebuilt Anthropic blocks, or the Responses terminal snapshot). Raw SSE text is stored only when raw capture is on (`log_raw_stream`, or `x-log-full: true` for one request). See ADR-0015.
_Avoid_: streaming body, raw stream (that is the opt-in raw form), SSE log

The reassembled body also carries the provider extras a non-streaming response would: the lifecycle reads the transformer's `get_terminal_provider_info()` verb before `finalize()` clears the pending state and hands the result to the protocol formatter as `provider_info`, so beta terminal fields (`stop_sequence`, `stop_details`, `container`, `diagnostics`) are not lost to streaming.

**Log retention window**:
The single UI-managed retention setting (`logging.retention_days`, Settings → Log Management) that governs `request_logs`, the audit rows that inherit it, and `usage_records`; `0` keeps rows indefinitely. See ADR-0016.
_Avoid_: usage retention, per-table retention

**Log intake**:
The single point where a request's facts become a stored log row: `llm_proxy.observability.log_intake`, one verb per situation (endpoint lifecycle from `EventContext`, early failure from the raw request, admin request, admin action, rejection, internal call, tool call, realtime turn). The verbs own classification, identity, masking, hostname, dispatch to the background writers, and which records a situation produces (an early failure writes a usage row too); they are idempotent, so the `audit_log_written` dedup latch is their private detail (a field of the request's facts, ADR-0022). `RequestLogCreate` is the module's assembly detail, never built at a call site. See ADR-0020.
_Avoid_: building a log row at a call site, writing `audit_log_written` outside the intake

### Billing

**Context pricing tier**:
A price band that applies once a request's input token count reaches its threshold: the whole request is billed at the band's rates, and any dimension the band leaves unset inherits the base rate. Stored per model and per provider mapping (`pricing_tiers` JSON), edited in the model form, and synced from models.dev `cost.tiers`; thresholds are inclusive and must be unique. See ADR-0014.
_Avoid_: long-context surcharge, tier pricing (ambiguous with routing quality tiers), marginal overage pricing
