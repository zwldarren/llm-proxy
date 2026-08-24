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

**Reasoning-field preference**:
The per-`(base_url, model)` learned cache of which assistant reasoning field the upstream expects (`reasoning` vs `reasoning_content`), held by `OpenAIRequestBuilder` with a TTL/LRU bound and a model-less fallback read. Every response path teaches the model before the client-facing rename — parsed non-stream (`OpenAIResponseParser`), verbatim wire-reuse, and streaming chunks (`_stream_transform_chunk`) — keyed by routed model plus upstream-reported model (aliasing), via the single shared write `record_reasoning_field_preference`. Request-side normalization and the reasoning-echo placeholder resolve the field per body model; never-seen models default to `reasoning_content`. See ADR-0013.
_Avoid_: reasoning convention, per-base_url reasoning cache, detect reasoning field

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
The loop that injects proxy-executed search results and re-calls the provider when a streamed turn ends waiting on `web_search` results. Owned by `WebSearchStreamProcessor` (`core/processing/web_search_streaming.py`): result processing, continuation request building, the loop itself (`generate_continuation`), state hand-off (`ContinuationState`), and usage merge (`merge_continuation_usage`).
_Avoid_: continuation logic in streaming_processor (that was the pre-ADR-0007 arrangement)

**Fallback re-parse**:
Each provider fallback attempt re-parses from the pristine client body (`PipelineState.original_raw_data` / `fallback_raw_data`), re-applies its own parameter overrides, and re-runs the per-provider request stages (`_rerun_per_provider_stages` in `core/processing/fallback.py`) so a failed provider's overrides and stage decisions never leak into the next attempt.
_Avoid_: fallback reusing the failed provider's mutated request

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
