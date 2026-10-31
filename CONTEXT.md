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

**Upstream session**:
The conversation-scoped affinity id a provider sends upstream — OpenCode's
`x-opencode-session`. It is stable across turns, retries, and workers,
resolved from the client's native session header, request metadata, or a
deterministic conversation fingerprint, and it is purely provider-scoped.
Do not confuse it with the request-fact `session_id`, which is the client's
log-correlation id for request logs and tracing; the upstream id is never
written to the request facts. See ADR-0052.
_Avoid_: session id (ambiguous on its own), session context, cache key

### Access control

**Key gate**:
The single gate every protected transport passes through before its traffic
proceeds: key extraction, verification (bcrypt against the cached key snapshot,
stale-entry cheap revalidation, session-key fallback), per-key rate limit, key
and user budget (fail-closed on a stats-DB error), lockout accounting, and the
throttled last-used claim. One module (`llm_proxy.api.gate`) owns the sequence;
each transport — the `/v1/*` middleware, the `/servers/*` MCP middleware, the
Realtime and OpenResponses WebSocket handshakes — declares a policy row (which
checks apply, how strictly a key is extracted) and renders rejections in its own
wire form. A rejection — auth failure or quota — is logged by the gate itself,
never by the transport. The verified principal is a fact of the request, not a
scope dict: identity goes through the request facts' `RequestIdentity`.
See ADR-0032.
_Avoid_: verify_api_key_for_mcp (the old misnomer for the shared verifier), a
per-transport hand-rolled gate sequence, `scope["llm_proxy_auth"]`

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
(`state` + `questions` map, kinds `noul`/`choice`/`score`, plus `box` for the
image-locating question only System One has) and Decisions (`input` + ordered
`questions`, kinds `predicate`/`choice`/`score`). A request names one
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

**Image channel**:
How one System One upstream carries an image, since the envelope has no image
field of its own: `images` (Ollama's shared base64 array), `state_parts`
(OpenRouter's `image_url` parts in `state`), or `assets` (TypeSafe's placeholder
map). Each adapter declares its own (`SYSTEMONE_IMAGE_CHANNEL`); the bridge
renders a Decisions request's images into that channel, and an adapter that
declares none receives the text alone — the hosted-image download is skipped for
it, since there is nothing to inline into. Where the upstream has a channel, a
hosted image URL is downloaded and inlined before the bridge runs, because no
System One upstream fetches one. See ADR-0045.
_Avoid_: image field, attachment (an image is not a first-class envelope field)

**Media download budget**:
The bounds on how much client-supplied media one request may make the proxy
fetch. `llm_proxy.http.media.RequestDownloads` is the single chokepoint — every
chat image/document URL and every bridged Decisions image goes through it. It
runs at most four downloads at once and charges each inlined payload to a
request-wide total (`HTTP_MAX_REQUEST_DOWNLOAD_BYTES`, 64 MiB), on top of the
per-URL SSRF check and size cap (`HTTP_MAX_DOWNLOAD_BYTES`). A failed download is
an outcome, not an exception, so the caller decides whether it fails the request
(bridged Decisions) or drops that one part (chat media). See ADR-0046.
_Avoid_: image cache, download pool (it bounds one request, it does not pool
anything)

### Retrieval endpoints

**Reranker**:
A model that scores a document list against a query and returns the documents
most relevant to it as an ordered `(index, relevance_score)` list, never generated
text. Reached at `/v1/rerank` in the shared Cohere-compatible wire format that
OpenRouter, Jina and the vLLM/SGLang `/rerank` route speak; `/v2/rerank` answers
the same handler as an alias because Cohere's official SDK targets that path.
Unlike an embedding
model, which maps each document to a vector independently, a reranker compares
the candidates jointly against the query. The capability flag `supports_rerank`
gates the endpoint. See ADR-0041.
_Avoid_: ranker, scoring model, embedding model (that is `/v1/embeddings`)

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
One independent in-process vote on the routing tier: metadata (A) or structural (B). The embedding signal (C) was removed with its heavy ML stack (ADR-0027). The ensemble combines the signals that did not abstain; a judge verdict is not a signal. See ADR-0018.
_Avoid_: classifier (that is signal B), heuristic, score

**Routing judge**:
A System One decision model the router consults when the signals are not confident. It answers typed questions about the request and returns calibrated probabilities, never text. Inert until an operator points it at a model marked as a System One model. Every consulted call is also logged as a row of its own (log type `judge`), attributed to the request it served. See ADR-0018.
_Avoid_: classifier, arbiter, LLM-as-a-judge, decision model

**Judge verdict**:
The judge's typed answers for one request: a tier choice with its distribution, and an escalation probability. An *ambiguous* answer is a verdict of no verdict.
_Avoid_: judge output, prediction

**Judge gate**:
The configured predicate that decides whether an eligible turn is worth asking the judge about: ensemble confidence below a threshold, complexity inside a half-open band, the shipped default `not_low_anchor` — any non-trivial placement, i.e. complexity away from the SIMPLE anchor 0.0 (ADR-0028) — or no predicate at all, which asks about every eligible turn. Eligibility itself is narrower: a first human turn, in a mode where the judge is enabled. The gate is the knob that sizes the judge's added latency and cost.
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

**Graceful degradation (routing)**:
The opt-in relaxation of two routing *preferences* — a virtual-model assignment that empties the pool, and the output half of the context window — so a request is served degraded (each relaxation recorded as a `degraded=…` reasoning note) instead of refused. Image capability, the allowlist and the `max_cost` budget are never relaxed: those are permissions and explicit caps, not preferences.
_Avoid_: fallback model (that is provider failover), loose routing

**Cascade (routing)**:
The proposed post-hoc escalation lane for `fast`: serve the cheap draft, score the `(ask, draft answer)` pair with a System One model, and re-serve from a stronger candidate only when the score says the draft is inadequate. Fail-open, shadow-first, latency-budget-decayed; both calls bill. See ADR-0033 (proposed).
_Avoid_: retry, failover, judge (it uses one, but the mechanism is the escalation lane)

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

**Segment accumulator**:
The one engine that owns how a streaming turn's segments — text, thinking
(with its signature and its redacted/encrypted variants), tool-call arguments
— interleave and close into canonical content blocks. Fed canonical deltas,
it exposes closed blocks incrementally (the web-search continuation reads
them before `finalize()`), and each protocol declares its own policy: which
block types exist, how blocks close, cursor semantics, indexing (flat vs
item-addressed), and what counts as a signature and where it lands. The
engine is the mechanism; the protocols stay the knowledge. Facts a previous
per-protocol copy dropped are decided once at the engine: no observation is
silently discarded (reasoning after text, signature-only segments, refusal,
thought signatures all reach the canonical blocks), and the wire-level rules
that legitimately differ per protocol (e.g. Anthropic wanting thinking first
on its wire) are declared where the wire is spoken.
_Avoid_: per-protocol accumulation state machines, text/thinking buffer
(those are the engine's private internals), a second hand-rolled
tool-args JSON fallback

### Conversion layer

**Provider serializer**:
The registered per-provider-key conversion class (`build_provider_request` / `parse_provider_response` / `get_chunk_converter`), living in `serialization/<family>/serializer.py` for the four dialect families (openai, anthropic, gemini, ollama). Adapters obtain it through the registry, never by direct import. Provider-local serializers stay next to their adapter: the small ones (chutes, nanogpt) and Cohere, whose native v2 vocabulary is a whole dialect but serves exactly one provider type (`providers/cohere/serializer.py`, ADR-0040).
_Avoid_: family serializer, provider converter

**Provider declaration**:
What a provider type *is*, before it has behaviour: the registry name (`@register_adapter`), the catalog branding (`DISPLAY_NAME_EN`/`_ZH`, `LOBE_ICON_ID`/`_VARIANT` — read only by `list_provider_types()`), `DEFAULT_BASE_URL`, and for the Chat-Completions family the upstream's wire facts — `native_protocols`, the Anthropic/Responses endpoint URLs (a fixed root, or one derived from the configured base URL), the `NATIVE_PASSTHROUGH_DEFAULT` opt-in and `DEFAULT_UNKNOWN_FIELDS_POLICY`. It lives in `providers/<type>/adapter.py`, next to any provider-local serializer; a type stays a pure declaration until it grows a method, at which point the method — and a test file for it — are the behaviour, and the declaration is unchanged. The shared half of the contract of the **declaration-only** Chat-Completions types is asserted once, from one row per type, in `tests/providers/test_declared_providers.py` (the local engines have their own table in `test_local_engine_adapters.py`, whose rows carry extra engine vocabulary). A type that never joins that family — a native-dialect behaviour provider such as Cohere, whose adapters are not Chat-Completions-shaped at all, or the OpenCode lanes with their per-model endpoints (ADR-0037) — cannot use a row there: its table asserts the Chat-Completions wire facts a row is made of. Such a type asserts its own declaration facts next to its behaviour, e.g. `tests/providers/cohere/test_cohere_adapter.py`.
_Avoid_: a per-provider test suite restating registration / base URL / branding / native protocols / endpoint derivation **for a declaration-only type** (the row is where they go); asserting branding by comparing the class attributes to themselves instead of reading the published catalog; a declaration-only provider split across more than one module; asserting an endpoint constant against itself — assert the URL a call actually resolves

**API variant** (Gemini):
The per-provider upstream dialect switch `metadata.api_variant` — `generate_content` (default, legacy) or `interactions` (Google's GA Interactions API, serializer at `serialization/gemini_interactions/`). The Gemini adapter picks the serializer and endpoint shape from it; embeddings/models are untouched. See ADR-0010.
_Avoid_: dialect, flavor, api_version

**Model family** (Gemini):
The `gemini-X.Y` version read off the model id, which decides the *request shape* the upstream accepts: `thinkingBudget` (2.5) vs `thinkingLevel` + `includeThoughts` (3.x, 4.x), the deprecated OpenAPI-subset `responseSchema`/`parameters` vs plain JSON Schema `responseFormat`/`parametersJsonSchema`, `includeServerSideToolInvocations` for built-in + function combinations (3.x on), and whether sampling parameters still reach the wire (ignored from 3.6 on, and on 3.5 Flash-Lite). One capability table, `serialization/gemini/model_family.py`, keyed by the version; gates are stated "from version V on", so a new family inherits the modern shape instead of falling into a legacy else-branch. An unparseable model id gets the legacy-safe answer (logged once per id); an operator can pin any id to a family via the provider's `metadata.model_family_overrides`, which wins over the id's own version. See ADR-0039.
_Avoid_: substring checks on the model id per call site (they are what made Gemini 4 lose its thinking config and its function-call signatures); silent legacy fallbacks (the unparseable-id fallback must be visible in the log, not discovered via upstream 400s)

**Wireless context** (Gemini):
Provider facts the client's protocol has no field for, so a stateless proxy must cache them and re-attach them to the next request: the built-in tools' `toolCall`/`toolResponse` parts (keyed by the function-call id they preceded), a `thoughtSignature` issued on a *text* part (keyed by the exact text, the only handle the client echoes), and the function-call `thoughtSignature` itself. All three are process-lifetime LRU caches on the Gemini adapter (`_tool_context_cache`, `_text_signature_cache`, `_thought_signature_cache`), written on every response path and read before the request is built (`_enrich_conversation_with_thought_signatures`). A cache miss degrades to the previous behaviour, never to a wrong field. See ADR-0039.
_Avoid_: a content block per provider-specific part (the raw parts must replay in order, with their signature siblings)

**Model dialect** (OpenCode):
The per-request upstream wire dialect of the OpenCode gateway adapters (`opencode-zen` / `opencode-go` — the two lanes) — which of Chat Completions / Anthropic Messages / OpenAI Responses / Google generateContent the *model* is served on, a property of the model (the gateway fronts many upstream families), not of the provider type. Resolved from, in order: a process-wide learned correction (a dialect the gateway rejected for that model), the models.dev catalog (`provider.npm` per model, 6h TTL cache; the same id legitimately speaks different dialects on the Console and Go lanes), then model-id prefix heuristics. Native passthrough is gated on it (protocol AND dialect must match), a rejected dialect walks a fallback chain that records the winner, and the adapter's own class-level endpoint predicates stay Chat-Completions-shaped. See ADR-0037.
_Avoid_: per-model provider, protocol inference (the client protocol is not what is inferred), dialect table (there is no static one)

**Canonical usage record**:
`Usage` expresses each billable fact in exactly one field — cache read → `cache_read_input_tokens`, cache write → `cache_creation_input_tokens`, thinking → `reasoning_tokens`. Dialect aliases are normalized at parse time; the alias fields do not exist on the canonical record. It holds wire-reported facts only: token counts, `web_search_requests`, `audio_duration_seconds`, `provider_reported_cost`. Proxy-computed dimensions (`images_generated`, `tts_characters`, cost) belong to billing, not to the record.
The request's capture object holds one such record (`EventContext.usage`) rather than a field per fact, so a number cannot be stored twice and read from whichever copy a caller happens to know about; it is written only through `update_usage`, which folds each observation with the record's own merge rule. Two billable dimensions keep a field of their own there because they have a second route: the provider-reported cost (also a provider-info extra) and the web-search count (the searches the proxy executed itself add to the provider's own).
_Avoid_: cached_content_tokens, thoughts_tokens (deleted provider-flavored aliases), usage mirrors on the capture object

**Usage intake**:
The single point where a provider's wire payload becomes the canonical usage record:
`parse_usage(dialect, payload)` plus a per-dialect accumulator, in
`llm_proxy.serialization.usage` — one dialect adapter per wire shape
(chat-completions, responses, anthropic, realtime, ollama, gemini
`usageMetadata`, cohere's `ApiMeta` envelope — `tokens` + `billed_units` + `cached_tokens`,
seen at `usage`, `delta.usage` and `meta`, ADR-0040 — the shared `/rerank` block
(`total_tokens` + `search_units` + `cost`, whose count is prompt-side because a
reranker generates nothing, and whose `search_units` is a client-echo fact with no
canonical field — ADR-0041 — audio transcription/translation), keyed by shape and not
by provider, because one dialect serves many providers and the native-passthrough
tier has no provider serializer in play. Every tier reads usage through it:
provider serializers, streaming converters, protocol transformers and the
passthrough bookkeeping, which hold an accumulator and expose its `result()`
behind their existing `get_usage()` verb. Extraction only — canonical → client
wire stays with the protocol serializer. An adapter may also need a fact about the
turn that the payload cannot carry (gemini's `search_grounded`: grounding sits on the
candidate, and a usage-only stream chunk carries none) — the caller that saw the turn
states it in `turn`, a fact the payload states itself wins, and a running total is a
merge the accumulator already owns rather than a turn fact. Until a dialect's copies are deleted,
`tests/test_serialization/test_usage_parity_ratchet.py` holds the seam and each
still-live call site in agreement on the billable *amounts* (not the field
placement), one row per dialect and call site; a row that declares a divergence
must name the fact and the reason, and a fact a slice deliberately repaired keeps
its name in a `closed` entry beside the amount the repair produces. The dialects
above are the migrated ones — the wire shapes that still parse usage by hand (the
Gemini Interactions vocabulary, the gpt-image branch of the image stream tracker)
follow the same landing order; a provider whose usage block *is*
an existing dialect declares that dialect rather than adding a provider-named key
(Decisions speaks the responses shape). An emission site may read the
record instead of re-deriving the amounts (Ollama's chunk converter builds the
client's OpenAI-dialect usage object that way), but it still owns the shape it
emits. See ADR-0025.
_Avoid_: usage parser, token counter, per-tier usage extraction, provider-registry usage adapter

**Web-search continuation**:
The loop that injects proxy-executed search results and re-calls the provider when a streamed turn ends waiting on `web_search` results. Owned by `WebSearchStreamProcessor` (`core/processing/web_search_streaming.py`): result processing, continuation request building, the loop itself (`generate_continuation`), state hand-off (`ContinuationState`), and usage merge (`merge_continuation_usage`, delegating to the transformer's public `merge_terminal_state` verb). A protocol transformer takes part by declaring `supports_web_search_continuation`; the results, the usage frame and the next hop's transformer are then its own public verbs (`web_search_result_block`, `usage_delta`, `continuation`, `adopt_continuation_state`), so the loop names no private member and no protocol's `.state`. See ADR-0007.
_Avoid_: continuation logic in streaming_processor (that was the pre-ADR-0007 arrangement); getattr probes into transformer `_pending_*` / `_current_block_index`, `hasattr` probes for the result-block / usage-delta emitters or the `continuation` factory, and reaching into a transformer's `.state` bag (the pending terminal state is `PendingTerminalState` in `streaming/transformer.py`, reached via public verbs)

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
The usage shape carried by `response.done` — top-level `input_tokens`/`output_tokens` plus nested `input_token_details`/`output_token_details` (singular "token"). One of the wire shapes Usage intake parses; the same token fact is never counted twice.
_Avoid_: realtime usage format, audio usage

**Realtime close code**:
The WebSocket close code a Realtime connection ends with, from the endpoint's own table (`api/routers/realtime.py`). Two codes follow the official OpenAI Realtime scheme (4000-4009 client errors, 4100-4108 server errors) where a semantic match exists (4004 invalid model, 4007 rate limited); the rest are proxy conventions in the RFC 6455 private-use range, matching the OpenResponses WebSocket transport (4401 auth failure, 4403 forbidden, 1011 upstream/provider failure) — the official 4005 invalid-authentication and 4100-4108 server-error codes are intentionally not used so both proxy WS transports share one close-code language. The reason always precedes the close as a Realtime `error` event.
_Avoid_: reusing HTTP status codes as close codes

### Server settings

**Server settings**:
One UI-managed `server_config` key — logging, web search, smart routing, provider selection, request policy, resilience, security, keepalive, rate limits, CORS origins, MCP security policy — declared once as a descriptor in the server settings registry (`llm_proxy.config.server_settings`). The descriptor carries the key's canonical model (the runtime truth, living in `llm_proxy.config.types`), its write merge mode, its response mode, and its runtime read. Three consumers are generated from the registry: the admin endpoints (`api/routers/config/server.py`), the config manager's typed snapshot (`load_server_settings`), and the contract test. Adding a setting is adding a descriptor row. See ADR-0026.
_Avoid_: hand-written per-key config endpoints, a second schema module mirroring the runtime types (`api/schemas/server_config.py` is deleted)

### Request logging

**Logged response body**:
The response-side content stored on a request-log row. For a streaming request it is the protocol-native non-streaming body: reassembled from the streaming transformer's accumulated content blocks on the converted tiers, or from the native passthrough frames when the client's protocol and the provider speak the same wire format (`native_frame_accumulation` — accumulated Chat Completions chunks, rebuilt Anthropic blocks, or the Responses terminal snapshot). Raw SSE text is stored only when raw capture is on (`log_raw_stream`, or `x-log-full: true` for one request). See ADR-0015.
_Avoid_: streaming body, raw stream (that is the opt-in raw form), SSE log

The reassembled body also carries the provider extras a non-streaming response would: the lifecycle reads the transformer's `get_terminal_provider_info()` verb before `finalize()` clears the pending state and hands the result to the protocol formatter as `provider_info`, so beta terminal fields (`stop_sequence`, `stop_details`, `container`, `diagnostics`) are not lost to streaming.

**Log retention window**:
The single UI-managed retention setting (`logging.retention_days`, Settings → Log Management) that governs `request_logs`, the audit rows that inherit it, and `usage_records`; `0` keeps rows indefinitely. See ADR-0016.
_Avoid_: usage retention, per-table retention

**Log intake**:
The verbs that turn a request situation into stored records: `llm_proxy.observability.log_intake`, one verb per situation (endpoint lifecycle from `EventContext`, early failure from the raw request, admin request, admin action, rejection, internal call, tool call, realtime turn). The verbs own classification, identity, the server/service stamps, and which records a situation produces (an early failure writes a usage row too); they are idempotent, so the `audit_log_written` dedup latch is their private detail (a field of the request's facts, ADR-0022). The three endpoint-lifecycle verbs finalize the capture and emit one observation to the database bus instead of assembling a row; the remaining situations build theirs through the store's assembler. See ADR-0020 and ADR-0044.
_Avoid_: building a log row at a call site, writing `audit_log_written` outside the intake

**Log store plumbing**:
`llm_proxy.observability.store` — what the two record stores share and nobody above them should re-implement: `assemble_row` (the single `RequestLogCreate` construction point, incl. the deployment constants and the timestamp default), `build_log_metadata`, `build_usage_record`, `token_column`, `stream_body`, `dispatch_row` to the background writers, and the live `logging_config()` / `configure()` resolution. It sits below both the intake verbs and the sinks so the observation seam stays acyclic (ADR-0044); it reads no `EventContext`.
_Avoid_: a verb or a sink constructing a row itself, resolving the logging config on its own

**Observation**:
One finished request, captured once as an immutable record (`llm_proxy.observability.observation.model.RequestObservation`): core identity, who made it, which model/provider served it, usage/cost, timing, outcome, the captured content, and the request's content policy. The collector (`observation/collector.py`) is the only reader of `EventContext`; a per-sink policy projection narrows what a sink may see; the bus (`observation/bus.py`) fans the result out to sinks with a single `emit()`. The request-log and usage stores are the first two sinks (`observation/database.py`). The Langfuse exporter (`sinks/langfuse.py`) also reads only the observation plus the typed wire request/response; its handler (`tracing/handlers/providers/langfuse/handler.py`) is a thin lifecycle adapter because Langfuse needs an open span for the trace-id header and for TTFT. Content rules — the body switch, masking, the size cap, sampling — live once in `observation/policy.py`, so no sink can bypass the deployment's privacy choice. See ADR-0044.
_Avoid_: log row (that is the request-log sink's projection), trace event (that is one backend's payload), reading `EventContext` in a sink

**Content registry**:
`llm_proxy.observability.content.registry` — what each request type carries, rendered for a backend: the per-type input/output pair (`build_request_input_data`, `build_rerank_input_data`, …), the bounded text/list/vector previews, and `OBSERVATION_PROFILES`, which maps a `RequestType` to its pair and to the observation type a billed backend records it as (`generation`, or `embedding` for embeddings). Renderers take the typed internal request/response, because the per-type questions they answer (rerank documents, attachment summaries, tool-call extraction) are not modelled by the raw JSON. A backend's wire shape — usage buckets, cost details, metadata — stays in the backend's sink/exporter.
_Avoid_: attributes, SDK builders (those are one backend's shape), rendering from `EventContext`

### Billing

**Billing read model**:
What billing prices from: `token_usage_from(usage, dimensions)` in `llm_proxy.billing.tokens` projects the turn's canonical record onto billing's columns (`TokenUsage`), and `ProxyBillableDimensions` carries the quantities the proxy itself produced — generated images, synthesized characters, executed searches — which are not provider facts and so not the record's. One projection serves both the cost calculation and the `usage_records` row, so what the dashboard shows and what was charged cannot drift apart. `TokenUsage` is the row/pricing shape, not a second record of the wire facts.
_Avoid_: usage dict, dialect tolerance in billing (that lives in the intake adapters), pricing from `EventContext` fields

**Context pricing tier**:
A price band that applies once a request's input token count reaches its threshold: the whole request is billed at the band's rates, and any dimension the band leaves unset inherits the base rate. Stored per model and per provider mapping (`pricing_tiers` JSON), edited in the model form, and synced from models.dev `cost.tiers`; thresholds are inclusive and must be unique. See ADR-0014.
_Avoid_: long-context surcharge, tier pricing (ambiguous with routing quality tiers), marginal overage pricing
