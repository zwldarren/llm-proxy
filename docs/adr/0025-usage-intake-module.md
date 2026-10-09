# One usage intake owns every provider wire dialect

## Status

Accepted (2026-10).

## Context

Every billable fact the proxy records starts as a provider wire payload, and
each tier parses it on its own. Usage is extracted at roughly a dozen sites
across four layers:

- **provider serializers** — `serialization/openai/components/response_parser.py`
  (`parse_usage`), `serialization/openai/serializer.py`
  (`parse_usage_from_response`), the anthropic, gemini and ollama serializers;
- **streaming converters** — `serialization/openai/streaming_converter.py`
  (`_handle_response_finished`), `serialization/anthropic/streaming_converter.py`;
- **protocol transformers** — `protocols/openai/streaming.py` (`get_usage`),
  `protocols/anthropic/streaming.py`, `protocols/openresponses/serializer.py`
  (`_convert_usage`);
- **the native-passthrough bookkeeping** — `core/conversion.py` hand-parses three
  dialects (`_apply_openresponses_usage`, `process_anthropic_usage_event`,
  `_capture_openai_stream_usage`), including the Anthropic input-token + cache
  addition rule.

Three records keep the same facts in lockstep by hand — `Usage`
(`models/types.py`), `StreamingUsage` (`streaming/transformer.py`, a
near-duplicate carrying a dict-shaped details field and `provider_reported_cost`),
and billing's `TokenUsage` (`billing/tokens.py`), which `extract_tokens_from_usage`
fills from a raw dict by tolerating dialects. `EventContext` mirrors ~13 further
flat usage fields, and the cost path runs
`EventContext.to_usage_dict()` → `extract_tokens_from_usage()` → `TokenUsage` →
cost (`observability/cost.py:96`, `billing/tokens.py:66`, `billing/cost.py:398`) —
a dict-shaped detour between two records of identical facts.

The copies have drifted into shipped billing bugs: ADR-0008 and ADR-0017 record
"usage lost on one tier" fixes; `0acd0b4ea` fixed the Anthropic
input-tokens-exclusive-of-cache contract a second time; `5ab1c53cd` stopped
double-counted tokens; `8999150cf` deduped cache-read tokens across dialect
columns. The merge rule — how a later chunk's usage combines with an earlier one —
is implemented four times (`sum_usage_dicts`, `merge_terminal_state`,
`sum_usage`, `merge_continuation_usage`).

## Decision

**One canonical record.** `Usage` holds wire-reported facts only: token counts,
cache read/write, reasoning tokens, `web_search_requests`,
`audio_duration_seconds`, and the absorbed `provider_reported_cost`.
Proxy-computed dimensions (`images_generated`, `tts_characters`, computed cost)
stay in billing, which derives its read model from a `Usage`. Canonical → client
wire stays with the protocol serializer that owns that wire (ADR-0009) — the
Anthropic exclusive-of-cache emission rule included.

**One module.** `llm_proxy.serialization.usage` is keyed by wire shape
(chat-completions, responses, anthropic, realtime), not by provider: one dialect
serves many providers, and the native-passthrough tier has no provider serializer
in play. Its interface is `parse_usage(dialect, payload) -> Usage` and
`UsageAccumulator(dialect)` with `observe(raw_chunk)` and `result()`. The
accumulator owns both the per-chunk parse and the merge; transformers hold one
and expose its result behind their existing `get_usage()` verb.

**Extraction only.** The module never emits a client-facing shape, and it never
estimates: `count_tokens` and the other estimation helpers answer a different
question from request text and stay where they are.

**`result()` returns provider semantics.** Dialect aliases are folded into
canonical field names — one field per fact, ADR-0006's contract — but values are
as the provider reported them. `input_tokens` stays inclusive of the cache
counters; making it exclusive is an emission rule.

**The dialect is declared once per conversion path**: provider serializers expose
it as a capability, protocol transformers declare it for the native tier. No call
site re-decides it.

**EventContext carries the record**, `usage: Usage | None`, instead of the ~13
flat mirror fields; the flat fields and `to_usage_dict()` are deleted.

## Considered Options

- **Record constructors** (`Usage.from_anthropic(payload)`) instead of dialect
  adapters: rejected — it welds wire-dialect knowledge onto the canonical record
  and makes every new dialect an edit to the record.
- **A capability on the provider-serializer registry** instead of a dialect-keyed
  module: rejected — a dialect is a wire shape, not a provider capability, and the
  native tier has no serializer to ask.
- **Sniffing the payload shape** when no dialect is given: rejected — chat and
  responses payloads overlap, so a wrong guess silently mis-bills.
- **A bidirectional module** owning client emission too: rejected — it duplicates
  the protocol serializers' job and blurs the two registries of ADR-0009.
- **Absorbing token counting and estimation**: rejected — estimation has no wire
  payload and would double the interface for a second audience.
- **Converging the parse but keeping the three records**: rejected — the lockstep
  copies are the bug source, not the parsing alone.

## Consequences

- Adding one billable fact means teaching one dialect adapter, not twelve sites.
  `TestTierIndependentStreamUsage` and `TestTierFieldParity`
  (`tests/core/test_conversion_tiers.py`) keep the tiers' *request* bodies aligned;
  `tests/test_serialization/test_usage_parity_ratchet.py` is the ratchet for the
  migration itself: one row per (dialect, still-live call site), comparing the
  seam and the shipped path on the billable **amounts** rather than the field
  placement, so a fact expressed in a flatter field still passes while a changed
  amount fails. A row that declares a divergence must name the fact and the
  reason, and the declaration is checked for exactness in both directions. The
  ratchet earned its keep before the first slice landed: it caught the Responses
  body parser billing zero searches whenever a usage block accompanied the
  web-search items, because the adapter looked for a usage-level count the
  Responses wire never sends. A migrated row keeps its driver, so it still
  compares live code, *and* carries the amounts the shipped parser produced before
  the swap (`recorded`): without that anchor the two sides share the seam and move
  together, and the row would only prove the projection self-consistent.
- The ratchet distinguishes two ways a migration can move a fact. ``declared`` is a
  *live* difference between the seam and a call site (checked for exactness in both
  directions, so it cannot outlive the gap it describes). ``closed`` records a gap a
  landed slice has already repaired: the fact stays pinned by the live comparison,
  and ``recorded`` carries the amount the repair produces rather than the lossy
  pre-swap one — a repaired amount is a deliberate change, so it should be visible
  in the row instead of silently moving.
- Two shipped gaps the ratchet declared were closed by the anthropic slice:
  `AnthropicChunkConverter.get_usage` and `process_anthropic_usage_event` read only
  the input/output/cache counters, so Anthropic's
  `output_tokens_details.thinking_tokens` never reached the recorded usage
  (observability only — reasoning is priced inside the output count).
- The anthropic slice also repaired three facts the Anthropic *protocol*
  transformer's parser dropped when it served a converted OpenAI-family provider,
  each a billing difference: the provider's own cost, the web-search count (the
  transformer read `server_tool_use` for client emission only), and DeepSeek's
  `prompt_cache_hit_tokens` (priced at the full input rate instead of the cache
  rate). The same slice migrated the non-streaming Anthropic body parser, the
  fourth anthropic call site; it was already reporting the counts correctly, but
  where it had returned the search count through the `provider_info` extras the
  record now carries it too. That transformer reads canonical chat-dialect chunks,
  so it uses the chat adapter — a slice is a set of call sites, not a naming
  coincidence.
- The responses slice closed two more search-count gaps, both on native paths that
  read only a usage block: the passthrough handler now receives the whole terminal
  event, so a provider-executed `web_search_call` in the snapshot's output items is
  billed exactly as the non-streaming body parser already billed it, and the
  OpenResponses protocol transformer's Responses branch does the same. Neither
  tier runs the converted tier's web-search continuation, so those calls had no
  other counter.
- One count deliberately stays outside the seam: a provider-executed search that
  an upstream announces as a *tool call* in the chunk stream. The intake module
  reads usage payloads and a chunk's tool call says nothing about usage, so the
  OpenResponses transformer keeps its own counter and folds it into the record
  with the monotonic rule (the larger wins — an upstream that reports both the
  tool call and a usage-level count describes the same calls). This is the same
  split ADR-0006 draws elsewhere: proxy-computed dimensions stay with billing.
- The OpenResponses protocol transformer is the one call site whose dialect is not
  fixed: it is handed a converted upstream's canonical chat chunks or a
  Responses-shaped event, depending on the chunk. It holds one accumulator per
  dialect and merges the two on read, rather than sniffing the payload — a
  transformer instance sees one upstream shape in practice, and the merge (not a
  guess) is what makes the impossible case harmless.
- The seam normalizes an emptied nested block to `None`: once the cache fact is
  lifted into the flat field, a details object with nothing left is not `None` to a
  caller testing truthiness, and downstream emission would have read it as "has
  nested facts".
- The gemini `usageMetadata` dialect is the one whose mapping is a *heuristic* rather
  than a rename: `toolUsePromptTokenCount` is billed as input except when the turn
  used Google Search grounding (the search fee covers it), and `thoughtsTokenCount`
  is added to the output count only when the block's own total proves
  `candidatesTokenCount` excludes it. Two of the three call sites that read the block
  skipped both rules and took the raw counters — the image-generation body parser
  and the streaming image tracker — so the same response billed one way through the
  chat path and another through the image paths. The seam is now the single reader of
  the shape, and the two are repaired: on a block reporting 100 prompt + 5 tool-use +
  20 candidates + 7 thoughts + 30 cached, the image paths used to record 100/20 with
  no cache or thinking fact, and now record 105/27 with both.
- Extraction therefore has *two* inputs, and the seam says so: `turn`, a mapping of
  facts about the turn that the payload cannot carry. Gemini's
  `search_grounded` is the first — grounding lives on the candidate, so a usage-only
  stream chunk carries no statement about it, while the converter that saw the
  earlier chunk does. A fact the payload states itself wins; the dialects that need
  none ignore the argument. The alternative was to let the seam guess from the
  payload alone and accept an over-bill whenever grounding arrived in an earlier
  chunk, or to move the exclusion into pricing (which would need the tool-use token
  count as a new canonical field and a new rule on the bill) — a fact the caller
  holds is cheaper to pass than to reconstruct.
- A turn-level *count* stays the caller's counter, as it already did for the
  searches a provider announces as tool calls: the gemini converter ratchets the
  queries it sees and folds them into the record on read, with the monotonic rule,
  so a later chunk cannot withdraw a count and a usage chunk that arrives before the
  grounding still records it. The turn-facts channel carries the boolean the adapter
  needs to *decide* the input count; it does not carry running totals, which are
  merges the accumulator already owns.
- One call site turned out to speak a dialect that already existed. Decisions
  reports its usage as the Responses block: ``input_tokens`` / ``output_tokens``
  with the cache and thinking counts nested under ``input_tokens_details`` /
  ``output_tokens_details``, and no ``output`` items for the adapter's search count
  to find. Its parser therefore declares ``responses`` rather than registering a
  ``decisions`` alias, because the registry is keyed by wire *shape*: a
  provider-named key would make the key mean "provider flavour" and leave two
  implementations of one mapping to drift apart. The one fact this moved is the
  provider's own cost, which the shipped parser routed to ``provider_info``
  (``openrouter_cost``) instead of the record; the amount billed is unchanged.
- The dialect set starts with the four that carry the most copies — chat,
  responses, anthropic, realtime — and gains one per slice after that (ollama,
  which parses `prompt_eval_count`/`eval_count` and clamps the
  `prompt_eval_cached_count` subset into the input total). The wire shapes that
  still parse usage locally (gemini `usageMetadata`, the Interactions vocabulary,
  the SystemOne/Decisions evaluation wires, and the audio/transcription/image
  stream trackers) register adapters as they migrate; the registry grows, the
  interface does not. Ollama also shows the split between extraction and
  emission holding up in the small: its chunk converter still builds the client's
  OpenAI-dialect usage object, including the native `ollama_metrics` block that is
  observability rather than billing, but the amounts now come from the seam
  instead of a second pair of counter readers.
- The accumulator's merge rule — accounting counters never decrease, every
  other field takes the later non-``None`` value — supersedes the three rules it
  replaces. It matches the monotonic requirement the Anthropic path already
  encoded; it differs from `PendingTerminalState.fold_usage` only where a later
  event reports a *smaller* non-zero counter, which no supported wire does.
  `total_tokens` is the one exception to the monotonic rule: it is recomputed
  from the merged counters, because a dialect with no wire total (Anthropic)
  reports a per-event total that is only that event's partial sum, and
  max-merging those leaves a total the merged counters contradict.
  `tests/core/test_streaming_cache_fields.py` pins the cache-placement behaviour
  the streaming tiers must keep as they migrate. The same monotonic rule governs
  the EventContext projection, which is the native tier's route into the record:
  a sparse or repeated streaming frame adds facts but can never withdraw one.
- Billing's dict-shaped input disappears: cost reads a `Usage` directly and the
  `to_usage_dict()` detour goes with it.
- `StreamingUsage` and the dialect tolerance inside `extract_tokens_from_usage`
  are retired, so their importers migrate; the accumulator becomes the one place
  the streaming merge rule lives.
- The module sits under `serialization/`, so it may not import provider or
  protocol packages. Dialects are data, not registry entries.
- One placement change came with the chat slice: the OpenAI protocol transformer's
  streaming usage used to keep a cache read that only the nested
  `prompt_tokens_details.cached_tokens` expressed *as nested*, beside a flat zero,
  while the non-streaming parser lifted the same fact into
  `cache_read_input_tokens`. The streaming path now lifts it too — in the Anthropic
  protocol transformer as well — so a streamed and a non-streamed turn record the
  cache read in the same field. The amount is unchanged either way, which is why the
  ratchet passes on both sides of the swap and only the placement pins had to be
  rewritten.
- `StreamingUsage.from_usage` was the one projection of the canonical record onto
  the wire-shaped record the streaming tiers then spoke. Every migrated
  `get_usage()` returned it, so when the record itself was retired the bridge had a
  single implementation and disappeared from one place. Its only non-mechanical step
  moved the canonical flat `reasoning_tokens` into the nested completion detail,
  which was the only expression the old record had; the record now keeps the flat
  field, and client emission folds it back.
- `EventContext.update_usage` derives `total_tokens` from its merged counters when
  both are known, keeps a total-only record's own total, and lets a partial record
  leave the total alone: an adapter synthesizes a per-event total from whatever one
  event carried (Anthropic's `message_delta` reports the output only), so a turn
  whose prompt the provider never reports records the output as its total rather
  than no total. Unreachable on the Anthropic wire, whose `message_start` always
  reports the prompt.
- The request's capture object holds the record. `EventContext.usage` is the single
  home for every fact a provider reported, written only by `update_usage`, which
  copies the observation in and folds it with `merge_latest`; the thirteen flat
  mirrors and `to_usage_dict()` are gone, so a fact can no longer be stored twice
  and read from whichever copy a caller happened to know about. Two billable
  dimensions keep a field of their own on the context because they have a second
  route: `provider_reported_cost` also arrives as a provider-info extra, and
  `web_search_requests` also counts the searches the proxy executed itself (which
  add to the provider's own count in the record).
- Billing's read model is built from the record. `token_usage_from(usage,
  dimensions)` replaced `extract_tokens_from_usage`, which read one dict holding
  four dialects at once; the "is the cache inside the input count?" guesswork went
  with it, because the record's input count already includes the cache counters by
  the canonical invariant. The two cache columns survive, one dialect expression
  each, with the flat field winning — the rule the `1d85a77da088` dedup migration
  already encodes for stored rows. `ProxyBillableDimensions` carries what the proxy
  itself produced (generated images, synthesized characters, executed searches).
- `StreamingUsage` is retired: every `get_usage()` returns the record. The
  projection bridge (`from_usage`) and the inverse projection the context would
  otherwise have needed both disappeared with it. The three converters whose
  dialects have not had their slice (gemini, ollama, gemini_interactions) build the
  record directly and still parse by hand; their slices move that parse behind the
  seam and add their ratchet rows.
- Thinking tokens are the canonical flat field in every dialect. The chat and
  responses adapters lift the nested `completion_tokens_details.reasoning_tokens`
  (and SGLang's top-level alias for the same fact) into `reasoning_tokens` and clear
  the alias, as CONTEXT.md's canonical record requires — before this, two migrated
  dialects were still expressing one fact twice. Client emission folds it back into
  the dialect's own object (the OpenAI chat formatter and the Realtime audio
  serializer; OpenResponses already merged both), so no client wire changed.
- `sum_usage`, a hand-rolled duplicate of `add_usage`, is gone; the non-streaming
  web-search continuation calls the seam's rule. The streaming continuation's token
  sum still lives only in the client-wire pending usage, which is a pre-existing gap
  this slice does not close.
- A migrated row's live side now compares the record with itself, so its `recorded`
  anchor carries the comparison; the ratchet's docstring says so, and every row
  without an anchor was given one.

- Landing order: seam and adapters first, then one dialect migrated at a time,
  each commit deleting the copies it replaces. No release leaves two live parse
  paths for a fact the parity tests cannot see. Slice 1 (the native
  chat-completions usage capture in `core/conversion.py`) has landed: it parses
  through the chat adapter and projects through `EventContext.update_usage`.
  Slice 2 has landed: the OpenAI protocol transformer's `get_usage` no longer
  parses the frame — a `UsageAccumulator("chat")` observes the chunks and the
  method projects the canonical record. The verbatim wire copy the client is served
  (`_pending_usage`) is untouched, since client emission stays with the protocol
  serializer (ADR-0009). Slice 3 has landed: the anthropic dialect's four call
  sites (`AnthropicChunkConverter.get_usage`, `parse_usage_and_provider_extras`,
  `process_anthropic_usage_event`, and the native passthrough's
  `maybe_capture_native_streaming_usage`) plus the Anthropic protocol transformer
  (`UsageAccumulator("chat")`), with both `reasoning` gaps closed. Slice 4 has
  landed: the responses dialect's four call sites
  (`parse_usage_from_response`, `OpenAIResponsesChunkConverter.get_usage`,
  `_apply_openresponses_usage`, and the OpenResponses protocol transformer), the
  body parser losing its caller-supplied `web_search_call_count` parameter — the
  count is now the adapter's own reading of the body's items, so no caller can
  hand the seam a different number. The record slice has landed with it: the
  capture object holds the record, billing reads it, and `StreamingUsage` is gone,
  so the remaining provider-native dialects (gemini, ollama, the Interactions
  vocabulary, the audio/image stream trackers, the Decisions shape) have only their
  own parse left to move.
