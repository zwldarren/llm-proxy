# 0058 — Batch mode is one OpenAI surface over three upstream dialects

## Status

Accepted (2026-11).

## Context

OpenAI, OpenRouter and Gemini all sell asynchronous batch inference at roughly
half the synchronous price, and each sells it over a different wire:

- **OpenAI** is file-based. The client uploads a JSONL file to `POST /v1/files`
  (`purpose=batch`), creates a job with `POST /v1/batches`
  (`input_file_id`, `endpoint`, `completion_window`), polls
  `GET /v1/batches/{id}`, and downloads `output_file_id` / `error_file_id`
  through `GET /v1/files/{id}/content`.
- **OpenRouter** is inline. `POST /v1/batches` carries
  `{endpoint, model, requests: [{custom_id, body}]}` and returns `202` with the
  job; `GET /v1/batches/{id}` returns the same object with `results` inline
  (one entry per request, matched on `custom_id`). There is no cancel endpoint
  and no results file.
- **Gemini** is an operation. `POST /models/{model}:batchGenerateContent` takes
  `{batch: {displayName, inputConfig: {requests: {requests: [{request,
  metadata}]}}}}` with inline `GenerateContentRequest` objects (the whole
  payload must stay under 20 MB) and returns a long-running operation named
  `batches/{id}`; `GET /batches/{id}` reports `metadata.state`
  (`BATCH_STATE_*`) and `metadata.batchStats`, and the finished responses arrive
  as `response.inlinedResponses.inlinedResponses[]` in input order.

Before this decision the proxy had no batch surface at all, and
`docs/admin/providers.md` told Gemini users to use `service_tier: "flex"`
instead. Meanwhile the sync path already owns a client dialect (the OpenAI
protocol) and per-provider dialects, so batch is the same two-sided translation
problem with a longer-lived job attached.

Three properties of the upstreams drive the design:

1. **Only OpenAI's batch takes a file.** Any proxy file abstraction that mirrors
   OpenAI's must therefore be the proxy's own storage, not a pass-through of an
   upstream file id.
2. **OpenRouter requires one `model` per batch, Gemini is one operation per
   model.** A batch mixing models cannot be forwarded unchanged to two of the
   three upstreams.
3. **Results arrive in three shapes** (JSONL file, inline `results`, inline
   `inlinedResponses`) with three different ways of identifying a request
   (OpenAI `custom_id` field, OpenRouter `custom_id`, Gemini request order).

## Decision

**One client surface: OpenAI's.** `/v1/files` and `/v1/batches` are served in
OpenAI's shapes, because that is the dialect clients and SDKs already know. A
batch's lines are `/v1/chat/completions` requests only; anything else is
rejected at creation with `400`.

**One provider seam.** `BatchCapabilityMixin`
(`providers/capabilities/batch.py`) declares four adapter methods — submit,
fetch, fetch results, cancel — returning proxy-normalized types
(`BatchSubmission`, `BatchSnapshot`, `BatchResultItem`) defined in
`llm_proxy/batch/types.py`. Each adapter translates its own dialect
(`providers/{openai,openrouter,gemini}/batch.py`); the batch service never
inspects an upstream payload. Adapters that do not include the mixin do not
support batch, and a batch aimed at one is refused by name.

**Translation only where dialects differ.** A class flag states which side of
the seam a provider sits on:

- `BATCH_FORWARDS_CHAT_BODY = True` (OpenAI, OpenRouter): the upstream's batch
  entry *is* an OpenAI chat-completions body, so the client's line is forwarded
  verbatim with only `model` rewritten to the resolved upstream model name.
  Round-tripping it through `InternalRequest` would drop passthrough fields
  (OpenRouter's `provider`, `plugins`, `usage`) for no gain.
- `BATCH_FORWARDS_CHAT_BODY = False` (Gemini): the line is parsed with the
  OpenAI protocol serializer, rebuilt by the provider serializer, and each
  response is parsed back to `InternalResponse` and formatted by the OpenAI
  protocol serializer — the same canonical path the synchronous request takes.

**One model per batch.** Creation resolves the provider from the line models and
requires them all to name the same configured model. This is a documented
restriction, not an accident: OpenRouter rejects a batch whose requests disagree
with the batch `model`, and Gemini binds the whole operation to one model.

**Provider choice is the synchronous choice.** The batch goes to the
highest-priority enabled provider configured for that model, with the same
upstream model-name mapping the sync path uses. If that provider has no batch
API the request fails with an explanatory `400` instead of silently demoting to
a lower-priority provider — a batch that costs a different price than the sync
path it was configured alongside would be a surprise, not a feature.

**Job state is the proxy's.** Two tables (`batch_files`, `batch_jobs`) persist
the uploaded JSONL, the proxy↔upstream id mapping, a normalized OpenAI status,
counts and timestamps. Statuses are normalized to OpenAI's enum, so clients do
not have to learn `BATCH_STATE_*`:

| Upstream | → proxy |
| --- | --- |
| OpenAI `validating` / `in_progress` / `finalizing` / `completed` / `failed` / `expired` / `cancelling` / `cancelled` | identity |
| OpenRouter `validating` / `in_progress` / `finalizing` / `completed` / `failed` / `expired` / `cancelled` | identity |
| Gemini `BATCH_STATE_PENDING` / `RUNNING` / `SUCCEEDED` / `FAILED` / `CANCELLED` / `EXPIRED` | `validating` / `in_progress` / `completed` / `failed` / `cancelled` / `expired` |

**Results become a proxy file, lazily.** The first `GET /v1/batches/{id}` that
sees a terminal status pulls the upstream results (OpenAI's output/error file
content, OpenRouter's inline `results`, Gemini's inline responses), converts each
entry to OpenAI's result-line shape
(`{id, custom_id, response: {status_code, request_id, body}, error}`), stores it
as the proxy's own output file and sets `output_file_id`. There is no background
poller: a job whose status nobody reads is never fetched, and a client polling
the id gets the results on the call that observes completion. `GET /v1/batches`
deliberately does not refresh from upstream — it reports the last observed state
so listing stays one query.

**Authorization follows the file, not the body.** The per-key model allowlist
middleware reads a ``model`` field from the request body, which a batch does not
have — its model lives inside the uploaded JSONL. Creation therefore re-checks the
allowlist once the file is parsed, and refuses with ``403 model_not_allowed``
instead of letting a restricted key batch a model it cannot call directly.

**Limits, stated rather than hidden.** The accepted JSONL is capped at 16 MiB
and 50,000 lines (Gemini's inline limit is 20 MB for the *built* request, which
is measured after translation); `stream: true` lines are rejected; cancellation
is refused with a clear error on OpenRouter, whose batch API has none.

**Spend is priced once, at the batch.** A batch has no request row of its own and
its results are read hours later, so the poll that materializes them prices every
line the upstream reported and writes a single `usage_records` row
(`log_type='batch'`, `request_id` = the batch id, timestamp = the batch's creation
instant). One row rather than one per line, because the client made one call and a
batch is one model by construction; each line is read through the same usage
dialect and priced through the same `calculate_cost`, so a per-line context tier is
applied to the line that reached it instead of to the batch total. `batch` is a
spend-only log type: it is in the request-count-free side of the aggregates but it
*is* in `BILLABLE_SPEND_LOG_TYPES`, so batch spend lands in budgets and per-key
spend without inflating request volume or latency averages. It lands after the
fact — a batch consumes budget from the next request on, not before it runs.

**Retention is a sweep, not a manual chore.** `batch_retention_days` (logging
config, default 30, `0` = forever) drives a process-lifetime sweeper that deletes
job rows past the window and then the files no remaining job names, in bounded
`SKIP LOCKED` batches, then reclaims the freed pages on SQLite. Jobs go first so a
retained job's input, output or error file is never deleted out from under it. The
upstream's copy is the provider's to clean up, not the proxy's.

## Consequences

- Clients keep the OpenAI batch SDK surface: upload, create, poll, download.
  The same client code works against all three upstreams; only the configured
  provider changes.
- The proxy owns storage. Input and output files accumulate in `batch_files`
  (content, not metadata), so they are pruned by their own sweep rather than by
  the log window: a batch payload is far larger than a log row and nobody re-reads
  it weeks later.
- A batch produces one `usage_records` row, not one per line: per-line request
  accounting would require expanding results, which contradicts "forward to the
  upstream and get the discount". Consequently batch spend is visible in the
  dashboard, budgets and per-key spend, but a batch is not attributable to an
  individual request (there is none).
- Model-level `parameter_overrides` are not applied on the verbatim path; the
  line body is the client's, minus the model name. Documented on the batch page.
- A future provider with a batch API joins by implementing four methods and
  declaring which side of the seam it is on; no change to the service, the
  router, or the tables.
