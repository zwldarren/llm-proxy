# ADR-0043: Wire-contract tests for every protocol's response bodies

## Status

Accepted.

## Context

ADR-0042 pinned the OpenResponses streaming snapshot to its non-streaming body
and, one layer down, to the published Open Responses schema. The other
protocols had no equivalent: OpenAI Chat Completions and Anthropic Messages were
checked by hand-written expected dicts, and the JSON endpoints with no streaming
twin (embeddings, images, audio transcription/translation) had no response-shape
check at all.

Hand-written expectations encode whatever the implementation produced on the day
they were written. A field the published schema requires can disappear from the
builder and the test keeps passing, because the expectation was edited in the
same commit. That is exactly how the OpenResponses terminal snapshot drifted for
three separate usage facts (ADR-0042) before anyone noticed.

## Decision

**Validate emitted bodies against the published schema, not against a copy of
our own output.**

- `tests/protocols/schema_validation.py` is the single validator. It walks
  `$ref`, `anyOf`/`oneOf` (picking the branch with the fewest problems),
  `enum`/`const`, `type`, `required`, `properties`, `items` and
  `additionalProperties`.
- `tests/protocols/fixtures/trim_schema.py` produces the fixtures: it parses the
  published document (YAML or JSON), keeps the `$ref` closure of named root
  schemas **verbatim**, and records the source URL and commit. The fixtures are
  `openai_schema_contract.json` (33 schemas, from `openai-openapi`) and
  `anthropic_schema_contract.json` (82 schemas, from the Stainless "documented"
  Claude spec the Anthropic SDKs are generated from). Hand-editing a fixture is
  forbidden; regeneration is one command.
- `test_openai_wire_contract.py` covers `CreateChatCompletionResponse`,
  `CreateChatCompletionStreamResponse`, `CreateEmbeddingResponse`,
  `ImagesResponse` (generations and edits), `CreateTranscriptionResponseJson`,
  `CreateTranscriptionResponseVerboseJson` and `CreateTranslationResponseJson`.
- `test_anthropic_wire_contract.py` covers `Message`, `MessageStreamEvent` (every
  event body, plus the nested `Message` in `message_start` and
  `MessageDeltaUsage`), `CountMessageTokensResponse` and `ErrorResponse`.
- `test_openresponses_wire_contract.py` now uses the same validator.

**A required field with an explicit `default` may be absent.** Both published
documents mark fields `required` *and* give them a `default` (usually `null`);
the code generated from them supplies the default, so a body omitting the key is
not rejected by a client. Fields `required` without a default must be present.
Without this rule the validator demands keys no client needs and the tests would
push fabricated values onto the wire.

**One Anthropic usage builder.** `protocols/anthropic/usage_payload.py` is the
only place an Anthropic `usage` object is assembled: the non-streaming
`Message`, the streaming `message_start` and the terminal `message_delta` all
call it. It owns two Anthropic rules: `input_tokens` is exclusive of the cache
counters, and every counter the schema declares (`cache_creation`,
`inference_geo`, `server_tool_use`, `service_tier`, `output_tokens_details`, …)
is emitted — `null` when the provider did not report it. This is the same
structural cure ADR-0042 applied to the OpenResponses snapshot: three hand-built
copies was the way drift happened.

Following the established `stop_sequence: null` precedent, required-nullable
envelope fields are emitted with `null` rather than omitted: `Message`'s
`stop_details`/`container`, `MessageDelta`'s same two, a text block's
`citations`, and `Usage`/`MessageDeltaUsage`'s optional counters.

## What the contracts found immediately

- **OpenAI Chat Completions**: the non-streaming choice omitted `logprobs`
  (the spec defaults it to `null`, but real OpenAI sends the key on every
  choice). Now emitted.
- **Anthropic `Message`**: `stop_details`, `container`, `usage.cache_creation`,
  `usage.inference_geo`, `usage.server_tool_use`, `usage.service_tier` and the
  text block's `citations` were omitted.
- **Anthropic stream**: `message_start.message` omitted the same Message/Usage
  fields; `message_delta.usage` omitted `server_tool_use`; the terminal delta
  omitted `stop_details`/`container`; text `content_block_start` omitted
  `citations`.
- **Anthropic `caller` is not nullable** (its default is `{"type": "direct"}`),
  so it is emitted only when the provider reported one — a fabricated `null`
  would be a type violation.

## Rejected alternatives

- **Vendor the full OpenAPI documents** (4 MB each) — the fixtures are 51 KB and
  76 KB, and nothing either way auto-syncs; the trim script makes regeneration
  cheap without committing two large snapshots.
- **Validate against generated SDK types** — language-specific, and they carry
  the SDK's own defaults rather than the wire contract.
- **Keep hand-written expectations alongside the schema check** — the schema
  check is the guard; hand-written assertions are kept only for the specific
  fields this proxy's clients read (so a spec change that quietly makes our body
  pass while dropping a fact those clients use still fails).
- **Emit a fabricated value for every required-nullable field** — `caller` shows
  why not: some required fields are not nullable and their default is
  meaningful. Unknown nullable facts get `null`; fields with a non-null default
  are omitted and the client supplies the default.
- **Weaken the validator to ignore `required`** — the `logprobs` case would have
  gone unnoticed, and a dropped `id`/`object`/`choices` would too.

## Consequences

- Adding a field to an emitted body is cheap; removing or renaming one a schema
  declares required without a default fails a test that never mentions the
  field.
- **Not covered**: `/v1/audio/speech` returns binary audio (the serializer
  produces an internal envelope, not JSON); streamed transcription
  (`transcript.text.*`) is upstream SSE forwarded verbatim, so the upstream, not
  the proxy, is the authority on that body.
- The published OpenAI spec lags the live API for
  `CreateTranscriptionResponseVerboseJson.usage` (duration-only, while
  token-billed transcribers return `type: "tokens"` in verbose_json). The test
  validates the body against the root schema minus `usage`, then validates the
  emitted usage against `TranscriptTextUsageTokens`. The lag is documented in
  the test rather than papered over.
- Fixtures are schema snapshots. A spec release is not picked up until someone
  re-runs `trim_schema.py`; the recorded commit makes that visible.
