# One flag, two envelopes: serving Decisions alongside System One

## Status

Accepted (2026-10).

## Context

System One shipped first (ADR-0018, which consumes it as the routing judge).
Its endpoint is `/v1/systemone`, its wire format is TypeSafe's Jev — `state`
plus a map of typed questions, answering the primitives `noul` (a probability),
`choice` (one option out of a `criteria` map) and `score` (a point on an ordered
`criteria` array) — and three provider adapters speak it: `typesafe` (direct),
`openrouter` (TypeSafe's Jev resold) and `ollama` (local decision models,
v0.35+). Every one of them is the same upstream shape, so
`SystemOneCapabilityMixin` builds one body and parses one response for all
three.

OpenAI then shipped its own evaluation endpoint: `POST /v1/decisions`, model
`gpt-6-luna`, public beta at the time of writing. It answers the *same three
primitives* under different names and in a different envelope:

| | Decisions | System One |
| --- | --- | --- |
| Evidence | `input` — string, or user messages with `input_text`/`input_image` | `state` — string, object or array |
| Questions | ordered array, each carrying a `name` | map of id to question |
| Yes/no | `predicate` → `probability` | `noul` → `noul` |
| Pick one | `choices[{value, description}]`, values typed `str \| bool` | `criteria{value: description}`, keys are strings |
| Rate | `levels[{label, description}]` | `criteria[level, …]`, strings only |
| Answers | ordered array, each echoing `name` | map keyed by question id |
| Usage | `{input_tokens, input_tokens_details, output_tokens, output_tokens_details, total_tokens}` | `{input_tokens, output_tokens}`, plus OpenRouter's `cost` |

Nothing about this is a chat request: no conversation, no tools, no streaming,
and the upstream may decline one question (`refusal`) without disclosing a
score. The existing `InternalSystemOneRequest`/`Response` models are shaped like
the System One envelope — `state`, a `questions` map, an `answers` map — because
they were written for one wire format with three resellers, not as a neutral
vocabulary.

## Decision

**Add `/v1/decisions` as a protocol peer of `/v1/systemone`, and let every
decision model serve both endpoints by bridging at the provider.**

1. **A parallel internal model pair**, `InternalDecisionRequest` /
   `InternalDecisionResponse` (`llm_proxy.models.decisions`), shaped like the
   Decisions envelope the way the System One pair is shaped like its own. A new
   `RequestType.DECISIONS`, `DecisionsStrategy` and
   `DecisionsCapabilityMixin` mirror the System One ones. The System One path,
   including `call_systemone` and the routing judge, is untouched.

2. **One capability flag for both endpoints.** `supports_systemone` becomes the
   marker for "this is a decision model" rather than "this serves that one
   route". A model marked once serves `/v1/systemone` *and* `/v1/decisions`,
   whichever envelope its upstream actually speaks, because the provider
   translates.

3. **The bridge lives at the provider**, in
   `llm_proxy.models.decisions_bridge`, applied by two capability mixins:
   `DecisionsOverSystemOneMixin` (System One upstream, Decisions endpoint) and
   `SystemOneOverDecisionsMixin` (Decisions upstream, System One endpoint).
   `openai` takes the native `DecisionsCapabilityMixin` plus the second bridge;
   `typesafe`, `openrouter` and `ollama` take the first. The conversion is pure,
   takes both the request and the response, and is as lossless as the two
   envelopes allow.

4. **Bridging is delegated, not re-implemented.** The decisions direction reuses
   the adapter's System One transport — its endpoint resolution (including
   `endpoint_base_urls` overrides), retries, usage echo and rate-limit capture —
   through a route the provider can restate once, and the reverse calls
   `decisions()`.

## Consequences

- A Jev model answers `/v1/decisions` and a `gpt-6-luna` model answers
  `/v1/systemone`, so a client can pick the envelope it already writes and every
  decision model stays reachable. The routing judge gains the `openai` provider
  for free.
- Every lossy step is a fact one envelope has no field for, and is documented in
  the bridge module and in `docs/api/decisions.md`: a `predicate` has no rubric
  for a `noul` criteria map; score level descriptions move into the question
  text (labels stay put, because the client's labels are what the answer
  legend must echo); a structured `state` becomes JSON text; inline images ride
  Ollama's raw-base64 `images` field and are dropped by the upstreams that do
  not document it; `safety_identifier` and the OpenRouter-only fields have no
  counterpart in the other envelope.
- Two lossy steps would be silent data loss and are therefore rejected at the
  edge instead: question names must be unique (a duplicate collapses in a System
  One questions map), and choice option values must not collide once a boolean is
  written as its word (they would become one `criteria` key). `true` and
  `"true"` stay distinct options in both directions — the Decisions wire types
  them and the bridge restores the type by matching the request's own options.
- `openai_compatible` and its family deliberately gain neither endpoint. Their
  upstreams are third-party OpenAI-shaped surfaces whose evaluation route (if
  any) is unverifiable from here, and `supports_systemone` is a per-model
  operator claim that would silently become wrong if it also implied an envelope
  choice for a provider the proxy cannot inspect.
- OpenRouter serves the System One envelope on two routes: `/api/v1/systemone`,
  its compatibility alias, and `/api/alpha/decisions`, which its docs, SDKs and
  model catalog describe as the Decisions surface. The alpha one is deliberately
  *not* the default while it is alpha, so both proxy endpoints stay on the stable
  route and the two agree. The seam is still there for the day that route
  graduates: `/v1/decisions` resolves under its own `endpoint_base_urls.decisions`
  key, so a deployment can try the alpha route — a full URL, because it sits
  beside `/api/v1` rather than under it — without moving `/v1/systemone`, and
  `_systemone_decisions_url` is the one method to change to make it the default.
- The `systemone` protocol keeps its single canonical path while
  `openai_decisions` ships `openai`-family aliases (`/decisions`,
  `/v1/v1/decisions`), matching `openai` and `openresponses` rather than
  `systemone`. The evaluation endpoints are the family whose clients have
  `base_url` quirks; System One's existing path is left alone so nothing that
  works today changes behavior.
