# A budget rejects spending requests, not reads

## Status

Accepted (2026-11).

## Context

The `/v1/*` transport declared one `GatePolicy` for every path under its prefix
(ADR-0032), with `budget=True`. The gate therefore ran the key- and
account-level spend checks on `GET /v1/models` — and on every other read:
stored responses, response input items, files, file contents, batch listings.

Two consequences surfaced as one report: "configuring a budget on an API key
breaks my OpenCode provider; removing the budget fixes it."

- The model list is the first call a client makes. An over-budget key got a 429
  `budget_exceeded` there, and an OpenCode client that cannot resolve a model
  falls back to its own built-in default — it never sends the request the
  operator configured, so the provider looks broken rather than budget-blocked.
- The console's model picker and the console's own reads use the same paths, so
  an operator with an exhausted key could not even list what the key may call.

A budget caps *spend*. A `GET` creates no usage record: `usage_records` rows are
written by the pipeline for endpoint, judge and batch calls, and every one of
those is a `POST` (a batch's *results* are read, the batch itself is created and
billed on creation). Rejecting a read therefore enforces a cap against money that
cannot be spent, and it removes the client's only way to learn what the cap still
allows.

## Decision

**The `/v1/*` transport declares two policy rows: one for requests that can
spend, and a read-only variant with the budget step off.**

- `_READ_ONLY_METHODS = {"GET", "HEAD"}`, declared beside `V1_GATE_POLICY` in
  `api/middleware/api_key_auth.py`, selects `V1_READ_GATE_POLICY`
  (`replace(V1_GATE_POLICY, budget=False)`). Every other step — key
  verification, per-key RPM, IP lockout, the last-used claim, identity stamping,
  rejection rows — runs unchanged, so a read is still authenticated and still
  rate limited.
- The distinction is the *method*, not a path allowlist. A path list would have
  to be maintained against every new read route (files, batches, stored
  responses) and would silently rot; a `GET` under `/v1/*` cannot write a
  usage record no matter which route serves it.
- `POST /v1/responses/{id}/cancel` stays budget-checked. It is a `POST`, and
  the rule deliberately stops at the method: deciding request by request which
  writes "look free" is the judgement call that produced the original
  inconsistency. A client that must cancel its way out of a spent budget is
  better served by raising the cap or resetting the window.
- The WebSocket transports (Realtime, OpenResponses) are untouched: their
  turns are `response.create` events that do spend, and the per-turn gate is
  exactly what stops a socket from outliving its budget.
- The contract test `test_key_gate_contract.py` gains a `v1-read` row, so the
  policy pair and its steps are pinned like every other transport.

## Alternatives rejected

- **Exempting specific paths (`/v1/models`, `/v1/files*`).** Every future read
  route would need a matching entry, and a forgotten one is a stranded client,
  not a loud failure. The method gives the safe direction instead: a `GET` can
  never create a usage row, while some writes (a file upload) cannot either —
  and leaving those checked costs a client nothing.
- **Returning the model list with a warning header instead of 200.** The
  request either proceeds or it does not; a body the client may or may not
  parse is worse than a status that says what happened.
- **Bypassing the gate entirely on reads.** That would drop authentication,
  rate limiting and lockout from a surface a hostile client reaches cheaply.

## Consequences

- An exhausted key can still discover its catalog and read its own records;
  only calls that would add spend are rejected, with the same 429 body.
- A budget no longer hides a misconfiguration behind "the provider is broken":
  the model list answers 200 while the chat call answers 429 with the budget
  message.
- `GET /v1/models` no longer runs a spend query, so the console's model picker
  costs no usage-table read per keystroke.
- Reads remain inside the lockout and RPM checks, so the exemption is strictly
  about the spending cap.
