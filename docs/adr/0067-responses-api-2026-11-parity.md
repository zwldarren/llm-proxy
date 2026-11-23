# Responses API parity: open tool_choice, echoed tools, owned background mode

## Status

Accepted (2026-11).

## Context

An audit of OpenAI's Responses API against this proxy (changelog entries from
2026-09 through 2026-11) found three places where the proxy's schema was
narrower than the wire it claims to speak, and one behavioral hazard:

- **`tool_choice` was a closed union.** `Literal["auto","required","none"] |
  FunctionToolChoice | AllowedToolChoice` rejects every typed object form the
  API documents: hosted-tool choices (`{"type": "web_search_preview"}`,
  `file_search`, `code_interpreter`, `image_generation`, computer use) and
  `mcp` / `custom` / `shell` / `apply_patch`. A client using one got a 422 from
  the proxy without a request ever reaching an upstream.
- **The response echo was narrower than the request.** `format_response`
  echoed the client's raw tool declarations (the Open Responses
  `ResponseResource` carries the effective tool set), but `ResponsesResponse`
  declared `tools: list[FunctionToolParam]`. Any stored response created with a
  hosted tool therefore failed validation on `GET /v1/responses/{id}` — an
  unhandled pydantic `ValidationError`, i.e. a 500, for the most common
  agent-shaped request there is.
- **Background mode ran twice.** `background: true` was emulated locally
  (return the `in_progress` body, run the work in-process, overwrite the stored
  response), but the same flag also survived into the native Responses body:
  the passthrough tier forwards the client's body verbatim, and nothing vetoed
  it. A native OpenAI upstream answered `queued` immediately, and
  `_run_background_openresponses` only promotes `in_progress` to `completed` —
  so the client polled a permanently `queued` response. The WebSocket transport
  already dropped `background`, so the two transports disagreed.
- **`background_options` did not exist** in the schema (added 2026-11-17), so
  the 2h/8h discounted completion windows could not be expressed at all.

GPT-6.2 (2026-11-17) also changed two prefix rules: the API drops reasoning
items after mid-conversation prefix edits, and top-level `instructions` is
appended *after* `input` for prefix matching. Neither is a schema break, but
both make prefix fidelity load-bearing for quality.

## Decision

**Typed `tool_choice` is an open contract.** `TypedToolChoice` accepts any
object with a non-empty `type` discriminator (`extra="allow"`), and
`_parse_tool_choice` maps everything outside the modelled variants to
`ToolChoiceHosted(type, name, params)`. The client's spelling (including
variant-specific keys such as `server_label` on `mcp`) is preserved:

- native Responses upstreams receive the full object verbatim on both the
  passthrough and rebuilt paths (they are the authority on which discriminators
  and parameters they accept);
- every other provider degrades it to `auto` (Chat Completions and Anthropic
  to `"auto"`, Gemini to `AUTO`), because those providers have no equivalent
  hosted tools; the (still filtered) tool list constrains what the model may
  call.

**The Response echo reuses the request-side tool union.** `ResponsesResponse`
declares `tools: list[ToolParam]` and `tool_choice: ToolChoiceParam`, the same
shapes the request accepts. A body the proxy is willing to forward must be a
body it is willing to store and read back.

**Background mode belongs to the proxy.** `background_options` is modelled
(`completion_window: soonest | 2h | 8h`, `continuation_id`), and both
`background` and `background_options` are cleared from the protocol request
before the hidden upstream call, exactly as the WebSocket transport drops them
from `response.create` events. In-flight background tasks are tracked in
`RuntimeServices.background_response_tasks()` so `POST /v1/responses/{id}/cancel`
cancels the running asyncio task in addition to marking the stored record
`cancelled` (and a finishing task never overwrites a `cancelled` record).
`completion_window` is accepted and not honored, and documented as such: the
discounted windows require the upstream to hold the job for hours, and the
proxy's background mode returns as soon as the model answers.

**Prefix fidelity is pinned, not migrated.** On the materialized-continuation
path the rebuilt body re-emits the prior turn in order (prior input, prior
output including reasoning items, then the new turn) and carries `instructions`
as the first item of `input` rather than a top-level field — which is the
placement GPT-6.2 recommends for cacheable instructions. Stored `instructions`
are inherited only when the continuation turn omits its own `instructions`
(when the continuation supplies new or repeated `instructions`, the live value
wins instead of stacking duplicate system messages). A regression test pins
that shape. Moving the client's top-level `instructions` into `input`
unconditionally was rejected: it would change the echo contract for every
client and duplicate the field on the native path, where the body is forwarded
verbatim.

## Consequences

- Clients using hosted tool choices or `mcp`/`shell`/`apply_patch` work against
  the proxy, and stored responses echoing those tools are readable again
  (`GET /v1/responses/{id}`); the follow-up `POST .../cancel` and
  `GET .../input_items` paths return the stored body unvalidated and are
  unaffected.
- Degrading a forced hosted choice to `auto` is a behavior change for
  non-Responses upstreams: the model may answer without calling the tool. The
  alternative — a 422 — made the request impossible. Callers that need the
  force must stay on a native Responses upstream.
- An unknown discriminator is no longer a client error. A typo now surfaces as
  an upstream 400 on native providers and is silently degraded elsewhere.
- Background responses become terminal on every provider, including native
  OpenAI, at the cost of the 2h/8h discount. Supporting the discount means
  relaying upstream polling (`GET`/`cancel` proxied by upstream id) and is
  deferred rather than half-built.
- `docs/public/v1-openapi.json` is regenerated; `docs/api/tools.md`,
  `docs/api/endpoints.md` and their `docs/zh/` mirrors document the typed
  choice, the degradation and the background window.

## Alternatives considered

- **Keep `tool_choice` closed and document the gap.** Rejected: the failure is
  a 422 at the protocol boundary for a request the API documents, and each new
  hosted tool would need a proxy release.
- **Accept `dict[str, Any]` for `tool_choice`.** Rejected: it would swallow the
  modelled variants, and `allowed_tools` needs its collapse-to-mode/filter
  behavior.
- **Validate hosted types against a fixed literal.** Rejected: same release
  treadmill, one level down; the upstream is the authority.
- **Model `tools` on the response as `list[dict]` only.** Rejected: the
  request-side union already covers the modelled shapes and a catch-all, so
  reusing it keeps one definition of "a tool declaration".
- **Reject `completion_window: 2h|8h`.** Rejected: a client asking for
  background execution would hard-fail on a proxy setting that has a working
  (if undiscounted) answer.
- **Forward `background` to a native upstream and relay polling.** Rejected for
  now: it needs an upstream-id mapping plus `GET`/`cancel` relay and a
  migration story for responses already stored locally.
