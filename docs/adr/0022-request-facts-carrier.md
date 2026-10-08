# Request-scoped facts get one typed carrier

## Status

Accepted (2026-10).

## Context

A request accumulates ambient facts as it travels: who sent it, what it asked
for, where it was routed, what the auditor captured for it, and two latches
("the client is gone", "an audit row already exists"). Those facts lived as
untyped `request.state` attributes: **24 writer sites across 13 modules**, and
**42 readers across 15 modules** — 39 of them `getattr(request.state, "name",
None)` probes, including five scope-level keys written by the pure-ASGI
middlewares through `set_request_state`.

Every reader re-derived the same two things:

- **Whether absence is legal.** `None` meant "the writer never ran" (a bug),
  "the feature is off" (legal), and "not learned yet" (the normal state). The
  call sites papered over the difference with `or "unknown"` and `isinstance`
  checks, so a *lost* fact looked exactly like a legal default.
- **The attribute name.** A typo is a permanent `None`, and a rename compiles
  nothing.

Four concrete costs, each verified against the code this ADR replaces:

- **`request_id` had three mints and 17 readers.** `HttpLoggingMiddleware`
  minted `uuid4().hex` when the scope had none, `api/context.py` minted
  `str(uuid.uuid4())` as its fallback, and `core/ws_common.py` minted
  `ws_<24 hex>` for websocket connections: one fact, three formats, decided in
  three modules. Seventeen places read it, most defensively.
- **One fact had two homes.** `api_key_name` was written to `request.state`
  *and* carried in `RequestIdentity` by the same auth gate
  (`api_key_auth.py`, and `core/ws_common.py` for websocket connections), with
  the state copy read by the model-restriction middleware and the OpenResponses
  router. Changing one writer silently left the other copy stale.
- **Two keys were read but never written.** `session_id` was read twice — by
  early-failure row assembly and by admin audit row assembly — and written
  nowhere, so a trusted-proxy client's session id never reached either row even
  though `api/context.py` had resolved it. `error_message` was read by the
  logging middleware and written nowhere: the row it fed is written by
  `record_early_failure`, which knows the message itself, so the parameter could
  never be filled.
- **The latches failed silently.** `audit_log_written` (the intake module's
  dedupe flag) and `client_disconnected` (keepalive's cross-layer flag) were
  read as `getattr(..., False)`: a wrong name resets the latch and loses the
  abandonment instead of raising.

`core/identity.py` already had the answer for its own slice — a dataclass, one
writer, typed readers, and a docstring saying exactly that — but it was one
slice, and the identity slot was invisible to every other fact's readers.

## Decision

`llm_proxy.core.request_facts` owns the request-scoped facts and the single
`request.state` slot that holds them (`request_facts`).

- **One record, one slot.** `RequestFacts` is a dataclass whose fields are the
  facts, documented by writer in the module docstring: `request_id`, `identity`,
  `allowed_models`, `model`, `provider`, `session_id`, `parsed_request_body`,
  `client_disconnected`, `audit_log_written`, and the four audit capture buffers
  (`request_headers`/`request_body`/`response_headers`/`response_body`).
  `facts_for(request)` and `facts_from_scope(scope)` return it, creating it on
  first touch; `mint_request_id(prefix)` owns the id's shape.
- **The identity is a field of the record.** `core/identity.py` is absorbed:
  `RequestIdentity` keeps its name and shape, `get_request_identity` /
  `set_request_identity` stay as the identity sub-API (unchanged semantics,
  default-unauthenticated on absence), and its 60 call sites did not move. Two
  `request.state` slots would have been the same disease in a different place.
- **No new middleware.** The record is created by whichever code touches it
  first; for HTTP that is `HttpLoggingMiddleware`, which is the outermost
  middleware every request passes and already minted the id outside every auth
  gate. Adding an eleventh ASGI layer to make "one early middleware populates
  it" literal would cost every request a wrapper to do what an existing one
  already does.
- **`api_key_name` loses its state copy.** The identity already carries it;
  the two readers take it from there.
- **The two dead keys are resolved, differently.** `session_id` is a real fact
  the pipeline learns (`api/context.py`, for trusted-proxy clients), so it is
  written and its two readers finally receive data. `error_message` had no
  writer and no possible one, so the read and `record_admin_request`'s
  unfillable parameter are deleted rather than guessed at.
- **The scope-key helpers go.** The capture buffers are record fields, so
  `asgi_utils.set_request_state` / `get_request_state` and
  `core.constants.CLIENT_DISCONNECTED_STATE_KEY` are deleted.
- **`scope["llm_proxy_auth"]` stays.** It is a different carrier for a different
  boundary: the mounted MCP sub-app reads a dict of principal/permission keys
  straight off the ASGI scope, with a shape (`principal_type`,
  `allowed_mcp_servers`) the record does not model.

## Considered Options

- **A dedicated facts middleware.** Rejected: an extra ASGI layer for every
  request — including `/health` and the admin UI's static assets — to do what
  `HttpLoggingMiddleware` already does at the same position in the stack.
- **Keep `core/identity.py` as a second slot.** Rejected: two carriers means
  readers must know which holds what, and the identity would stay invisible from
  the record's readers. The accessor API is preserved instead of the slot.
- **A frozen record with typed setters per field.** Rejected: the facts are
  written progressively by different stages (the auth gate, the pipeline, the
  logger, the intake module), so a frozen record would need a builder per
  writer. The interface is the field list, each field documented with its
  writer; `client_disconnected` and `audit_log_written` keep exactly one writer
  each.
- **Keep the string keys as properties over the record.** Rejected: the point is
  that no module spells these names. A compatibility shim keeps the typos alive
  and the probing idiom with them.

## Consequences

- Facts are renamed by the type checker, and a missing fact is a typed default
  instead of a `None` that might be a typo.
- Ordering knowledge is in one place: `facts_for` creates the record, and the
  module docstring names who writes each field. The auth gates no longer depend
  on another middleware having run to have a request id for a rejection row.
- Early-failure and admin audit rows carry `session_id` for trusted-proxy
  clients (they never did).
- The latches are fields (`facts.audit_log_written`,
  `facts.client_disconnected`), so a wrong name is an `AttributeError` rather
  than a silently reset flag.
- Tests seed a record instead of poking string keys, which removes the
  request-double trap: `facts_for(MagicMock())` returns a real record, where
  `getattr(mock.state, "request_id", None)` returned a mock.
- `RequestFacts` is process-internal and unversioned: a field added here is
  available to every module at once, which is the point, but it also means the
  record is not a place to model per-protocol or per-provider detail.
