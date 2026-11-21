# A preflight answers the headers the client asks for

## Status

Accepted (2026-11).

## Context

`api/middleware/cors.py` answered every preflight with one fixed
`Access-Control-Allow-Headers` list: `Authorization`, `Content-Type`, the trace
and request-id headers, `X-CSRF-Token`, and the `Accept*` family.

That list describes the *console's* requests. The API's header surface is
defined by each client protocol instead:

- `x-api-key` is a first-class credential — `gate._extract_key` accepts it
  alongside `Authorization`.
- Anthropic clients send `anthropic-version`, `anthropic-beta`,
  `anthropic-workspace-id` and the `x-stainless-*` fingerprint.
- OpenAI/Codex clients send `OpenAI-Beta`, `originator`, `chatgpt-account-id`,
  `conversation_id` and `x-codex-*`.
- OpenCode clients send their identity set (`x-opencode-session`,
  `x-opencode-request`, `x-opencode-client`, `x-opencode-project`).
- Provider-specific headers (`x-app`, `x-session-id`, `x-log-full`) come from
  coding agents and debugging clients.

A browser client behind a CORS-configured origin sends a preflight naming the
headers it wants. Any name outside the fixed list fails the preflight — and a
preflight failure happens **before** the request reaches auth, so the proxy logs
nothing. The client reports a generic CORS error, and the operator has no row,
no status code and no error message to work from. This is the failure mode
`docs/deployment/troubleshooting.md` already describes as "reproduce with
`curl` to see the true status", except that here the true status only exists
once CORS stops refusing the header set.

The list cannot simply be extended: it would have to track every client's
headers forever, and each miss is silent.

## Decision

**The allow-headers answer is derived per request from
`Access-Control-Request-Headers`, and only for an origin already on the allow
list.**

- Each requested name is validated against the RFC 9110 `field-name`
  production; invalid names are dropped rather than echoed. A preflight that
  names nothing (or an empty value) still gets the built-in list.
- The reflected list is bounded at 64 names, and each name at 64 characters, so
  an unauthenticated preflight cannot inflate the response.
- Names that match the built-in list are answered in their canonical spelling
  (`x-api-key` → `X-API-Key`); everything else is echoed as sent, which the
  header grammar allows.
- The preflight response carries
  `Vary: Origin, Access-Control-Request-Headers`, because the answer now depends
  on both. Without it a shared cache could hand one client's answer to another.
- `X-API-Key` is added to the built-in list, and `Retry-After` /
  `X-Request-Id` to `Access-Control-Expose-Headers` — a cross-origin client
  needs the retry delay it is being told to honour.

## Alternatives rejected

- **Extending the fixed list with every known client header.** The list would
  have to be updated with each new SDK and each new provider dialect, and a
  forgotten entry fails silently at the preflight.
- **Echoing `Access-Control-Request-Headers` verbatim.** A header value is not
  a list of field names; reflecting it unchecked puts whatever the caller sent
  into a response header and lets one preflight produce an arbitrarily large
  answer.
- **Reflecting the list for any origin.** The origin check stays first: a
  disallowed origin is answered with a 400 and no allow-headers at all.
- **Wildcard (`*`) for allowed headers.** It cannot be combined with
  `Access-Control-Allow-Credentials: true`, which the console's cookie
  transport needs.

## Consequences

- Browser SDK clients can send their protocol and identity headers; the
  preflight no longer decides which clients are allowed to authenticate.
- The CORS layer is no longer a second, undocumented allowlist of header names
  that has to be kept in sync with `providers/*/client_headers.py`.
- One extra `Vary` value on preflight responses, and a bounded amount of work
  per preflight (name validation over at most 64 names).
