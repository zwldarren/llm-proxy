# SSRF checks at connect time, not only at save time

## Status

Accepted (2026-10).

## Context

Every URL the proxy accepts from outside its own code — a provider `base_url`, a
non-admin's personal tracing collector, an MCP remote server, a web-search
endpoint, a client's image URL — goes through `validate_server_url`
(`http/client.py`), which rejects loopback, RFC1918, link-local, CGNAT, cloud
metadata hosts, URLs carrying credentials, and non-http(s) schemes, resolving
hostnames to catch the obvious rebinding trick.

The check happens **when the value is stored**, and again for a media download
immediately before each hop. The connection, however, is made later, by a fresh
`getaddrinfo` inside the transport. Three consequences followed:

1. A hostname can answer with a public address when the settings row is written
   and a private one when the request runs. The window is the whole life of the
   configuration, which is not a window at all.
2. The shared httpx2 client follows redirects, and its response hook validated
   each `Location` with `resolve_dns=False`: a redirect to `10.0.0.5` was refused,
   a redirect to `internal.corp` (a name that resolves there) was followed. A
   non-admin can write their own tracing collector URL, so that gap crossed a
   privilege boundary.
3. The MCP policy made the same check-then-use bet: `_resolve_and_check_blocked`
   resolves at config time and the docstring calls it rebinding protection.

## Decision

**The last check happens where the address is chosen: in the resolver.**

1. **`_SsrfGuardResolver` (`http/aiohttp_backend.py`)** wraps aiohttp's
   `ThreadedResolver` and rejects any answer in a banned network, at
   `TCPConnector` construction, so every request the shared outbound session makes
   — provider calls, web search, media downloads, the admin config probes, the
   update check — is checked as it connects. It reads `HTTP_ALLOWED_HOSTS` per
   resolution, so an opted-in LAN provider keeps working, and it logs the refusal:
   aiohttp rewrites the resolver's `OSError` into a generic "cannot connect", so the
   reason would otherwise be invisible.

2. **Literal hosts are passed through.** `_resolve_host` never calls the resolver
   for an IP literal, and a host that *is* `localhost` or a metadata name is
   rejected by the pre-flight check on every path that accepts a URL. Filtering
   them again at connect time would only break the loopback upstreams a developer
   runs locally. What the resolver adds is the case **only DNS reveals**.

3. **The redirect hooks resolve too.** `_on_redirect` (httpx2) and
   `_validate_redirect` (aiohttp) now run `validate_server_url(..., fail_closed=True)`
   in a worker thread instead of `resolve_dns=False`. A redirect target is as
   attacker-controlled as the URL that produced it.

4. **Media downloads keep their own per-hop check.** `download_image_as_base64`
   disables the client's redirect handling and validates each hop itself; the
   resolver is defence in depth beneath it, not a replacement.

## Consequences

- With the default backend (`HTTP_CLIENT_BACKEND=auto` → aiohttp) the
  check-then-use gap is closed for the initial connection and every redirect.
- Forcing `HTTP_CLIENT_BACKEND=httpx2` (also what `auto` does when an outbound
  proxy is set, because aiohttp's connector ignores `HTTP_PROXY`) keeps the
  save-time check plus a DNS check before each redirect; the initial connection
  can still re-resolve to a private address. httpx2 gives httpcore no resolver
  hook to hang this on, so the difference is documented rather than papered over.
- aiohttp's DNS cache (`ttl_dns_cache=300`) caches only answers that passed the
  check. A rejected answer is never cached, and a cached public answer is the
  safe direction: rebinding toward a private address means the connection goes to
  the public address that was cached.
- A refused connection surfaces as `ConnectError` (a retryable network error), so
  a retry policy may try the same provider again and fail the same way. The log
  line is the diagnostic; the exception text is not, because aiohttp builds it.
- Tests keep their loopback servers: literals and `localhost` are untouched.
- **Not covered:** code that builds its own `httpx2.AsyncClient` instead of using
  the shared session — the MCP remote client (`mcp/backend.py`, which the MCP SDK
  insists on owning), the flywheel/judge clients under `routing/`, and the Langfuse
  SDK's client. Their URLs are admin-configured and validated when saved; moving
  them onto `AsyncSession` would give them the cap and the resolver as well, and is
  the natural follow-up.

## Alternatives considered

- **Re-resolve immediately before every request.** Still a check-then-use gap, just
  a smaller one, and it adds a DNS lookup to the hot path.
- **Resolve once and pin the IP** (connect to the address, set `Host`/SNI
  manually). Closes the window fully, but bypasses connection pooling by hostname,
  breaks HTTP/2 coalescing, and re-implements TLS verification decisions.
- **Rely on an egress firewall.** The right defence in depth, and the wrong only
  defence: the proxy is often deployed inside the network it would have to be
  protected from.
