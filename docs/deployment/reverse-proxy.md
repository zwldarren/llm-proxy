# Reverse Proxy & TLS

LLM Proxy speaks plain HTTP and expects TLS to terminate in front of it. This page
covers what the proxy needs from the proxy in front.

## What the app expects

- **TLS termination**: uvicorn is started without certificate options, so the reverse
  proxy (nginx, Caddy, Traefik, Cloudflare Tunnel…) owns HTTPS.
- **Forwarded headers**: only peers listed in `TRUSTED_PROXIES` may set
  `X-Forwarded-For` / `X-Real-IP`. The default is the RFC1918 + loopback + link-local
  set:

  ```
  10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,127.0.0.0/8,169.254.0.0/16,::1/128,fc00::/7,fe80::/10
  ```

  Client IP resolution walks `X-Forwarded-For` right-to-left and uses the first hop
  **outside** the trusted ranges (max 50 hops). `X-Real-IP` is used only when
  `X-Forwarded-For` is absent. Spoofed headers from untrusted peers cannot influence
  rate limiting, lockouts, or audit attribution.

  !!! warning "`TRUSTED_PROXIES` replaces the default list"
      Setting it drops the built-in ranges — always include your local/Docker
      networks. Setting it to an empty value trusts nobody (the TCP peer address is
      used). Invalid CIDRs fail startup.

- **HTTP/1.1 to the app** (HTTP/2 terminates at the proxy): WebSocket upgrade paths
  (`/v1/responses`, `/v1/realtime`) must be passed through, and SSE streaming must not
  be buffered.
- **Keep-alive**: `UVICORN_TIMEOUT_KEEPALIVE` defaults to 600 seconds.

## Cloudflare and other impatient CDNs

Cloudflare's free/pro plans abort proxied requests that stay silent for ~100 s
(error 524). Long reasoning requests can exceed that. The proxy mitigates it on two
levels — both hot-reloadable under **Settings → Advanced → Response Keepalive**:

| Situation | Mechanism |
| --- | --- |
| Non-streaming request runs longer than `grace_seconds` (default 60 s) | The proxy sends `200 application/json` immediately and writes a single space byte every `interval_seconds` (default 15 s) until the real body is ready |
| Streaming request with upstream silence | SSE comment frames `: keep-alive` every interval; SSE clients ignore comments by definition |

Consequences you must know:

- Once heartbeat mode has started, the status is already committed to **200**. A
  failure that surfaces later is delivered as `200` plus an error JSON body — clients
  that only check status codes will treat it as success. Inner response headers
  (including some usage hints) are lost.
- The `audio/speech` endpoint is excluded (binary).
- If the client or CDN gives up, the proxy cancels the upstream call and logs the
  request as **499** instead of silently succeeding.

::: note Runtime default vs Settings form
With nothing stored, the runtime defaults are **enabled, 60 s grace, 15 s
interval**. The Settings form shows its own defaults (disabled, 30 s) until the
section is saved, and saving applies exactly what the form shows. Review the
values before saving if you never touched this section.
:::

## Example: nginx

```nginx
server {
    listen 443 ssl;
    server_name llm.example.com;

    ssl_certificate     /etc/letsencrypt/live/llm.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/llm.example.com/privkey.pem;

    client_max_body_size 10m;   # keep in sync with max_request_body_size_bytes

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;

        proxy_set_header Host              $host;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streaming: no buffering, no read timeout for long generations
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;

        # WebSockets (/v1/responses, /v1/realtime)
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Set `TRUSTED_PROXIES` to the network the reverse proxy connects from (e.g.
`127.0.0.1/32` for the snippet above), otherwise clients appear as the proxy's IP and
all rate limiting collapses onto one bucket.

## Security headers

The app emits these itself on every response:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`
- `Content-Security-Policy`:
  `default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains` when
  `hsts_enabled` (default **true** — toggle in
  [Server Settings](../admin/settings.md#security-rate-limiting); disable only for
  local HTTP development)

There is no need to duplicate them at the reverse proxy, though adding
`X-Forwarded-Proto`-aware redirects (HTTP → HTTPS) at the edge is good practice.

## CORS

CORS is only needed when the admin UI is served from a **different origin** than the
API. Configured origins live in **Settings → Advanced → CORS Origins** and are
hot-reloaded; an empty list disables CORS entirely.

::: warning CORS is the innermost layer
Requests rejected by auth, body-size, or rate-limit middleware return **before**
CORS headers are added, so a browser sees a generic CORS failure instead of the
real 401/413/429. Check the proxy logs or `curl` when debugging cross-origin
setups.
:::

## Request body size

`max_request_body_size_bytes` defaults to **10 MiB** and is enforced with a Content-Length
check plus an actual byte count (chunked transfer encoding is rejected — it would
bypass the check). Configure it in
[Server Settings](../admin/settings.md#security-rate-limiting) and keep the reverse
proxy's own body limit at or above the same value.
