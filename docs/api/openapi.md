---
pageClass: api-openapi
aside: false
---

# OpenAPI Specification

The machine-readable contract of every `/v1` endpoint, generated from the running
proxy: request schemas come from the exact Pydantic models the server validates
with, so they cannot drift from behaviour.

::: info Scope
Only the client-facing `/v1` surface is published here. The admin API (`/api/*`),
the MCP proxy (`/servers/*`), the SPA fallback, and the path aliases
(`/v1/v1/...`, trailing slash) are intentionally excluded. WebSocket transports
(`WS /v1/responses`, `WS /v1/realtime`) have no OpenAPI representation — see
[Streaming](streaming.md) and the [Endpoint Index](endpoints.md).
:::

::: tip Regenerating
`uv run llm-proxy-openapi` rewrites `docs/public/v1-openapi.json`.
`tests/api/test_openapi_spec.py` fails if the committed file is stale, if a
`/v1` route is missing from this document, or if an alias route leaks in.
:::

<OASpec groupByTags hideBranding />
