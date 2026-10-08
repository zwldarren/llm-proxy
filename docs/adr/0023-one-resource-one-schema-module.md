# One resource, one schema module

## Status

Accepted (2026-10).

## Context

`api/schemas/admin.py` held the request and response shapes for the whole
dashboard: **52 classes across 1732 lines**, serving ten unrelated resources —
model config, providers, API keys, MCP servers, auth/setup, and every section of
the server config (logging, web search, smart routing, provider selection,
request policy, resilience, security). Nine routers imported it, and it was the
hottest file in the repo: **13 of the last 200 commits touched it**.

The name describes the audience, not the content. "Admin" is *every* dashboard
endpoint, so the file had no owner — the second axis is the resource.

Three concrete costs:

- **Every resource change is a whole-file change.** One new field on
  `ModelUpdate` lands in the same file as `ApiKeyCreate` and `McpServerRead`;
  unrelated changes conflict with each other, and a reviewer has to work out
  which of the ten resources a hunk belongs to.
- **One resource's shapes cannot be read on their own.** Understanding the MCP
  schemas means scrolling past roughly 1600 lines of everything else.
- **Shared bases are buried.** `ValidatorMixin` (the optional-field coercions
  the model, provider and MCP write schemas all rely on) sat at line 38 of the
  grab-bag, so "which schemas share these coercions?" had no answer short of
  searching.

The repo already had two schema conventions, and the split had to pick one:

- `api/schemas/<resource>.py`, mirroring `api/routers/<resource>.py` —
  `logs.py`, `me.py`, `team.py`, `tracing.py`.
- Schemas next to the router inside the config sub-package —
  `routers/config/pricing_schemas.py`, `routers/config/metadata_schemas.py` —
  plus small response shapes defined inline in a router
  (`routers/config/circuit_breaker.py`, `routers/system.py`).

## Decision

A schema lives in the module of the resource it belongs to, under
`llm_proxy.api.schemas`, which mirrors `api/routers/` one resource per file:

| module | serves | shapes |
| --- | --- | --- |
| `models.py` | `routers/config/models.py`, `routers/catalog.py`, `routers/models.py` (`GET /v1/models`) | 9 classes, 760 lines |
| `providers.py` | `routers/config/providers.py`, `routers/config/provider_models.py` | 9 classes, 206 lines |
| `api_keys.py` | `routers/api_keys.py` | 8 classes, 242 lines |
| `mcp.py` | `routers/mcp.py` | 7 classes, 89 lines |
| `auth.py` | `routers/auth.py` | 4 classes, 51 lines |
| `server_config.py` | `routers/config/server.py` (every section) | 14 classes, 381 lines |
| `common.py` | shared by the three write-schema modules | `ValidatorMixin` |

One module per resource, whatever the surface count: three routers serve the
models resource and their shapes overlap (the catalog and the model config API
share `normalize_model_status` and the capability derivation), so they share the
module.

- **One resource, one module.** The module docstring names the routers that
  serve it, so `api/schemas/` is navigable the same way `api/routers/` is.
- **`common.py` holds what more than one resource reuses** — today only
  `ValidatorMixin`. A validator moves there when a second resource needs it,
  not in anticipation.
- **A resource may embed a peer's read shape.** `providers.py` imports
  `ModelRead` for `ProviderDetails.models`. No resource imports a peer's *write*
  schema; `ty` and the import graph keep the direction one-way.
- **`admin.py` is deleted, not kept as a façade.** All 14 importers (9 routers,
  5 test modules) point at the module that owns the resource, and the tests
  follow the same split (`tests/api/schemas/test_model_schemas.py`,
  `test_api_key_schemas.py`, `test_server_config_schema_defaults.py`).
- **Small router-local shapes may still be inline** (unchanged precedent), and
  the config sub-package keeps its `*_schemas.py` neighbours; both are
  single-owner files, which is the property this ADR is about.

## Considered Options

- **Schemas next to their router** (`routers/models_schemas.py`), the shape
  suggested by the architecture review that raised this. Rejected:
  `api/routers/` is the routers package, and `api/schemas/` already mirrors it
  by resource for four resources; the config sub-package's `*_schemas.py` files
  exist because that package holds several routers over one sub-resource family.
- **Keep `admin.py` as a re-export façade** over the new modules. Rejected: it
  keeps the grab-bag's gravity (the next schema gets added there, where it is
  "already imported") and hides the module that actually owns each resource.
- **Split by endpoint or by feature** rather than by resource. Rejected: one
  resource has several endpoints and they share shapes — `ModelRead` is used by
  the model config API and embedded by provider details, and the catalog reuses
  the model config's status/capability logic.
- **Split further** (a module per server-config section, a module for the four
  auth shapes). Rejected: 381 lines with no cross imports is one owner — the
  server-config router — and the seam that matters is the owner, not the size.

## Consequences

- A change to one resource touches one schema module next to the router that
  serves it, plus one test module. The hottest file in the repo dissolves into
  files with one owner each.
- The split is invisible at the API surface: the generated OpenAPI document is
  identical before and after (187 component schemas, 123 paths, every `$ref`
  included), because FastAPI names components after the class and not the
  module. The committed `docs/public/v1-openapi.json` did not move and
  `tests/api/test_openapi_spec.py` passes unchanged.
- No definition changed: all 58 classes, functions and aliases were moved with
  byte-identical sources, verified by comparing the parsed definitions before
  and after.
- The cost is in imports: nine routers and five test modules had to be
  repointed, and there is no shim, so a stale import fails at import time
  instead of silently resolving.
- New schemas have one default home, and the shared base is now visible by
  name rather than by memory of line 38.
