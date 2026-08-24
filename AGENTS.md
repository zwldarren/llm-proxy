# LLM Proxy - Agent Guidelines

## OVERVIEW

LLM API proxy with multi-provider support (OpenAI, Anthropic, Gemini, Ollama, OpenRouter), protocol translation, streaming, rate limiting, MCP integration, and Vue 3 admin UI.

## COMMANDS

### Backend

```bash
uv sync                           # Install dependencies
uv run ruff check --fix           # Lint (auto-fix)
uv run ruff format                # Format code
uv run ty check                   # Type check (ty, not mypy/pyright)
uv run pytest                     # Run tests
uv run llm-proxy                  # Run production server
uv run llm-proxy-dev              # Start backend + frontend dev servers concurrently
```

### Verification order

```bash
uv run ruff check --fix && uv run ruff format && uv run ty check && uv run pytest
```

## ARCHITECTURE

### Request Flow

1. Incoming request → Protocol endpoint (`protocols/`) mounts the route
2. Parse to `InternalRequest` via the protocol serializer (`protocols/<name>/serializer.py`)
3. Route via `UnifiedProcessor` (`core/processing/unified.py`) to adapter
4. Adapter calls provider API via the provider serializer (`serialization/`)
5. Response → `InternalResponse` → Format for client via the protocol serializer

### Serialization Pattern

Protocol serializers (client wire format ↔ internal models) and provider
serializers (internal models ↔ provider API format) are **two separate
classes in two separate registries** — one serializer never does both
sides.

```python
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.serialization import get_provider_serializer

# Client side: wire format -> InternalRequest
protocol_serializer = get_protocol_serializer("openai")
unified_request = protocol_serializer.parse_request(raw_data)

# Upstream side: InternalRequest -> provider body
provider_serializer = get_provider_serializer("openai")
provider_body = provider_serializer.build_provider_request(unified_request)
```

### MCP Integration

- `mcp/manager.py`: MCP server lifecycle management
- `mcp/proxy.py`: Request proxying to MCP servers
- Configure MCP servers in provider config with `mcp_servers` key

## DATABASE MIGRATIONS

Migrations are the **single source of truth** for schema. There is no `create_all` fallback — if migrations fail, startup fails loudly.

- `connection.py:init_db()` calls `alembic upgrade head` on startup via `asyncio.to_thread()`
- `env.py` creates an async engine with `NullPool`, runs migrations, and explicitly commits
- `render_as_batch` is enabled **only for SQLite**; PostgreSQL uses native DDL
- Generated migrations are auto-formatted by the ruff post-write hook in `alembic.ini`

### Commands

```bash
# After modifying models in database/tables.py:
uv run alembic revision --autogenerate -m "description_of_change"
```

## TESTS

- `tests/conftest.py` provides `MockResponse` and `create_mock_client` for mocking `httpx2`

## DEV SERVER

`uv run llm-proxy-dev` (`llm_proxy.__main__:main_wrapper_dev`) starts:

- Backend uvicorn with `--reload` on port `9911`
- Frontend Vite dev server on `0.0.0.0:5173`, proxying API to backend via `VITE_API_BASE_URL`

Requires `bun` in PATH. `node_modules` is auto-installed on first start if missing.

## STYLE & CONVENTIONS

- **Python**: 3.14+ (`pyproject.toml`)
- **Comments**: All code comments must be in English
- **Ruff**: line-length 100, excludes `**/versions/**`
- **Type checker**: `ty` (configured in `pyproject.toml` with `include = ["src"]`)

### Exception Handling

- **Python 3.14 (PEP 758) allows omitting parentheses in multi-exception `except` clauses when there is no `as` binding.**
  Ruff format will therefore collapse `except (ValueError, ConfigurationError):` into
  `except ValueError, ConfigurationError:` — this is valid Python 3.14 and catches both exceptions.

  Correct without `as`:

  ```python
  except ValueError, ConfigurationError:
      pass
  ```

  Correct with `as` (parentheses are still required):

  ```python
  except (ValueError, ConfigurationError) as exc:
      ...
  # or, when the exception object is unused:
  except (ValueError, ConfigurationError) as _:
      pass
  ```

  Do not write:

  ```python
  except ValueError, ConfigurationError as exc:  # WRONG: binds only to ValueError
      ...
  ```

## DESIGN CONTEXT

Strategic + visual context for the admin UI. Full detail in [`PRODUCT.md`](PRODUCT.md) (strategy: register, users, purpose, brand, anti-references, principles, accessibility) and [`DESIGN.md`](DESIGN.md) (visual system: tokens, typography, elevation, components, do's/don'ts). The frontend's own `frontend/AGENTS.md` carries the tactical DESIGN CONTEXT for the Vue app; `PRODUCT.md` / `DESIGN.md` are the root source of truth and win on conflicts.

- **Register**: `product` — design serves the product (admin console / dashboard / tool).
- **Primary users**: operators who self-host the proxy and run the dashboard (configure providers/models/keys, monitor usage, cost, logs). API consumers are a non-design surface.
- **Brand personality**: Technical Sophistication, Reliable, Modern. Direct, precise, professional; confident, not arrogant.
- **Aesthetic**: Monochrome Editorial — dark-first (deep cool-neutral, hue 220°), light mode a crisp mirror. Space Grotesk display + Manrope body + IBM Plex Mono for all data. Low-saturation semantic tints (violet/blue/amber/rose + status) for differentiation only, never decoration.
- **Design principles**: Clarity Over Cleverness · Confidence Through Feedback · Technical Sophistication · Dark-First Design · Respectful of Time.
- **Anti-references**: NOT playful/consumer; NOT boring/outdated admin panels; NO loud color accents; NOT the generic 2026 AI-tool aesthetic (no cream/sand bodies, gradient-text, identical card grids, hero-metric template).
- **Accessibility**: WCAG AA both themes; `prefers-reduced-motion` respected; visible focus rings + skip link; color never the sole status signal; i18n (English primary, Chinese secondary).
- **Frontend stack**: Vue 3 + Vite (rolldown) + TypeScript + Tailwind v4 + shadcn-vue (new-york, neutral) + reka-ui + Pinia + vue-router + vue-i18n + vue-sonner. Tokens in `frontend/src/assets/main.css`; primitives in `frontend/src/components/ui/`.

## Agent skills

### Issue tracker

Issues live as GitHub issues in this repo; use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
