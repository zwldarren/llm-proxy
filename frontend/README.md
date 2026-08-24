# LLM Proxy — Admin Dashboard

The Vue 3 admin console for [LLM Proxy](https://github.com/zwldarren/llm-proxy): usage analytics, request logs, model catalog, chat & image playgrounds, and full configuration (providers, models, pricing, API keys, MCP servers, team, security policy).

## Stack

- **Vue 3 + TypeScript** (Vite / rolldown-vite)
- **Tailwind CSS v4** + [shadcn-vue](https://www.shadcn-vue.com) (new-york, neutral)
- Pinia, vue-router, vue-i18n (English + 中文), vue-sonner
- Vitest for unit tests

## Development

The frontend is usually started together with the backend via `uv run llm-proxy-dev` from the repository root (backend on `:9911`, frontend dev server on `:5173`, proxying API calls to the backend).

To run it standalone:

```bash
bun install
bun dev
```

By default the dev server proxies API requests to the backend at `http://localhost:8000`. Point it elsewhere with `VITE_API_BASE_URL`, e.g.:

```bash
VITE_API_BASE_URL=http://localhost:9911 bun dev
```

## Commands

| Command | Description |
| --- | --- |
| `bun dev` | Vite dev server with hot reload |
| `bun run build` | Type-check + production build |
| `bun run preview` | Preview the production build |
| `bun run test` | Run Vitest unit tests |
| `bun run lint` / `lint:check` | ESLint (auto-fix) / check only |
| `bun run format` / `format:check` | Prettier (write) / check only |

## Project Layout

```
src/
  components/        # UI components (common/, chat/, ui/ shadcn-vue primitives…)
  views/             # Route-level pages
  stores/            # Pinia stores
  i18n/              # en.ts / zh.ts locale catalogs
  adapters/          # Client-side protocol adapters (openai / anthropic / openresponses)
  api/               # Backend API client
  assets/            # main.css design tokens (dark-first monochrome system)
```

## Design system

Tokens and visual conventions live in `src/assets/main.css` and `src/components/ui/`; the system is documented at the repository root in [`DESIGN.md`](../DESIGN.md) (colors, typography, elevation, do's and don'ts). Dark-first by default, with light and system theme options, `prefers-reduced-motion` support, and i18n for English and Chinese.
