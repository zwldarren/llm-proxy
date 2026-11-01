# Contributing to LLM Proxy

Thanks for taking the time to improve the proxy. This project is a self-hosted gateway,
so changes that affect request handling, credentials, or documented defaults get extra
scrutiny — the notes below describe how to make that easy.

## Before you start

- **Open an issue first** for anything larger than a typo or a one-line fix. Describe the
  behavior you see, the behavior you expect, and the provider/protocol involved.
- Security problems do **not** belong in an issue — see [SECURITY.md](SECURITY.md).
- Check [`docs/adr/`](docs/adr/) before changing a subsystem. The ADRs record decisions
  and the reasons behind them; a change that contradicts one needs to say why in the PR.

## Development setup

```bash
uv sync                     # backend dependencies
uv run llm-proxy-dev        # backend :9911 + Vite dev server :5173 (requires bun)
```

`uv run llm-proxy-dev` installs `frontend/node_modules` on first start if it is missing.
To run the backend alone, use `uv run llm-proxy` (production build) or
`uv run python -m llm_proxy` (no dev frontend).

## Verification

Run the full sequence before opening a pull request:

```bash
uv run ruff check --fix
uv run ruff format
uv run ty check
uv run pytest
```

Frontend changes additionally need:

```bash
cd frontend
bun run type-check
bun run lint
bun run test
```

If you touched the docs site:

```bash
cd docs
bun install
bun run docs:build
```

The repo ships a `.pre-commit-config.yaml` that runs the ruff and ty checks on commit;
`pre-commit install` wires it up. CI runs the same checks, so a green local run usually
means a green PR.

## Where things live

| Area | Path |
| --- | --- |
| Protocol endpoints and client serializers | `src/llm_proxy/protocols/` |
| Provider serializers | `src/llm_proxy/serialization/` and `src/llm_proxy/providers/` |
| Request pipeline, routing, retries, budgets | `src/llm_proxy/core/` |
| Admin API routers and schemas | `src/llm_proxy/api/` |
| Database tables and migrations | `src/llm_proxy/database/` |
| Vue console | `frontend/src/` |
| User documentation | `docs/` |

Schema changes go through Alembic — migrations are the single source of truth, and there
is no `create_all` fallback:

```bash
uv run alembic revision --autogenerate -m "describe_the_change"
```

## Writing docs

- Behavior changes and new settings must update the matching page under `docs/`. A quick
  map: provider fields → `docs/admin/providers.md`, model/pricing → `docs/admin/models.md`,
  server settings → `docs/admin/settings.md`, environment defaults →
  `docs/reference/environment.md`, endpoints → `docs/api/`.
- Prefer concrete values (defaults, status codes, limits) over adjectives. If you quote a
  default, quote the one in the code.
- The docs site is built in English and Chinese. When you edit a page that has a
  translated counterpart under `docs/zh/`, update both in the same pull request. Pages
  without a translation are linked to their English original.

## Commits and pull requests

- Use conventional commit subjects (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`,
  `chore:`), optionally scoped — releases are generated from them by release-please.
- Keep a pull request to one concern. Include the commands you ran, and any provider or
  protocol combination you tested manually.
- Add or update tests for behavior changes. Bug fixes should include a regression test
  that fails without the fix.

## License

By contributing, you agree that your contribution is licensed under the project's MIT
license (see [LICENSE](LICENSE)).
