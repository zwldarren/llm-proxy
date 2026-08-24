# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

**Primary — admin-UI operators.** Individual hobbyists and developers who self-host the proxy and run its dashboard. They configure providers and models, manage API keys and MCP servers, set rate limits and security policy, and monitor usage, cost, and request logs. They are technical users who integrate multiple LLM providers and value efficiency and clarity over hand-holding. Their context: they are operating infrastructure they or others depend on, often opening the dashboard mid-incident or while debugging an integration — they need to trust what they see, fast.

**Secondary — API consumers (non-design surface).** Developers whose applications call the proxy's `/v1/*` and `/servers/*` endpoints. They interact via code, never the UI. Their needs (protocol compatibility, faithful streaming, predictable translation, sensible fallback) shape the proxy's behavior, which the dashboard exists to surface and control. They are not a design audience, but the UI must represent their world accurately.

**Secondary — teammates with dashboard access (access control, not a separate audience).** In deployments where the operator shares access, admin and user roles gate what a teammate can configure or see. This is a real but secondary dimension; most deployments are run by a single operator.

## Product Purpose

LLM Proxy is a self-hostable, multi-provider LLM API gateway. It normalizes many upstream providers (Anthropic, OpenAI, Gemini, DeepSeek, OpenRouter, Ollama, NanoGPT, Chutes, and any OpenAI-compatible backend) and many client protocols (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, plus utility endpoints for embeddings, images, and audio) behind one consistent API, with bidirectional protocol translation, priority-based provider fallback, SSE streaming, web-search interception, MCP tool bridging, distributed tracing and audit logging, per-token billing, and sliding-window rate limiting. A Vue 3 admin dashboard provides the operational control plane: configuration, monitoring, log analysis, and cost tracking.

It exists so a single self-hosted gateway can stand between many clients and many providers, with full operational visibility and live configuration. **Success** = an operator — including a stranger deploying it for the first time — can wire up providers, models, and keys once, point any compatible client at the gateway, and then confidently monitor usage, cost, and behavior in real time, trusting that what the dashboard shows is accurate and that every action gave clear feedback.

## Positioning

One self-hosted gateway that normalizes **N client protocols × M upstream providers with bidirectional translation.** Most proxies normalize one axis — many providers behind one client protocol, or one provider behind many protocols — but not both directions at once. The differentiator a neighbor could not truthfully copy is that an OpenAI-protocol client can be routed to an Anthropic provider (and vice versa) with faithful streaming, tool-calling, and MCP bridging across the translation boundary, all behind a single consistent API the operator owns.

## Operating Context

- **Deployment:** self-hosted. MIT-licensed single codebase; runs via `uv run llm-proxy` (default `http://localhost:9911`) or Docker / `docker-compose`. The frontend is served by the backend in the default same-port deployment, and reverse-proxyable to a separate origin when separating the admin UI.
- **First run:** when no admin exists yet, the dashboard redirects to a one-time **Setup** screen to create the first admin account before anything else is usable. After that, login gates every route.
- **Primary surfaces:** the **config console** (Providers, Models, API Keys, MCP Servers, Settings) and **monitoring / logs** (request logs, usage, cost, status). These are where operator time and the strongest visual identity live.
- **Secondary surfaces:** an in-dashboard **Chat** playground and **Images** playground exist as operator test beds against configured models — diagnostic tools for verifying a wiring works, not consumer-facing products. They must not pull the product toward a chat-app or creative-tool aesthetic.
- **Real usage scene:** operators open the dashboard mid-incident or while debugging an integration, often with several providers or a fallback chain in play. They need to confirm "did that request hit the right provider, stream correctly, cost what I expect" without leaving the page.
- **Data the dashboard must represent faithfully:** API details, protocol specifics, HTTP methods, status codes, token counts, costs, latencies, and timestamps. Hiding these behind friendly abstractions is a failure mode, not a polish.

## Capabilities and Constraints

- **Protocols:** OpenAI Chat Completions (`/v1/chat/completions`), OpenAI Responses (`/v1/responses`), Anthropic Messages (`/v1/messages`), plus utility endpoints (embeddings, images, audio). Bidirectional translation across all of them.
- **Providers:** Anthropic, OpenAI, Gemini, DeepSeek, OpenRouter, Ollama, NanoGPT, Chutes natively; any OpenAI-compatible backend via a generic adapter.
- **Routing & fallback:** priority-based provider selection with fallback chains; per-provider / per-model parameter overrides.
- **Streaming:** SSE throughout, with protocol-specific streaming transformers.
- **Tooling:** web-search interception (converts `web_search` to function calls); MCP server bridging (servers configured per provider via `mcp_servers`).
- **Observability & cost:** distributed tracing, audit logging (with configurable sampling / retention / sensitive-data masking), per-token billing, sliding-window rate limiting.
- **Access control:** admin and user roles; rate-limited login with lockout after repeated failures.
- **Storage:** migrations are the single source of truth for schema (no `create_all` fallback); SQLite (batch-mode DDL) or PostgreSQL.
- **Constraints:** Python 3.14+; `uv` for backend, `bun` for frontend dev server. The dashboard must read as trustworthy to someone seeing it cold, because the proxy is self-hosted by others — not just by its author.
- **Undecided / open:** whether a hosted tier will ever exist (currently none); whether the playgrounds grow beyond operator test beds. Treat both as undecided rather than committed.

## Brand Commitments

- **Name:** LLM Proxy.
- **License:** MIT (recorded in README badges; no proprietary claims).
- **Personality:** Technical Sophistication, Reliable, Modern.
- **Voice:** Direct, precise, professional without being corporate.
- **Tone:** Confident but not arrogant; helpful without being verbose.
- **Emotional goals:** The operator feels in control, trusts the system, and accomplishes tasks quickly without confusion.
- **No binding logo, identity asset, or external brand reference has been confirmed.** Do not invent one.

## Evidence on Hand

- `README.md` — public overview, supported protocols/providers table, quick start, architecture diagram.
- `AGENTS.md` / `CLAUDE.md` — architecture, commands, style conventions, design context pointers.
- `frontend/src/router/index.ts` — the real route list (login, setup, dashboard, chat, images, logs, models, config/*, team).
- `frontend/src/stores/auth.ts` — first-run setup gate, admin/user roles, login flow.
- `.env.example` — user-facing config surface and the UI-managed-vs-env-only split.
- **Absences future work must not fabricate:** no real testimonials, customer logos, case studies, usage metrics, or benchmarks have been provided. Do not invent any.

## Product Principles

1. **Clarity Over Cleverness** — Every element has a clear purpose. Labels are explicit, icons are standard, hierarchy is obvious. A stranger reading the dashboard cold should understand it without a tour.
2. **Confidence Through Feedback** — Actions have visible, immediate feedback. The operator never wonders "did that work?"
3. **Technical Sophistication** — Embrace the technical nature of the product. Show API details, protocol specifics, and metrics clearly; do not hide them behind friendly abstractions.
4. **Dark-First Design** — The primary experience is dark mode (deep cool-neutral). Light mode is well-supported but secondary. Both are first-class engineering, not afterthoughts.
5. **Respectful of Time** — Power users navigate quickly: keyboard shortcuts, collapsible sections, efficient tables, dense-but-legible layouts. Density is earned through clarity, never through clutter.

## Accessibility & Inclusion

- **Contrast:** WCAG AA minimum — body text ≥4.5:1, large text ≥3:1 — against both dark and light backgrounds. Muted text is tuned for readability, not for "elegance."
- **Motion:** Respect `prefers-reduced-motion` everywhere; provide crossfade or instant alternatives for every animation.
- **Focus:** Visible focus rings on all interactive elements; skip-to-content link on every screen.
- **Color redundancy:** Color is never the sole signal for status, HTTP method, or error — always paired with text, icon, or shape, so the low-saturation semantic tints remain legible to color-blind users.
- **Internationalization:** All user-facing text is internationalized (English primary, Chinese secondary); layout must survive translation length variance.