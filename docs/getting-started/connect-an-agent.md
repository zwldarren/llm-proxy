# Connect an AI Agent

Coding agents are just API clients — they work through the proxy like any SDK
call. Create an [API key](../admin/api-keys.md), pick a model, then point the
agent's provider config at the proxy.

Every agent below uses one of the two wire formats the proxy already serves:

| Agent | Configuration file | Wire |
| --- | --- | --- |
| [Claude Code](#claude-code) | `~/.claude/settings.json` | Anthropic Messages (`/v1/messages`) |
| [OpenAI Codex](#openai-codex) | `~/.codex/config.toml` | Responses (`/v1/responses`) |
| [OpenCode](#opencode) | `opencode.json` | Chat Completions (`/v1/chat/completions`) |
| [Pi](#pi) | `~/.pi/agent/models.json` | Chat Completions |
| [OMP](#omp) | `~/.omp/agent/models.yml` | Chat Completions |
| [Hermes Agent](#hermes-agent) | `~/.hermes/config.yaml` | Chat Completions |

## Before you start

1. **Create an API key** in **Configuration → API Keys** — see
   [API Keys](../admin/api-keys.md). Agents authenticate with it exactly like an SDK.
2. **Note the model name to use** — a configured model name, or the virtual models
   `auto` / `fast` / `best` when [smart routing](../api/routing.md) is enabled:

   ```bash
   curl http://localhost:8080/v1/models -H "Authorization: Bearer sk-your-proxy-key"
   ```

3. **Base URL** — `http://localhost:8080/v1` for OpenAI-wire agents,
   `http://localhost:8080` for Anthropic-wire agents (they append `/v1/messages`
   themselves). Replace `localhost` with the host running the proxy if the agent
   runs in a container, VM, or another machine.

Most agents make small background calls too (session titles, summaries) with a
different model than the main one. Make sure those model names are configured and
allowed for the key as well, or point them at the same model.

## Claude Code

Claude Code speaks the Anthropic Messages API. Set the endpoint and key through
environment variables, either in your shell or once in the `env` block of
`~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8080",
    "ANTHROPIC_AUTH_TOKEN": "sk-your-proxy-key",
    "ANTHROPIC_MODEL": "your-model-name",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "your-cheap-model"
  }
}
```

`ANTHROPIC_AUTH_TOKEN` is sent as `Authorization: Bearer …`; the proxy accepts it
(and `ANTHROPIC_API_KEY`, which is sent as `x-api-key`) on every `/v1/*` route.

- `ANTHROPIC_MODEL` is the main model. `ANTHROPIC_DEFAULT_HAIKU_MODEL` covers
  background work (titles, summaries) — point it at something cheap.
- To keep using the `/model` picker, set `ANTHROPIC_DEFAULT_SONNET_MODEL`,
  `ANTHROPIC_DEFAULT_OPUS_MODEL`, and `ANTHROPIC_DEFAULT_HAIKU_MODEL` to proxy
  model names, or add `"CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1"` so
  Claude Code lists the proxy's `GET /v1/models` output directly.

Shell equivalent:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=sk-your-proxy-key
export ANTHROPIC_MODEL=your-model-name
claude
```

::: tip Keep Claude's default model IDs working
Claude Code resolves its built-in aliases (`sonnet`, `opus`, `haiku`) to real
Anthropic model IDs unless you override them. Instead of configuring every
machine, create model records in the proxy whose **Name** matches those IDs
(e.g. `claude-sonnet-4-5`) and map each to an upstream provider model — then a
stock Claude Code installation works unchanged. See
[Models & Pricing](../admin/models.md).
:::

If the proxy rejects beta request fields with errors like
`Unexpected value(s) for the anthropic-beta header` or
`Extra inputs are not permitted`, set `"CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1"`.
On a non-Anthropic base URL, MCP tool search is disabled by default; set
`"ENABLE_TOOL_SEARCH": "true"` only if your upstreams forward `tool_reference`
blocks.

## OpenAI Codex

Codex talks to the Responses API, which the proxy serves at `/v1/responses`.
Define a custom provider in `~/.codex/config.toml` (provider settings are
user-level only — project-local `.codex/config.toml` files can't set them):

```toml
# ~/.codex/config.toml
model = "your-model-name"
model_provider = "llm-proxy"

[model_providers.llm-proxy]
name = "LLM Proxy"
base_url = "http://localhost:8080/v1"
env_key = "LLM_PROXY_KEY"
wire_api = "responses"
```

```bash
export LLM_PROXY_KEY=sk-your-proxy-key   # add to ~/.bashrc / ~/.zshrc
codex
```

- `env_key` names the environment variable Codex reads the key from; it is never
  stored in the config file.
- `wire_api = "responses"` is the only supported value on current Codex versions
  and can be omitted (it is the default).
- If Codex warns about an unknown context window for your model ID, add
  `model_context_window = 200000` (or the value configured for that model) at the
  top level so auto-compaction triggers at the right point.

## OpenCode

OpenCode reads providers from `opencode.json` — project-local or global at
`~/.config/opencode/opencode.json`. Any OpenAI-compatible endpoint plugs in
through `@ai-sdk/openai-compatible`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "llm-proxy/your-model-name",
  "small_model": "llm-proxy/your-cheap-model",
  "provider": {
    "llm-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LLM Proxy",
      "options": {
        "baseURL": "http://localhost:8080/v1",
        "apiKey": "sk-your-proxy-key"
      },
      "models": {
        "your-model-name": { "name": "Your Model" },
        "your-cheap-model": { "name": "Your Cheap Model" }
      }
    }
  }
}
```

- Each key under `models` is the ID the proxy expects (from `GET /v1/models`).
- `small_model` covers title generation and other side jobs; without it OpenCode
  falls back to a model hosted by the OpenCode team.
- Set `options.apiKey` from `opencode auth login` → **Other** instead of writing it
  into the file, if you prefer credentials in the auth store — use `llm-proxy` as
  the provider ID.

## Pi

Pi loads custom providers from `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "llm-proxy": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "apiKey": "$LLM_PROXY_KEY",
      "models": [
        { "id": "your-model-name", "name": "Your Model" },
        { "id": "your-cheap-model", "name": "Your Cheap Model" }
      ]
    }
  }
}
```

```bash
export LLM_PROXY_KEY=sk-your-proxy-key
pi --model llm-proxy/your-model-name
```

- `apiKey` supports `$ENV_VAR` interpolation, so the key stays out of the file.
- The file is re-read whenever `/model` opens — no restart needed after edits.
- `api` also accepts `openai-responses` or `anthropic-messages` if you prefer a
  different wire per provider.

## OMP

OMP (oh-my-pi) reads custom providers from `~/.omp/agent/models.yml`:

```yaml
# ~/.omp/agent/models.yml
providers:
  llm-proxy:
    baseUrl: http://localhost:8080/v1
    api: openai-completions
    apiKey: LLM_PROXY_KEY # env-var name, or the literal key
    models:
      - id: your-model-name
        name: Your Model
      - id: your-cheap-model
        name: Your Cheap Model
```

```bash
export LLM_PROXY_KEY=sk-your-proxy-key
omp --model llm-proxy/your-model-name
```

`apiKey` is resolved as an environment-variable name first and as a literal
second, so both forms work. To have OMP discover the proxy's models instead of
listing them, drop the `models` block and add:

```yaml
    api: openai-completions
    discovery:
      type: openai-models-list
```

Discovered models carry no pricing metadata, so OMP reports local estimates as
unknown until you override `cost` per model.

## Hermes Agent

Hermes treats any OpenAI-compatible endpoint as a custom provider. The quickest
path is the interactive wizard:

```bash
hermes model
# → Custom endpoint (self-hosted / VLLM / etc.)
# → Base URL: http://localhost:8080/v1
# → API key:  sk-your-proxy-key
# → Model:    your-model-name
```

Or write it directly — secrets belong in `~/.hermes/.env`:

```dotenv
# ~/.hermes/.env
LLM_PROXY_KEY=sk-your-proxy-key
```

```yaml
# ~/.hermes/config.yaml
model:
  provider: custom
  default: your-model-name
  base_url: http://localhost:8080/v1
  api_key: ${LLM_PROXY_KEY}
```

To also switch to it mid-session (or as an auxiliary model), register it as a
named provider and use `/model custom:llm-proxy:<model>`:

```yaml
providers:
  llm-proxy:
    api: http://localhost:8080/v1
    api_key: ${LLM_PROXY_KEY}
    models:
      - your-model-name
      - your-cheap-model
```

`hermes model` changes apply to new sessions; use `/model` inside a running chat
to switch immediately.

## Related

- [API Keys](../admin/api-keys.md) — create and scope the key agents authenticate with
- [Models & Pricing](../admin/models.md) — client-facing names, mappings, pricing
- [Virtual Models & Routing](../api/routing.md) — `auto` / `fast` / `best`
- [Streaming](../api/streaming.md) — SSE details, heartbeats, cancellation
