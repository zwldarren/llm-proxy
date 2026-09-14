---
pageClass: api-reference kicker-advanced
aside: false
---

# Tools, Reasoning & Web Search

## Tool calling

Tool calls work across protocols: an OpenAI client can call tools served by an
Anthropic upstream and vice versa.

| Aspect | Behavior |
| --- | --- |
| Tool types | `function`, `custom` (grammar/format), and built-in `web_search`/`web_search_preview` definitions |
| `tool_choice` | `auto` / `none` / `required`, `{"type":"function","name":…}`; Responses-style `allowed_tools` collapses to its mode and filters the tool list |
| Parallel calls | Supported; `parallel_tool_calls: false` disables parallel tool use |
| Unknown tool types | Accepted during validation, dropped when building the upstream request |
| Results | Send `tool` messages (Chat) / `tool_result` blocks (Anthropic) / `function_call_output` items (Responses) as usual |
| Custom tool calls | Returned re-wrapped as `type: "function"` with `arguments = {"content": <input>}` |

The proxy preserves protocol-specific extras such as tool `strict` flags, grammars,
and thought signatures where the upstream supports them.

## Reasoning

Thinking controls are accepted in every dialect and normalized:

| Client | Inputs |
| --- | --- |
| OpenAI Chat | `reasoning_effort` (`none`…`max`), `thinking` (bool or `{type, budget_tokens, effort, display}`), or a Responses-shaped `reasoning: {effort}` / `output_config: {effort}` |
| Anthropic | `thinking: {"type":"enabled","budget_tokens":N}` or an effort form |
| Responses | `reasoning: {effort, mode, context, summary}` |

Provider-specific behavior worth knowing:

- For Claude targets, `budget_tokens` must be ≥ 1024 and < `max_tokens` (400
  `invalid_parameter` otherwise), and `temperature`/`top_p` are omitted while thinking
  is enabled — a forced `tool_choice` (`any`/`tool`) disables thinking and restores
  them.
- Outbound, the proxy emits `reasoning_effort` when derivable, otherwise a `thinking`
  object.

How reasoning comes back:

| Protocol | Field |
| --- | --- |
| Chat Completions | `reasoning_content` (never `reasoning`), plus `reasoning_signature`; redacted blocks → `reasoning_content: "[redacted]"` + `reasoning_is_redacted: true` |
| Anthropic | `thinking` blocks with `signature`, and `redacted_thinking` blocks |
| Responses | Reasoning output items plus `response.reasoning_summary_text.delta` events while streaming; token counts in `output_tokens_details.reasoning_tokens` |

Round-tripping multi-turn reasoning: pass back what you received. On Chat Completions
input the proxy reads `reasoning_content` (with its signature) — an assistant
`reasoning` field is not read back.

## Web search

The proxy can answer `web_search` tool calls itself, so **every** model becomes
web-aware through your own search backend.

### Enable

1. **Settings → General → Web Search** — turn on interception and configure a backend:
   - **SearXNG**: `url` required; optional API key or basic auth; `engines` list;
     `timeout` (default 30 s); `max_results` (default 10, max 20). Queries over 4000
     characters are rejected.
   - **Ollama**: `api_key` required; `base_url` default `https://ollama.com`;
     `timeout`; `max_results` (max 10).
2. Nothing else — the change is hot-reloaded.

### How interception works

- Only providers **without native web search** are intercepted (the provider's
  *Native web search* flag controls this). With native support, your tools are passed
  through unchanged.
- In Chat Completions, a `web_search_options` field (or a `web_search` tool definition)
  enables it; the proxy replaces the tool with a plain function named `web_search`
  taking a single `query` string, and executes the model's call against your backend.
- The proxy re-calls the provider with the results until the model stops searching, and
  returns the **final answer** to the client — intermediate tool chatter is not exposed
  on non-streaming paths; usage across the re-calls is summed.
- `max_uses` on the tool definition is enforced per request.
- Requests taking the native-passthrough tier disable interception for that request.

The interceptor also reads further per-tool options; only some reach the backend:

| Option | Behavior |
| --- | --- |
| `allowed_domains`, `blocked_domains` | SearXNG filters results to/excluding these domains; ignored by Ollama |
| `user_location` | SearXNG: passed as `location` (city/region/country) with `language` derived from `timezone`; ignored by Ollama |
| `search_context_size`, `external_web_access`, `return_token_budget`, `search_content_types` | Captured from the tool definition but not enforced by any backend |

### What clients see

- Anthropic-protocol clients receive native `server_tool_use` + `web_search_tool_result`
  blocks, and the billing counter
  `provider_info.server_tool_use.web_search_requests`.
- Other protocols receive result blocks: `{"type":"web_search_result", "url", "title",
  "encoded_content": <base64 snippet>, "encoded_index": <base64 sha256>, "page_age"?}`.
  Note: this is base64 encoding, **not** Anthropic's encryption.
- Backend failures do **not** fail the HTTP request: the model receives a
  `web_search_tool_result_error` with a code —
  `too_many_requests` (429/timeout), `invalid_input` (400), `query_too_long`,
  `unavailable` (5xx), `invalid_api_key` (Ollama 401), `max_uses_exceeded`.

### Observability

Every intercepted search is logged (Logs → **Web Search**): query, provider, status,
result count, and current-vs-maximum use.

## MCP is separate

MCP servers are not injected into model requests — they are re-exposed as MCP endpoints
at `/servers/<name>/mcp` for MCP clients, with the same API keys, budgets, and logs.
See [MCP Servers](../admin/mcp.md).
