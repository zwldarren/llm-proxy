# A hosted tool is a dialect fact, not a function

## Status

Accepted (2026-10).

## Context

Clients declare *hosted* tools — tools the upstream executes: OpenAI's
`web_search` / `file_search`, Anthropic's `web_search_20250305` family, Gemini's
`google_search` / `url_context` / `google_maps`, xAI's `x_search`, Qwen's
`web_extractor`, Z.ai's `retrieval`, Kimi's `builtin_function` `$web_search`.
The proxy modelled three of them individually (`WebSearchTool` on both families,
`OpenAIToolSearchTool`) and handled every other one in one of two unsatisfying
ways:

- the Chat Completions intake (`_parse_tools`) had no `else` branch, so an
  unmodelled type was dropped silently — no warning, no tool, no trace;
- the Responses intake preserved unmodelled types as raw dicts in
  `extra["responses_tools"]`, a carrier only a native Responses wire could
  forward, and which the Chat Completions builder strips wholesale.

The outbound side had the mirror-image problem: `OpenAIToolsHandler.build_tools`
rewrote *every* hosted `web_search` into a `{"type": "function", "name":
"web_search"}` bridge. That is right only when a function tool is what the
request needs. For a hosted declaration it is wrong in both directions — the
caller cannot execute a function it never declared, and the upstreams that do
document server-side search on their chat wire (Z.ai's
`{"type": "web_search", "web_search": {...}}`, Qwen's `enable_search`, OpenRouter's
`openrouter:web_search` server tool, NanoGPT's `web_search` body object) never
saw the declaration. `native_web_search: true` was consequently a dead switch on
every Chat Completions provider: the pipeline stage stopped intercepting, and
the serializer then downgraded the tool anyway.

## Decision

Hosted tools are carried as data and rendered per provider dialect.

1. **`HostedTool`** (`models/tools/hosted.py`) carries a hosted declaration the
   proxy has no typed model for: the wire `type` plus the remaining fields
   verbatim. Intake on *both* protocols produces it for the provider-hosted types
   (`google_search`, `google_search_retrieval`, `url_context`, `google_maps`,
   `x_search`, `web_extractor`, `retrieval`, `builtin_function`) and, with a
   warning, for types the proxy has never seen.

   OpenAI's *own* Responses-only tools without typed models (`mcp`, computer
   use, image generation, future types) are the deliberate exception: they are
   native to a Responses wire, which accepts them verbatim, so the Responses
   intake keeps them in the preserved raw-tools carrier and only a native
   Responses provider forwards them; every other wire drops the carrier with a
   warning naming the carried types. Converting them to `HostedTool` would hand
   the dialect table — not OpenAI — the decision whether a future built-in
   reaches OpenAI.
2. **One dialect table per wire family** (`serialization/hosted_tools.py`), keyed
   by registered adapter type (`BuildContext.provider_type`):
   `CHAT_SEARCH_DIALECTS` (how a chat wire asks for server-side search, as a
   `tools[]` entry *or* a body field), `CHAT_SEARCH_UNSUPPORTED` (the chat wires
   that cannot search, with the reason), `CHAT_HOSTED_TYPES`,
   `RESPONSES_HOSTED_TYPES`, `GEMINI_HOSTED_TYPES`, `INTERACTIONS_HOSTED_TYPES`.
   A provider absent from a table keeps the pre-dialect behaviour: the Responses
   and Gemini wires forward the declaration verbatim (their `tools[]` union is
   open-ended and aggregators extend it), while the chat wire falls back to the
   function bridge, because every Chat Completions upstream accepts functions.
3. **Hosted search is rendered, or dropped with its reason.** On a chat wire with
   a dialect it becomes that dialect's shape; on a wire whose documentation says
   chat has no server-side search (DeepSeek, Mistral, Kimi, xAI, MiniMax, Ollama,
   …) it is dropped with a warning naming the reason. It is never turned into a
   function tool the caller cannot execute.
4. **Deprecated shapes are preserved, not generated.** Kimi's `$web_search`,
   OpenRouter's `plugins` / `:online` and Mistral's `web_search` are documented
   upstream as deprecated or unsupported on the wire this proxy speaks; the proxy
   never synthesizes them. Forwarding applies to what a client's declaration
   survives intake as: Kimi's `builtin_function` arrives as a `HostedTool` and is
   forwarded with a deprecation warning. OpenRouter's `plugins` / `:online` are
   body-level spellings outside the tool list, and Mistral's `web_search` exists
   only on the Agents / Conversations APIs — neither arrives as a tool
   declaration the intake can carry, so for them "never synthesized" is the
   operative half; a search declared in a portable spelling follows decision 3.
5. The pipeline stage stays the policy owner (`native_web_search`: intercept or
   pass through) and the serializer stays the wire owner (what shape the selected
   upstream consumes). No flag is threaded into the serializer: if a hosted tool
   reaches it, either the provider opted into native handling or no interceptor is
   configured, and rendering the provider's own shape is correct in both cases.

## Consequences

- `native_web_search` now means the same thing on every wire: a hosted `web_search`
  survives to the upstream in that upstream's own shape.
- A hosted declaration can no longer come back as a `function_call` the caller
  cannot answer — the failure mode becomes "search dropped, warning names why",
  which an operator can act on. The one survivor is a Chat Completions upstream
  absent from both tables, where the pre-dialect function bridge remains the
  fallback by decision 2 (every Chat Completions wire accepts functions).
- One table per wire family is the only place that knows a provider's hosted-tool
  vocabulary; adding a provider is a table entry plus a test, not a serializer
  branch. The tables are transcribed from vendor docs, so a vendor change is a
  one-line diff.
- `web_search_options`, the Chat-Completions spelling of the same search intent,
  is translated into the hosted tool on wires that have no such field
  (`with_web_search_options`). A serializer may synthesize the hosted tool from
  a *parameter*; it never invents a provider's deprecated shape.
- The two cross-family spellings of one hosted tool (OpenAI's `web_search` and
  Anthropic's `web_search_20250305`) project through sibling helpers in
  `serialization/hosted_tools.py` (`openai_web_search_to_anthropic` /
  `anthropic_web_search_to_openai`) — the same single home as the tables.
- History replay is unaffected: `web_search_call` items in a Responses history
  still replay as `web_search` tool calls (that path is about a *response*, not a
  declaration), and the interception bridge still supplies the function tool the
  proxy itself executes.
- Billing for the new wires is not folded in yet: Qwen reports search counts under
  `usage.x_tools.web_search.count`, which the usage intake does not read.
