# Gemini: family-aware request shapes and the caches the wire cannot carry

## Status

Accepted (2026-10).

## Context

Google's `generateContent` API changed shape between model families, and every
change is a shape the *upstream* accepts or rejects — not a preference:

- **2.5** takes `thinkingConfig.thinkingBudget`; `thinkingLevel` there is an
  error. Structured output uses the OpenAPI-subset `responseSchema` /
  `FunctionDeclaration.parameters`.
- **3.x** takes `thinkingLevel` (with `includeThoughts` to actually receive
  thought parts), enforces thought signatures for function calling, and accepts
  plain JSON Schema through `generationConfig.responseFormat` /
  `FunctionDeclaration.parametersJsonSchema` — which supersede the deprecated
  `responseSchema` / `parameters` pair.
- **3.6+ / 4.x, and 3.5 Flash-Lite** ignore `temperature`, `top_p` and `top_k`
  (deprecation headers today, 400s announced). The release notes start the
  deprecation at Gemini 3.6 Flash **and Gemini 3.5 Flash-Lite**; 3.5 Flash
  proper and 3.1 Flash-Lite still accept the parameters.
- **Gemini 4** (`gemini-4-argon`) is a thinking-level family like 3.x, does not
  support `thinkingBudget` alongside a level, and enforces signatures as
  strictly as Gemini 3.

The proxy keyed all of this off substrings (`"gemini-2.5" in model`,
`"gemini-3" in model`), so a brand-new family silently fell into the "not
supported" branch: Gemini 4 requests lost their thinking config, and — worse —
their function-calling history got no thought signature at all, which the API
rejects with a 400.

Two facts Gemini returns have no field on the OpenAI/Anthropic wires the client
answers on, so a stateless proxy cannot receive them back:

1. **`toolCall` / `toolResponse` parts** — Gemini 3's tool context circulation
   returns the built-in tools' invocations and results inside the model content.
   The docs ask for them back verbatim, in place, in every subsequent turn; they
   are also what "combine google_search with function calling" depends on.
2. **A `thoughtSignature` on a text part** — Gemini 3/4 put one on the last part
   of a response. Omitting it is not an error, but it is a documented quality
   loss.

## Decision

**One capability table keyed by the model family's version**, in
`serialization/gemini/model_family.py`: `gemini-X.Y` is read off the model id and
every gate (`supports_thinking_config`, `uses_thinking_level`,
`supports_thought_signatures`, `supports_json_schema_fields`,
`supports_minimal_thinking_level`, `ignores_sampling_params`) answers from it.
Gates are stated as "from version V on", so the next family inherits the modern
shape instead of landing in a legacy else-branch; an unparseable model id keeps
the legacy-safe answer rather than guessing.

**A disabled request asks for the lowest level the model accepts.** Gemini 3/4
cannot turn thinking off. `MINIMAL` ("little to no thinking") is emitted where
Google documents it, `LOW` where `MINIMAL` is documented as an error (3.8 Flash
and 4.x — note Gemini 3.1 Flash-Lite **Image** takes `minimal` as its
*default* while `low`/`medium` are the levels it rejects), and the request is
left untouched when the family has no documented floor.

**`computer_use` carries the documented `environment`.** The reference marks
`ComputerUse.environment` required and every example sets it, so the proxy
supplies `ENVIRONMENT_BROWSER` unless the client named one; `file_search` is
dropped rather than sent without its required `fileSearchStoreNames`.

**Three class-level caches hold what the wire cannot echo**, all LRU-bounded like
the existing thought-signature cache (`GeminiAdapter`):

- `_thought_signature_cache` — unchanged: `tool_call_id → thoughtSignature`.
- `_tool_context_cache` — `tool_call_id → [toolCall, toolResponse, …]`, replayed
  immediately before the same `functionCall` part on the next request.
- `_text_signature_cache` — `exact text → thoughtSignature`. A text part has no
  id, so the text the client echoes back is the only handle available.

**Function ids are round-tripped**: `FunctionCall.id` and `FunctionResponse.id`
now carry the same value, which the docs state as a requirement from Gemini 3 on.

**The dummy signature goes on the first `functionCall` of each step only** —
Google's own responses leave subsequent parallel calls unsigned, and the
validated part is the first one.

## Considered Options

- **Keep substring checks, add `"gemini-4"`**: rejected — the same bug returns
  with the next family, and the checks were already duplicated (two identical
  `_is_gemini_25_model` definitions).
- **Send `VALIDATED` explicitly when tool context circulation is on**: rejected —
  the docs say the flag *defaults* to `VALIDATED` and that `AUTO` is unsupported
  there, so omitting `functionCallingConfig` is the documented default without
  inventing an enum value the v1beta reference does not list.
- **Drop `toolCall` / `toolResponse` parts and accept the loss**: rejected —
  they are load-bearing for built-in + function combinations and carry
  signatures.
- **Represent the parts as a new content block**: rejected — a block would have
  to be re-assembled into the exact part sequence and its `thoughtSignature`
  sibling; carrying the raw parts on the `ToolUseBlock` that preceded them
  (`extra["gemini_tool_parts"]`) is the smallest faithful carrier, and it rides
  the same cache-and-reattach path as the signature.

## Consequences

- Gemini 4 (and any future family) gets the thinking, signature, JSON-Schema and
  sampling behaviour its documented shape requires.
- Gemini 3 built-in + function combinations now send
  `includeServerSideToolInvocations` and replay the tool context, which is the
  documented way to make tool context circulation work.
- Tool schemas from MCP/OpenAI clients keep `anyOf`, `$ref` and
  `additionalProperties` on Gemini 3+, where they used to be stripped.
- The three caches are process-lifetime state keyed by ids/texts the client
  echoes; a cache miss degrades to the previous behaviour (no signature / no tool
  context), never to a wrong field.
- `CONTINUATION` folds into `length`, and the token that resumes the truncated
  turn is surfaced on both wires: `provider_info["continuationToken"]` on a
  buffered response, and a `continuation_token` field on the streaming chunk
  (which the fidelity-first OpenAI transformer forwards verbatim).
- Family rules are transcribed facts with dates; a new family means updating the
  one table, not hunting for substring checks. A model id the table cannot
  parse (Google aliases like `gemini-flash-latest`, aggregator nicknames)
  falls back to the legacy-safe answer and logs a warning once per id; an
  operator can pin such an id to a family in the provider metadata
  (`metadata.model_family_overrides`, exact model id → version like `"4.0"`)
  — the pin is consulted before the regex, which turns "new model with
  surprising capabilities" from a code change into a config change.
