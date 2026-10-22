---
pageClass: api-reference kicker-rerank
aside: false
---

# Rerank documents

Score a document list against a query and get the most relevant documents back,
highest score first. Cohere-compatible.

::: endpoint POST /v1/rerank
:::

`POST /v1/rerank` is the canonical route. `/v2/rerank` answers the same handler as
an alias, so Cohere's official SDK (`ClientV2.rerank`, which targets that path)
works unchanged.

```bash [Request]
curl http://localhost:8080/v1/rerank \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cohere/rerank-v3.5",
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital of France.",
      "Berlin is the capital of Germany."
    ],
    "top_n": 1
  }'
```

```python [Request — Cohere SDK]
import cohere

# ``ClientV2.rerank`` posts to ``/v2/rerank``; the proxy answers that path as an
# alias of the canonical ``/v1/rerank`` route, so the SDK works unchanged.
client = cohere.ClientV2(api_key="sk-your-proxy-key", base_url="http://localhost:8080")
res = client.v2.rerank(
    model="rerank-v3.5",
    query="What is the capital of France?",
    documents=[
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
    ],
    top_n=1,
)
print(res.results[0].index, res.results[0].relevance_score)
```

```json [Response]
{
  "id": "gen-rerank-1234567890-abc",
  "model": "cohere/rerank-v3.5",
  "provider": "Cohere",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.98,
      "document": {"text": "Paris is the capital of France."}
    }
  ],
  "usage": {"total_tokens": 150, "search_units": 1, "cost": 0.001}
}
```

## Request fields

`model` (required), `query` (required), `documents` (required, non-empty).

- `documents` entries are plain strings, or structured objects with `text` and/or
  `image` (a remote URL or base64 data URI) for multimodal rerankers. A document
  object without either is rejected with a `422`.
- `top_n` (`≥ 1`) caps how many documents come back. A value larger than the
  document count is clamped to it at the proxy edge — Cohere documents only that
  an omitted `top_n` returns every document, so clamping keeps a client that
  over-asks working instead of relying on an unstated upstream behaviour.
  Omitted, the upstream's own default applies.
- `return_documents` (default `true`): when `false`, the `document` field is
  omitted from every result — indexes and scores only. The projection is applied
  by the proxy; it is not forwarded upstream.
- `user` is accepted and forwarded to the providers that document it for this
  endpoint (OpenRouter); a provider that does not (Cohere) never sees it.

OpenRouter also documents `provider` (its routing block), `session_id` and
`trace`. They are accepted here and forwarded only to providers that document
them.

Cohere's v2-only knobs are accepted and forwarded to a `cohere` provider:
`max_tokens_per_doc` (truncate long documents), `priority` (admission order) and
`relevant_spans` (`{mode: enabled|disabled, threshold?, max_spans?}`, Rerank 5
and newer — turn it on to get the spans that drove each score back on
`results[].relevant_spans`).

A structured document is open-ended: a key the proxy does not model is carried
through to an upstream that documents it, not rejected.

## Response

`results` carries `index` (position in the request's `documents` list),
`relevance_score` (higher is better) and, unless suppressed, the `document`
echoed by the upstream. When the request enabled `relevant_spans` and the
upstream reported them, `relevant_spans` rides the result verbatim — the
upstream's own `{start, end, score}` objects. `id` appears when the upstream
reports one, and `provider` when a router (OpenRouter) names the provider that
answered.
`usage` is reported when the upstream sends one, in one vocabulary across
providers: `total_tokens`, `search_units` and, on a provider that reports it,
the billed `cost`. OpenRouter states those keys itself; Cohere's `meta.tokens`
and `meta.billed_units` are re-shaped onto them, and any other key a provider
adds to its usage block stays upstream instead of reaching the client.

There is no streaming variant: rerank answers in a single JSON body. A
provider-reported `cost` is used for billing in preference to the proxy's own
price estimate.

## Provider support

`cohere` (its native `/v2/rerank`, e.g. `rerank-v3.5`, `rerank-v5.0-pro`),
`openrouter` (its `/api/v1/rerank` route, e.g. `cohere/rerank-v3.5`,
`cohere/rerank-4-fast`, `cohere/rerank-4-pro`) and the `openai-compatible`
family, whose `/rerank` route is resolved from the provider's base URL (or a
`rerank` entry in its `endpoint_base_urls`) — e.g. vLLM, SGLang, Jina or a
self-hosted Cohere-compatible server. Native Cohere refuses an image-only
document (v2 takes text documents only). See the
[capability matrix](endpoints.md#capability-matrix).

The named model must carry the `supports_rerank` flag; a request naming a model
without it is rejected with a `400 invalid_request_error` before any provider is
called.

## Related

- [Errors & Rate Limits](errors.md)
