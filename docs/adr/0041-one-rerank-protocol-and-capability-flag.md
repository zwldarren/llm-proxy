# One rerank protocol behind one capability flag

## Status

Accepted (2026-11).

## Context

Retrieval had one client-facing endpoint: `/v1/embeddings`. Reranking is a
different operation — an embedding maps each document to a vector independently,
while a reranker scores the candidate documents *jointly* against the query and
returns an ordered `(index, relevance_score)` list, never generated text. The
providers the proxy supports happen to agree on the wire: Cohere's
`POST /v1/rerank` shape is what OpenRouter's `/api/v1/rerank` and the
vLLM/SGLang/Jina `/rerank` route also speak. Cohere's own v2 API serves the same
body at `POST /v2/rerank`, and its official SDK targets that path.

Two questions had to be answered before writing the endpoint: whether a rerank
request rides the existing retrieval capability (`supports_embedding`) or needs
its own flag, and whether the five backends each get a wire or share one.

## Decision

**One protocol, one capability flag, one shared wire, and adapters that differ
only where their vendor does.**

- **The protocol owns `/v1/rerank`.** `protocols/rerank/` is a normal protocol
  module: `ProtocolEndpoint`, a client schema, and a `ProtocolSerializer`
  registered in the protocol registry. The wire is the shared Cohere-compatible
  shape; `/v2/rerank` is registered as an alias of the same handler (Cohere's
  SDK posts there) by declaring both paths on the endpoint, not by a second
  module.
- **The internal pair is protocol-agnostic.** `InternalRerankRequest` /
  `InternalRerankResponse` live in `models/rerank.py`. Documents are
  `str | RerankDocument`, where a structured document carries `text` and/or
  `image` (multimodal rerankers); a document object with neither is a client
  error.
- **The provider side is one mixin plus serializer defaults.**
  `RerankCapabilityMixin` owns the call (`_build_outbound_body(request_type=
  "rerank")`, one `_post_json_response_with_retry`, rate-limit headers), and
  `ProviderSerializer` carries the shared body builder and response parser. A
  provider overrides only its own deviation: Cohere takes plain-string documents
  and echoes none back (the adapter projects the request's own documents back,
  since the client-facing protocol reports them), and an image-only document is
  refused there rather than flattened.
- **`supports_rerank` is its own flag.** A reranker is not an embedding model,
  and the two endpoints are served by disjoint model sets at most vendors, so
  reusing `supports_embedding` would either over-grant or under-grant one of
  them. The flag is enforced in `api/context.py` before provider selection, so a
  chat-only model naming `/v1/rerank` gets a `400 invalid_request_error` instead
  of an upstream 404; the flag does not restrict the model's chat service.
- **The shape is deliberately small.** No streaming (one query, one scored
  list); `top_n` above the document count is clamped at the proxy edge — Cohere
  documents only that an omitted `top_n` returns every document, so clamping
  keeps a client that over-asks working without depending on an upstream
  behaviour nobody states; `return_documents` is a response projection the proxy
  applies itself and is never forwarded; `user`, `provider`, `session_id` and
  `trace` ride `extra` and reach only upstreams that document them, through the
  existing unknown-field policy.
- **A request field the proxy forwards is a response field it carries back.**
  Cohere's v2-only knobs (`max_tokens_per_doc`, `relevant_spans`, `priority`)
  are declared on the request schema and forwarded through `extra`, and
  `results[].relevant_spans` is passed through verbatim on the way back: a
  client that enabled the feature upstream would otherwise get an empty list.
  Spans are the upstream's own `{start, end, score}` objects, so the proxy
  re-models nothing. A structured document is open for the same reason — a
  document key the proxy does not model (`RerankDocument.extra`) rides through
  to the upstream that documents it rather than being rejected.
- **Usage lands in the intake seam.** The client-facing usage echo is normalized
  to `total_tokens` / `search_units` / `cost`, which is what OpenRouter states
  verbatim and what Cohere's `ApiMeta` is re-shaped onto.

## Consequences

- Adding a rerank backend is a capability mixin on the adapter plus, at most, a
  serializer override; the client wire and the endpoint do not move.
- The published capability matrix and the model-form flag list have one more
  row each, and `/v1/rerank` names its gate in the endpoint table.
- Cohere's native provider is the only implementation that needs a rerank
  override today (text-only documents, no echoed document, a normalized usage
  block); a future provider with the same deviations can reuse that serializer
  or share its own.
- Cohere's SDK works unchanged against the proxy because the alias exists; a
  client that hard-codes `/v2/rerank` on a rerank provider without a v2 path
  still fails, which is the correct amount of knowledge for the proxy to have.

## Alternatives rejected

- **Reuse `supports_embedding`.** Conflates two capabilities and forces a
  provider that serves only one of them to lie, or makes the gate untestable per
  endpoint.
- **One wire per provider.** OpenRouter, vLLM, SGLang, Jina and Cohere all serve
  the same body shape; five serializers would restate one.
- **An OpenAI-shaped rerank endpoint.** No vendor defines one; Cohere's is the
  de-facto shared wire, so inventing a different one would translate on both
  sides for no gain.
