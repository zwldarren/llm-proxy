---
pageClass: api-reference kicker-decisions
aside: false
---

# Decisions

Answer a list of typed questions about shared evidence and get one typed answer
per question. This is OpenAI's Decisions endpoint (`POST /v1/decisions`,
model `gpt-6-luna`), and it is also served by the decision models the proxy
already reaches through [System One](systemone.md) — the two endpoints are one
capability in two envelopes, and the proxy translates between them.

::: endpoint POST /v1/decisions
:::

```bash [Request]
curl http://localhost:8080/v1/decisions \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-6-luna",
    "input": "I was charged twice for my order.",
    "questions": [
      {
        "type": "predicate",
        "name": "urgent",
        "instructions": "Does this customer need a reply today?"
      },
      {
        "type": "choice",
        "name": "department",
        "instructions": "Which department should handle this complaint?",
        "choices": [
          {"value": "billing", "description": "Payments, invoices, and refunds."},
          {"value": "technical", "description": "Problems using the product."},
          {"value": "other", "description": "Requests outside these categories."}
        ]
      },
      {
        "type": "score",
        "name": "severity",
        "instructions": "How severe is this issue?",
        "levels": [
          {"label": "Cosmetic", "description": "Appearance only; no lost functionality."},
          {"label": "Workaround available", "description": "A task fails, but another way works."},
          {"label": "Fully blocked", "description": "A task fails with no workaround."}
        ]
      }
    ]
  }'
```

```json [Response]
{
  "model": "gpt-6-luna",
  "answers": [
    {"type": "predicate", "name": "urgent", "probability": 0.92},
    {
      "type": "choice",
      "name": "department",
      "choice": "billing",
      "probabilities": [
        {"value": "billing", "probability": 0.95},
        {"value": "technical", "probability": 0.02},
        {"value": "other", "probability": 0.03}
      ],
      "confidence": 0.93
    },
    {
      "type": "score",
      "name": "severity",
      "score": 1.1,
      "probabilities": [
        {"value": 0, "label": "Cosmetic", "probability": 0.1},
        {"value": 1, "label": "Workaround available", "probability": 0.7},
        {"value": 2, "label": "Fully blocked", "probability": 0.2}
      ],
      "confidence": 0.55
    }
  ],
  "usage": {
    "input_tokens": 296,
    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 296
  }
}
```

## Request fields

`input` (required) is the shared evidence every question reads: a plain string,
or an array of user messages holding `input_text` and `input_image` parts.
Images must be **inline base64 data URLs** (`data:image/png;base64,…`); hosted
HTTP(S) URLs and file ids are not supported. At most 128 image parts are allowed
across all messages in one request.

`model` (required) selects the decision model. Point the model at a provider
whose type supports evaluation — `openai` for the Decisions endpoint itself, or
`typesafe`, `openrouter` and `ollama` for the System One models the proxy
bridges onto it.

`questions` (required) is the ordered list of questions; answers come back in the
same order. Every question has a `type` and `instructions`, and may carry a
`name` — a string when given (explicit `null` is rejected) — that the answer
echoes. Names must be unique, and a name starting with
`__unnamed_question_` is rejected: the proxy uses that prefix to key unnamed
questions when translating for a System One upstream, so a client name under
it would collide with one. An unnamed question answers with `"name": null`,
the shape the endpoint itself emits.

| `type` | Shape | Extra fields |
| --- | --- | --- |
| `predicate` | Yes/no question; returns the probability the condition is true | — |
| `choice` | Picks one option; returns the option and the full distribution | `choices`: non-empty list of `{value, description?}` (at most 255). `value` is a **string or a boolean** — `true` and `"true"` are distinct options |
| `score` | Rates along a rubric; returns a probability-weighted value | `levels`: 2–26 ordered `{label, description?}`, lowest first |

`safety_identifier` is accepted and forwarded to upstreams that document it
(`openai`); the System One upstreams have no equivalent field and drop it.

Unknown top-level fields are rejected with a `422`.

### One endpoint, two envelopes

System One and Decisions answer the same three primitives. The proxy keeps both
endpoints available on every evaluation model and translates at the provider:

| | Decisions | System One |
| --- | --- | --- |
| Evidence | `input` (string or messages) | `state` (string, object or array) |
| Questions | ordered array, each with a `name` | map of id to question |
| Yes/no | `predicate` | `noul` |
| Pick one | `choices[{value, description}]` | `criteria{value: description}` |
| Rate | `levels[{label, description}]` | `criteria[level, …]` |
| Answers | ordered array, echoing `name` | map keyed by question id |

So a request to `/v1/decisions` naming a Jev model is converted to a System One
request, sent to `/v1/systemone`, and its answers converted back — and a request
to `/v1/systemone` naming a `gpt-6-luna` model takes the reverse path. The
conversion is as lossless as the two envelopes allow; see
[Cross-endpoint losses](#cross-endpoint-losses) for what each direction cannot
carry.

## Limits

The proxy rejects a malformed request up front with a `422` naming the offending
question. Where the upstreams disagree on a limit, the proxy accepts the union
and lets the upstream that does not support the request answer for itself:

- `score` takes 2–26 ordered levels — Ollama's ceiling. TypeSafe and OpenRouter
  stop at 10, so a rubric above that is served by Ollama and rejected upstream
  by the other two.
- `choice` takes at most 255 options.
- Question names must be unique when given, and choice option values must stay
  distinct after a boolean is written as its word: the bridge keys a System One
  `criteria` map by string, where `true` and `"true"` would collapse into one
  option.

## Response

`answers` carries one typed answer per question, in question order. Every answer
has a `type` — `predicate`, `choice`, `score`, or `refusal` (the host declined
one question without disclosing a score).

- `predicate` → `probability`, 0 to 1
- `choice` → `choice` (a supplied value, string or boolean), `probabilities` for
  every option, and `confidence`
- `score` → `score` (the probability-weighted average of the level indices, so
  it can fall between levels), `probabilities` with each level's `value` and
  `label`, and `confidence`

`model` identifies the model that performed the evaluation and `usage` reports
input and output tokens. There is no streaming variant: a decision is a single
synchronous evaluation. `usage` is echoed verbatim from the upstream when it
reported one.

An unnamed question has no `name` in its answer; correlate it by position. A
bridged answer omits `confidence` when the System One upstream reported none.

## Provider support

The model named in the request must be marked as a decision model
(`supports_systemone`) in the admin model configuration; a chat model is
rejected with a `400 invalid_request_error` before any provider is called. The
same flag gates [`/v1/systemone`](systemone.md) and the
[routing judge](../api/routing.md#routing-judge).

| Provider type | Upstream endpoint | Notes |
| --- | --- | --- |
| `openai` | `POST /v1/decisions` | Native. The only provider that accepts inline images for a decision. |
| `typesafe` | `POST /v1/systemone` | Bridged. Images are dropped (no image channel upstream). |
| `openrouter` | `POST /v1/systemone` | Bridged. OpenRouter serves the same envelope on a second, still-alpha route (`/api/alpha/decisions`); this stays on the stable one. Images are dropped. |
| `ollama` | `POST /v1/systemone` | Bridged. Inline images become the local `images` field (raw base64, as Ollama documents). |

Typesafe and Ollama serve one route for both endpoints, so an
`endpoint_base_urls.systemone` override moves both. OpenRouter names a second
route for the same envelope, so its `/v1/decisions` resolves under its own
`endpoint_base_urls.decisions` key, which is how a deployment opts into the
alpha route without moving `/v1/systemone`:

```yaml
endpoint_base_urls:
  decisions: "https://openrouter.ai/api/alpha/decisions"
```

## Cross-endpoint losses

Both directions are lossless except where one envelope has no field for a fact:

**Decisions → System One**

- A `predicate` has no rubric, so the System One `noul` `criteria` map is left
  empty rather than invented.
- `score` level `description`s move into the question text (as a numbered
  rubric) because System One's `criteria` array holds the level strings the
  answer legend echoes. The labels the client sent come back unchanged.
- `input` messages are flattened to one text block — every message is a user
  message, so the evidence keeps its order — and their inline images move to
  the local `images` field, which only `ollama` documents. `safety_identifier`
  has no System One counterpart and is dropped.

**System One → Decisions**

- A `noul` rubric (`criteria.true`/`criteria.false`) is folded into the
  instructions, since a `predicate` has no rubric field.
- A structured `state` (object or array) becomes JSON text, and `images` become
  inline `input_image` parts whose media type is read from the image's leading
  bytes. The OpenRouter-only fields (`provider`, `session_id`, `trace`, `user`)
  and Ollama's `keep_alive` have no Decisions counterpart and are dropped.

## Related

- [System One evaluation](systemone.md) — the sibling envelope
- [Endpoint Index](endpoints.md)
- [Errors & Rate Limits](errors.md)
