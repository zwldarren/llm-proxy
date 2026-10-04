---
pageClass: api-reference kicker-systemone
aside: false
---

# System One evaluation

Evaluate a state against a map of typed questions and get back one structured
answer per question. This endpoint mirrors [TypeSafe's System One API](https://docs.typesafe.ai/api)
and is also served by OpenRouter's `/systemone` endpoint.

::: endpoint POST /v1/systemone
:::

```bash [Request]
curl http://localhost:8080/v1/systemone \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
      "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoicing, refunds",
          "technical": "Bugs, outages, integrations"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      }
    }
  }'
```

```python [Request — TypeSafe SDK]
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(base_url="http://localhost:8080", api_key="sk-your-proxy-key")

response = client.system_one(
    state="Help! My payouts have been failing for 3 days.",
    questions={
        "is_urgent": Noul(instructions="Does this convey urgency?"),
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Payments", "technical": "Bugs"},
        ),
        "frustration": Score(
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated", "Very angry"],
        ),
    },
)
print(response.answers["is_urgent"].noul)
```

```json [Response]
{
  "model": "jev-1.13.0",
  "answers": {
    "is_urgent": {"type": "noul", "noul": 0.95},
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.88, "technical": 0.12},
      "confidence": 0.81
    },
    "frustration": {
      "type": "score",
      "score": 1.05,
      "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
      "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
      "confidence": 0.92
    }
  },
  "usage": {"input_tokens": 392, "output_tokens": 65}
}
```

## Request fields

`state` (required) is the content to evaluate — a plain string, or a JSON object
or array of related context such as a chat log or application state.

`model` (required) selects the System One model, e.g. `jev-latest`. Point the
model at a provider whose type supports System One (`typesafe`, `openrouter` or
`ollama`) in the console.

`questions` (required) maps a question id you choose to a typed question. The id
is not sent to the model; answers come back under the same id.

| `type` | Shape | `criteria` |
| --- | --- | --- |
| `noul` | Yes/no question; returns the probability the answer is yes | Optional object with `true` / `false` descriptions; either key may be given on its own (OpenRouter requires both when `criteria` is present) |
| `choice` | Picks one option; returns the option and the full distribution | Required map of option → rubric description (`null` for no extra detail; at most 255) |
| `score` | Rates along a rubric; returns a probability-weighted value | Required ordered array of level descriptions (2–26; TypeSafe and OpenRouter cap at 10) |

`instructions` can be a string, or a structured object/array that holds the
question in one field and the data it references in others. The optional
OpenRouter-only fields `provider` (routing preferences), `session_id`, `trace`
and `user` are accepted and forwarded to OpenRouter; TypeSafe does not document
them, so they are stripped before the request reaches it.

All of that is shared by every System One upstream. Ollama adds two fields of
its own, which the proxy validates and forwards to Ollama only — TypeSafe and
OpenRouter strip them:

| Field | Shape | Meaning |
| --- | --- | --- |
| `images` | Array of base64 strings | Images shared by every question, in request order. Needs a vision-capable System One model (Clef). URLs and data URLs are not accepted. |
| `keep_alive` | Duration string (`"5m"`) or seconds (number) | How long to keep the model loaded after the request. Zero unloads it; a negative value keeps it loaded. Defaults to the server's setting. |

Unknown top-level fields are rejected with a `422`.

## Limits

The proxy rejects a malformed question up front with a `422` naming it. Where
the upstreams disagree on a limit, the proxy accepts the union and lets the
upstream that does not support the request answer for itself:

- `score` takes 2–26 ordered levels — Ollama's ceiling. TypeSafe and OpenRouter
  stop at 10, so a rubric above that is served by Ollama and rejected upstream
  by the other two.
- `choice` takes at most 255 options (Ollama's own ceiling is 26).

Ollama adds two limits of its own that the proxy does not enforce, because only
it knows the loaded context window: a request body of 64 KiB without images and
32 MiB with them, and the whole input must fit the loaded context. Input is
never truncated.

## Response

`model` identifies the model that performed the evaluation, and `answers`
carries one answer per question, keyed by the ids you supplied. `usage` reports
`input_tokens` / `output_tokens`. When OpenRouter serves the request, the
response also carries `id` and `provider`, and `usage.cost` is preserved.
There is no streaming variant: System One is a single synchronous evaluation.

## Provider support

| Provider type | Base URL | Notes |
| --- | --- | --- |
| `typesafe` | `https://api.typesafe.ai/v1` | TypeSafe direct (the Jev model). Chat requests are rejected. An OpenRouter-namespaced `typesafe/…` model id is reduced to the bare id before the request is sent. |
| `openrouter` | `https://openrouter.ai/api/v1` | TypeSafe's Jev resold through OpenRouter. OpenRouter itself maps bare `jev-*` model ids onto its `typesafe/` namespace. |
| `ollama` | `http://localhost:11434` | Local System One models (`nimble`, `clef`, `tev`) on Ollama v0.35 or later. Chat and embeddings use the native `/api/*` surface; `/v1/systemone` is served from local models only — cloud models are rejected with a `400`. |

## Related

- [Endpoint Index](endpoints.md)
- [Errors & Rate Limits](errors.md)
