---
pageClass: api-reference kicker-batch
aside: false
---

# Batch mode

Run thousands of chat requests as one asynchronous job and pay roughly half the
synchronous price. The proxy speaks OpenAI's batch API (`/v1/files` +
`/v1/batches`) and forwards each job to the upstream's own batch endpoint, so the
discount is the upstream's, not a proxy-side rebate.

::: endpoint POST /v1/batches
:::

::: endpoint POST /v1/files
:::

## Which upstreams can run a batch

| Provider type | Upstream API | Cancel |
| --- | --- | --- |
| `openai` | Files + Batches (JSONL upload, results in `output_file_id`) | Yes |
| `openrouter` | Inline `POST /api/v1/batches` (results inline in the status object) | No |
| `gemini` | `models/{model}:batchGenerateContent` (inline requests, long-running operation) | Yes |

Other provider types have no batch API. A batch naming a model whose provider
cannot do it is rejected at creation, so nothing is submitted and nothing is
billed:

```json
{
  "error": {
    "message": "Provider 'ollama-local' (ollama) has no batch API, so model 'llama-3' cannot be batched. Configure a provider with batch support (openai, openrouter, gemini) for this model.",
    "type": "invalid_request_error",
    "code": "batch_unsupported"
  }
}
```

## Workflow

1. Upload the requests as one JSON object per line, in the shape OpenAI's batch
   API documents: `custom_id`, `method`, `url`, `body`.
2. Create the batch from that file id.
3. Poll the batch until `status` is terminal.
4. Download the results from `output_file_id` (successes) and `error_file_id`
   (per-request failures).

```bash [Upload]
curl http://localhost:8080/v1/files \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -F purpose=batch \
  -F file=@requests.jsonl
```

```json [Upload response]
{
  "id": "file-9f1c2b7a4d3e5f60718293a4",
  "object": "file",
  "bytes": 214,
  "created_at": 1782097200,
  "filename": "requests.jsonl",
  "purpose": "batch"
}
```

```bash [Create]
curl http://localhost:8080/v1/batches \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-9f1c2b7a4d3e5f60718293a4",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h",
    "metadata": {"job": "nightly-classification"}
  }'
```

```bash [Poll]
curl http://localhost:8080/v1/batches/batch_4d3e5f60718293a49f1c2b7a \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```bash [Results]
curl http://localhost:8080/v1/files/file-1a2b3c4d5e6f708192a3b4c5/content \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

The same flow with the OpenAI SDK, pointed at the proxy and nothing else changed:

```python [Python — OpenAI SDK]
import json
from openai import OpenAI

client = OpenAI(api_key="sk-your-proxy-key", base_url="http://localhost:8080/v1")

with open("requests.jsonl", "wb") as fh:
    fh.write(
        b"\n".join(
            json.dumps(
                {
                    "custom_id": f"review-{index}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5",
                        "messages": [{"role": "user", "content": text}],
                    },
                }
            ).encode()
            for index, text in enumerate(reviews)
        )
    )

uploaded = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
batch = client.batches.create(
    input_file_id=uploaded.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

while batch.status not in ("completed", "failed", "expired", "cancelled"):
    batch = client.batches.retrieve(batch.id)

for line in client.files.content(batch.output_file_id).text.splitlines():
    record = json.loads(line)
    print(record["custom_id"], record["response"]["body"]["choices"][0]["message"]["content"])
```

## The batch object

```json [Response — GET /v1/batches/{id}]
{
  "id": "batch_4d3e5f60718293a49f1c2b7a",
  "object": "batch",
  "endpoint": "/v1/chat/completions",
  "errors": null,
  "input_file_id": "file-9f1c2b7a4d3e5f60718293a4",
  "completion_window": "24h",
  "status": "completed",
  "output_file_id": "file-1a2b3c4d5e6f708192a3b4c5",
  "error_file_id": "file-5c4b3a29180f7e6d5c4b3a29",
  "created_at": 1782097200,
  "in_progress_at": 1782097500,
  "expires_at": 1782183600,
  "finalizing_at": null,
  "completed_at": 1782099000,
  "failed_at": null,
  "expired_at": null,
  "cancelling_at": null,
  "cancelled_at": null,
  "request_counts": {"total": 500, "completed": 497, "failed": 3},
  "metadata": {"job": "nightly-classification"}
}
```

Statuses are OpenAI's, whatever the upstream calls them:

| Status | Meaning |
| --- | --- |
| `validating` | Accepted; the upstream is checking the requests (Gemini's `BATCH_STATE_PENDING` lands here) |
| `in_progress` | Running |
| `finalizing` | Finished, the upstream is assembling results |
| `completed` | Results are downloadable |
| `failed` | The batch itself failed; see `errors` |
| `expired` | The completion window closed first |
| `cancelling` | Cancel requested, upstream still working |
| `cancelled` | Cancelled |

Result files use OpenAI's line shape:

```json [Output file — one line]
{
  "id": "batch_req_8f2c1d3e4a5b6c7d8e9f0a1b",
  "custom_id": "review-1",
  "response": {
    "status_code": 200,
    "request_id": null,
    "body": {"object": "chat.completion", "choices": [{"message": {"content": "positive"}}]}
  },
  "error": null
}
```

A failed request has `"response": null` and an `error` object, and lands in
`error_file_id`. The body carries the model name the client asked for, not the
upstream's name, so a mapped model looks the same in batch as in a synchronous
response.

## Rules the proxy enforces at creation

These fail with a `400` before anything reaches an upstream, because the two
inline batch APIs cannot express them:

| Rule | Why |
| --- | --- |
| Every line targets `/v1/chat/completions` | OpenRouter and Gemini both bind a batch to one endpoint/model pair |
| Every line names the same model | OpenRouter requires one `model` for the whole batch; Gemini creates one operation per model |
| `custom_id` is unique and non-empty | Results are matched on it |
| No `stream: true` | A batch request has no stream to write to |
| `completion_window` is `24h` | The only value all three upstreams accept |
| Uploaded file ≤ 16 MiB and ≤ 50,000 lines | Gemini's inline request cap is 20 MB; the smaller bound leaves room for the envelope |

A virtual model (`auto`, `fast`, `best`) is refused: the router picks a model per
request, and a batch is submitted once.

The calling key's model allowlist is enforced on the batch's model at creation.
The allowlist middleware reads the `model` field of the request body, which a
batch does not carry — its model lives inside the uploaded file — so the check
runs once the file is parsed.

## How each line reaches the upstream

The proxy resolves the batch's model the same way it resolves a synchronous
request — the highest-priority enabled provider configured for that model, mapped
to that provider's upstream model name — and then:

- OpenAI and OpenRouter receive the client's `body` unchanged except for `model`,
  because both speak the chat-completions wire their batch entries use.
- Gemini receives a translated `GenerateContentRequest` built through the same
  serializer a synchronous request uses, and each response is translated back
  into an OpenAI chat completion.

::: warning What batch mode does not do
- Model-level `parameter_overrides` are **not** applied on the OpenAI/OpenRouter
  path: the body is the client's.
- No streaming, no images, no embeddings, no Responses API lines.
:::

## Reading results

`GET /v1/batches/{id}` is the only call that talks to the upstream. When it
observes a terminal status for the first time, the proxy downloads the upstream's
results and stores them as its own files, so `output_file_id` and `error_file_id`
are ids you can fetch from `/v1/files/{id}/content`. A job whose status nobody
reads is never fetched — there is no background poller.

That pull happens once per job, including for a batch that finished with no
results at all (both file ids then stay `null`), so polling a finished batch does
not re-download anything. If the download itself fails, the next poll retries it.

`GET /v1/batches` reports each job's **last observed** state and does not poll
upstreams, so listing stays one database query. Poll the ids you care about.

## Spend and budgets

The poll that materializes a batch's results also bills it. Every line the
upstream reported usage for is priced through the same cost calculation a
synchronous request uses, at the batch's **creation** instant (the instant the
work was submitted, which is what peak/off-peak windows use), and the batch
writes **one** `usage_records` row:

- `request_id` is the batch id, the timestamp is `created_at`, the model and
  provider are the batch's, and the token counts are the sum over its lines;
- its log type is `batch`, so it counts as spend and tokens but not as a
  request: the dashboards' request counts and latency averages stay the ones the
  request logs describe;
- the dashboard's usage views, per-key spend and budget windows all include it.

Two consequences worth knowing:

- A billable batch is charged **after** it finishes. The spend lands when the
  results are materialized, so a budget can be exceeded by the batch that
  crosses it — the next request is the first one that sees the new total.
- Cache savings are not reported for a batch: a line reports how many tokens
  came from cache, not what they saved. A batch whose lines reported no usage at
  all writes no row, so nothing is recorded as a known `$0`.

## Cancelling

```bash [Cancel]
curl -X POST http://localhost:8080/v1/batches/batch_4d3e5f60718293a49f1c2b7a/cancel \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

A terminal batch answers with its current object. If the cancel itself lands on a
final status (Gemini can answer `cancelled` at once with the partial responses
attached), the proxy materializes the results in that same call. OpenRouter has
no cancel endpoint, so a cancel on an OpenRouter-backed batch returns:

```json
{
  "error": {
    "message": "Provider 'openrouter-main' (openrouter) has no batch cancel endpoint, so this batch cannot be cancelled upstream.",
    "type": "invalid_request_error",
    "code": "batch_cancel_unsupported"
  }
}
```

Gemini's cancel is asynchronous: while Google still reports the operation as
pending or running, the proxy answers `status: "cancelling"` and sets
`cancelling_at`.

## Files and storage

Every file — the requests you upload and the results the proxy materializes — is
stored in the proxy's database and scoped to the API key that created it. The
same key must be used to read or delete it; another key gets a `404`.

An input file that a running batch reads cannot be deleted: the proxy answers
`400 file_in_use`, so a job's `input_file_id` never dangles. Delete it once the
batch is terminal.

There is no per-line `request_logs` row: a batch is one client call, and expanding
its results into request rows would undo the discount it exists for.

Batch files are pruned automatically. `batch_retention_days` (logging settings,
default 30, `0` = keep forever) drives a sweep that deletes job rows older than the
window and then the files no remaining job names, so a file a retained job still
references — its input, its output file or its error file — is never deleted out
from under it. The sweep runs a few times a day, in bounded batches, and reclaims
freed space on SQLite.

Only the proxy's own copies are managed here. Whatever an upstream stores —
OpenAI's uploaded input and result files, in particular — is neither tracked nor
deleted by the proxy; clean those up with the provider's own tooling. The
client's `metadata` is likewise proxy-side only: it is stored and returned on the
batch object, not forwarded upstream.

## Provider-specific notes

- OpenAI: the proxy uploads your JSONL as a `purpose: "batch"` file, creates the
  batch against `/v1/chat/completions`, and downloads both the output and error
  files when it materializes results.
- OpenRouter: the whole batch goes inline (`POST /batches` with
  `requests: [{custom_id, body}]`), and results come back inside the status
  object. OpenRouter deletes requests and results 30 days after submission, so
  fetch and store the output before then.
- Gemini: requests are sent inline, so the assembled payload must stay under
  20 MB (the proxy measures it and answers `413` with `batch_too_large` if not).
  Batch uses the `generate_content` dialect; a provider configured with
  `api_variant: "interactions"` refuses batch with a `400` naming the setting.

### Endpoint overrides

The provider metadata key `endpoint_base_urls` can move the batch endpoints off
the configured base URL, under the keys `files` (OpenAI's Files API, the upload
and the result downloads) and `batch` (create and poll).

```json
{"endpoint_base_urls": {"files": "https://api.openai.com/v1", "batch": "https://api.openai.com/v1"}}
```

An override replaces the whole path rather than a prefix. That matters for
Gemini, whose create route is model-scoped: an override for `batch` has to carry
`{model}` itself, e.g. `https://host/v1beta/models/{model}:batchGenerateContent`,
or the model name is lost.
