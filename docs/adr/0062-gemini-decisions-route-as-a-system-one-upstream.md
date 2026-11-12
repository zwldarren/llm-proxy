# Gemini's Decisions route is a System One upstream

## Status

Accepted (2026-11).

## Context

Google shipped the **Decisions API** and the model behind it, Gemini Decision 1
(`gemini-decision-1`). Its resource is `POST {base}/decisions`, and its payload
is a third spelling of the proxy's evaluation primitives — and a close one:

| | Gemini Decisions | System One | OpenAI Decisions |
| --- | --- | --- | --- |
| Evidence | `input`: string or Interactions content blocks | `state`: string, object or array | `input`: string or user messages |
| Questions | **map** keyed by id | **map** keyed by id | ordered array with `name` |
| Yes/no | `noul` | `noul` | `predicate` |
| Pick one | `criteria{option: description}` | `criteria{option: description}` | `choices[{value, description}]` |
| Rate | `criteria[level description, …]` | `criteria[level description, …]` | `levels[{label, description}]` |
| Answers | **map** keyed by id | **map** keyed by id | ordered array |
| Usage | `total_input_tokens`, `total_output_tokens`, `input_tokens_by_modality` | `input_tokens`, `output_tokens` | Responses-shaped usage |

Gemini also documents a `noul` rubric, under the keys `true_description` /
`false_description`, and text-only question instructions ("Media inside questions
isn't supported").

ADR-0019 makes every decision model serve `/v1/systemone` and `/v1/decisions`,
with the provider bridging the envelope it does not speak; ADR-0045 gives each
System One upstream exactly one declared image channel. The question is which
side of that seam Gemini belongs on.

## Decision

**Gemini declares the System One capability against its `/decisions` route, and
`/v1/decisions` reaches it through the existing bridge.**

1. **The envelope decides, not the route name.** Gemini's payload *is* the
   System One envelope: a map of typed questions answered under the same keys, the
   same three kind names, the same `criteria` shapes. A native Decisions
   capability would have to re-derive the ordered-array/`predicate` mapping the
   bridge already owns. So the adapter takes `SystemOneCapabilityMixin` with
   `SYSTEMONE_ENDPOINT = "/decisions"` — the same shape OpenRouter's second,
   alpha route already has — plus `DecisionsOverSystemOneMixin` for the other
   endpoint. Both endpoints resolve to one upstream route, because Gemini
   publishes one evaluation surface.

2. **The wire differences are a provider serializer's job.** They live in
   `serialization/gemini/evaluation.py` (`GeminiEvaluationMixin`): `state` →
   `input`, a structured `state`/`instructions` value → JSON text (Gemini's own
   guidance for structured evidence), the `noul` rubric key spelling, and a
   structured `score` level folded into one description string, because Gemini's
   criteria *are* the descriptions it echoes back as the answer legend.

3. **A fourth image channel: `content_parts`.** Gemini carries media as typed
   content blocks inside `input` (`{"type": "image", "data", "mime_type"}` for an
   inlined image, `uri` for a hosted one), and spells text as a block too
   (adjacent text blocks concatenate with no separator). Because question
   instructions are text-only upstream, a bridged question's own image joins the
   shared evidence every question is asked about — the same trade-off Ollama's
   shared `images` array documents. A data URL splits into `data` plus
   `mime_type`; a URL that is not a data URL is refused rather than given a
   guessed media type. Hosted images are downloaded and inlined one layer up,
   like every other evaluation upstream, so the same request behaves identically
   everywhere.

4. **The usage block is mapped, not echoed.** Gemini reports the Interactions
   vocabulary (`total_input_tokens` / `total_output_tokens` /
   `input_tokens_by_modality`), which is not what either client wire promises, so
   the parser maps it onto the canonical record through the shared Interactions
   helpers (the modalities land in `prompt_tokens_details`, where per-modality
   pricing reads them) and does not set `systemone_usage`/`decisions_usage`.

5. **Evaluation requests are outside the chat dialect switch.** `api_variant` picks
   the chat dialect (generateContent vs Interactions); the `/decisions` route is
   the same resource either way, so the Interactions extra whitelist that protects
   chat bodies must not rewrite an evaluation body.

6. **What has no counterpart is refused, not forwarded.** `box` — the
   image-locating question only Jev has — is a `400` naming the model, like the
   `SystemOneOverDecisionsMixin` direction already does.

## Consequences

- A `gemini-decision-1` model marked `supports_systemone` serves `/v1/systemone`
   and `/v1/decisions`, in the envelopes their clients already write, and becomes
   eligible as the routing judge.
- Gemini's endpoint is named after Decisions while carrying the System One
   question vocabulary. Both docs say so, because the route name is a trap for the
   next reader.
- The proxy's own evaluation limits still bind: a `score` rubric above 26 levels
   is rejected at the edge although Gemini accepts 255. Widening the edge would
   loosen it for every System One upstream (TypeSafe and OpenRouter stop at 10,
   Ollama at 26), which is a decision of its own rather than a side effect of
   adding an upstream.
- Media Gemini supports beyond still images — audio, video, PDF documents — has
   no field in either client envelope, so it remains unreachable through these
   endpoints until one of them grows one.
- Adding the channel cost one class attribute and one branch in the image sink,
   which is what ADR-0045 anticipated; the bridge stayed a pure function with no
   client of its own.
