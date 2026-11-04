# 0057 — Drop the judge warm-up

## Status

Accepted (2026-11). Amends ADR-0018's startup-warm-up rule.

## Context

ADR-0018 warmed the judge twice over: at startup and on every config reload
(`startup_judge_warmup`). The trap it removed was measured and real — a judge
call cancelled at its deadline does not leave the model loaded, so on a judge
that owns a cold start, a deadline shorter than that start makes every gated
turn pay the load, lose it, and fall back to the ensemble: the feature is on
and never answers.

That measurement was made against a *local* stand-in, and the rationale is a
local-load claim. ADR-0018 said as much in the same paragraph — "a *remote*
upstream still owns its own cold start, which no proxy-side warm-up can
remove" — and ADR-0028 went further, making judge tail latency a deployment
prerequisite rather than a code concern. Most deployments point the judge at a
hosted API, where the warm-up buys almost nothing:

- **No connection stays warm.** The pooled client carries no idle keepalive
  setting (`http/client.py` builds `httpx2.Limits` without a keepalive expiry),
  and the gate is designed to fire minutes apart. Whatever socket the warm-up
  opened is gone before the first judged turn arrives.
- **Prompt-cache priming was never measured.** It could only cover the system
  prompt and rubric prefix, and the warm-up body (`"Warm-up request."`) does not
  resemble a real ask. ADR-0018 recorded no number for it.
- **Static misconfiguration does not need a call.** An unknown judge model and a
  model without `supports_systemone` are already reported as warnings on the
  consult path, without spending tokens.

What it did cost was real. The warm-up is a billed upstream call that ADR-0018
deliberately kept out of the log store ("makes no client request and is not
logged as one"), so the spend is invisible in the dashboard; it was paid again
on every config reload, not just at boot; and its only trace was an INFO line on
`llm-proxy.routing.judge` at every start, which reads like an error report to
anyone watching the console.

## Decision

Remove the warm-up and everything that existed only to serve it:

- `warm_judge` and `JUDGE_WARMUP_DEADLINE_S` in `routing/judge/consult.py`;
- `startup_judge_warmup` / `_schedule_judge_warmup` and the config-reload
  listener they registered, plus the lifespan step that called them;
- the `RuntimeServices.judge_warmup_task` / `set_judge_warmup_task` slots.

`consult_judge` also loses its `deadline_s` override. Two deadlines existed only
because two kinds of caller did: a generous one for loading a model off the
request path, and the configured one for a client that is waiting. With one
caller left, `judge.deadline_s` is the only deadline.

## Consequences

- A deployment whose judge is local (ollama, vLLM, SGLang on the same box)
  returns to the failure mode ADR-0018 measured: the first gated turn after a
  start pays the cold load, and if that exceeds `deadline_s` the turn falls back
  to the ensemble. For such a deployment, sizing the deadline to the model's cold
  start — or keeping the model resident — is now a prerequisite, which is the
  posture ADR-0028 already took for remote upstreams.
- Nothing else regresses. The judge still abstains instead of failing, and a
  misconfigured judge still says so on the consult path (unknown model, model not
  marked `supports_systemone`).
- Startup no longer spends the operator's tokens, no longer re-spends them on
  every settings save, and no longer prints an INFO line that looked like a
  fault. Judge traffic now shows up only where it is paid for and recorded: the
  per-turn `judge` log row.
- A future warm-up, if one is ever wanted, should be an explicit per-judge
  choice for deployments that actually host the model, not an unconditional
  side effect of the process starting.
