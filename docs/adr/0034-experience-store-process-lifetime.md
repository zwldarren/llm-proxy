# The routing experience store is process-lifetime, and its read path is warm

## Status

Accepted (2026-10).

## Context

`ModelExperienceStore` (per-model EWMA reliability/latency/cache-affinity stats
feeding Thompson sampling in the selector) was constructed **per request**,
bound to a session opened just for the resolution, with an always-empty
in-memory cache. The consequences were all bad at once:

- **The learning loop never closed.** `_experience_snapshot` reads the store
  through `get()`/`snapshot()`, which consult only the in-memory cache —
  `load_from_db` ran solely inside the *write* paths. Every decision therefore
  saw `samples=0` neutral priors, and the rows the proxy dutifully upserted
  after each request influenced nothing. Explicit feedback
  (`POST /api/me/feedback`) landed in the same void.
- **Dead I/O on the hot path.** `orchestrate_smart_routing` opened a DB session
  around the resolution and committed it — but resolution writes nothing (the
  observation happens at request completion, in its own session). One
  connection checkout + `BEGIN`/`COMMIT` per virtual-model request for no
  effect.
- **Write-side overhead per observation**: a fresh store re-`SELECT`ed the row
  before every update, and snapshotted it with `copy.deepcopy` for revert-on-
  failure — the read half of a read-modify-write cycle that existed only to
  feed a cache nobody read.

## Decision

**One store per worker, hydrated once; persistence through an explicitly
passed session.**

- The lifespan constructs the store after migrations (`startup_database` runs
  `alembic upgrade head` first), hydrates every persisted row into memory
  (`store.hydrate(session)`), and installs it behind `RuntimeServices` as an
  optional service (`experience_store()`), per ADR-0021's pattern. Hydration
  failure logs and starts cold; it never blocks startup.
- `resolve_virtual_model` takes the shared store from services (falling back to
  a request-local empty store in tests/early startup — neutral priors, never a
  failure), and the resolution no longer opens any session.
- `observe_async` / `record_feedback_async` no longer re-read the row before
  updating: with a hydrated store the in-process cache *is* the authority, and
  re-reading would cost a query and clobber newer in-memory samples. They
  update memory inside the per-model lock, then persist through the caller's
  session (the completion callback's short-lived session; the feedback
  endpoint's request session), reverting the in-memory entry if the persist
  fails. The caller commits.
- `_experience_snapshot` prefers `get(name)` over `snapshot()`: one warm entry
  per candidate instead of a full-cache copy per candidate per request.

Concurrency: updates to one model's entry happen synchronously inside that
model's `asyncio.Lock`, reads are synchronous, so a single-loop reader never
sees a torn EWMA update. Across workers the cache is process-local — each
worker hydrates at startup and persists its own observations (last-writer-wins
upserts); EWMA statistics are approximate by construction, so the divergence
is acceptable and needs no distributed coordination.

## Consequences

- The routing read path is pure memory: zero queries per decision, and the
  persisted EWMAs (including explicit feedback) actually reach the next
  selection. The per-request dead session and the per-observation
  pre-read are gone.
- `ModelExperienceStore` no longer takes a session in its constructor and
  `load_from_db` is gone; call sites that opened a session around a store now
  pass one to the persisting method instead.
- A worker restarted after an outage resumes from the last persisted row per
  model, not from a stale per-request snapshot.
- Selection behavior changes in the direction the stats were always meant to
  push: a model that has been failing or slow now loses selections it used to
  win by default priors. The eval rig's `session=None` store is unaffected.

## Considered Options

- **Batch/debounce persistence** (flush every N observations or on a timer):
  rejected for now — it trades row freshness for queries the process no longer
  needs to spend, and a crashed worker would silently drop up to N observations.
  Revisit if the per-observation upsert shows up in profiles under load.
- **Redis as the shared experience store** (true cross-worker state): rejected —
  it couples routing quality to the Redis availability class that continuity
  already degrades around, and EWMA divergence across a handful of workers is
  statistically irrelevant next to the EWMA's own smoothing window.
- **Keep per-request stores and read rows at decision time** (N queries per
  route): rejected — it makes the hot path strictly worse to preserve a broken
  cache abstraction.
