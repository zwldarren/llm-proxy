/**
 * Single read contract for log-entry usage fields (tokens, TTFT, cost).
 *
 * Every log entry carries usage twice: in dedicated flat columns
 * (`prompt_tokens`, `ttft_ms`, …) and inside the free-form `log_metadata`
 * bag. Both are written from the same usage event, but the flat columns were
 * extracted from `log_metadata` after the fact, so legacy rows and some
 * metadata-only paths leave them unset — readers need a fallback. That
 * fallback used to be re-implemented with three different semantics across
 * LogListItem, LogsView, LogResponseView and LogDetailsMetrics, so the same
 * log could show different token counts on different UI surfaces.
 *
 * The priority rule is decided ONCE, here, and it keeps "recorded as zero"
 * distinct from "not recorded" — the backend stores NULL when the provider
 * reported no usage and 0 when it reported a real zero, and the UI must not
 * collapse the two:
 *
 * - Counts (tokens, cache): a finite flat value is the measurement, including
 *   a real 0. A null / undefined / non-numeric / negative flat value means
 *   the column has nothing, so `log_metadata` is consulted next (legacy rows
 *   predate the flat columns). One exception remains for legacy rows whose
 *   flat column was defaulted to 0 while `log_metadata` carried the
 *   measurement: there a positive metadata value wins. Returns null when
 *   nothing was recorded, and 0 when zero was reported.
 *
 * - TTFT: 0ms is not a measurement (it would mean the first byte arrived
 *   before the request), so only a positive value counts.
 *
 * - Money (cost_usd): 0 is a genuine measurement (a free request), so only
 *   null / undefined / non-number fall through. Returns null when unrecorded.
 *
 * Both `LogRead` (details payload) and `LogListItem` (list payload) satisfy
 * `LogUsageSource` structurally, so every consumer resolves identical values.
 * `tokenBreakdown` goes one step further and turns those values into the one
 * string every list surface renders, so no surface can drift.
 */

/** Minimal structural view of a log entry's usage fields. */
export interface LogUsageSource {
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  cached_prompt_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
  ttft_ms?: number | null;
  cost_usd?: number | null;
  log_metadata?: Record<string, unknown> | null;
}

const asFiniteNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const metadataNumber = (log: LogUsageSource, key: string): number | null =>
  asFiniteNumber(log.log_metadata?.[key]);

/** First positive finite value wins; 0 / negative / non-number are skipped. */
const firstPositive = (...values: Array<number | null | undefined>): number | null => {
  for (const value of values) {
    if (value != null && Number.isFinite(value) && value > 0) return value;
  }
  return null;
};

/**
 * Flat-first count resolution with metadata fallback.
 *
 * Returns the recorded count — `0` included, because the backend stores a
 * reported zero as `0` and an unreported count as `NULL` — or `null` when
 * neither source recorded anything. A positive metadata value still wins over
 * a flat `0`, which is how legacy rows (flat column defaulted, metadata the
 * real measurement) keep rendering correctly.
 */
const count = (flat: number | null | undefined, meta: number | null): number | null => {
  const recorded = asFiniteNumber(flat);
  if (recorded != null && recorded >= 0) {
    if (recorded === 0 && meta != null && meta > 0) return meta;
    return recorded;
  }
  return meta != null && meta > 0 ? meta : null;
};

/** Input (prompt) tokens, including cache-read/creation tokens; null if unrecorded. */
export const promptTokens = (log: LogUsageSource): number | null =>
  count(log.prompt_tokens, metadataNumber(log, "prompt_tokens"));

/** Output (completion) tokens; null if unrecorded. */
export const completionTokens = (log: LogUsageSource): number | null =>
  count(log.completion_tokens, metadataNumber(log, "completion_tokens"));

/** Total tokens as recorded (not derived from prompt + completion); null if unrecorded. */
export const totalTokens = (log: LogUsageSource): number | null =>
  count(log.total_tokens, metadataNumber(log, "total_tokens"));

/** OpenAI-dialect cache reads (`prompt_tokens_details.cached_tokens`). */
export const cachedPromptTokens = (log: LogUsageSource): number | null =>
  count(log.cached_prompt_tokens, metadataNumber(log, "cached_prompt_tokens"));

/** Anthropic/Gemini-dialect cache reads (`cache_read_input_tokens`). */
export const cacheReadTokens = (log: LogUsageSource): number | null =>
  count(log.cache_read_input_tokens, metadataNumber(log, "cache_read_input_tokens"));

/** Anthropic-dialect cache writes (`cache_creation_input_tokens`). */
export const cacheCreationTokens = (log: LogUsageSource): number | null =>
  count(log.cache_creation_input_tokens, metadataNumber(log, "cache_creation_input_tokens"));

/**
 * Tokens served from cache, merging the two dialects into one figure. The
 * backend never reports both dialects for one request (they express the same
 * billable fact), so a single precedence order is unambiguous: the first
 * positive value wins, and if every dialect was reported as 0 the result is a
 * recorded 0. Null when no dialect reported anything.
 */
export const cachedTokens = (log: LogUsageSource): number | null => {
  // Order matters: both flat dialects before either metadata copy, so a legacy
  // metadata value can never outrank a real flat measurement.
  const values = [
    asFiniteNumber(log.cached_prompt_tokens),
    asFiniteNumber(log.cache_read_input_tokens),
    metadataNumber(log, "cached_prompt_tokens"),
    metadataNumber(log, "cache_read_input_tokens"),
  ];
  const positive = firstPositive(...values);
  if (positive != null) return positive;
  return values.some((value) => value === 0) ? 0 : null;
};

/** Time to first token in milliseconds; null when unrecorded. */
export const ttftMs = (log: LogUsageSource): number | null =>
  firstPositive(log.ttft_ms, metadataNumber(log, "ttft_ms"));

/** Request cost in USD; null when unrecorded. 0 is a real measurement. */
export const costUsd = (log: LogUsageSource): number | null =>
  asFiniteNumber(log.cost_usd) ?? metadataNumber(log, "cost_usd");

/**
 * Compact "input → output" breakdown for list surfaces. Shared so the table
 * row, the mobile card and the details panel can never disagree about which
 * token counts a request consumed.
 *
 * Returns "-" only when nothing was recorded. A recorded zero is rendered as
 * "0" — the whole prompt served from cache and an empty completion are facts,
 * not missing measurements — and when just one side is recorded only that
 * figure is shown ("12,500 → 0" would read as an empty completion even though
 * the backend reported nothing for it).
 */
export const tokenBreakdown = (log: LogUsageSource): string => {
  const input = promptTokens(log);
  const output = completionTokens(log);
  const total = totalTokens(log);
  if (input == null && output == null && total == null) return "-";
  if (input == null || output == null) {
    const recorded = input ?? output ?? total;
    return recorded == null ? "-" : recorded.toLocaleString();
  }
  return `${input.toLocaleString()} → ${output.toLocaleString()}`;
};
