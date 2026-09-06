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
 * The priority rule is decided ONCE, here:
 *
 * - Counts (tokens, TTFT): the first POSITIVE finite number in
 *   [flat field, log_metadata field] wins. 0 / null / undefined / non-number
 *   all mean "not recorded" — a request that really ran never consumed 0
 *   prompt tokens, so a flat 0 with a nonzero metadata value means the flat
 *   column was never populated and the metadata value is the measurement.
 *   Token accessors return 0 and `ttftMs` returns null when unrecorded.
 *
 * - Money (cost_usd): 0 is a genuine measurement (a free request), so only
 *   null / undefined / non-number fall through. Returns null when unrecorded.
 *
 * Both `LogRead` (details payload) and `LogListItem` (list payload) satisfy
 * `LogUsageSource` structurally, so every consumer resolves identical values.
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

/** First positive finite value wins; 0 / negative / non-number mean "not recorded". */
const firstPositive = (...values: Array<number | null | undefined>): number | null => {
  for (const value of values) {
    if (value != null && Number.isFinite(value) && value > 0) return value;
  }
  return null;
};

/** Flat-first count resolution with metadata fallback; 0 when unrecorded. */
const count = (flat: number | null | undefined, meta: number | null): number =>
  firstPositive(flat, meta) ?? 0;

/** Input (prompt) tokens, including cache-read/creation tokens. */
export const promptTokens = (log: LogUsageSource): number =>
  count(log.prompt_tokens, metadataNumber(log, "prompt_tokens"));

/** Output (completion) tokens. */
export const completionTokens = (log: LogUsageSource): number =>
  count(log.completion_tokens, metadataNumber(log, "completion_tokens"));

/** Total tokens as recorded (not derived from prompt + completion). */
export const totalTokens = (log: LogUsageSource): number =>
  count(log.total_tokens, metadataNumber(log, "total_tokens"));

/** OpenAI-dialect cache reads (`prompt_tokens_details.cached_tokens`). */
export const cachedPromptTokens = (log: LogUsageSource): number =>
  count(log.cached_prompt_tokens, metadataNumber(log, "cached_prompt_tokens"));

/** Anthropic/Gemini-dialect cache reads (`cache_read_input_tokens`). */
export const cacheReadTokens = (log: LogUsageSource): number =>
  count(log.cache_read_input_tokens, metadataNumber(log, "cache_read_input_tokens"));

/** Anthropic-dialect cache writes (`cache_creation_input_tokens`). */
export const cacheCreationTokens = (log: LogUsageSource): number =>
  count(log.cache_creation_input_tokens, metadataNumber(log, "cache_creation_input_tokens"));

/**
 * Tokens served from cache, merging the two dialects into one figure. The
 * backend never reports both dialects for one request (they express the same
 * billable fact), so a single precedence order is unambiguous.
 */
export const cachedTokens = (log: LogUsageSource): number =>
  firstPositive(
    log.cached_prompt_tokens,
    log.cache_read_input_tokens,
    metadataNumber(log, "cached_prompt_tokens"),
    metadataNumber(log, "cache_read_input_tokens")
  ) ?? 0;

/** Time to first token in milliseconds; null when unrecorded. */
export const ttftMs = (log: LogUsageSource): number | null =>
  firstPositive(log.ttft_ms, metadataNumber(log, "ttft_ms"));

/** Request cost in USD; null when unrecorded. 0 is a real measurement. */
export const costUsd = (log: LogUsageSource): number | null =>
  asFiniteNumber(log.cost_usd) ?? metadataNumber(log, "cost_usd");
