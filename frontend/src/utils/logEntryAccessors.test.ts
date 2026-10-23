import { describe, expect, it } from "vitest";
import {
  cacheCreationTokens,
  cachedPromptTokens,
  cachedTokens,
  cacheReadTokens,
  completionTokens,
  costUsd,
  promptTokens,
  tokenBreakdown,
  ttftMs,
  totalTokens,
  type LogUsageSource,
} from "./logEntryAccessors";

/**
 * Pins the single priority rule for log-entry usage fields: flat column first,
 * `log_metadata` as fallback, and — since the backend stores a reported zero as
 * `0` and an unreported count as `NULL` — "recorded as zero" must stay distinct
 * from "not recorded". These are the boundary cases that made the four consumer
 * components disagree before this module existed.
 */

const log = (fields: Partial<LogUsageSource>): LogUsageSource => fields;

describe("count accessors (prompt/completion/total)", () => {
  it("prefers the flat column when both sources are populated", () => {
    const entry = log({
      prompt_tokens: 100,
      completion_tokens: 50,
      total_tokens: 150,
      log_metadata: { prompt_tokens: 2500, completion_tokens: 900, total_tokens: 3400 },
    });
    expect(promptTokens(entry)).toBe(100);
    expect(completionTokens(entry)).toBe(50);
    expect(totalTokens(entry)).toBe(150);
  });

  it("lets a positive metadata value win over a legacy flat 0", () => {
    // Legacy rows defaulted the flat column to 0 while log_metadata carried the
    // measurement; they must keep rendering that measurement.
    const entry = log({
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
      log_metadata: { prompt_tokens: 2500, completion_tokens: 900, total_tokens: 3400 },
    });
    expect(promptTokens(entry)).toBe(2500);
    expect(completionTokens(entry)).toBe(900);
    expect(totalTokens(entry)).toBe(3400);
  });

  it("honours a reported zero when nothing contradicts it", () => {
    // The backend stores "provider reported 0" as 0 and "provider reported
    // nothing" as NULL; collapsing the two is the bug this rule prevents.
    expect(promptTokens(log({ prompt_tokens: 0 }))).toBe(0);
    expect(completionTokens(log({ completion_tokens: 0 }))).toBe(0);
    expect(totalTokens(log({ total_tokens: 0 }))).toBe(0);
    expect(promptTokens(log({ prompt_tokens: 0, log_metadata: { prompt_tokens: 0 } }))).toBe(0);
  });

  it("falls back to log_metadata when the flat column is missing or null", () => {
    const missing = log({ log_metadata: { prompt_tokens: 2500 } });
    expect(promptTokens(missing)).toBe(2500);

    const nulled = log({ prompt_tokens: null, log_metadata: { prompt_tokens: 2500 } });
    expect(promptTokens(nulled)).toBe(2500);

    const noMetadata = log({ prompt_tokens: null });
    expect(promptTokens(noMetadata)).toBeNull();
  });

  it("ignores non-numeric, NaN, infinite and negative values", () => {
    const entry = log({
      prompt_tokens: Number.NaN,
      completion_tokens: Number.POSITIVE_INFINITY,
      total_tokens: -5,
      log_metadata: {
        prompt_tokens: "2500",
        completion_tokens: null,
        total_tokens: undefined,
      },
    });
    expect(promptTokens(entry)).toBeNull();
    expect(completionTokens(entry)).toBeNull();
    expect(totalTokens(entry)).toBeNull();
  });

  it("returns null when nothing is recorded anywhere", () => {
    const entry = log({});
    expect(promptTokens(entry)).toBeNull();
    expect(completionTokens(entry)).toBeNull();
    expect(totalTokens(entry)).toBeNull();
  });
});

describe("cache token accessors", () => {
  it("resolves each cache dialect with flat-first metadata fallback", () => {
    const entry = log({
      cache_read_input_tokens: 0,
      cached_prompt_tokens: 120,
      cache_creation_input_tokens: null,
      log_metadata: { cache_read_input_tokens: 800, cache_creation_input_tokens: 60 },
    });
    expect(cacheReadTokens(entry)).toBe(800);
    expect(cachedPromptTokens(entry)).toBe(120);
    expect(cacheCreationTokens(entry)).toBe(60);
  });

  it("keeps a reported zero cache write distinct from an unrecorded one", () => {
    expect(cacheCreationTokens(log({ cache_creation_input_tokens: 0 }))).toBe(0);
    expect(cacheCreationTokens(log({ cache_creation_input_tokens: null }))).toBeNull();
  });

  it("merges dialects in cachedTokens with flat cached_prompt first", () => {
    expect(
      cachedTokens(
        log({
          cached_prompt_tokens: 120,
          cache_read_input_tokens: 800,
          log_metadata: { cached_prompt_tokens: 999, cache_read_input_tokens: 999 },
        })
      )
    ).toBe(120);

    expect(
      cachedTokens(log({ cache_read_input_tokens: 800, log_metadata: { cached_prompt_tokens: 9 } }))
    ).toBe(800);

    expect(cachedTokens(log({ log_metadata: { cached_prompt_tokens: 300 } }))).toBe(300);
    expect(cachedTokens(log({ log_metadata: { cache_read_input_tokens: 700 } }))).toBe(700);
    // Both dialects reported zero: that is "no cache hit", not "unknown".
    expect(cachedTokens(log({ cached_prompt_tokens: 0, cache_read_input_tokens: 0 }))).toBe(0);
    // Neither dialect reported anything.
    expect(cachedTokens(log({}))).toBeNull();
  });
});

describe("ttftMs", () => {
  it("prefers the flat column and falls back to log_metadata", () => {
    expect(ttftMs(log({ ttft_ms: 245, log_metadata: { ttft_ms: 300 } }))).toBe(245);
    expect(ttftMs(log({ log_metadata: { ttft_ms: 300 } }))).toBe(300);
    expect(ttftMs(log({ ttft_ms: null, log_metadata: { ttft_ms: 300 } }))).toBe(300);
  });

  it("treats 0 as unrecorded and returns null when nothing is recorded", () => {
    // A first token cannot arrive before the request; 0ms is not a measurement.
    expect(ttftMs(log({ ttft_ms: 0, log_metadata: { ttft_ms: 180 } }))).toBe(180);
    expect(ttftMs(log({ ttft_ms: 0 }))).toBeNull();
    expect(ttftMs(log({}))).toBeNull();
  });
});

describe("costUsd", () => {
  it("respects a flat 0 as a genuine measurement (no fall-through)", () => {
    expect(costUsd(log({ cost_usd: 0, log_metadata: { cost_usd: 0.005 } }))).toBe(0);
  });

  it("falls back to log_metadata only when the flat column is missing or null", () => {
    expect(costUsd(log({ cost_usd: null, log_metadata: { cost_usd: 0.005 } }))).toBe(0.005);
    expect(costUsd(log({ log_metadata: { cost_usd: 0.005 } }))).toBe(0.005);
    expect(costUsd(log({ cost_usd: 0.0123 }))).toBe(0.0123);
  });

  it("returns null when unrecorded or non-numeric", () => {
    expect(costUsd(log({}))).toBeNull();
    expect(costUsd(log({ cost_usd: null }))).toBeNull();
    expect(costUsd(log({ log_metadata: { cost_usd: "0.01" } }))).toBeNull();
  });
});

describe("tokenBreakdown", () => {
  it("renders input followed by output", () => {
    expect(tokenBreakdown(log({ prompt_tokens: 12500, completion_tokens: 820 }))).toBe(
      "12,500 → 820"
    );
  });

  it("renders a reported zero instead of hiding it", () => {
    expect(tokenBreakdown(log({ prompt_tokens: 31, completion_tokens: 0 }))).toBe("31 → 0");
    expect(tokenBreakdown(log({ prompt_tokens: 0, completion_tokens: 820 }))).toBe("0 → 820");
  });

  it("omits a side that was never recorded", () => {
    expect(tokenBreakdown(log({ prompt_tokens: 31 }))).toBe("31");
    expect(tokenBreakdown(log({ completion_tokens: 820 }))).toBe("820");
    expect(tokenBreakdown(log({ total_tokens: 900 }))).toBe("900");
  });

  it('renders "-" only when nothing was recorded', () => {
    expect(tokenBreakdown(log({}))).toBe("-");
    expect(tokenBreakdown(log({ prompt_tokens: null, completion_tokens: null }))).toBe("-");
  });
});
