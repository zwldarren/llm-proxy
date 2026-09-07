import { describe, expect, it } from "vitest";
import {
  cacheCreationTokens,
  cachedPromptTokens,
  cachedTokens,
  cacheReadTokens,
  completionTokens,
  costUsd,
  promptTokens,
  ttftMs,
  totalTokens,
  type LogUsageSource,
} from "./logEntryAccessors";

/**
 * Pins the single priority rule for log-entry usage fields: flat column
 * first, `log_metadata` as fallback, with 0 / null / non-number counting as
 * "not recorded" for counts but not for money. These are the boundary cases
 * that made the four consumer components disagree before this module existed.
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

  it("falls back to log_metadata when the flat column is 0 (unrecorded)", () => {
    // The reported bug: the same entry rendered 2500 in list rows but 0 in
    // the response view. Now every surface resolves 2500.
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

  it("falls back to log_metadata when the flat column is missing or null", () => {
    const missing = log({ log_metadata: { prompt_tokens: 2500 } });
    expect(promptTokens(missing)).toBe(2500);

    const nulled = log({ prompt_tokens: null, log_metadata: { prompt_tokens: 2500 } });
    expect(promptTokens(nulled)).toBe(2500);

    const noMetadata = log({ prompt_tokens: null });
    expect(promptTokens(noMetadata)).toBe(0);
  });

  it("treats a metadata 0 as unrecorded too", () => {
    expect(promptTokens(log({ prompt_tokens: 100, log_metadata: { prompt_tokens: 0 } }))).toBe(100);
    expect(promptTokens(log({ prompt_tokens: 0, log_metadata: { prompt_tokens: 0 } }))).toBe(0);
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
    expect(promptTokens(entry)).toBe(0);
    expect(completionTokens(entry)).toBe(0);
    expect(totalTokens(entry)).toBe(0);
  });

  it("returns 0 when nothing is recorded anywhere", () => {
    const entry = log({});
    expect(promptTokens(entry)).toBe(0);
    expect(completionTokens(entry)).toBe(0);
    expect(totalTokens(entry)).toBe(0);
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
    expect(cachedTokens(log({}))).toBe(0);
  });
});

describe("ttftMs", () => {
  it("prefers the flat column and falls back to log_metadata", () => {
    expect(ttftMs(log({ ttft_ms: 245, log_metadata: { ttft_ms: 300 } }))).toBe(245);
    expect(ttftMs(log({ log_metadata: { ttft_ms: 300 } }))).toBe(300);
    expect(ttftMs(log({ ttft_ms: null, log_metadata: { ttft_ms: 300 } }))).toBe(300);
  });

  it("treats 0 as unrecorded and returns null when nothing is recorded", () => {
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
