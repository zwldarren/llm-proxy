import { describe, expect, it } from "vitest";
import {
  ABSENT_METRIC,
  formatChartDate,
  formatContextLength,
  formatCost,
  formatDate,
  formatDuration,
  formatFullDate,
  formatTokens,
} from "./format";
import { setActiveLocale } from "./locale";

describe("absent metrics", () => {
  it("renders an em dash rather than a hyphen", () => {
    expect(ABSENT_METRIC).toBe("—");
    expect(formatCost(null)).toBe(ABSENT_METRIC);
    expect(formatDuration(undefined)).toBe(ABSENT_METRIC);
    expect(formatTokens(null)).toBe(ABSENT_METRIC);
    expect(formatContextLength(0)).toBe(ABSENT_METRIC);
  });
});

describe("date formatting follows the console locale", () => {
  it("formats a date in the active locale", () => {
    setActiveLocale("zh");
    const chinese = formatDate("2026-10-30T19:42:11Z");
    setActiveLocale("en");
    const english = formatDate("2026-10-30T19:42:11Z");

    // The two locales render the same instant differently: a console set to
    // Chinese must not show the browser's US format.
    expect(chinese).not.toBe(english);
    expect(chinese).toContain("2026");
    expect(english).toContain("2026");
  });

  it("accepts an explicit locale", () => {
    expect(formatDate("2026-10-30T19:42:11Z", "zh")).toContain("2026");
  });

  it("renders a chart label in the active locale", () => {
    setActiveLocale("zh");
    expect(formatChartDate("2026-11-11")).toContain("11");
    expect(formatFullDate("2026-11-11")).toContain("2026");
  });
});
