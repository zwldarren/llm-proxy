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

describe("cost formatting follows the magnitude of the amount", () => {
  it("prints whole cents, not six decimals, for amounts that reach a cent", () => {
    // The dashboard total used to read "$0.000000" for a rounding error and
    // five zeros wide for every real total.
    expect(formatCost(0)).toBe("$0.00");
    expect(formatCost(12.5)).toBe("$12.50");
    expect(formatCost(1_234.567)).toBe("$1234.57");
  });

  it("keeps the significant digits of a sub-cent amount", () => {
    // Rounding these to two decimals — as every other amount is — would print
    // "$0.00" for the number the operator opened the row to read.
    expect(formatCost(0.000027)).toBe("$0.000027");
    expect(formatCost(0.001)).toBe("$0.001");
    expect(formatCost(-0.000004)).toBe("-$0.000004");
  });

  it("falls back to cents when nothing is left at six decimals", () => {
    expect(formatCost(1e-9)).toBe("$0.00");
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
