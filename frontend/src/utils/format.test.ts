import { describe, expect, it } from "vitest";
import {
  ABSENT_METRIC,
  formatContextLength,
  formatCost,
  formatDuration,
  formatTokens,
} from "./format";

describe("absent metrics", () => {
  it("renders an em dash rather than a hyphen", () => {
    expect(ABSENT_METRIC).toBe("—");
    expect(formatCost(null)).toBe(ABSENT_METRIC);
    expect(formatDuration(undefined)).toBe(ABSENT_METRIC);
    expect(formatTokens(null)).toBe(ABSENT_METRIC);
    expect(formatContextLength(0)).toBe(ABSENT_METRIC);
  });
});
