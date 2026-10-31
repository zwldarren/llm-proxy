// frontend/src/utils/pricingTiers.test.ts
import { describe, expect, it } from "vitest";
import { formatRate, overlappingWindowIndexes } from "./pricingTiers";
import type { PricingTimeWindow } from "@/types/schemas";

function window(overrides: Partial<PricingTimeWindow> = {}): PricingTimeWindow {
  return { start: "22:00", end: "08:00", days: [], timezone: "UTC", ...overrides };
}

describe("formatRate", () => {
  it("keeps a configured zero distinct from an unset rate", () => {
    expect(formatRate(0)).toBe("$0");
    expect(formatRate(null)).toBe("—");
    expect(formatRate(undefined)).toBe("—");
  });
});

describe("overlappingWindowIndexes", () => {
  it("flags two windows over the same hours", () => {
    const windows = [
      window({ start: "22:00", end: "08:00" }),
      window({ start: "23:00", end: "23:30" }),
    ];

    expect([...overlappingWindowIndexes(windows)]).toEqual([0, 1]);
  });

  it("flags a midnight wrap that runs into a peer's disjoint start day", () => {
    // Sunday 22:00-08:00 covers Monday 01:00, which the Monday band also prices.
    const windows = [
      window({ start: "00:00", end: "02:00", days: [0] }),
      window({ start: "22:00", end: "08:00", days: [6] }),
    ];

    expect([...overlappingWindowIndexes(windows)]).toEqual([0, 1]);
  });

  it("leaves windows over different days alone", () => {
    const windows = [
      window({ start: "08:00", end: "20:00", days: [0] }),
      window({ start: "08:00", end: "20:00", days: [1] }),
    ];

    expect(overlappingWindowIndexes(windows).size).toBe(0);
  });

  it("flags the same clock band whose zones overlap on a real instant", () => {
    // 08:00-20:00 Shanghai is 00:00-12:00 UTC, which shares 08:00-12:00 UTC.
    const windows = [
      window({ start: "08:00", end: "20:00", timezone: "UTC" }),
      window({ start: "08:00", end: "20:00", timezone: "Asia/Shanghai" }),
    ];

    expect([...overlappingWindowIndexes(windows)]).toEqual([0, 1]);
  });

  it("leaves regional bands that are disjoint in UTC alone", () => {
    // Shanghai 20:00-22:00 is 12:00-14:00 UTC, clear of the 00:00-08:00 UTC band.
    const windows = [
      window({ start: "00:00", end: "08:00", timezone: "UTC" }),
      window({ start: "20:00", end: "22:00", timezone: "Asia/Shanghai" }),
    ];

    expect(overlappingWindowIndexes(windows).size).toBe(0);
  });

  it("does not invent a DST pairing the two zones never shared", () => {
    // London is UTC+0/+1, Berlin UTC+1/+2: their offsets always differ, so the
    // same nominal local band never prices the same real instant.
    const windows = [
      window({ start: "09:00", end: "10:00", timezone: "Europe/London" }),
      window({ start: "09:00", end: "10:00", timezone: "Europe/Berlin" }),
    ];

    expect(overlappingWindowIndexes(windows).size).toBe(0);
  });

  it("treats an empty day list as every day", () => {
    const windows = [
      window({ start: "08:00", end: "20:00", days: [] }),
      window({ start: "09:00", end: "10:00", days: [5] }),
    ];

    expect(overlappingWindowIndexes(windows).size).toBe(2);
  });
});
