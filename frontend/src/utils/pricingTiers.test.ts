// frontend/src/utils/pricingTiers.test.ts
import { describe, expect, it } from "vitest";
import type { ComposerTranslation } from "vue-i18n";
import {
  flattenWindowBands,
  formatRate,
  groupWindowBands,
  overlappingWindowIndexes,
  windowRateRows,
} from "./pricingTiers";
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

describe("groupWindowBands", () => {
  it("folds windows that share timezone, days and rates into one band", () => {
    const bands = groupWindowBands([
      window({ start: "06:00", end: "10:00", input_cost_per_1m: 2 }),
      window({ start: "01:00", end: "04:00", input_cost_per_1m: 2 }),
    ]);

    expect(bands).toHaveLength(1);
    expect(bands[0]?.ranges).toEqual([
      { start: "01:00", end: "04:00" },
      { start: "06:00", end: "10:00" },
    ]);
    expect(bands[0]?.rates).toEqual({ input_cost_per_1m: 2 });
  });

  it("treats a differently ordered day list as the same band", () => {
    const bands = groupWindowBands([
      window({ start: "01:00", end: "04:00", days: [0, 2] }),
      window({ start: "06:00", end: "10:00", days: [2, 0] }),
    ]);

    expect(bands).toHaveLength(1);
    expect(bands[0]?.days).toEqual([0, 2]);
  });

  it("keeps a window that prices or schedules differently in its own band", () => {
    const bands = groupWindowBands([
      window({ start: "01:00", end: "04:00", input_cost_per_1m: 2 }),
      window({ start: "06:00", end: "10:00", input_cost_per_1m: 5 }),
      window({ start: "06:00", end: "10:00", days: [1] }),
      window({ start: "06:00", end: "10:00", timezone: "Asia/Shanghai" }),
    ]);

    expect(bands).toHaveLength(4);
  });

  it("treats an explicit null rate as an unset one", () => {
    const bands = groupWindowBands([
      window({ start: "01:00", end: "04:00", input_cost_per_1m: null }),
      window({ start: "06:00", end: "10:00" }),
    ]);

    expect(bands).toHaveLength(1);
    expect(bands[0]?.rates).toEqual({});
  });
});

describe("flattenWindowBands", () => {
  it("writes one window per range with the band's timezone, days and rates", () => {
    const windows = flattenWindowBands([
      {
        timezone: "UTC",
        days: [1],
        ranges: [
          { start: "01:00", end: "04:00" },
          { start: "06:00", end: "10:00" },
        ],
        rates: { input_cost_per_1m: 2 },
      },
    ]);

    expect(windows).toEqual([
      { start: "01:00", end: "04:00", days: [1], timezone: "UTC", input_cost_per_1m: 2 },
      { start: "06:00", end: "10:00", days: [1], timezone: "UTC", input_cost_per_1m: 2 },
    ]);
  });

  it("round-trips a saved list through the band grouping", () => {
    // Stored order is by timezone then clock, which is what the editor emits.
    const saved = [
      window({ start: "22:00", end: "08:00", timezone: "Asia/Shanghai" }),
      window({ start: "01:00", end: "04:00", days: [0], input_cost_per_1m: 2 }),
      window({ start: "06:00", end: "10:00", days: [0], input_cost_per_1m: 2 }),
    ];

    expect(flattenWindowBands(groupWindowBands(saved))).toEqual(saved);
  });
});

describe("windowRateRows", () => {
  const translate = ((key: string) =>
    ({
      "models.inputShort": "IN",
      "models.outputShort": "OUT",
      "models.pricingWindowEveryDay": "every\u00A0day",
    })[key] ?? key) as unknown as ComposerTranslation;

  it("lists every range of a band in a single row", () => {
    const rows = windowRateRows(translate, [
      window({ start: "01:00", end: "04:00", input_cost_per_1m: 2 }),
      window({ start: "06:00", end: "10:00", input_cost_per_1m: 2 }),
    ]);

    expect(rows).toHaveLength(1);
    expect(rows[0]?.label).toBe("01:00–04:00,\u00A006:00–10:00 · every\u00A0day · UTC");
    expect(rows[0]?.text).toBe("IN $2 · OUT —");
  });

  it("keeps a band that prices differently on its own row", () => {
    const rows = windowRateRows(translate, [
      window({ start: "01:00", end: "04:00", input_cost_per_1m: 2 }),
      window({ start: "06:00", end: "10:00", input_cost_per_1m: 5 }),
    ]);

    expect(rows.map((row) => row.text)).toEqual(["IN $2 · OUT —", "IN $5 · OUT —"]);
  });
});

describe("overlappingWindowIndexes with an unresolvable zone", () => {
  it("skips the comparison instead of throwing out of the form", () => {
    const windows = [
      window({ start: "01:00", end: "04:00", timezone: "Asia/Atlantis" }),
      window({ start: "06:00", end: "10:00", timezone: "UTC" }),
    ];

    // The editor already flags the name itself; a RangeError here would take
    // the whole pricing form down with it.
    expect(() => overlappingWindowIndexes(windows)).not.toThrow();
    expect(overlappingWindowIndexes(windows).size).toBe(0);
  });
});
