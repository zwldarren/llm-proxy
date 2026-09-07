// frontend/src/utils/modelPricing.test.ts
import { describe, expect, it } from "vitest";
import {
  compareModelsByCost,
  costRange,
  costValues,
  effectiveCost,
  MODEL_COST_KEYS,
} from "./modelPricing";
import type { ModelProviderMapping, ModelRead } from "@/types/schemas";

function provider(overrides: Partial<ModelProviderMapping> = {}): ModelProviderMapping {
  return {
    provider_name: "p",
    provider_model_name: "m",
    ...overrides,
  };
}

function model(overrides: Partial<ModelRead> = {}): ModelRead {
  return { id: 1, name: "m", providers: [], ...overrides };
}

describe("costValues", () => {
  it("collects provider overrides and the model default", () => {
    const m = model({
      input_cost_per_1m: 1,
      providers: [provider({ input_cost_per_1m: 0.5 }), provider({ input_cost_per_1m: 2 })],
    });
    expect(costValues(m, "input_cost_per_1m")).toEqual([0.5, 2, 1]);
  });

  it("skips null and undefined entries", () => {
    const m = model({
      providers: [provider({ input_cost_per_1m: null }), provider({})],
    });
    expect(costValues(m, "input_cost_per_1m")).toEqual([]);
  });

  it("treats a missing provider list as empty", () => {
    const m = model({ output_cost_per_1m: 3 });
    delete (m as Partial<ModelRead>).providers;
    expect(costValues(m, "output_cost_per_1m")).toEqual([3]);
  });
});

describe("costRange", () => {
  it("returns null when nothing is priced", () => {
    expect(costRange(model(), "input_cost_per_1m")).toBeNull();
  });

  it("collapses to a single point when all values agree", () => {
    const m = model({
      input_cost_per_1m: 1,
      providers: [provider({ input_cost_per_1m: 1 })],
    });
    expect(costRange(m, "input_cost_per_1m")).toEqual({ min: 1, max: 1 });
  });

  it("spans provider overrides and the model default", () => {
    const m = model({
      input_cost_per_1m: 5,
      providers: [provider({ input_cost_per_1m: 0.25 }), provider({})],
    });
    expect(costRange(m, "input_cost_per_1m")).toEqual({ min: 0.25, max: 5 });
  });

  it("works for every cost dimension", () => {
    const m = model(Object.fromEntries(MODEL_COST_KEYS.map((k) => [k, 7])));
    for (const key of MODEL_COST_KEYS) {
      expect(costRange(m, key)).toEqual({ min: 7, max: 7 });
    }
  });
});

describe("effectiveCost", () => {
  it("is the lowest configured value", () => {
    const m = model({
      output_cost_per_1m: 4,
      providers: [provider({ output_cost_per_1m: 2 })],
    });
    expect(effectiveCost(m, "output_cost_per_1m")).toBe(2);
  });

  it("is null when unpriced", () => {
    expect(effectiveCost(model(), "cached_read_cost_per_1m")).toBeNull();
  });
});

describe("compareModelsByCost", () => {
  const cheap = model({ name: "cheap", input_cost_per_1m: 1 });
  const spread = model({
    name: "spread",
    input_cost_per_1m: 1,
    providers: [provider({ input_cost_per_1m: 100 })],
  });
  const pricey = model({ name: "pricey", input_cost_per_1m: 2 });
  const unpriced = model({ name: "unpriced" });

  it("orders by range minimum, the value shown as the effective price", () => {
    const sorted = [pricey, cheap].sort(compareModelsByCost("input_cost_per_1m", 1));
    expect(sorted.map((m) => m.name)).toEqual(["cheap", "pricey"]);
  });

  it("breaks min ties by range maximum so wider ranges sort after tight ones", () => {
    const sorted = [spread, cheap].sort(compareModelsByCost("input_cost_per_1m", 1));
    expect(sorted.map((m) => m.name)).toEqual(["cheap", "spread"]);
    const desc = [cheap, spread].sort(compareModelsByCost("input_cost_per_1m", -1));
    expect(desc.map((m) => m.name)).toEqual(["spread", "cheap"]);
  });

  it("always sorts unpriced models last, regardless of direction", () => {
    const asc = [unpriced, cheap].sort(compareModelsByCost("input_cost_per_1m", 1));
    expect(asc.map((m) => m.name)).toEqual(["cheap", "unpriced"]);
    const desc = [unpriced, cheap].sort(compareModelsByCost("input_cost_per_1m", -1));
    expect(desc.map((m) => m.name)).toEqual(["cheap", "unpriced"]);
  });

  it("falls back to name order between equally priced models", () => {
    const b = model({ name: "b", input_cost_per_1m: 1 });
    const a = model({ name: "a", input_cost_per_1m: 1 });
    const sorted = [b, a].sort(compareModelsByCost("input_cost_per_1m", 1));
    expect(sorted.map((m) => m.name)).toEqual(["a", "b"]);
  });
});
