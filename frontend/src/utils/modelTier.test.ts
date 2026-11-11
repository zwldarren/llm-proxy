import { describe, expect, it } from "vitest";
import { modelTierLabel } from "./modelTier";

describe("modelTierLabel", () => {
  it("translates the known tiers", () => {
    const t = (key: string) => `translated:${key}`;
    expect(modelTierLabel(t, "ECONOMY")).toBe("translated:models.qualityTierEconomy");
    expect(modelTierLabel(t, "BALANCED")).toBe("translated:models.qualityTierBalanced");
    expect(modelTierLabel(t, "PREMIUM")).toBe("translated:models.qualityTierPremium");
  });

  it("passes an unknown tier through so a new backend value is still readable", () => {
    const t = (key: string) => `translated:${key}`;
    expect(modelTierLabel(t, "ULTRA")).toBe("ULTRA");
  });
});
