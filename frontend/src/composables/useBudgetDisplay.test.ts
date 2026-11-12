import { describe, expect, it } from "vitest";
import { formatBudgetAmount, useBudgetDisplay } from "./useBudgetDisplay";

const t = (key: string) => key;

describe("useBudgetDisplay", () => {
  it("uses the key table's two-decimal budget precision everywhere", () => {
    expect(formatBudgetAmount(0.000027, "en")).toBe("$0.00");
    expect(formatBudgetAmount(1_234.567, "en")).toBe("$1,234.57");
    expect(formatBudgetAmount(null, "en")).toBe("—");
  });

  it("keeps an unknown spend null instead of turning it into zero", () => {
    const display = useBudgetDisplay(
      {
        budgetUsd: 10,
        budgetPeriod: "monthly",
        budgetResetDay: 1,
        periodSpendUsd: null,
        fallbackSpendUsd: null,
      },
      t
    );

    expect(display.budgetSpend.value).toBeNull();
    expect(display.budgetRatio.value).toBeNull();
  });
});
