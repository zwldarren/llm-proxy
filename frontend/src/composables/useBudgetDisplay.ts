import { computed, toValue, type MaybeRefOrGetter } from "vue";
import type { BudgetPeriod } from "@/services/api/apiKeys";

type TFn = (key: string, ...args: unknown[]) => string;

interface BudgetDisplayInput {
  budgetUsd: MaybeRefOrGetter<number | null | undefined>;
  budgetPeriod: MaybeRefOrGetter<BudgetPeriod | null | undefined>;
  budgetResetDay: MaybeRefOrGetter<number | null | undefined>;
  /** Spend inside the enforced budget window (null when unknown). */
  periodSpendUsd: MaybeRefOrGetter<number | null | undefined>;
  /** Spend used when the window spend is unknown (e.g. selected-range total). */
  fallbackSpendUsd?: MaybeRefOrGetter<number | null | undefined>;
}

/**
 * Shared budget-window display state for the API-key list cell, list item,
 * and usage sheet: exceed flag, ratio, period label, title, and bar color.
 * Keeps the three views' progress bars and labels from drifting apart.
 */
export function useBudgetDisplay(input: BudgetDisplayInput, t: TFn) {
  const budgetUsd = computed(() => toValue(input.budgetUsd) ?? null);
  const budgetPeriod = computed(() => toValue(input.budgetPeriod) ?? null);
  const budgetResetDay = computed(() => toValue(input.budgetResetDay) ?? null);
  const periodSpendUsd = computed(() => toValue(input.periodSpendUsd) ?? null);
  const fallbackSpendUsd = computed(() => toValue(input.fallbackSpendUsd) ?? null);

  const hasBudget = computed(() => budgetUsd.value !== null);

  /** Spend counted against the budget; falls back when the window spend is unknown. */
  const budgetSpend = computed(() => periodSpendUsd.value ?? fallbackSpendUsd.value ?? 0);

  const isBudgetExceeded = computed(
    () =>
      budgetUsd.value !== null &&
      periodSpendUsd.value !== null &&
      periodSpendUsd.value >= budgetUsd.value
  );

  const budgetRatio = computed(() => {
    const budget = budgetUsd.value;
    if (budget === null) return null;
    const spend = periodSpendUsd.value ?? fallbackSpendUsd.value;
    if (spend === null) return null;
    if (budget <= 0) return 0;
    return Math.min(1, spend / budget);
  });

  const budgetPeriodLabel = computed(() => {
    const period = budgetPeriod.value;
    if (period === "daily") return t("apiKeys.periodDaily");
    if (period === "weekly") return t("apiKeys.periodWeekly");
    if (period === "monthly") {
      return budgetResetDay.value && budgetResetDay.value !== 1
        ? t("apiKeys.periodMonthlyOnDay", { day: budgetResetDay.value })
        : t("apiKeys.periodMonthly");
    }
    // A period-less budget is a lifetime cap on cumulative spend.
    return hasBudget.value ? t("apiKeys.periodNone") : "";
  });

  const spendTitle = computed(() => {
    if (!hasBudget.value) return t("apiKeys.spend");
    return budgetPeriod.value
      ? t("apiKeys.spendWindowTitle", { period: budgetPeriodLabel.value })
      : t("apiKeys.spendLifetimeTitle");
  });

  const barClass = computed(() =>
    budgetRatio.value !== null && budgetRatio.value >= 1
      ? "bg-destructive"
      : budgetRatio.value !== null && budgetRatio.value >= 0.9
        ? "bg-status-warning"
        : "bg-primary"
  );

  return {
    hasBudget,
    budgetSpend,
    isBudgetExceeded,
    budgetRatio,
    budgetPeriodLabel,
    spendTitle,
    barClass,
  };
}
