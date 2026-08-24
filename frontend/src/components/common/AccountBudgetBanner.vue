<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { Wallet } from "@lucide/vue";
import { useBudgetDisplay } from "@/composables/useBudgetDisplay";
import type { MeBudget } from "@/services/api/me";
import { formatCost } from "@/utils/format";

interface Props {
  budget: MeBudget;
}

const props = defineProps<Props>();

const { t } = useI18n();

const { isBudgetExceeded, budgetRatio, budgetPeriodLabel, spendTitle, barClass } = useBudgetDisplay(
  {
    budgetUsd: computed(() => props.budget.budget_usd),
    budgetPeriod: computed(() => props.budget.budget_period),
    budgetResetDay: computed(() => props.budget.budget_reset_day),
    periodSpendUsd: computed(() => props.budget.period_spend_usd),
  },
  t
);
</script>

<template>
  <!-- Account-level budget envelope (admin-set): caps total spend across all
       of the user's keys. Only rendered when a budget is configured. -->
  <div
    v-if="budget.budget_usd !== null"
    class="flex items-center gap-3 rounded-lg border px-4 py-3"
    :class="
      isBudgetExceeded ? 'border-destructive/50 bg-destructive/5' : 'border-border/60 bg-muted/20'
    "
    :title="t('apiKeys.accountBudgetHelp')"
  >
    <Wallet class="h-4 w-4 shrink-0 text-muted-foreground" />
    <div class="min-w-0 flex-1">
      <div class="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <span class="text-xs font-medium">{{ t("apiKeys.accountBudget") }}</span>
        <span class="text-data text-muted-foreground" :title="spendTitle">
          <span :class="{ 'text-destructive font-medium': isBudgetExceeded }">
            {{ formatCost(budget.period_spend_usd) }}
          </span>
          / {{ formatCost(budget.budget_usd) }}
        </span>
        <span class="text-[10px] text-muted-foreground/80">{{ budgetPeriodLabel }}</span>
      </div>
      <div
        class="mt-1.5 h-1.5 w-full max-w-48 rounded-full bg-muted overflow-hidden"
        role="progressbar"
        :aria-label="spendTitle"
        :aria-valuenow="Math.round((budgetRatio ?? 0) * 100)"
        aria-valuemin="0"
        aria-valuemax="100"
      >
        <div
          class="h-full rounded-full transition-all"
          :class="barClass"
          :style="{ width: `${(budgetRatio ?? 0) * 100}%` }"
        />
      </div>
    </div>
  </div>
</template>
