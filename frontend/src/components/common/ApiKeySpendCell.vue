<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { useBudgetDisplay } from "@/composables/useBudgetDisplay";
import type { ApiKeySpendSummary } from "@/services/api/apiKeys";
import { ABSENT_METRIC, formatCostWithPrecision } from "@/utils/format";

interface Props {
  spend?: ApiKeySpendSummary;
}

const props = withDefaults(defineProps<Props>(), {
  spend: undefined,
});

const { t, locale } = useI18n();

const { isBudgetExceeded, budgetRatio, budgetPeriodLabel, spendTitle, barClass } = useBudgetDisplay(
  {
    budgetUsd: computed(() => props.spend?.budget_usd ?? null),
    budgetPeriod: computed(() => props.spend?.budget_period ?? null),
    budgetResetDay: computed(() => props.spend?.budget_reset_day ?? null),
    periodSpendUsd: computed(() => props.spend?.period_spend_usd ?? null),
  },
  t
);
</script>

<template>
  <div v-if="spend && spend.budget_usd !== null" class="min-w-28" :title="spendTitle">
    <div class="flex items-center gap-1.5 text-data text-muted-foreground">
      <!-- Unknown window spend is not zero: the cell has no range total to fall
           back to, so it says it does not know instead of claiming $0.00 spent
           against a limit that is being enforced. -->
      <span :class="{ 'text-destructive font-medium': isBudgetExceeded }">
        {{
          spend.period_spend_usd === null
            ? ABSENT_METRIC
            : formatCostWithPrecision(spend.period_spend_usd, 2, locale)
        }}
      </span>
      <span>/ {{ formatCostWithPrecision(spend.budget_usd, 2, locale) }}</span>
    </div>
    <div class="mt-1 flex items-center gap-2">
      <BudgetMeter class="w-24" :ratio="budgetRatio" :label="spendTitle" :bar-class="barClass" />
      <span class="text-[10px] text-muted-foreground">
        {{ budgetPeriodLabel }}
      </span>
    </div>
  </div>
  <span v-else-if="spend" class="text-data text-muted-foreground">
    {{ formatCostWithPrecision(spend.total_spend_usd, 2, locale) }}
  </span>
  <span v-else class="text-data text-muted-foreground">{{ ABSENT_METRIC }}</span>
</template>
