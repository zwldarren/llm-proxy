<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { useBudgetDisplay } from "@/composables/useBudgetDisplay";
import type { ApiKeySpendSummary } from "@/services/api/apiKeys";
import { formatCost } from "@/utils/format";

interface Props {
  spend?: ApiKeySpendSummary;
}

const props = withDefaults(defineProps<Props>(), {
  spend: undefined,
});

const { t } = useI18n();

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
      <span :class="{ 'text-destructive font-medium': isBudgetExceeded }">
        {{ formatCost(spend.period_spend_usd) }}
      </span>
      <span>/ {{ formatCost(spend.budget_usd) }}</span>
    </div>
    <div class="mt-1 flex items-center gap-2">
      <div
        class="h-1.5 w-24 rounded-full bg-muted overflow-hidden"
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
      <span class="text-[10px] text-muted-foreground/80">
        {{ budgetPeriodLabel }}
      </span>
    </div>
  </div>
  <span v-else-if="spend" class="text-data text-muted-foreground">
    {{ formatCost(spend.total_spend_usd) }}
  </span>
  <span v-else class="text-data text-muted-foreground">-</span>
</template>
