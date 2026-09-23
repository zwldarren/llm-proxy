<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { useBudgetDisplay } from "@/composables/useBudgetDisplay";
import type { TeamMember } from "@/services/api/team";
import { formatCostWithPrecision } from "@/utils/format";

interface Props {
  member: TeamMember;
}

const props = defineProps<Props>();

const { t, locale } = useI18n();

// Mirrors ApiKeySpendCell, but for the member's account-level budget envelope:
// spend aggregates all of the member's keys within the current window.
const { isBudgetExceeded, budgetRatio, budgetPeriodLabel, spendTitle, barClass } = useBudgetDisplay(
  {
    budgetUsd: computed(() => props.member.budget_usd),
    budgetPeriod: computed(() => props.member.budget_period),
    budgetResetDay: computed(() => props.member.budget_reset_day),
    periodSpendUsd: computed(() => props.member.budget_spend_usd),
  },
  t
);
</script>

<template>
  <div v-if="member.budget_usd !== null" class="min-w-28" :title="spendTitle">
    <div class="flex items-center gap-1.5 text-data text-muted-foreground">
      <span :class="{ 'text-destructive font-medium': isBudgetExceeded }">
        {{ formatCostWithPrecision(member.budget_spend_usd ?? 0, 2, locale) }}
      </span>
      <span>/ {{ formatCostWithPrecision(member.budget_usd, 2, locale) }}</span>
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
          class="h-full w-full origin-left rounded-full transition-transform"
          :class="barClass"
          :style="{ transform: `scaleX(${budgetRatio ?? 0})` }"
        />
      </div>
      <span class="text-[10px] text-muted-foreground">
        {{ budgetPeriodLabel }}
      </span>
    </div>
  </div>
  <span v-else class="text-data text-muted-foreground">{{ t("team.budgetUnlimited") }}</span>
</template>
