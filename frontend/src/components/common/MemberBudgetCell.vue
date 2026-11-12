<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { formatBudgetAmount, useBudgetDisplay } from "@/composables/useBudgetDisplay";
import type { TeamMember } from "@/services/api/team";
import BudgetMeter from "@/components/common/BudgetMeter.vue";

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
        {{ formatBudgetAmount(member.budget_spend_usd, locale) }}
      </span>
      <span>/ {{ formatBudgetAmount(member.budget_usd, locale) }}</span>
    </div>
    <div class="mt-1 flex items-center gap-2">
      <BudgetMeter class="w-24" :ratio="budgetRatio" :label="spendTitle" :bar-class="barClass" />
      <span class="text-[10px] text-muted-foreground">
        {{ budgetPeriodLabel }}
      </span>
    </div>
  </div>
  <span v-else class="text-data text-muted-foreground">{{ t("team.budgetUnlimited") }}</span>
</template>
