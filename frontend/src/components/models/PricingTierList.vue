<script setup lang="ts">
import { computed } from "vue";
import { useI18n } from "vue-i18n";
import { tierThresholdLabel } from "@/utils/pricingTiers";
import type { PricingTier } from "@/types/schemas";

/**
 * Read-only tier breakdown: one row per threshold showing the IN/OUT rates
 * (plus cached read when the tier sets one) that apply from it.
 *
 * Shared by the model-default and per-provider sections of the pricing
 * popover, which render identical rows one density step apart.
 */
defineOptions({ name: "PricingTierList" });

const props = withDefaults(
  defineProps<{
    tiers: PricingTier[];
    /** Nested sections step the heading down: 10px/mb-1 instead of 11px/mb-1.5. */
    dense?: boolean;
  }>(),
  { dense: false }
);

const { t } = useI18n();

const sectionClass = computed(() => (props.dense ? "mt-2" : "mt-2.5"));
const headingClass = computed(() => (props.dense ? "mb-1 text-[10px]" : "mb-1.5 text-[11px]"));

function formatCost(v: number): string {
  // Trim to at most 6 decimal places without trailing zeros; very small
  // non-zero rates fall back to exponent form so they never render as "$0".
  const formatted = Number.parseFloat(v.toFixed(6));
  if (formatted === 0 && v > 0) return `$${v.toExponential(2)}`;
  return `$${formatted}`;
}

function fmtRate(v: number | null | undefined): string {
  return v == null ? "—" : formatCost(v);
}

/** IN/OUT (+ cached read when set) summary for one tier. */
function tierText(tier: PricingTier): string {
  const parts = [
    `${t("models.inputShort")} ${fmtRate(tier.input_cost_per_1m)}`,
    `${t("models.outputShort")} ${fmtRate(tier.output_cost_per_1m)}`,
  ];
  if (tier.cached_read_cost_per_1m != null) {
    parts.push(`${t("models.cachedShort")} ${formatCost(tier.cached_read_cost_per_1m)}`);
  }
  return parts.join(" · ");
}
</script>

<template>
  <div :class="sectionClass">
    <p class="font-semibold uppercase tracking-wider text-muted-foreground" :class="headingClass">
      {{ t("models.pricingTiersSection") }}
    </p>
    <div class="divide-y divide-border/40">
      <div
        v-for="tier in tiers"
        :key="tier.threshold"
        class="flex items-baseline justify-between gap-3 py-1"
      >
        <span class="text-xs text-muted-foreground">
          {{ tierThresholdLabel(t, tier.threshold) }}
        </span>
        <span class="text-data text-xs text-foreground">{{ tierText(tier) }}</span>
      </div>
    </div>
  </div>
</template>
